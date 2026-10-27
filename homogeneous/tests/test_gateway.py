import builtins
import json
import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from heteroserve.gateway import create_app, validate
from heteroserve.routing.runtime import RoutingDecision, RouterSettings


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "deploy/gateway.json").read_text())


def test_upstream_connection_pool_scales_with_the_pipeline_budget():
    cfg=config();cfg.update(max_inflight=640,max_upstream_connections=672,upstream_keepalive_seconds=2)
    app=create_app(cfg,httpx.MockTransport(lambda request:httpx.Response(200)),Runtime())
    with TestClient(app):
        assert app.state.upstream_limits.max_connections==672
        assert app.state.upstream_limits.max_keepalive_connections==640
        assert app.state.upstream_limits.keepalive_expiry==2
    cfg['max_upstream_connections']=100
    with pytest.raises(ValueError,match='connection pool'):validate(cfg)


class Runtime:
    chat_template = "canonical-template"
    def __init__(self, expert="awq"):
        self.calls = 0
        self.expert = expert
    def route(self, messages, generation, kwargs):
        self.calls += 1
        if messages[0]["content"] == "too long":
            raise ValueError("prompt exceeds limit; no truncation")
        return RoutingDecision(self.expert, {"awq": .3, "gptq": .7}, 28, 7.)


@pytest.mark.parametrize("stream", [False, True])
def test_auto_invokes_router_once_and_uses_pool_service_with_canonical_template(stream):
    def backend(request):
        assert request.url.host == "awq-ascend910b-vllm-router.heteroserve.svc.cluster.local"
        payload = json.loads(request.content)
        assert payload["model"] == "awq" and payload["chat_template"] == "canonical-template"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        headers = {"x-heteroserve-worker-headers-ms": "1", "x-heteroserve-l2-core-headers-ms": "2"}
        if stream:
            return httpx.Response(200, content=b'data: {"model":"backend","choices":[]}\n\ndata: [DONE]\n\n', headers=headers)
        return httpx.Response(200, json={"model": "backend", "choices": []}, headers=headers)
    runtime = Runtime()
    app = create_app(config(), httpx.MockTransport(backend), runtime)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "stream": stream, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == "awq"
        assert response.headers["x-moqe-input-tokens"] == "28" and runtime.calls == 1
        assert response.headers['x-heteroserve-worker-headers-ms'] == '1'
        assert response.headers['x-heteroserve-l2-core-headers-ms'] == '2'
        assert float(response.headers['x-heteroserve-l1-rpc-ms']) >= 0
        assert float(response.headers['x-heteroserve-gateway-headers-ms']) >= 0
        assert app.state.gateway["inflight"] == 0
        assert '[DONE]' in response.text if stream else response.json()["model"] == "auto"
        # Preserve checkpoint policy: selected AWQ even with GPTQ probability > AWQ.


def test_undeployed_predicted_expert_is_explicit_and_never_silently_remapped():
    def backend(request):
        pytest.fail("undeployed pool must not be dispatched")
    cfg = config()
    cfg["pools"]["gptq"]["enabled"] = False
    cfg["pools"]["gptq"]["base_url"] = None
    app = create_app(cfg, httpx.MockTransport(backend), Runtime("gptq"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 503 and "fallback is disabled" in response.text
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("stream", [False, True])
def test_auto_selected_gptq_uses_gptq_pool(stream):
    def backend(request):
        assert request.url.host == "gptq-ascend910b-vllm-router.heteroserve.svc.cluster.local"
        assert json.loads(request.content)["model"] == "gptq"
        if stream:
            return httpx.Response(200, content=b'data: {"choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "backend", "choices": []})
    runtime = Runtime("gptq")
    app = create_app(config(), httpx.MockTransport(backend), runtime)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "stream": stream, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == "gptq"
        assert runtime.calls == 1 and app.state.gateway["inflight"] == 0


def test_ready_coverage_requires_both_experts_healthy():
    def backend(request):
        return httpx.Response(200 if request.url.host.startswith("awq-") else 503)
    app = create_app(config(), httpx.MockTransport(backend), Runtime())
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready_experts"] == ["awq"]
        assert response.json()["auto_expert_coverage_complete"] is False


@pytest.mark.parametrize("payload", [[], {"model": []}, {"model": "auto", "messages": []},
    {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "max_tokens": True},
    {"model": "auto", "messages": [{"role": "user", "content": "hello"}], "chat_template": "override"},
    {"model": "auto", "messages": [{"role": "user", "content": "too long"}]}])
def test_invalid_requests_release_admission_without_backend_call(payload):
    def backend(request):
        pytest.fail("invalid request must not reach backend")
    app = create_app(config(), httpx.MockTransport(backend), Runtime())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert app.state.gateway["inflight"] == 0


def test_downstream_capacity_rejection_is_preserved_and_fixed_expert_skips_learned_router():
    runtime = Runtime()
    app = create_app(config(), httpx.MockTransport(lambda request: httpx.Response(429)), runtime)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 429 and runtime.calls == 0 and app.state.gateway["inflight"] == 0


def test_checkpoint_hash_and_distinct_mapping_checked_before_loading_npu(tmp_path):
    cfg = config()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"changed")
    cfg["router"]["checkpoint"] = str(path)
    with pytest.raises(ValueError, match="checksum"):
        validate(cfg).verify_checkpoint()
    cfg["router"]["expert_mapping"]["qwen3-14b/gptq-w4a16/v1"] = "awq"
    with pytest.raises(ValueError, match="distinct"):
        validate(cfg)


def explicit_config():
    return {"learned_routing": {"enabled": False}, "max_inflight": 4,
            "pools": {name: {"base_url": f"http://{name}/v1", "model": "Qwen3-14B-AWQ",
                             "health_path": "/health", "verify_model_identity": True}
                      for name in ("ascend", "musa", "iluvatar")}}


def test_explicit_gateway_starts_without_router_assets_or_accelerator_imports(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "torch_npu"} or name.endswith(("routing.ascend", "routing.client")):
            pytest.fail("disabled learned routing must not load a local or remote runtime")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cfg = explicit_config()
    # A retained L1 address must not cause an RPC client to be initialized.
    cfg["router_service"] = {"base_url": "http://unused-l1"}
    def backend(request):
        pytest.fail("startup and model listing must not send downstream requests")
    app = create_app(cfg, httpx.MockTransport(backend))
    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert {row["id"] for row in client.get("/v1/models").json()["data"]} == {"ascend", "musa", "iluvatar"}
        assert app.state.runtime is None
        assert app.state.gateway["router_calls"] == 0


@pytest.mark.parametrize("model", ["ascend", "musa", "iluvatar"])
@pytest.mark.parametrize("stream", [False, True])
def test_explicit_vendor_forwarding_preserves_payload_and_rewrites_response_model(model, stream):
    def backend(request):
        assert request.url.host == model and request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload == {"model": "Qwen3-14B-AWQ", "messages": [{"role": "user", "content": "hello"}],
                           "stream": stream, "max_tokens": 12, "temperature": 0}
        assert request.headers["x-request-id"]
        if stream:
            return httpx.Response(200, content=b'data: {"model":"Qwen3-14B-AWQ","choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "Qwen3-14B-AWQ", "choices": [{"message": {"content": "hello"}}]})
    app = create_app(explicit_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hello"}],
                                                           "stream": stream, "max_tokens": 12, "temperature": 0})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == model
        assert "x-moqe-router-ms" not in response.headers
        if stream:
            assert response.headers["content-type"].startswith("text/event-stream")
            events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert events[-1] == "[DONE]"
            assert json.loads(events[0])["model"] == model
        else:
            assert response.json()["model"] == model
        assert app.state.gateway["inflight"] == 0 and app.state.gateway["router_calls"] == 0


def test_explicit_readiness_checks_health_and_model_identity_with_complete_coverage():
    calls = []
    def backend(request):
        calls.append((request.url.host, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
    cfg = explicit_config()
    cfg["pools"]["ascend"]["base_url"] += "/"
    with TestClient(create_app(cfg, httpx.MockTransport(backend))) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        ready = response.json()
        assert set(ready["ready_experts"]) == {"ascend", "musa", "iluvatar"}
        assert ready["expert_coverage_complete"] is True
        assert ready["learned_routing_enabled"] is False and ready["l1_router_ready"] is None
        assert ready["auto_expert_coverage_complete"] is False
    assert len(calls) == 6


@pytest.mark.parametrize("bad_identity", [
    {"data": [{"id": "Qwen3-14B"}]}, {"data": []}, {"data": None}, {"data": [None]}, [], "not-json",
])
def test_identity_mismatch_or_malformed_response_excludes_only_that_pool(bad_identity):
    def backend(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.host == "musa":
            return httpx.Response(200, content=bad_identity) if isinstance(bad_identity, str) else httpx.Response(200, json=bad_identity)
        return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
    with TestClient(create_app(explicit_config(), httpx.MockTransport(backend))) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready_experts"] == ["ascend", "iluvatar"]
        assert response.json()["expert_coverage_complete"] is False


def test_identity_check_is_opt_in_and_health_path_defaults_to_ready():
    cfg = explicit_config()
    for pool in cfg["pools"].values():
        del pool["health_path"], pool["verify_model_identity"]
    calls = []
    def backend(request):
        calls.append(request.url.path)
        assert request.url.path == "/ready"
        return httpx.Response(200)
    with TestClient(create_app(cfg, httpx.MockTransport(backend))) as client:
        assert client.get("/ready").json()["expert_coverage_complete"] is True
    assert len(calls) == 3


@pytest.mark.parametrize("failure", ["unhealthy", "identity_unavailable", "timeout"])
def test_no_healthy_identity_verified_pool_is_not_ready(failure):
    def backend(request):
        if failure == "timeout":
            raise httpx.ConnectTimeout("unavailable", request=request)
        return httpx.Response(200 if failure == "identity_unavailable" and request.url.path == "/health" else 503)
    with TestClient(create_app(explicit_config(), httpx.MockTransport(backend))) as client:
        assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("model", ["auto", "unknown", None, []])
def test_explicit_gateway_rejects_auto_and_unknown_models_without_dispatch(model):
    def backend(request):
        pytest.fail("unsupported model must not reach a backend")
    app = create_app(explicit_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 400 and app.state.gateway["inflight"] == 0


def test_disabled_explicit_pool_is_hidden_and_never_falls_back():
    cfg = explicit_config()
    cfg["pools"]["musa"].update(enabled=False, base_url=None)
    def backend(request):
        pytest.fail("disabled pool must not dispatch or fall back")
    app = create_app(cfg, httpx.MockTransport(backend))
    with TestClient(app) as client:
        assert {row["id"] for row in client.get("/v1/models").json()["data"]} == {"ascend", "iluvatar"}
        response = client.post("/v1/chat/completions", json={"model": "musa", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 503 and app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("error,status", [(httpx.ReadTimeout, 504), (httpx.ConnectTimeout, 504), (httpx.ConnectError, 502)])
def test_transport_errors_have_distinct_status_and_release_admission(error, status):
    def backend(request):
        raise error("backend unavailable", request=request)
    app = create_app(explicit_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "ascend", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == status and app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("status", [429, 500, 503])
def test_explicit_upstream_errors_preserve_status_retry_after_and_close_response(status):
    closed = []
    class ErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error":"overloaded"}'
        async def aclose(self):
            closed.append(True)
    app = create_app(explicit_config(), httpx.MockTransport(lambda request: httpx.Response(
        status, headers={"retry-after": "7"}, stream=ErrorStream())))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "ascend", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == status and response.headers["retry-after"] == "7"
        assert response.json()["error"]["upstream_status"] == status
        assert app.state.gateway["inflight"] == 0 and closed == [True]


def test_body_read_timeout_closes_upstream_and_releases_admission():
    closed = []
    class TimedOutStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadTimeout("response body stalled")
            yield b""  # Keep this an async iterator.
        async def aclose(self):
            closed.append(True)
    app = create_app(explicit_config(), httpx.MockTransport(lambda request: httpx.Response(200, stream=TimedOutStream())))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "ascend", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 504 and app.state.gateway["inflight"] == 0 and closed == [True]


def test_learned_routing_does_not_accept_unmapped_hardware_aliases():
    cfg = config()
    cfg["pools"]["musa"] = explicit_config()["pools"]["musa"]
    with pytest.raises(ValueError, match="checkpoint expert"):
        validate(cfg)


@pytest.mark.parametrize("learned", [None, False, {"enabled": "false"}, {"enabled": 0}])
def test_invalid_learned_mode_does_not_silently_disable_or_load_runtime(learned):
    cfg = explicit_config()
    cfg["learned_routing"] = learned
    with pytest.raises(ValueError, match="learned_routing.enabled"):
        validate(cfg)


@pytest.mark.parametrize("value", ["hello", ["hello", "world"], [123, 456], [[123], [456]]])
def test_embedding_endpoint_preserves_input_dimensions_and_rewrites_model(value):
    cfg = explicit_config()
    cfg["pools"]["embedding"] = dict(base_url="http://embedding/v1", model="Qwen3-Embedding-0.6B", task="embed")
    def backend(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/embeddings"
        assert body == dict(model="Qwen3-Embedding-0.6B", input=value, dimensions=64)
        return httpx.Response(200, json={"model": body["model"], "data": [{"embedding": [0.5] * 64}]})
    app = create_app(cfg, httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json=dict(model="embedding", input=value, dimensions=64))
        assert response.status_code == 200 and response.json()["model"] == "embedding"
        assert len(response.json()["data"][0]["embedding"]) == 64
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("endpoint,payload", [
    ("embeddings", {"model": "ascend", "input": "hello"}),
    ("chat/completions", {"model": "embedding", "messages": [{"role": "user", "content": "hi"}]}),
    ("embeddings", {"model": "embedding", "input": "hello", "stream": True}),
    ("embeddings", {"model": "embedding", "input": []}),
])
def test_task_mismatch_and_invalid_embedding_requests_do_not_dispatch(endpoint, payload):
    cfg = explicit_config()
    cfg["pools"]["embedding"] = dict(base_url="http://embedding/v1", model="embedding", task="embed")
    app = create_app(cfg, httpx.MockTransport(lambda request: pytest.fail("invalid request dispatched")))
    with TestClient(app) as client:
        assert client.post("/v1/" + endpoint, json=payload).status_code == 400
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize('healthy,complete', [(1,False),(2,True),(3,False)])
def test_readiness_requires_exact_healthy_replica_registry(healthy, complete):
    cfg = explicit_config()
    urls = ['http://w1:8000','http://w2:8000','http://unexpected:8000']
    cfg['worker_registry'] = {'url':'http://router/workers','expected_urls':urls[:2]}
    def backend(request):
        if request.url.path == '/workers':
            return httpx.Response(200,json={'workers':[{'url':u,'is_healthy':True} for u in urls[:healthy]]})
        if request.url.path == '/v1/models':
            return httpx.Response(200,json={'data':[{'id':'Qwen3-14B-AWQ'}]})
        return httpx.Response(200)
    with TestClient(create_app(cfg,httpx.MockTransport(backend))) as client:
        ready = client.get('/ready').json()
        assert ready['expert_coverage_complete'] is complete
        assert ready['worker_coverage_complete'] is complete and ready['expected_workers']==2


def random_config():
    cfg = explicit_config()
    cfg["random_routes"] = {"awq": ["ascend", "musa", "iluvatar"]}
    cfg["pools"]["embedding"] = dict(base_url="http://embedding/v1", model="Qwen3-Embedding-0.6B", task="embed")
    return cfg


@pytest.mark.parametrize("stream", [False, True])
def test_random_route_selects_equal_pool_candidates_and_preserves_request_response(monkeypatch, stream):
    chosen = []
    checks = []
    names = ["ascend", "musa", "iluvatar"]
    def choose(candidates):
        # Each pool has one entry regardless of how many workers its L2 Router owns.
        assert candidates == names
        selected = names[len(chosen) % len(names)]
        chosen.append(selected)
        return selected
    monkeypatch.setattr("heteroserve.gateway.random.choice", choose)
    def backend(request):
        if request.method == "GET":
            checks.append((request.url.host, request.url.path))
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
            return httpx.Response(200)
        assert request.url.host == chosen[-1]
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["x-request-id"]
        payload = json.loads(request.content)
        assert payload == {"model": "Qwen3-14B-AWQ", "messages": [{"role": "user", "content": "hello"}],
                           "stream": stream, "temperature": 0, "max_tokens": 12}
        if stream:
            return httpx.Response(200, content=b'data: {"model":"Qwen3-14B-AWQ","choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "Qwen3-14B-AWQ", "choices": []})
    app = create_app(random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        assert "awq" in {row["id"] for row in client.get("/v1/models").json()["data"]}
        for index in range(6):
            response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hello"}],
                                                                              "stream": stream, "temperature": 0, "max_tokens": 12})
            assert response.status_code == 200 and response.headers["x-moqe-expert"] == names[index % 3]
            if stream:
                events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
                assert events[-1] == "[DONE]" and json.loads(events[0])["model"] == "awq"
            else:
                assert response.json()["model"] == "awq"
        assert app.state.gateway["inflight"] == 0 and app.state.gateway["router_calls"] == 6
        # Health results are reused during the configured 5-second TTL.
        assert len(checks) == 6


def test_random_selection_excludes_disabled_unhealthy_and_wrong_identity_pools(monkeypatch):
    cfg = random_config()
    cfg["pools"]["disabled"] = dict(base_url=None, model="Qwen3-14B-AWQ", enabled=False)
    cfg["random_routes"]["awq"].append("disabled")
    def choose(candidates):
        assert candidates == ["ascend"]
        return candidates[0]
    monkeypatch.setattr("heteroserve.gateway.random.choice", choose)
    def backend(request):
        assert request.url.host != "disabled"
        if request.method == "POST":
            assert request.url.host == "ascend"
            return httpx.Response(200, json={"model": "backend"})
        if request.url.host == "iluvatar":
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "wrong" if request.url.host == "musa" else "Qwen3-14B-AWQ"}]})
    with TestClient(create_app(cfg, httpx.MockTransport(backend))) as client:
        ready = client.get("/ready").json()
        assert ready["ready_random_routes"] == ["awq"]
        assert ready["random_route_pools"] == {"awq": ["ascend"]}
        assert ready["random_routing_enabled"] is True
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == "ascend"


def test_random_route_without_healthy_generation_pool_returns_503_even_if_embedding_is_healthy(monkeypatch):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda _: pytest.fail("no healthy candidates"))
    def backend(request):
        assert request.method == "GET"
        return httpx.Response(200 if request.url.host == "embedding" else 503)
    app = create_app(random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        ready = client.get("/ready").json()
        assert ready["ready_random_routes"] == []
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 503 and "no ready pool" in response.text
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("failure", [429, 503, httpx.ConnectError, httpx.ReadTimeout])
def test_random_selected_pool_failure_is_not_retried_and_preserves_attribution(monkeypatch, failure):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda candidates: "musa")
    posts = []
    def backend(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
        posts.append(request.url.host)
        if isinstance(failure, type):
            raise failure("unavailable", request=request)
        return httpx.Response(failure, headers={"retry-after": "3"})
    app = create_app(random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        expected = 502 if failure is httpx.ConnectError else 504 if failure is httpx.ReadTimeout else failure
        assert response.status_code == expected
        assert response.headers["x-moqe-expert"] == "musa" and response.headers["x-request-id"]
        assert posts == ["musa"] and app.state.gateway["inflight"] == 0
        if not isinstance(failure, type):
            assert response.headers["retry-after"] == "3"


def test_explicit_vendor_and_embedding_requests_bypass_random_routing(monkeypatch):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda _: pytest.fail("explicit model must bypass random routing"))
    def backend(request):
        assert request.method == "POST"
        assert request.url.host in {"ascend", "embedding"}
        return httpx.Response(200, json={"model": "backend", "data": []})
    app = create_app(embedded_random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"model": "ascend", "messages": [{"role": "user", "content": "hi"}]}).status_code == 200
        assert client.post("/v1/embeddings", json={"model": "embedding", "input": "hello"}).status_code == 200
        assert client.post("/v1/embeddings", json={"model": "awq", "input": "hello"}).status_code == 400
        assert app.state.gateway["router_calls"] == 0


@pytest.mark.parametrize("routes", [None, [], {"awq": []}, {"auto": ["ascend"]}, {"ascend": ["musa"]},
    {"awq": ["unknown"]}, {"awq": ["ascend", "ascend"]}, {"awq": ["embedding"]}, {"awq": [None]}])
def test_random_route_configuration_rejects_invalid_aliases_or_pools(routes):
    cfg = random_config()
    cfg["random_routes"] = routes
    with pytest.raises(ValueError, match="[Rr]andom"):
        validate(cfg)


@pytest.mark.parametrize("seconds", [True, -1, float("nan"), float("inf"), "5"])
def test_random_health_cache_requires_finite_nonnegative_seconds(seconds):
    cfg = random_config()
    cfg["random_route_health_seconds"] = seconds
    with pytest.raises(ValueError, match="random_route_health_seconds"):
        validate(cfg)


@pytest.mark.parametrize("failure", [None, "unhealthy", "registry_error", "malformed"])
def test_multiple_l2_worker_registries_are_aggregated_with_partial_failure_reported(failure):
    cfg = random_config()
    cfg["worker_registry"] = {"urls": ["http://ascend/workers", "http://musa/workers", "http://iluvatar/workers"],
                              "expected_urls": ["http://w-ascend:8000", "http://w-musa:8000", "http://w-iluvatar:8000"]}
    def backend(request):
        if request.url.path == "/workers":
            if request.url.host == "musa" and failure in {"registry_error", "malformed"}:
                return httpx.Response(503 if failure == "registry_error" else 200, json={"workers": None})
            return httpx.Response(200, json={"workers": [{"url": f"http://w-{request.url.host}:8000",
                                                         "is_healthy": not (failure == "unhealthy" and request.url.host == "musa")}]})
        return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
    with TestClient(create_app(cfg, httpx.MockTransport(backend))) as client:
        ready = client.get("/ready").json()
        assert ready["expected_workers"] == 3
        assert ready["healthy_workers"] == (3 if failure is None else 2)
        assert ready["worker_coverage_complete"] is (failure is None)


@pytest.mark.parametrize("urls", [None, [], "http://router/workers", ["https://router/workers"],
                                  ["http://router/workers", "http://router/workers"], [None]])
def test_worker_registry_urls_require_unique_http_endpoints(urls):
    cfg = random_config()
    cfg["worker_registry"] = {"urls": urls, "expected_urls": ["http://w1:8000"]}
    with pytest.raises(ValueError, match="worker_registry"):
        validate(cfg)


def embedded_random_config():
    cfg = random_config()
    cfg["route_embeddings"] = {"awq": "embedding"}
    return cfg


@pytest.mark.parametrize("stream", [False, True])
def test_random_route_completes_real_embedding_before_pool_choice_and_generation(monkeypatch, caplog, stream):
    events = []
    traces = []
    class EmbeddingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            events.append("embedding_body")
            yield b'{"data":[{"embedding":[0.25,-0.5,1.0]}]}'
        async def aclose(self):
            events.append("embedding_closed")
    def choose(candidates):
        assert events == ["embedding_request", "embedding_body", "embedding_closed"]
        assert candidates == ["ascend", "musa", "iluvatar"]
        events.append("random_choice")
        return "iluvatar"
    monkeypatch.setattr("heteroserve.gateway.random.choice", choose)
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "你好\n世界"}]
    def backend(request):
        if request.method == "GET":
            assert request.url.host != "embedding"
            return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
        assert app.state.gateway["inflight"] == 1
        traces.append(request.headers["x-request-id"])
        body = json.loads(request.content)
        if request.url.host == "embedding":
            assert request.url.path == "/v1/embeddings"
            assert body == {"model": "Qwen3-Embedding-0.6B", "input": "system: Be concise.\nuser: 你好\n世界", "encoding_format": "float"}
            events.append("embedding_request")
            return httpx.Response(200, stream=EmbeddingStream())
        assert request.url.host == "iluvatar" and events[-1] == "random_choice"
        assert body == {"model": "Qwen3-14B-AWQ", "messages": messages, "stream": stream, "max_tokens": 12}
        events.append("generation_request")
        if stream:
            return httpx.Response(200, content=b'data: {"model":"backend","choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "backend", "choices": []})
    app = create_app(embedded_random_config(), httpx.MockTransport(backend))
    caplog.set_level("INFO", logger="heteroserve.gateway")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": messages, "stream": stream, "max_tokens": 12})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == "iluvatar"
        assert float(response.headers["x-heteroserve-embedding-ms"]) >= 0
        assert traces == [response.headers["x-request-id"]] * 2
        assert app.state.gateway["inflight"] == 0 and app.state.gateway["router_calls"] == 1
        if stream:
            data = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert json.loads(data[0])["model"] == "awq" and data[-1] == "[DONE]"
        else:
            assert response.json()["model"] == "awq"
    finished = [json.loads(record.message) for record in caplog.records if record.name == "heteroserve.gateway"][-1]
    assert finished["embedding_pool"] == "embedding" and finished["embedding_ms"] >= 0


@pytest.mark.parametrize("failure", [429, 500, 503, httpx.ConnectError, httpx.ReadTimeout])
def test_embedding_failure_stops_before_random_choice_and_releases_admission(monkeypatch, failure):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda _: pytest.fail("embedding must succeed before choosing a pool"))
    closed = []
    class ErrorStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error":"unavailable"}'
        async def aclose(self):
            closed.append(True)
    def backend(request):
        assert request.url.host == "embedding" and request.url.path == "/v1/embeddings"
        if isinstance(failure, type):
            raise failure("unavailable", request=request)
        return httpx.Response(failure, headers={"retry-after": "7"}, stream=ErrorStream())
    app = create_app(embedded_random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        expected = 502 if failure is httpx.ConnectError else 504 if failure is httpx.ReadTimeout else failure
        assert response.status_code == expected
        assert response.headers["x-request-id"] and float(response.headers["x-heteroserve-embedding-ms"]) >= 0
        assert "x-moqe-expert" not in response.headers
        assert app.state.gateway["inflight"] == 0 and app.state.gateway["router_calls"] == 0
        if not isinstance(failure, type):
            assert response.headers["retry-after"] == "7" and closed == [True]


@pytest.mark.parametrize("body", [
    None, {}, [], {"data": []}, {"data": [None]}, {"data": [{"embedding": []}]},
    {"data": [{"embedding": [1]}, {"embedding": [2]}]}, {"data": [{"embedding": [True]}]},
    {"data": [{"embedding": ["0.1"]}]}, {"data": [{"embedding": [float("nan")]}]},
    {"data": [{"embedding": [float("inf")]}]}, {"data": [{"embedding": [10**400]}]}, "invalid-json",
])
def test_malformed_or_nonfinite_embedding_never_reaches_generation(monkeypatch, body):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda _: pytest.fail("invalid embedding must stop routing"))
    def backend(request):
        assert request.url.host == "embedding"
        content = body if isinstance(body, str) else json.dumps(body)
        return httpx.Response(200, content=content)
    app = create_app(embedded_random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 502 and "invalid vector" in response.text
        assert float(response.headers["x-heteroserve-embedding-ms"]) >= 0
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("status", [429, 503])
def test_generation_rejection_after_embedding_keeps_both_stage_attribution_and_time(monkeypatch, status):
    monkeypatch.setattr("heteroserve.gateway.random.choice", lambda _: "ascend")
    def backend(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
        if request.url.host == "embedding":
            return httpx.Response(200, json={"data": [{"embedding": [0.1]}]})
        return httpx.Response(status, headers={"retry-after": "2"})
    app = create_app(embedded_random_config(), httpx.MockTransport(backend))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == status and response.headers["x-moqe-expert"] == "ascend"
        assert float(response.headers["x-heteroserve-embedding-ms"]) >= 0 and response.headers["retry-after"] == "2"
        assert app.state.gateway["inflight"] == 0


@pytest.mark.parametrize("messages", [[None], [{"content": "hi"}], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]])
def test_embedding_route_requires_plain_text_messages_before_any_backend_call(messages):
    app = create_app(embedded_random_config(), httpx.MockTransport(lambda _: pytest.fail("invalid message reached backend")))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "awq", "messages": messages})
        assert response.status_code == 400 and app.state.gateway["inflight"] == 0


def test_embedded_random_alias_is_not_ready_when_embedding_dependency_is_unhealthy():
    def backend(request):
        if request.url.host == "embedding":
            return httpx.Response(503)
        return httpx.Response(200, json={"data": [{"id": "Qwen3-14B-AWQ"}]})
    with TestClient(create_app(embedded_random_config(), httpx.MockTransport(backend))) as client:
        ready = client.get("/ready").json()
        assert ready["random_route_pools"] == {"awq": ["ascend", "musa", "iluvatar"]}
        assert ready["route_embeddings"] == {"awq": "embedding"} and ready["ready_random_routes"] == []


@pytest.mark.parametrize("route_embeddings", [None, [], {"unknown": "embedding"}, {"awq": "unknown"},
    {"awq": "awq"}, {"awq": "ascend"}, {"awq": None}])
def test_route_embedding_configuration_requires_random_alias_and_embedding_target(route_embeddings):
    cfg = embedded_random_config()
    cfg["route_embeddings"] = route_embeddings
    with pytest.raises(ValueError, match="route_embeddings"):
        validate(cfg)


def test_route_embedding_configuration_rejects_disabled_embedding_pool():
    cfg = embedded_random_config()
    cfg["pools"]["embedding"].update(enabled=False, base_url=None)
    with pytest.raises(ValueError, match="route_embeddings"):
        validate(cfg)
