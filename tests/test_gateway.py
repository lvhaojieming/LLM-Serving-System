import json
import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from heteroserve.gateway import create_app, validate
from heteroserve.routing.runtime import RoutingDecision, RouterSettings


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "deploy/gateway.json").read_text())


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
        if stream:
            return httpx.Response(200, content=b'data: {"model":"backend","choices":[]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"model": "backend", "choices": []})
    runtime = Runtime()
    app = create_app(config(), httpx.MockTransport(backend), runtime)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "stream": stream, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200 and response.headers["x-moqe-expert"] == "awq"
        assert response.headers["x-moqe-input-tokens"] == "28" and runtime.calls == 1
        assert app.state.gateway["inflight"] == 0
        assert '[DONE]' in response.text if stream else response.json()["model"] == "auto"
        # Preserve checkpoint policy: selected AWQ even with GPTQ probability > AWQ.


def test_undeployed_predicted_expert_is_explicit_and_never_silently_remapped():
    def backend(request):
        pytest.fail("undeployed pool must not be dispatched")
    app = create_app(config(), httpx.MockTransport(backend), Runtime("gptq"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 503 and "fallback is disabled" in response.text
        assert app.state.gateway["inflight"] == 0


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
