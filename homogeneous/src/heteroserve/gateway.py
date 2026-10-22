"""Explicit pool forwarding with optional learned expert selection.

Routing validation and response-model rewriting are migrated from
MLsys_inference. Kubernetes and the official L2 Router own replica lifecycle.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import random
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .routing.runtime import RouterSettings

log = logging.getLogger("heteroserve.gateway")


def validate(config):
    learned = config.get("learned_routing", {})
    if not isinstance(learned, dict) or type(learned.get("enabled", True)) is not bool:
        raise ValueError("learned_routing.enabled must be boolean")
    pools = config.get("pools")
    if not isinstance(pools, dict) or not pools or any(not isinstance(name, str) or not name or name == "auto" for name in pools):
        raise ValueError("Gateway requires named pools; auto is reserved for learned routing")
    settings = RouterSettings(**config["router"]) if learned.get("enabled", True) else None
    if settings is not None and set(settings.expert_mapping.values()) != set(pools):
        raise ValueError("Every checkpoint expert must have an explicit pool entry")
    for pool in pools.values():
        if not isinstance(pool, dict) or not isinstance(pool.get("model"), str) or not pool["model"]:
            raise ValueError("Every pool requires an expected model identity")
        if pool.get("enabled", True) and (not isinstance(pool.get("base_url"), str) or not pool["base_url"].startswith("http://")):
            raise ValueError("Enabled pools need an internal HTTP Service URL")
        health_path = pool.get("health_path", "/ready")
        if not isinstance(health_path, str) or not health_path.startswith("/") or health_path.startswith("//") or any(c in health_path for c in "?#"):
            raise ValueError("Pool health_path must be an absolute path on its configured service")
        if type(pool.get("verify_model_identity", False)) is not bool:
            raise ValueError("Pool verify_model_identity must be boolean")
        if pool.get("task", "generate") not in {"generate", "embed"}:
            raise ValueError("Pool task must be generate or embed")
    random_routes = config.get("random_routes", {})
    if not isinstance(random_routes, dict):
        raise ValueError("random_routes must map model aliases to named generation pools")
    for alias, names in random_routes.items():
        if (not isinstance(alias, str) or not alias or alias == "auto" or alias in pools
                or not isinstance(names, list) or not names
                or any(not isinstance(name, str) or name not in pools or pools[name].get("task", "generate") != "generate" for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Random routes require a distinct alias and unique named generation pools")
    route_embeddings = config.get("route_embeddings", {})
    if (not isinstance(route_embeddings, dict)
            or any(alias not in random_routes or not isinstance(name, str) or name not in pools
                   or pools[name].get("task", "generate") != "embed" or not pools[name].get("enabled", True)
                   for alias, name in route_embeddings.items())):
        raise ValueError("route_embeddings must map random route aliases to enabled embedding pools")
    health_seconds = config.get("random_route_health_seconds", 5)
    if (isinstance(health_seconds, bool) or not isinstance(health_seconds, (int, float))
            or not math.isfinite(health_seconds) or health_seconds < 0):
        raise ValueError("random_route_health_seconds must be a finite nonnegative number")
    if type(config.get("max_inflight", 16)) is not int or config.get("max_inflight", 16) < 1:
        raise ValueError("Gateway requires a positive request budget")
    registry = config.get("worker_registry")
    if registry is not None:
        if not isinstance(registry, dict):
            raise ValueError("worker_registry requires HTTP registry URLs and unique expected worker URLs")
        urls = registry.get("urls", [registry.get("url")])
        if ("url" in registry and "urls" in registry
                or not isinstance(urls, list) or not urls
                or any(not isinstance(url, str) or not url.startswith("http://") for url in urls)
                or len(set(urls)) != len(urls) or not isinstance(registry.get("expected_urls"), list)
                or not registry["expected_urls"] or any(not isinstance(u, str) or not u.startswith("http://") for u in registry["expected_urls"])
                or len(set(registry["expected_urls"])) != len(registry["expected_urls"])):
            raise ValueError("worker_registry requires HTTP registry URLs and unique expected worker URLs")
    return settings


def create_app(config, transport=None, router_runtime=None):
    settings = validate(config)
    learned_enabled = settings is not None
    state = {"inflight": 0, "draining": False, "router_calls": 0, "rejected": 0}
    runtime = router_runtime if learned_enabled else None
    routing_slot = asyncio.Semaphore(1)
    random_routes = config.get("random_routes", {})
    route_embeddings = config.get("route_embeddings", {})
    pool_health = {}
    health_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime
        if learned_enabled and runtime is None:
            if config.get("router_service"):
                from .routing.client import RemoteRouterClient
                runtime = RemoteRouterClient(settings, config["router_service"], httpx.AsyncClient(timeout=30, trust_env=False))
            else:
                from .routing.ascend import AscendRouterRuntime
                runtime = await asyncio.to_thread(AscendRouterRuntime, settings)
        async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(config.get("timeout_seconds", 180)), trust_env=False) as client:
            app.state.client = client
            app.state.runtime = runtime
            yield
            state["draining"] = True
            if getattr(runtime, "remote", False):
                await runtime.client.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.gateway = state

    async def check_pool(name, pool):
        if not pool.get("enabled", True):
            return None
        try:
            base_url = pool["base_url"].rstrip("/")
            response = await app.state.client.get(base_url.removesuffix("/v1") + pool.get("health_path", "/ready"), timeout=3)
            if response.status_code != 200:
                return None
            if pool.get("verify_model_identity", False):
                response = await app.state.client.get(base_url + "/models", timeout=3)
                if response.status_code != 200:
                    return None
                data = response.json()
                models = data.get("data") if isinstance(data, dict) else None
                if not isinstance(models, list) or not any(isinstance(model, dict) and model.get("id") == pool["model"] for model in models):
                    return None
            return name
        except (httpx.HTTPError, ValueError):
            return None

    async def healthy_pools(refresh=False, names=None):
        names = list(config["pools"]) if names is None else names
        async with health_lock:
            pending = [name for name in names if refresh or name not in pool_health
                       or time.monotonic() - pool_health[name][0] >= config.get("random_route_health_seconds", 5)]
            results = await asyncio.gather(*(check_pool(name, config["pools"][name]) for name in pending))
            checked_at = time.monotonic()
            pool_health.update({name: (checked_at, result is not None) for name, result in zip(pending, results)})
            return [name for name in names if pool_health[name][1]]

    @app.get("/live")
    async def live():
        return {"live": True}

    @app.get("/ready")
    async def ready():
        if state["draining"] or (learned_enabled and runtime is None):
            raise HTTPException(503, "gateway is unavailable")
        healthy = await healthy_pools(refresh=True)
        if healthy:
            l1_ready = (await runtime.ready() if getattr(runtime, "remote", False) else True) if learned_enabled else None
            worker_coverage = True
            expected_workers = healthy_workers = None
            if config.get("worker_registry"):
                registry = config["worker_registry"]
                expected = set(registry["expected_urls"])
                expected_workers = len(expected)
                healthy_workers = 0
                urls = registry.get("urls", [registry.get("url")])
                async def registry_workers(url):
                    try:
                        response = await app.state.client.get(url, timeout=3)
                        if response.status_code == 200:
                            rows = response.json().get("workers")
                            if isinstance(rows, list):
                                return {r["url"] for r in rows if isinstance(r, dict) and isinstance(r.get("url"), str) and r.get("is_healthy") is True}
                    except (httpx.HTTPError, ValueError, AttributeError, TypeError):
                        pass
                    return None
                registries = await asyncio.gather(*(registry_workers(url) for url in urls))
                actual = set().union(*(workers for workers in registries if workers is not None))
                healthy_workers = len(expected & actual)
                worker_coverage = all(workers is not None for workers in registries) and actual == expected
            coverage_complete = len(healthy) == len(config["pools"]) and worker_coverage
            eligible_routes = {alias: [name for name in names if name in healthy] for alias, names in random_routes.items()}
            return {"ready": True, "ready_experts": healthy,
                    "learned_routing_enabled": learned_enabled,
                    "random_routing_enabled": bool(random_routes),
                    "ready_random_routes": [alias for alias, names in eligible_routes.items()
                                            if names and (alias not in route_embeddings or route_embeddings[alias] in healthy)],
                    "random_route_pools": eligible_routes,
                    "route_embeddings": route_embeddings,
                    "expert_coverage_complete": coverage_complete,
                    "worker_coverage_complete": worker_coverage,
                    "expected_workers": expected_workers, "healthy_workers": healthy_workers,
                    "l1_router_ready": l1_ready,
                    "auto_expert_coverage_complete": learned_enabled and l1_ready and coverage_complete}
        raise HTTPException(503, "no ready downstream expert pool")

    @app.get("/v1/models")
    async def models():
        ids = (["auto"] if learned_enabled else []) + list(random_routes) + [key for key, pool in config["pools"].items() if pool.get("enabled", True)]
        return {"object": "list", "data": [{"id": key, "object": "model", "owned_by": "heteroserve"} for key in ids]}

    @app.get("/metrics")
    async def metrics():
        from fastapi.responses import Response
        return Response("\n".join([f"heteroserve_gateway_inflight {state['inflight']}",
            f"heteroserve_gateway_router_calls_total {state['router_calls']}",
            f"heteroserve_gateway_rejected_total {state['rejected']}", ""]), media_type="text/plain")

    @app.post("/drain")
    async def drain(request: Request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "drain is local-only")
        state["draining"] = True
        deadline = time.monotonic() + config.get("drain_seconds", 240)
        while state["inflight"] and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return {"draining": True, "remaining_requests": state["inflight"]}

    @app.post("/v1/embeddings")
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        embedding = request.url.path == "/v1/embeddings"
        entered = time.perf_counter()
        if state["draining"]:
            raise HTTPException(503, "gateway is draining")
        if state["inflight"] >= config.get("max_inflight", 16):
            state["rejected"] += 1
            raise HTTPException(429, "gateway request budget reached", headers={"Retry-After": "1"})
        state["inflight"] += 1
        trace = uuid.uuid4().hex
        upstream = None
        streaming = False
        decision = None
        l1_identity = None
        selected = None
        l1_rpc_ms = 0.0
        l1_timing = {}
        embedding_ms = None
        embedding_pool = None
        def trace_headers():
            headers = {"x-request-id": trace}
            if selected is not None:
                headers["x-moqe-expert"] = selected
            if embedding_ms is not None:
                headers["x-heteroserve-embedding-ms"] = str(embedding_ms)
            return headers
        async def cleanup():
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                state["inflight"] -= 1
                log.info(json.dumps({"event": "gateway_request_finished", "request_id": trace, "selected_expert": selected,
                    "embedding_pool": embedding_pool, "embedding_ms": embedding_ms,
                    "decision": asdict(decision) if decision else None}))
        try:
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > config.get("max_request_bytes", 2 * 1024 * 1024):
                    raise HTTPException(413, "request body exceeds budget")
            try:
                payload = json.loads(body)
            except ValueError:
                raise HTTPException(400, "request must be valid JSON")
            if not isinstance(payload, dict):
                raise HTTPException(400, "request must be a JSON object")
            requested = payload.get("model")
            allowed = set(config["pools"]) | set(random_routes)
            if learned_enabled:
                allowed.add("auto")
            if not isinstance(requested, str) or requested not in allowed:
                raise HTTPException(400, "model must name a configured expert" + (" or auto" if learned_enabled else ""))
            task = config["pools"].get(requested, {}).get("task", "generate")
            if (embedding and (requested == "auto" or task != "embed")) or (not embedding and task != "generate"):
                raise HTTPException(400, "model does not support this endpoint")
            if not isinstance(payload.get("stream", False), bool):
                raise HTTPException(400, "stream must be boolean")
            messages = payload.get("messages")
            if embedding:
                if payload.get("stream", False) or not isinstance(payload.get("input"), (str, list)) or not payload["input"]:
                    raise HTTPException(400, "embeddings require nonempty input and do not support streaming")
            elif not isinstance(messages, list) or not messages:
                raise HTTPException(400, "messages must be a nonempty list")
            selected = None if requested in random_routes else requested
            if requested in random_routes:
                if requested in route_embeddings:
                    if any(not isinstance(message, dict) or not isinstance(message.get("role"), str)
                           or not message["role"] or not isinstance(message.get("content"), str) for message in messages):
                        raise HTTPException(400, "embedding before routing requires plain text messages")
                    embedding_pool = route_embeddings[requested]
                    embed_config = config["pools"][embedding_pool]
                    embed_input = "\n".join(message["role"] + ": " + message["content"] for message in messages)
                    embedding_started = time.perf_counter()
                    try:
                        async with app.state.client.stream("POST", embed_config["base_url"].rstrip("/") + "/embeddings",
                                json={"model": embed_config["model"], "input": embed_input, "encoding_format": "float"},
                                headers={"x-request-id": trace}) as embed_response:
                            if embed_response.status_code != 200:
                                headers = trace_headers()
                                if "retry-after" in embed_response.headers:
                                    headers["retry-after"] = embed_response.headers["retry-after"]
                                raise HTTPException(embed_response.status_code, "embedding pool rejected request", headers=headers)
                            await embed_response.aread()
                            try:
                                data = embed_response.json()
                                rows = data.get("data") if isinstance(data, dict) else None
                                vector = rows[0].get("embedding") if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict) else None
                                if not isinstance(vector, list) or not vector or not all(type(value) in (int, float) and math.isfinite(value) for value in vector):
                                    raise ValueError("invalid embedding vector")
                            except (ValueError, TypeError, OverflowError) as exc:
                                raise HTTPException(502, "embedding pool returned an invalid vector", headers=trace_headers()) from exc
                    finally:
                        embedding_ms = (time.perf_counter() - embedding_started) * 1000
                routing_started = time.perf_counter()
                healthy = await healthy_pools(names=random_routes[requested])
                candidates = [name for name in random_routes[requested] if name in healthy]
                if not candidates:
                    raise HTTPException(503, "no ready pool for random route")
                selected = random.choice(candidates)
                l1_rpc_ms = (time.perf_counter() - routing_started) * 1000
                state["router_calls"] += 1
            elif requested == "auto":
                if any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str) for m in messages):
                    raise HTTPException(400, "automatic routing requires plain text messages")
                if any(k in payload for k in ("tools", "tool_choice", "chat_template", "documents", "add_generation_prompt", "continue_final_message", "truncate_prompt_tokens", "add_special_tokens", "chat_template_content_format")):
                    raise HTTPException(400, "automatic routing uses the canonical chat template")
                generation = payload.get("max_tokens", config.get("default_max_tokens", 128))
                if "max_completion_tokens" in payload or type(generation) is not int or generation <= 0:
                    raise HTTPException(400, "use a positive integer max_tokens")
                kwargs = payload.get("chat_template_kwargs", {"enable_thinking": False})
                if not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"} or type(kwargs.get("enable_thinking", False)) is not bool:
                    raise HTTPException(400, "only boolean enable_thinking is supported")
                kwargs = {"enable_thinking": kwargs.get("enable_thinking", False)}
                try:
                    routing_started = time.perf_counter()
                    if getattr(runtime, "remote", False):
                        decision = await runtime.route(messages, generation, kwargs)
                        l1_identity = dict(runtime.last_identity)
                        l1_timing = dict(getattr(runtime, "last_timing", {}))
                    else:
                        async with routing_slot:
                            decision = await asyncio.to_thread(runtime.route, messages, generation, kwargs)
                    l1_rpc_ms = (time.perf_counter() - routing_started) * 1000
                    state["router_calls"] += 1
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
                if set(decision.probabilities) != set(config["pools"]) or not all(math.isfinite(p) and 0 <= p <= 1 for p in decision.probabilities.values()) or not math.isclose(sum(decision.probabilities.values()), 1, abs_tol=1e-4):
                    raise HTTPException(503, "invalid router expert probabilities")
                selected = decision.expert
                payload.update(max_tokens=generation, chat_template=runtime.chat_template, chat_template_kwargs=kwargs)
            pool = config["pools"].get(selected)
            if not pool or not pool.get("enabled", True):
                raise HTTPException(503, "router selected an undeployed expert; automatic fallback is disabled")
            payload["model"] = pool["model"]
            dispatch_started = time.perf_counter()
            endpoint = "/embeddings" if embedding else "/chat/completions"
            upstream = await app.state.client.send(app.state.client.build_request("POST", pool["base_url"].rstrip("/") + endpoint,
                json=payload, headers={"x-request-id": trace}), stream=True)
            gateway_headers_ms = (time.perf_counter() - entered) * 1000
            headers = trace_headers()
            for name in ["x-heteroserve-pod-uid", "x-heteroserve-node", "x-heteroserve-worker-headers-ms",
                         "x-heteroserve-worker-engine-headers-ms", "x-heteroserve-l2-pre-dispatch-ms", "x-heteroserve-l2-core-headers-ms"]:
                if name in upstream.headers:
                    headers[name] = upstream.headers[name]
            headers.update({"x-heteroserve-gateway-headers-ms": str(gateway_headers_ms),
                "x-heteroserve-gateway-pre-dispatch-ms": str((dispatch_started - entered) * 1000 - l1_rpc_ms),
                "x-heteroserve-l1-rpc-ms": str(l1_rpc_ms)})
            if decision:
                headers.update({"x-moqe-router-ms": str(decision.elapsed_ms), "x-moqe-input-tokens": str(decision.input_tokens)})
                for key in ("queue_ms", "service_ms", "graph_enabled", "graph_used", "graph_fallback"):
                    if key in l1_timing:
                        headers["x-heteroserve-l1-" + key.replace("_", "-")] = str(l1_timing[key]).lower()
                if getattr(runtime, "remote", False):
                    headers.update({"x-moqe-l1-instance": l1_identity["instance_id"], "x-moqe-l1-node": l1_identity["node"], "x-moqe-l1-pod-uid": l1_identity["pod_uid"]})
            if upstream.status_code != 200:
                if "retry-after" in upstream.headers:
                    headers["retry-after"] = upstream.headers["retry-after"]
                return JSONResponse({"error": {"message": "expert pool rejected request", "upstream_status": upstream.status_code}},
                                    status_code=upstream.status_code, headers=headers)
            if payload.get("stream", False):
                async def events():
                    try:
                        async for line in upstream.aiter_lines():
                            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                                event = json.loads(line[5:])
                                if isinstance(event, dict) and "model" in event:
                                    event["model"] = requested
                                line = "data: " + json.dumps(event, ensure_ascii=False)
                            yield (line + "\n").encode()
                    finally:
                        await asyncio.shield(cleanup())
                streaming = True
                return StreamingResponse(events(), media_type="text/event-stream", headers=headers)
            await upstream.aread()
            value = upstream.json()
            value["model"] = requested
            return JSONResponse(value, headers=headers)
        except HTTPException as exc:
            if embedding_ms is not None:
                exc.headers = {**(exc.headers or {}), **trace_headers()}
            raise
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "embedding pool request timed out" if embedding_pool and selected is None else "expert pool request timed out",
                                headers=trace_headers()) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "embedding pool is unavailable" if embedding_pool and selected is None else "expert pool is unavailable",
                                headers=trace_headers()) from exc
        finally:
            if not streaming:
                await asyncio.shield(cleanup())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(config), host=config.get("host", "0.0.0.0"), port=config.get("port", 8000), workers=1,
                timeout_graceful_shutdown=config.get("drain_seconds", 240) + 10)


if __name__ == "__main__":
    main()
