"""Learned expert selection above Kubernetes pool Services.

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
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .routing.runtime import RouterSettings

log = logging.getLogger("heteroserve.gateway")


def validate(config):
    settings = RouterSettings(**config["router"])
    if set(settings.expert_mapping.values()) != set(config["pools"]):
        raise ValueError("Every checkpoint expert must have an explicit pool entry")
    for pool in config["pools"].values():
        if pool.get("enabled", True) and (not isinstance(pool.get("base_url"), str) or not pool["base_url"].startswith("http://")):
            raise ValueError("Enabled pools need an internal HTTP Service URL")
    if type(config.get("max_inflight", 16)) is not int or config.get("max_inflight", 16) < 1:
        raise ValueError("Gateway requires a positive request budget")
    return settings


def create_app(config, transport=None, router_runtime=None):
    settings = validate(config)
    state = {"inflight": 0, "draining": False, "router_calls": 0, "rejected": 0}
    runtime = router_runtime
    routing_slot = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime
        if runtime is None:
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

    @app.get("/live")
    async def live():
        return {"live": True}

    @app.get("/ready")
    async def ready():
        if runtime is None or state["draining"]:
            raise HTTPException(503, "learned gateway is unavailable")
        healthy = []
        for name, pool in config["pools"].items():
            if not pool.get("enabled", True):
                continue
            try:
                response = await app.state.client.get(pool["base_url"].removesuffix("/v1") + "/ready", timeout=3)
                if response.status_code == 200:
                    healthy.append(name)
            except httpx.HTTPError:
                pass
        if healthy:
            l1_ready = await runtime.ready() if getattr(runtime, "remote", False) else True
            return {"ready": True, "ready_experts": healthy,
                    "l1_router_ready": l1_ready,
                    "auto_expert_coverage_complete": l1_ready and len(healthy) == len(config["pools"])}
        raise HTTPException(503, "no ready downstream expert pool")

    @app.get("/v1/models")
    async def models():
        ids = ["auto"] + [key for key, pool in config["pools"].items() if pool.get("enabled", True)]
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

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
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
        async def cleanup():
            try:
                if upstream is not None:
                    await upstream.aclose()
            finally:
                state["inflight"] -= 1
                log.info(json.dumps({"event": "gateway_request_finished", "request_id": trace, "selected_expert": selected,
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
            if not isinstance(requested, str) or requested not in {"auto", *config["pools"]}:
                raise HTTPException(400, "model must name auto or a configured expert")
            if not isinstance(payload.get("stream", False), bool):
                raise HTTPException(400, "stream must be boolean")
            messages = payload.get("messages")
            if not isinstance(messages, list) or not messages:
                raise HTTPException(400, "messages must be a nonempty list")
            selected = requested
            if requested == "auto":
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
                    if getattr(runtime, "remote", False):
                        decision = await runtime.route(messages, generation, kwargs)
                        l1_identity = dict(runtime.last_identity)
                    else:
                        async with routing_slot:
                            decision = await asyncio.to_thread(runtime.route, messages, generation, kwargs)
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
            upstream = await app.state.client.send(app.state.client.build_request("POST", pool["base_url"].rstrip("/") + "/chat/completions",
                json=payload, headers={"x-request-id": trace}), stream=True)
            headers = {"x-request-id": trace, "x-moqe-expert": selected}
            for name in ["x-heteroserve-pod-uid", "x-heteroserve-node"]:
                if name in upstream.headers:
                    headers[name] = upstream.headers[name]
            if decision:
                headers.update({"x-moqe-router-ms": str(decision.elapsed_ms), "x-moqe-input-tokens": str(decision.input_tokens)})
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
        except httpx.HTTPError as exc:
            raise HTTPException(502, "expert pool is unavailable") from exc
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
