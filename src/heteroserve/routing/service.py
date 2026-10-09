"""Independent one-NPU probability inference; HTTP lifecycle stays on CPU."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException, Request
import uvicorn

from .runtime import RouterSettings


def create_app(config, runtime=None):
    settings = RouterSettings(**config["router"])
    if type(config.get("max_inflight")) is not int or config["max_inflight"] < 1:
        raise ValueError("L1 requires a positive request budget")
    devices = sorted(int(p.name[7:]) for p in Path("/dev").glob("davinci*") if p.name[7:].isdigit())
    state = {"draining": False, "tasks": set(), "completed": 0, "rejected": 0, "started": None, "failed": False}
    slot = asyncio.Semaphore(1)
    identity = {"pod_uid": os.environ.get("POD_UID", "local"), "node": os.environ.get("NODE_NAME", "local"),
                "instance_id": config["instance_id"], "pool": "l1-router", "physical_devices": devices,
                "checkpoint_sha256": settings.checkpoint_sha256, "max_inflight": config["max_inflight"],
                "parallelism": {"tp": 1, "pp": 1}, "encoder_device": settings.device, "head_device": settings.device}

    @asynccontextmanager
    async def lifespan(app):
        nonlocal runtime
        if runtime is None:
            if len(devices) != 1:
                raise RuntimeError("L1 instance must receive exactly one NPU")
            from .ascend import AscendRouterRuntime
            runtime = await asyncio.to_thread(AscendRouterRuntime, settings)
            identity["head_device"] = str(next(runtime.router.parameters()).device)
            encoder = runtime.encoder if runtime.contextual else runtime.embedding
            identity["encoder_device"] = str(next(encoder.parameters()).device)
            if not all(identity[k].startswith("npu:") for k in ("head_device", "encoder_device")):
                raise RuntimeError("Encoder and classifier must both execute on NPU")
        template_sha = hashlib.sha256(runtime.chat_template.encode()).hexdigest()
        if template_sha != config["chat_template_sha256"]:
            raise RuntimeError("L1 canonical chat template differs from the pinned contract")
        identity["chat_template_sha256"] = template_sha
        yield
        state["draining"] = True
        if state["tasks"]:
            await asyncio.wait(state["tasks"], timeout=config.get("drain_seconds", 240))

    app = FastAPI(lifespan=lifespan)
    app.state.l1 = state
    def unhealthy():
        return state["failed"] or (state["started"] is not None and time.monotonic()-state["started"] > config.get("compute_timeout_seconds",120))

    @app.get("/identity")
    async def get_identity():
        return identity

    @app.get("/live")
    async def live():
        if unhealthy():raise HTTPException(503,"L1 compute requires recovery")
        return {"live": True}

    @app.get("/ready")
    async def ready():
        if runtime is None or state["draining"] or unhealthy():
            raise HTTPException(503, "L1 is not accepting requests")
        return {"ready": True, **identity}

    @app.get("/metrics")
    async def metrics():
        from fastapi.responses import Response
        return Response(f"heteroserve_l1_inflight {len(state['tasks'])}\nheteroserve_l1_completed_total {state['completed']}\nheteroserve_l1_rejected_total {state['rejected']}\n", media_type="text/plain")

    @app.post("/drain")
    async def drain(request: Request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "drain is local-only")
        state["draining"] = True
        deadline = time.monotonic() + config.get("drain_seconds", 240)
        while state["tasks"] and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return {"draining": True, "remaining": len(state["tasks"])}

    @app.post("/route")
    async def route(request: Request):
        if runtime is None or state["draining"] or unhealthy():
            raise HTTPException(503, "L1 is unavailable")
        if len(state["tasks"]) >= config["max_inflight"]:
            state["rejected"] += 1
            raise HTTPException(429, "L1 request budget reached", headers={"Retry-After": "1"})
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > config.get("max_request_bytes", 2 * 1024 * 1024):
                raise HTTPException(413, "L1 request body exceeds budget")
        try:
            payload = json.loads(body)
            messages, generation = payload["messages"], payload["max_new_tokens"]
            kwargs = payload.get("template_kwargs", {"enable_thinking": False})
            if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str) for m in messages):
                raise ValueError("L1 requires plain text messages")
            if type(generation) is not int or generation < 1 or not isinstance(kwargs, dict) or set(kwargs) != {"enable_thinking"} or type(kwargs["enable_thinking"]) is not bool:
                raise ValueError("Invalid generation/template settings")
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(400, str(error)) from error
        # Check again after reading a body; parsing must not oversubscribe the device.
        if state["draining"] or len(state["tasks"]) >= config["max_inflight"]:
            raise HTTPException(429 if not state["draining"] else 503, "L1 no longer has admission capacity")
        async def compute():
            async with slot:
                if state["draining"]:
                    raise HTTPException(503, "L1 is draining")
                state["started"] = time.monotonic()
                try:
                    return await asyncio.to_thread(runtime.route, messages, generation, kwargs)
                except (RuntimeError,OSError):
                    state["failed"] = True
                    raise
                finally:
                    state["started"] = None
        task = asyncio.create_task(compute())
        state["tasks"].add(task)
        def completed(future):
            state["tasks"].discard(future)
            if not future.cancelled():
                future.exception()  # Consume errors even if the HTTP caller disconnected.
        task.add_done_callback(completed)
        try:
            decision = await asyncio.shield(task)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        state["completed"] += 1
        return {"decision": asdict(decision), "chat_template": runtime.chat_template, "identity": identity}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(config), host="0.0.0.0", port=8000, workers=1,
                timeout_graceful_shutdown=config.get("drain_seconds", 240) + 10)


if __name__ == "__main__":
    main()
