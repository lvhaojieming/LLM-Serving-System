"""Supervise vLLM, admit real model output, and drain direct Pod traffic safely."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import re
import signal
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

log = logging.getLogger("heteroserve.worker")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "content-length"}


def identity(config):
    devices = sorted(int(m.group(1)) for p in Path("/dev").glob("davinci*")
                     if (m := re.fullmatch(r"davinci([0-9]+)", p.name)))
    return {"pod_uid": os.environ.get("POD_UID", "local"), "node": os.environ.get("NODE_NAME", "local"),
            "physical_devices": devices, "device_type": config["device_type"], "pool": config["pool"],
            "model": config["model"], "weights_manifest_sha256": config["weights_manifest_sha256"],
            "image": config["image"], "instance_id": config.get("instance_id", "legacy"),
            "parallelism": config.get("parallelism", {"tp": 1, "pp": 1}),
            "expected_devices": config.get("device_count", 1)}


def create_app(config, transport=None, start_engine=True):
    state = {"admitted": False, "draining": False, "health_failures": 0, "leases": {},
             "engine": None, "last_error": None, "identity": identity(config), "completed": 0, "cancelled": 0}

    async def admit(client):
        response = await client.get("/v1/models", timeout=10)
        response.raise_for_status()
        if config["model"] not in {m["id"] for m in response.json()["data"]}:
            raise RuntimeError("Engine reports an unexpected model identity")
        body = {"model": config["model"], "messages": [{"role": "user", "content": "What is 17 + 25? Reply with the number only."}],
                "temperature": 0, "max_tokens": 32, "chat_template_kwargs": {"enable_thinking": False}}
        response = await client.post("/v1/chat/completions", json=body, timeout=config["admission_timeout_seconds"])
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        if answer != "42":
            raise RuntimeError("Real model admission did not produce the expected answer")

    async def monitor(client):
        while True:
            engine = state["engine"]
            if engine is not None and engine.returncode is not None:
                state.update(admitted=False, last_error="engine exited", health_failures=config["liveness_failures"])
                log.error(json.dumps({"event": "engine_exited", "returncode": engine.returncode, **state["identity"]}))
                os.kill(os.getpid(), signal.SIGTERM)
                return
            try:
                response = await client.get("/health", timeout=config["health_timeout_seconds"])
                response.raise_for_status()
                if not state["admitted"] and not state["draining"]:
                    await admit(client)
                    state["admitted"] = True
                    log.info(json.dumps({"event": "model_admitted", **state["identity"]}))
                state.update(health_failures=0, last_error=None)
            except Exception as exc:
                state["health_failures"] += 1
                state["last_error"] = type(exc).__name__ + ": " + str(exc)[:300]
                if state["health_failures"] >= config["readiness_failures"]:
                    state["admitted"] = False
            await asyncio.sleep(config["health_interval_seconds"])

    @asynccontextmanager
    async def lifespan(app):
        if start_engine:
            if len(state["identity"]["physical_devices"]) != config.get("device_count", 1):
                raise RuntimeError("Actual NPU allocation does not match the configured parallel world size")
            state["engine"] = await asyncio.create_subprocess_exec(*config["engine_command"], start_new_session=True,
                env={**os.environ, **config.get("engine_env", {})})
        async with httpx.AsyncClient(base_url=config["engine_url"], transport=transport,
                                    timeout=httpx.Timeout(connect=10, read=config["request_timeout_seconds"], write=30, pool=10),
                                    trust_env=False) as client:
            app.state.client = client
            watcher = asyncio.create_task(monitor(client))
            try:
                yield
            finally:
                state["draining"] = True
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                for lease in list(state["leases"].values()):
                    if lease.get("response"):
                        await lease["response"].aclose()
                engine = state["engine"]
                if engine is not None and engine.returncode is None:
                    os.killpg(engine.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(engine.wait(), config["engine_stop_seconds"])
                    except asyncio.TimeoutError:
                        os.killpg(engine.pid, signal.SIGKILL)
                        await engine.wait()

    app = FastAPI(lifespan=lifespan)
    app.state.worker = state

    @app.get("/identity")
    async def get_identity():
        return state["identity"]

    @app.get("/ready")
    @app.get("/health")
    async def ready():
        if state["draining"] or not state["admitted"]:
            raise HTTPException(503, "worker unavailable")
        return {"ready": True, **state["identity"]}

    @app.get("/live")
    async def live():
        if not state["draining"] and state["health_failures"] >= config["liveness_failures"]:
            raise HTTPException(503, "engine requires recovery")
        return {"live": True, "inflight": len(state["leases"])}

    @app.post("/drain")
    async def drain(request: Request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "drain is local-only")
        state["draining"] = True
        deadline = time.monotonic() + config["drain_seconds"]
        while state["leases"] and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        cancelled = len(state["leases"])
        for lease in list(state["leases"].values()):
            lease["cancelled"] = True
            if lease.get("response"):
                await lease["response"].aclose()
            task = lease.get("task")
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        state["cancelled"] += cancelled
        log.info(json.dumps({"event": "drained", "cancelled": cancelled, **state["identity"]}))
        return {"draining": True, "cancelled": cancelled, "inflight": len(state["leases"])}

    @app.get("/metrics")
    async def metrics():
        text = "\n".join([
            "# TYPE heteroserve_worker_inflight gauge",
            f"heteroserve_worker_inflight {len(state['leases'])}",
            "# TYPE heteroserve_worker_ready gauge",
            f"heteroserve_worker_ready {int(state['admitted'] and not state['draining'])}",
            "# TYPE heteroserve_worker_completed_total counter",
            f"heteroserve_worker_completed_total {state['completed']}",
            "# TYPE heteroserve_worker_cancelled_total counter",
            f"heteroserve_worker_cancelled_total {state['cancelled']}", ""])
        try:
            response = await app.state.client.get("/metrics", timeout=3)
            if response.status_code == 200:
                text += response.text
        except httpx.HTTPError:
            pass
        return Response(text, media_type="text/plain")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if not state["admitted"] or state["draining"]:
            raise HTTPException(503, "worker is not accepting new requests")
        if request.method == "POST" and len(state["leases"]) >= config.get("max_inflight", 8):
            raise HTTPException(429, "worker request budget reached", headers={"Retry-After": "1"})
        lease_id = uuid.uuid4().hex
        lease = {"task": asyncio.current_task(), "response": None, "cancelled": False}
        state["leases"][lease_id] = lease
        upstream = None
        try:
            body = await request.body()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS | {"host"}}
            upstream_request = app.state.client.build_request(request.method, "/v1/" + path,
                                                              params=request.query_params, headers=headers, content=body)
            upstream = await app.state.client.send(upstream_request, stream=True)
            lease["response"] = upstream
            response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
            response_headers.update({"x-heteroserve-pod-uid": state["identity"]["pod_uid"],
                                     "x-heteroserve-node": state["identity"]["node"]})
            if "text/event-stream" in upstream.headers.get("content-type", ""):
                async def stream():
                    lease["task"] = asyncio.current_task()
                    try:
                        async for chunk in upstream.aiter_raw():
                            if lease["cancelled"]:
                                break
                            yield chunk
                        state["completed"] += 1
                    finally:
                        state["leases"].pop(lease_id, None)
                        await asyncio.shield(upstream.aclose())
                return StreamingResponse(stream(), status_code=upstream.status_code, headers=response_headers,
                                         media_type="text/event-stream")
            result = await upstream.aread()
            await upstream.aclose()
            state["leases"].pop(lease_id, None)
            state["completed"] += 1
            return Response(result, status_code=upstream.status_code, headers=response_headers)
        except BaseException:
            state["leases"].pop(lease_id, None)
            if upstream is not None:
                await asyncio.shield(upstream.aclose())
            raise

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(config), host="0.0.0.0", port=config["port"], workers=1,
                access_log=False, timeout_graceful_shutdown=config["drain_seconds"] + 10)


if __name__ == "__main__":
    main()
