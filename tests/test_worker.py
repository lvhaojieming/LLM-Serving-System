"""Model admission, direct-Pod draining, and stream leases are safety boundaries."""
import asyncio
import importlib.util
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worker", ROOT / "src/heteroserve/worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def config():
    return {"pool": "awq", "model": "model", "device_type": "Ascend910B4", "weights_manifest_sha256": "a" * 64,
            "image": "image@sha256:" + "b" * 64, "engine_url": "http://engine", "health_timeout_seconds": 1,
            "health_interval_seconds": .01, "admission_timeout_seconds": 1, "request_timeout_seconds": 2,
            "readiness_failures": 2, "liveness_failures": 4, "drain_seconds": 1, "engine_stop_seconds": 1}


async def until(predicate):
    for _ in range(200):
        if predicate():
            return
        await asyncio.sleep(.01)
    raise AssertionError("state did not reach the expected condition")


def test_wrong_model_never_becomes_ready_and_health_failure_removes_admission():
    async def scenario():
        mode = {"correct": False, "healthy": True}
        async def engine(request):
            if request.url.path == "/health":
                return httpx.Response(200 if mode["healthy"] else 500)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "model" if mode["correct"] else "wrong"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})
        app = worker.create_app(config(), transport=httpx.MockTransport(engine), start_engine=False)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://worker") as client:
            await until(lambda: app.state.worker["health_failures"] >= 2)
            assert (await client.get("/ready")).status_code == 503
            assert (await client.post("/v1/chat/completions", json={})).status_code == 503
            mode["correct"] = True
            await until(lambda: app.state.worker["admitted"])
            assert (await client.get("/ready")).status_code == 200
            mode["healthy"] = False
            await until(lambda: app.state.worker["health_failures"] >= 4)
            assert (await client.get("/ready")).status_code == 503
            assert (await client.get("/live")).status_code == 503
    asyncio.run(scenario())


def test_drain_rejects_new_requests_but_waits_for_complete_sse():
    async def scenario():
        started = asyncio.Event()
        finish = asyncio.Event()
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b'data: {"choices":[{"delta":{"content":"42"}}]}\n\n'
                await finish.wait()
                yield b"data: [DONE]\n\n"
        async def engine(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "model"}]})
            if json.loads(request.content).get("stream"):
                return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())
            return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})
        app = worker.create_app(config(), transport=httpx.MockTransport(engine), start_engine=False)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://worker") as client:
            await until(lambda: app.state.worker["admitted"])
            ongoing = asyncio.create_task(client.post("/v1/chat/completions", json={"stream": True}))
            await started.wait()
            assert len(app.state.worker["leases"]) == 1
            draining = asyncio.create_task(client.post("/drain"))
            await until(lambda: app.state.worker["draining"])
            assert (await client.get("/ready")).status_code == 503
            assert (await client.post("/v1/chat/completions", json={})).status_code == 503
            assert not draining.done()
            finish.set()
            result = await ongoing
            assert "[DONE]" in result.text
            assert (await draining).json()["cancelled"] == 0
            assert not app.state.worker["leases"]
    asyncio.run(scenario())
