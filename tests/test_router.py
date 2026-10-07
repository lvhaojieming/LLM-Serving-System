"""Discovery must be UID-bound, ready-only, and expire safely during API failure."""
import asyncio
import importlib.util
from pathlib import Path
import sys

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("heteroserve_router", ROOT / "src/heteroserve/router.py")
router = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = router
spec.loader.exec_module(router)


def inventory(uid="new"):
    pods = {"items": [{"metadata": {"name": "worker", "uid": uid, "labels": {"heteroserve.io/pool": "awq"}},
                      "spec": {"nodeName": "node"}, "status": {"phase": "Running", "podIP": "10.0.0.1", "conditions": [{"type": "Ready", "status": "True"}]}}]}
    slices = {"items": [{"metadata": {"labels": {"kubernetes.io/service-name": "awq"}}, "endpoints": [{
        "conditions": {"ready": True}, "targetRef": {"kind": "Pod", "uid": uid}, "addresses": ["10.0.0.1"]}]}]}
    return pods, slices


def test_old_endpoint_uid_cannot_admit_rebuilt_pod_at_same_address():
    pods, slices = inventory()
    slices["items"][0]["endpoints"][0]["targetRef"]["uid"] = "old"
    assert router.discover(pods, slices, "awq") == {}
    pods, slices = inventory()
    pods["items"][0]["metadata"]["deletionTimestamp"] = "now"
    assert router.discover(pods, slices, "awq") == {}
    pods, slices = inventory()
    slices["items"][0]["endpoints"][0]["conditions"]["terminating"] = True
    assert router.discover(pods, slices, "awq") == {}


def test_discovery_error_does_not_refresh_cache_age():
    view = router.DiscoveryView(ttl=15)
    view.update({"uid": {}}, now=100)
    view.error = "API unavailable"
    assert view.fresh(now=114)
    assert not view.fresh(now=116)


def test_official_worker_api_removes_stale_addresses_and_rebinds_reused_address():
    async def scenario():
        calls = []
        async def core(request):
            calls.append((request.method, str(request.url)))
            if request.method == "GET":
                return httpx.Response(200, json={"workers": [{"url": "http://10.0.0.1:8000"}, {"url": "http://10.0.0.2:8000"}]})
            return httpx.Response(200, json={"success": True})
        async with httpx.AsyncClient(base_url="http://core", transport=httpx.MockTransport(core)) as client:
            await router.reconcile_workers(client, {"old": {"ip": "10.0.0.1"}}, {"new": {"ip": "10.0.0.1"}})
        assert len([x for x in calls if x[0] == "DELETE"]) == 2
        assert len([x for x in calls if x[0] == "POST"]) == 1
    asyncio.run(scenario())


def test_api_outage_expires_cache_and_stops_forwarding_without_router_restart():
    async def scenario():
        status = {"api": True, "calls": 0}
        pods, slices = inventory()
        async def api(request):
            if not status["api"]:
                return httpx.Response(500)
            return httpx.Response(200, json=pods if request.url.path.endswith("/pods") else slices)
        async def core(request):
            status["calls"] += 1
            if request.url.path == "/workers":
                return httpx.Response(200, json={"workers": [{"url": "http://10.0.0.1:8000"}]})
            return httpx.Response(200, json={"ok": True}, headers={"x-heteroserve-pod-uid": "new"})
        config = {"pool": "awq", "namespace": "ns", "expert": "awq", "model": "model", "api_url": "http://api", "core_url": "http://core",
                  "discovery_cache_seconds": .03, "discovery_interval_seconds": .005, "requests_per_worker": 1, "max_request_bytes": 100, "drain_seconds": 1}
        app = router.create_app(config, api_transport=httpx.MockTransport(api), core_transport=httpx.MockTransport(core), start_core=False)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://router") as client:
            for _ in range(100):
                if app.state.discovery.endpoints:
                    break
                await asyncio.sleep(.005)
            assert (await client.post("/v1/chat/completions", json={"model": "awq"})).status_code == 200
            status["api"] = False
            await asyncio.sleep(.05)
            before = status["calls"]
            assert (await client.post("/v1/chat/completions", json={"model": "awq"})).status_code == 503
            assert status["calls"] == before
            assert app.state.router["active"] == 0
    asyncio.run(scenario())


def test_single_router_rejects_over_budget_and_cancellation_releases_its_slot():
    async def scenario():
        started = asyncio.Event()
        blocked = asyncio.Event()
        pods, slices = inventory()
        async def api(request):
            return httpx.Response(200, json=pods if request.url.path.endswith("/pods") else slices)
        async def core(request):
            if request.url.path == "/workers":
                return httpx.Response(200, json={"workers": [{"url": "http://10.0.0.1:8000"}]})
            started.set()
            await blocked.wait()
            return httpx.Response(200, json={"ok": True}, headers={"x-heteroserve-pod-uid": "new"})
        config = {"pool": "awq", "namespace": "ns", "expert": "awq", "model": "model", "api_url": "http://api", "core_url": "http://core",
                  "discovery_cache_seconds": 15, "discovery_interval_seconds": .005, "requests_per_worker": 1, "max_request_bytes": 100, "drain_seconds": 1}
        app = router.create_app(config, api_transport=httpx.MockTransport(api), core_transport=httpx.MockTransport(core), start_core=False)
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://router") as client:
            for _ in range(100):
                if app.state.discovery.endpoints:
                    break
                await asyncio.sleep(.005)
            ongoing = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "awq"}))
            try:
                await asyncio.wait_for(started.wait(), 2)
                rejection = await client.post("/v1/chat/completions", json={"model": "awq"})
                assert rejection.status_code == 429 and rejection.headers["retry-after"] == "1"
                assert app.state.router["active"] == 1
            finally:
                ongoing.cancel()
                await asyncio.gather(ongoing, return_exceptions=True)
            assert app.state.router["active"] == 0
            blocked.set()
            assert (await client.post("/v1/chat/completions", json={"model": "awq"})).status_code == 200
            assert app.state.router["active"] == 0
    asyncio.run(scenario())
