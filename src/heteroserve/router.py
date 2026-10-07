"""Bounded Kubernetes discovery guard around the official vLLM Router data plane."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import ipaddress
import json
import logging
import os
from pathlib import Path
import signal
import ssl
import time
import uuid
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

log = logging.getLogger("heteroserve.router")
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "content-length"}


def discover(pods, slices, pool):
    eligible = {}
    for pod in pods["items"]:
        meta, status = pod["metadata"], pod.get("status", {})
        if meta.get("deletionTimestamp") or meta.get("labels", {}).get("heteroserve.io/pool") != pool:
            continue
        if status.get("phase") != "Running" or not any(c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions", [])):
            continue
        eligible[meta["uid"]] = pod
    result = {}
    for item in slices["items"]:
        if item["metadata"].get("labels", {}).get("kubernetes.io/service-name") != pool:
            continue
        for endpoint in item.get("endpoints", []):
            conditions = endpoint.get("conditions", {})
            reference = endpoint.get("targetRef", {})
            uid = reference.get("uid")
            if conditions.get("ready") is not True or conditions.get("terminating") or conditions.get("serving") is False or reference.get("kind") != "Pod" or uid not in eligible:
                continue
            pod = eligible[uid]
            for address in endpoint.get("addresses", []):
                ipaddress.ip_address(address)
                if address != pod.get("status", {}).get("podIP"):
                    continue
                result[uid] = {"uid": uid, "name": pod["metadata"]["name"], "ip": address, "node": pod["spec"]["nodeName"]}
    return result


@dataclass
class DiscoveryView:
    ttl: float
    endpoints: dict = field(default_factory=dict)
    last_success: float | None = None
    error: str | None = None

    def fresh(self, now=None):
        return self.last_success is not None and (time.monotonic() if now is None else now) - self.last_success <= self.ttl

    def update(self, endpoints, now=None):
        self.endpoints = endpoints
        self.last_success = time.monotonic() if now is None else now
        self.error = None


async def reconcile_workers(client, previous, desired):
    """Use the official Router management API; never implement worker selection here."""
    def urls(endpoints):
        return {"http://" + ("[" + item["ip"] + "]" if ":" in item["ip"] else item["ip"]) + ":8000": uid for uid, item in endpoints.items()}
    before, wanted = urls(previous), urls(desired)
    response = await client.get("/workers", timeout=5)
    response.raise_for_status()
    current = {item["url"] for item in response.json()["workers"]}
    replaced = {url for url in current & set(wanted) if url in before and before[url] != wanted[url]}
    for url in (current - set(wanted)) | replaced:
        response = await client.delete("/workers/" + quote(url, safe=""), timeout=10)
        if response.status_code not in {200, 404}:
            response.raise_for_status()
        current.discard(url)
        log.info(json.dumps({"event": "official_router_worker_removed", "url": url}))
    for url in set(wanted) - current:
        response = await client.post("/workers", json={"url": url}, timeout=15)
        response.raise_for_status()
        log.info(json.dumps({"event": "official_router_worker_added", "url": url, "uid": wanted[url]}))


def create_app(config, api_transport=None, core_transport=None, start_core=True):
    view = DiscoveryView(config["discovery_cache_seconds"])
    state = {"active": 0, "draining": False, "core": None, "completed": 0, "rejected": 0}

    async def refresh(client):
        while True:
            try:
                pod_response = await client.get("/api/v1/namespaces/" + config["namespace"] + "/pods",
                    params={"labelSelector": "heteroserve.io/pool=" + config["pool"]})
                slice_response = await client.get("/apis/discovery.k8s.io/v1/namespaces/" + config["namespace"] + "/endpointslices",
                    params={"labelSelector": "kubernetes.io/service-name=" + config["pool"]})
                pod_response.raise_for_status()
                slice_response.raise_for_status()
                endpoints = discover(pod_response.json(), slice_response.json(), config["pool"])
                await reconcile_workers(app.state.core_client, view.endpoints, endpoints)
                if endpoints != view.endpoints:
                    log.info(json.dumps({"event": "discovery_changed", "pool": config["pool"], "endpoints": list(endpoints.values())}))
                view.update(endpoints)
            except Exception as exc:
                view.error = type(exc).__name__
            await asyncio.sleep(config["discovery_interval_seconds"])

    @asynccontextmanager
    async def lifespan(app):
        account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
        headers = {"Authorization": "Bearer " + (account / "token").read_text().strip()} if account.exists() else {}
        verify = ssl.create_default_context(cafile=str(account / "ca.crt")) if account.exists() else True
        api_url = config.get("api_url", "https://" + os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc") + ":" + os.environ.get("KUBERNETES_SERVICE_PORT", "443"))
        if start_core:
            state["core"] = await asyncio.create_subprocess_exec(*config["core_command"], start_new_session=True)
        async with httpx.AsyncClient(base_url=api_url, headers=headers, verify=verify, transport=api_transport, timeout=5, trust_env=False) as api, \
                httpx.AsyncClient(base_url=config["core_url"], transport=core_transport, timeout=httpx.Timeout(connect=10, read=180, write=30, pool=10), trust_env=False) as core:
            app.state.core_client = core
            watcher = asyncio.create_task(refresh(api))
            try:
                yield
            finally:
                state["draining"] = True
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
                process = state["core"]
                if process is not None and process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), 30)
                    except asyncio.TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()

    app = FastAPI(lifespan=lifespan)
    app.state.discovery = view
    app.state.router = state

    def available():
        return not state["draining"] and view.fresh() and bool(view.endpoints)

    @app.get("/ready")
    async def ready():
        if not available():
            raise HTTPException(503, "no fresh ready worker discovery")
        response = await app.state.core_client.get("/health", timeout=3)
        if response.status_code != 200:
            raise HTTPException(503, "vLLM Router is unavailable")
        return {"ready": True, "workers": len(view.endpoints)}

    @app.get("/live")
    async def live():
        process = state["core"]
        if process is not None and process.returncode is not None:
            raise HTTPException(503, "vLLM Router exited")
        return {"live": True}

    @app.get("/discovery")
    async def discovery():
        return {"fresh": view.fresh(), "error": view.error, "endpoints": list(view.endpoints.values())}

    @app.get("/metrics")
    async def metrics():
        text = "\n".join(["# TYPE heteroserve_router_inflight gauge", f"heteroserve_router_inflight {state['active']}",
            "# TYPE heteroserve_discovery_fresh gauge", f"heteroserve_discovery_fresh {int(view.fresh())}",
            "# TYPE heteroserve_ready_workers gauge", f"heteroserve_ready_workers {len(view.endpoints) if view.fresh() else 0}",
            "# TYPE heteroserve_router_rejected_total counter", f"heteroserve_router_rejected_total {state['rejected']}", ""])
        try:
            response = await app.state.core_client.get("http://127.0.0.1:29000/metrics", timeout=3)
            if response.status_code == 200:
                text += response.text
        except httpx.HTTPError:
            pass
        return Response(text, media_type="text/plain")

    @app.post("/drain")
    async def drain(request: Request):
        if request.client is None or request.client.host not in {"127.0.0.1", "::1"}:
            raise HTTPException(403, "drain is local-only")
        state["draining"] = True
        deadline = time.monotonic() + config["drain_seconds"]
        while state["active"] and time.monotonic() < deadline:
            await asyncio.sleep(.05)
        return {"draining": True, "remaining_requests": state["active"]}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if not available():
            raise HTTPException(503, "discovery is absent or expired")
        if state["active"] >= len(view.endpoints) * config["requests_per_worker"]:
            state["rejected"] += 1
            raise HTTPException(429, "pool request capacity reached", headers={"Retry-After": "1"})
        state["active"] += 1
        admitted_uids = set(view.endpoints)
        upstream = None
        streaming = False
        trace = request.headers.get("x-request-id", uuid.uuid4().hex)
        try:
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > config["max_request_bytes"]:
                    raise HTTPException(413, "request body is too large")
                chunks.append(chunk)
            body = b"".join(chunks)
            if request.method == "POST":
                try:
                    value = json.loads(body)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    raise HTTPException(400, "request body must be JSON")
                if not isinstance(value, dict):
                    raise HTTPException(400, "request body must be a JSON object")
                if value.get("model") in {config["pool"], config["expert"]}:
                    value["model"] = config["model"]
                    body = json.dumps(value).encode()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS | {"host"}}
            headers["x-request-id"] = trace
            if not available():
                raise HTTPException(503, "discovery expired before dispatch")
            admitted_uids = set(view.endpoints)
            upstream = await app.state.core_client.send(app.state.core_client.build_request(request.method, "/v1/" + path,
                params=request.query_params, content=body, headers=headers), stream=True)
            response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS}
            response_headers["x-request-id"] = trace
            worker_uid = upstream.headers.get("x-heteroserve-pod-uid")
            if request.method == "POST" and upstream.status_code == 200 and worker_uid not in admitted_uids:
                raise HTTPException(503, "worker response identity is absent from ready discovery")
            if "text/event-stream" in upstream.headers.get("content-type", ""):
                async def stream():
                    try:
                        async for chunk in upstream.aiter_raw():
                            yield chunk
                        state["completed"] += 1
                    finally:
                        state["active"] -= 1
                        await asyncio.shield(upstream.aclose())
                        log.info(json.dumps({"event": "request_finished", "trace": trace, "worker_uid": worker_uid, "stream": True}))
                streaming = True
                return StreamingResponse(stream(), status_code=upstream.status_code, headers=response_headers)
            result = await upstream.aread()
            state["completed"] += 1
            return Response(result, status_code=upstream.status_code, headers=response_headers)
        finally:
            if not streaming:
                state["active"] -= 1
                if upstream is not None:
                    await asyncio.shield(upstream.aclose())
                log.info(json.dumps({"event": "request_finished", "trace": trace, "worker_uid": upstream.headers.get("x-heteroserve-pod-uid") if upstream else None, "stream": False}))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(config), host="0.0.0.0", port=8000, workers=1,
                access_log=False, timeout_graceful_shutdown=config["drain_seconds"] + 5)


if __name__ == "__main__":
    main()
