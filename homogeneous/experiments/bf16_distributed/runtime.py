"""Own a Ray stage; reuse the formal worker supervisor for the serving head."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import signal
import socket
import time

from fastapi import FastAPI, HTTPException
import uvicorn
import worker


def ray_command(config):
    cluster = config["cluster"]
    command = ["ray", "start", "--node-ip-address=" + os.environ["POD_IP"],
        "--num-cpus=8", "--num-gpus=0", "--object-store-memory=268435456", "--resources=" + json.dumps({"NPU": cluster["tp"]}),
        "--node-manager-port=30001", "--object-manager-port=30002",
        "--min-worker-port=30010", "--max-worker-port=30109",
        "--dashboard-agent-listen-port=30003", "--dashboard-agent-grpc-port=30004",
        "--runtime-env-agent-port=30005", "--metrics-export-port=30006", "--block"]
    if cluster["stage"] == 0:
        command += ["--head", "--port=6379", "--include-dashboard=false"]
    else:
        command += ["--address=" + cluster["address"]]
    return command


def uses_ray(config):
    return (config["cluster"]["tp"] * config["cluster"]["pp"] > 1 and
            config.get("distributed_executor_backend", "ray") == "ray")


def create_app(config):
    cluster = config["cluster"]
    if not uses_ray(config):
        if cluster["pp"] != 1 or cluster["stage"] != 0:
            raise ValueError("Local execution requires one node and stage zero")
        app = worker.create_app(config)
        app.state.worker["identity"]["cluster"] = cluster

        @app.get("/cluster")
        async def single_topology():
            return {"stage": 0, "nodes_at_startup": [{"node_id": "local", "ip": os.environ["POD_IP"],
                "npu_resources": cluster["tp"]}], "scope": ("Single-process model execution" if cluster["tp"] == 1
                else "Local multiprocessing model execution"), **app.state.worker["identity"]}
        return app
    import ray
    state = {"process": None, "connected": False, "draining": False, "nodes": []}
    info = {**worker.identity(config), "cluster": cluster}
    is_head = cluster["stage"] == 0
    if is_head:
        app = worker.create_app(config)
        app.state.worker["identity"] = info
        original = app.router.lifespan_context
    else:
        app = FastAPI()
        @app.get("/identity")
        async def identity():
            return info

        @app.get("/ready")
        async def ready():
            if not state["connected"] or state["draining"] or state["process"].returncode is not None:
                raise HTTPException(503, "Ray stage is unavailable")
            return {"ready": True, "scope": "Ray worker membership; frontend is on stage 0", **info}

        @app.get("/live")
        async def live():
            if state["process"] is not None and state["process"].returncode is not None:
                raise HTTPException(503, "Ray worker exited")
            return {"live": True}

    @app.get("/cluster")
    async def topology():
        return {"stage": cluster["stage"], "nodes_at_startup": state["nodes"], **info}

    async def connect():
        address = "127.0.0.1:6379" if is_head else cluster["address"]
        deadline = time.monotonic() + config["cluster_startup_seconds"]
        while time.monotonic() < deadline:
            if state["process"].returncode is not None:
                raise RuntimeError("Ray exited during initialization")
            host, port = address.rsplit(":", 1)
            try:
                with socket.create_connection((host, int(port)), timeout=2):
                    break
            except OSError:
                await asyncio.sleep(2)
        else:
            raise TimeoutError("Ray control-plane socket did not become available")
        # A reachable head socket can precede local Ray-node registration during the first startup.
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((os.environ["POD_IP"], 30001), timeout=2):
                    break
            except OSError:
                await asyncio.sleep(1)
        else:
            raise TimeoutError("Local Ray node manager did not become available")
        await asyncio.wait_for(asyncio.to_thread(ray.init, address=address,
            ignore_reinit_error=True, log_to_driver=False, _node_ip_address=os.environ["POD_IP"]), config["cluster_startup_seconds"])
        while time.monotonic() < deadline:
            nodes = await asyncio.to_thread(ray.nodes)
            alive = [n for n in nodes if n["Alive"]]
            if len(alive) == cluster["pp"] and all(n["Resources"].get("NPU") == cluster["tp"] for n in alive):
                state["nodes"] = [{"node_id": n["NodeID"], "ip": n["NodeManagerAddress"],
                                   "npu_resources": n["Resources"]["NPU"]} for n in alive]
                state["connected"] = True
                print(json.dumps({"event": "ray_cluster_verified", "stage": cluster["stage"],
                                  "nodes": state["nodes"], "physical_devices": info["physical_devices"]}), flush=True)
                return
            await asyncio.sleep(2)
        raise TimeoutError("Ray membership or NPU resource count disagrees with TP/PP")

    @asynccontextmanager
    async def lifespan(app):
        if len(info["physical_devices"]) != cluster["tp"]:
            raise RuntimeError("Kubernetes allocation differs from the per-node TP size")
        state["process"] = await asyncio.create_subprocess_exec(*ray_command(config), start_new_session=True)
        try:
            await connect()
            if is_head:
                async with original(app):
                    yield
            else:
                yield
        finally:
            state["draining"] = True
            ray.shutdown()
            process = state["process"]
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()

    app.router.lifespan_context = lifespan
    return app










def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    # Ray actors must inherit dispatch settings before their interpreter imports
    # TorchNPU; setting only the frontend subprocess environment is too late.
    host_env = config.get("host_env", {})
    if set(host_env) - {"TASK_QUEUE_ENABLE", "CPU_AFFINITY_CONF"}:
        raise ValueError("Unknown host environment setting")
    os.environ.update(host_env)
    os.environ.update(VLLM_HOST_IP=os.environ["POD_IP"], HCCL_IF_IP=os.environ["POD_IP"],
        GLOO_SOCKET_IFNAME="eth0", TP_SOCKET_IFNAME="eth0", RAY_USAGE_STATS_ENABLED="0",
        RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="1",
        VLLM_USE_V1="0", VLLM_CONFIG_ROOT="/scratch/vllm-config")
    root = Path(os.environ["VLLM_CONFIG_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    if uses_ray(config):
        os.environ["RAY_ADDRESS"] = config["cluster"]["address"]
        (root / "ray_non_carry_over_env_vars.json").write_text(json.dumps(["HCCL_IF_IP", "POD_IP", "POD_UID", "NODE_NAME", "ASCEND_RT_VISIBLE_DEVICES"]))
    else:
        os.environ.pop("RAY_ADDRESS", None)
    uvicorn.run(create_app(config), host="0.0.0.0", port=8000, workers=1,
                access_log=False, timeout_graceful_shutdown=config["drain_seconds"] + 5)


if __name__ == "__main__":
    main()
