"""Compose retained vendor AWQ servers, official vLLM Router and the shared Gateway.

Run the controller on the workstation with --ssh-config, or inside an existing
container with --local --only SERVICE. Never creates containers or starts old PP jobs.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "heterogeneous_compute/configs/serving.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def http(url, body=None, timeout=10):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def gateway_config(config):
    result = {k: v for k, v in config["gateway"].items() if k != "address"}
    result.update(learned_routing={"enabled": False}, pools=dict(config["backends"]))
    router = config["router"]
    result["pools"]["awq"] = dict(model="Qwen3-14B-AWQ",
        base_url=f"http://{router['host']}:{router['port']}/v1", health_path="/health")
    return result


def stamp(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return [Path("/proc/sys/kernel/random/boot_id").read_text().strip(), fields[19]]
    except FileNotFoundError:
        return None


def live(marker):
    if not marker.exists():
        return False
    data = json.loads(marker.read_text())
    actual = stamp(data["pid"])
    return actual is not None and actual == data["stamp"]


def fingerprint(config, name):
    identity = {"service": config["services"][name]}
    if name == "gateway":
        identity["gateway"] = gateway_config(config)
    elif name == "router":
        identity["router"] = config["router"]
        identity["workers"] = {k: v for k, v in config["backends"].items() if v.get("task", "generate") == "generate"}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def commands(config, name):
    service = config["services"][name]
    if service["driver"] == "embedding":
        if not (Path(service["model_path"]) / "DOWNLOAD_COMPLETE").exists():
            raise RuntimeError("Embedding download is not verified complete yet")
        return service["argv"], service.get("env", {})
    if service["driver"] == "gateway":
        target = ROOT / config["result_dir"] / "gateway.json"
        save(target, gateway_config(config))
        return [sys.executable, "-m", "heteroserve.gateway", "--config", str(target)], {}
    if service["driver"] == "router":
        router = config["router"]
        workers = [v["base_url"].removesuffix("/v1") for v in config["backends"].values() if v.get("task", "generate") == "generate"]
        # Preserve the vendor's PYTHONPATH; the router target contains only its wheels.
        bootstrap = "import sys,runpy; sys.path.insert(0," + repr(router["runtime"]) + "); sys.argv="
        argv = ["launch_router", "--host", router["host"], "--port", str(router["port"]),
                "--worker-urls", *workers, "--policy", router["policy"], "--disable-retries",
                "--request-timeout-secs", "180", "--health-check-interval-secs", "10",
                "--worker-startup-check-interval", "5", "--worker-startup-timeout-secs", "600",
                "--prometheus-port", "18602", "--max-payload-size", "2097152"]
        bootstrap += repr(argv) + "; runpy.run_module('vllm_router.launch_router',run_name='__main__')"
        return [sys.executable, "-c", bootstrap], {}
    raise ValueError("Unknown service driver")


def local(config, action, name):
    directory = ROOT / config["result_dir"] / name
    marker = directory / "process.json"
    service = config["services"][name]
    if service["driver"] == "ascend-int4":
        from heterogeneous_compute.quantization.deploy import start_server, stop_server
        spec = json.loads((ROOT / service["config"]).read_text())
        spec["result_dir"] = config["result_dir"] + "/ascend-awq"
        spec["cudagraph_metrics"] = service.get("cudagraph_metrics", False)
        if action == "start":
            start_server(spec, "awq")
        elif action == "stop":
            stop_server(spec, "awq")
        else:
            print(json.dumps({"service": name, "marker": str(directory / "awq/server.json")}))
        return
    if action == "status":
        print(json.dumps({"service": name, "running": live(marker)}))
        return
    if action == "stop":
        if live(marker):
            state = json.loads(marker.read_text())
            pid = state["pid"]
            assert os.getpgid(pid) == pid, "Process group ownership changed"
            if name == "gateway":
                http(f"http://127.0.0.1:{config['gateway']['port']}/drain", {},
                     timeout=config["gateway"].get("drain_seconds", 240) + 10)
            os.killpg(pid, signal.SIGTERM)
            for _ in range(120):
                if not live(marker):
                    break
                time.sleep(.25)
            if live(marker):
                os.killpg(pid, signal.SIGKILL)
        print(json.dumps({"service": name, "stopped": True}))
        return
    if live(marker):
        state = json.loads(marker.read_text())
        digest = fingerprint(config, name)
        if state["config_sha256"] != digest:
            raise RuntimeError("Running service configuration changed; stop it before restarting")
        print(json.dumps({"service": name, "reused": True, "pid": state["pid"]}))
        return
    argv, extra = commands(config, name)
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = str(ROOT / "homogeneous/src") + ":" + str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    endpoint = urlsplit(config["backends"][name]["base_url"]) if name == "embedding" else None
    port = endpoint.port if endpoint else config[name]["port"]
    host = endpoint.hostname if endpoint else config[name]["host"]
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind((host, port))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "server.log").open("a") as log:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=log,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    identity = stamp(process.pid)
    if identity is None:
        raise RuntimeError("Service exited during startup; inspect server.log")
    save(marker, {"pid": process.pid, "stamp": identity, "argv": argv,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "config_sha256": fingerprint(config, name)})
    print(json.dumps({"service": name, "pid": process.pid, "log": str(directory / "server.log")}))


def status(config):
    endpoints = {name: spec["base_url"].removesuffix("/v1") for name, spec in config["backends"].items()}
    endpoints["gateway"] = f"http://{config['gateway']['address']}:{config['gateway']['port']}"
    endpoints["router"] = f"http://{config['router']['host']}:{config['router']['port']}"
    result = {}
    for name, url in endpoints.items():
        try:
            path = "/ready" if name == "gateway" else "/health"
            response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + path, timeout=4)
            with response:
                detail = response.read().decode()
            result[name] = {"healthy": True, "detail": detail[:2000]}
        except Exception as exc:
            result[name] = {"healthy": False, "error": str(exc)}
    save(ROOT / config["result_dir"] / "status.json", result)
    print(json.dumps(result, ensure_ascii=False))


def verify(config):
    base = f"http://{config['gateway']['address']}:{config['gateway']['port']}"
    ready = http(base + "/ready")
    assert ready["expert_coverage_complete"], ready
    router_url = f"http://{config['router']['host']}:{config['router']['port']}"
    workers = http(router_url + "/workers")["workers"]
    expected = {v["base_url"].removesuffix("/v1") for v in config["backends"].values() if v.get("task", "generate") == "generate"}
    assert {w["url"] for w in workers if w["is_healthy"]} == expected, workers
    tests = []
    for model in [n for n, spec in config["backends"].items() if spec.get("task", "generate") == "generate"] + ["awq"]:
        payload = {"model": model, "messages": [{"role": "user", "content": "What is 2+3? Reply with only the digit."}],
                   "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        response = http(base + "/v1/chat/completions", payload, timeout=120)
        text = response["choices"][0]["message"]["content"].strip()
        assert response["model"] == model and text == "5", (model, text)
        payload["stream"] = True
        request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as stream:
            lines = stream.read().decode().splitlines()
        events = [line[5:].strip() for line in lines if line.startswith("data:")]
        assert events and events[-1] == "[DONE]", model
        chunks = [json.loads(line) for line in events[:-1]]
        assert all(event.get("model", model) == model for event in chunks)
        text = "".join(choice.get("delta", {}).get("content") or "" for event in chunks for choice in event.get("choices", []))
        assert text.strip() == "5", (model, text)
        tests.append({"model": model, "json": True, "sse_done": True, "answer": text.strip()})
    def counters():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://{config['router']['host']}:18602/metrics", timeout=10) as response:
            raw = response.read().decode()
        return {worker: float(value) for worker, value in re.findall(
            r'^vllm_router_processed_requests_total\{worker="([^"]+)"\}\s+([0-9.eE+\-]+)', raw, re.M)}
    before = counters()
    def pooled(_):
        response = http(base + "/v1/chat/completions", {"model": "awq", "messages": [{"role": "user", "content": "What is 2+3? Reply with only the digit."}],
            "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
        assert response["choices"][0]["message"]["content"].strip() == "5"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(pooled, range(24)))
    after = counters()
    dispatched = {worker: after.get(worker, 0) - before.get(worker, 0) for worker in expected}
    assert all(n > 0 for n in dispatched.values()), dispatched
    assert sum(dispatched.values()) >= 24, dispatched
    embedding = http(base + "/v1/embeddings", {"model": "embedding", "input": ["The capital of China is Beijing.", "The capital of China is Beijing.", "An apple is a fruit."]}, timeout=120)
    vectors = [row["embedding"] for row in embedding["data"]]
    import math
    assert embedding["model"] == "embedding" and len(vectors) == 3
    assert all(len(v) == 1024 and all(math.isfinite(x) for x in v) for v in vectors)
    assert max(abs(x-y) for x, y in zip(vectors[0], vectors[1])) < .001
    assert sum((x-y)**2 for x,y in zip(vectors[0], vectors[2])) > .01
    report = {"passed": True, "ready": ready, "generation": tests,
              "official_router": {"policy": config["router"]["policy"], "pooled_requests": 24, "concurrency": 4, "worker_dispatch_deltas": dispatched},
              "embedding": {"count": 3, "dimensions": 1024, "finite": True, "duplicate_consistency": True},
              "scope": "Functional acceptance only; no throughput/stability/Kubernetes or cross-vendor PP claim",
              "completed_at": datetime.now(timezone.utc).isoformat()}
    save(ROOT / config["result_dir"] / "acceptance.json", report)
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sync", "start", "stop", "status", "verify"])
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["resource_limits_enabled"] is not False:
        raise ValueError("This deployment keeps resource limits disabled")
    names = args.only or list(config["services"])
    if any(n not in config["services"] for n in names):
        parser.error("Unknown service")
    if args.local:
        if args.action == "sync":
            parser.error("sync must run from the workstation")
        if args.action == "verify" or (args.action == "status" and not args.only):
            return {"status": status, "verify": verify}[args.action](config)
        for name in reversed(names) if args.action == "stop" else names:
            local(config, args.action, name)
        return
    if args.ssh_config is None:
        parser.error("--ssh-config required for workstation controller")
    ssh = ["ssh", "-F", str(args.ssh_config), "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    if args.action == "sync":
        sources = [Path(__file__), args.config.resolve(), ROOT / "homogeneous/src/heteroserve/gateway.py", ROOT / "homogeneous/src/heteroserve/routing/runtime.py",
                   ROOT / "homogeneous/src/heteroserve/__init__.py", ROOT / "homogeneous/src/heteroserve/routing/__init__.py", ROOT / "heterogeneous_compute/__init__.py",
                   ROOT / "homogeneous/deploy/gateway.json", ROOT / "homogeneous/tests/test_gateway.py", ROOT / "heterogeneous_compute/quantization/deploy.py",
                   ROOT / "heterogeneous_compute/quantization/deployment.json", ROOT / "heterogeneous_compute/tests/test_quantization_deployment.py",
                   ROOT / "heterogeneous_compute/tests/test_model_download.py", ROOT / "heterogeneous_compute/tests/test_int4_compilation.py",
                   ROOT / "heterogeneous_compute/tests/test_serving.py", ROOT / "shared/download_model.py", ROOT / "shared/ascend_int4/vllm_adapter.py"]
        manifest = ROOT / config["result_dir"] / "source-manifest.json"
        save(manifest, {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
        sources.append(manifest)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as bundle:
            for source in sources:
                bundle.add(source, arcname=source.relative_to(ROOT).as_posix())
        for node in dict.fromkeys(config["services"][n]["node"] for n in names):
            service = next(s for s in config["services"].values() if s["node"] == node)
            command = ["docker", "exec", "-i", service["container"], "tar", "-xf", "-", "-C", "/workspace"]
            subprocess.run([*ssh, node, shlex.join(command)], input=archive.getvalue(), check=True)
        return
    if args.action in {"status", "verify"}:
        names = ["gateway"]
    for name in reversed(names) if args.action == "stop" else names:
        spec = config["services"][name]
        relative = args.config.resolve().relative_to(ROOT).as_posix()
        command = ["docker", "exec", spec["container"], "python3", "/workspace/heterogeneous_compute/serving.py", args.action, "--local", "--config", "/workspace/" + relative]
        if args.action not in {"status", "verify"}:
            command += ["--only", name]
        subprocess.run([*ssh, spec["node"], shlex.join(command)], check=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
