"""Compose retained vendor AWQ servers, official vLLM Router and the shared Gateway.

Run the controller on the workstation with --ssh-config, or inside an existing
container with --local --only SERVICE. Never creates containers or starts old PP jobs.
"""
import argparse
import concurrent.futures
import copy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
import re
from pathlib import Path
import shlex
import shutil
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


def router_configs(config):
    """Official L2 routers, with the original single-router format supported."""
    common = {k: v for k, v in config["router"].items() if k != "pools"}
    pools = config["router"].get("pools")
    if not pools:
        return {"router": dict(common, metrics_port=common.get("metrics_port", 18602))}
    return {"router-" + name: dict(common, **pool, pool=name) for name, pool in pools.items()}


def router_workers(config, name):
    spec = router_configs(config)[name]
    nodes = spec.get("nodes")
    return {key: value for key, value in generation_workers(config).items()
            if nodes is None or value["node"] in nodes}


def expand_config(config):
    """Expand device partitions into replicas without duplicating deployment code."""
    result = copy.deepcopy(config)
    if not config.get("replica_groups"):
        return result
    workers = {}
    templates = copy.deepcopy(config["services"])
    active_nodes = set()
    for group in config["replica_groups"]:
        if not group.get("enabled", True):
            continue
        if group["node"] in active_nodes:
            raise ValueError("Duplicate device inventory for node: " + group["node"])
        active_nodes.add(group["node"])
        used = list(group.get("reserved_devices", []))
        for index, devices in enumerate(group["device_sets"]):
            if config.get("required_awq_tp") is not None and len(devices) != config["required_awq_tp"]:
                raise ValueError("AWQ partition does not match required tensor parallel size")
            if not devices or any(d in used for d in devices) or len(set(devices)) != len(devices):
                raise ValueError("Overlapping or empty device partition: " + group["name"])
            used.extend(devices)
            key = ",".join(map(str, devices))
            reused = group.get("reuse", {}).get(key)
            name = reused or f"{group['name']}-{index}"
            if reused:
                service = result["services"][name]
                if service["node"] != group["node"] or service["container"] != group["container"]:
                    raise ValueError("Reused worker belongs to another node/container")
                source = json.loads((ROOT / service["config"]).read_text())
                actual_devices = source["devices"] if service["driver"] == "ascend-int4" else source["vendors"][service["vendor"]]["devices"]
                if actual_devices != devices:
                    raise ValueError("Reused worker device allocation differs")
                endpoint = config["backends"][name]["base_url"]
            else:
                service = copy.deepcopy(templates[group["template"]])
                service.update(node=group["node"], container=group["container"])
                env = {"GLOO_SOCKET_IFNAME": group["interface"]}
                if service["driver"] == "ascend-int4":
                    base = group["communication_port_start"] + index * 128
                    env.update(HCCL_HOST_SOCKET_PORT_RANGE=f"{base}-{base+31}",
                               HCCL_NPU_SOCKET_PORT_RANGE=f"{base+32}-{base+63}", HCCL_IF_BASE_PORT=str(base+64))
                    service["cudagraph_metrics"] = group.get("cudagraph_metrics", False)
                elif service["vendor"] == "musa":
                    env.update(MUSA_VISIBLE_DEVICES=key, MTHREADS_VISIBLE_DEVICES=key,
                               MCCL_SOCKET_IFNAME=group["interface"])
                else:
                    env.update(CUDA_VISIBLE_DEVICES=key, IX_VISIBLE_DEVICES=key,
                               NCCL_SOCKET_IFNAME=group["interface"])
                port = group["port_start"] + index
                service["overrides"] = dict(host=group["host"], port=port, devices=devices,
                                            model_path=group["model_path"], tp=len(devices), env=env)
                for option in ("enforce_eager", "engine_args"):
                    if option in group:
                        service["overrides"][option] = copy.deepcopy(group[option])
                result["services"][name] = service
                endpoint = f"http://{group['host']}:{port}/v1"
            workers[name] = dict(base_url=endpoint, model="Qwen3-14B-AWQ", node=group["node"], devices=devices)
        if sorted(used) != list(range(group["device_count"])):
            raise ValueError("Device coverage is incomplete: " + group["name"])
    result["workers"] = workers
    if config["router"].get("pools"):
        template = templates.get("router") or next(s for s in templates.values() if s["driver"] == "router")
        result["services"] = {n:s for n,s in result["services"].items() if s["driver"] != "router"}
        assigned = set()
        for name, spec in router_configs(result).items():
            names = set(router_workers(result, name))
            if not names or assigned.intersection(names):
                raise ValueError("Empty or overlapping official Router worker pools")
            assigned.update(names)
            result["services"][name] = dict(template, router_pool=spec["pool"])
        if assigned != set(workers):
            raise ValueError("Official Router pools do not cover all generation workers")
    # Keep frontend services last so a default stop drains ingress before workers.
    result["services"] = {**{n:s for n,s in result["services"].items() if s["driver"] not in {"gateway","router"}},
                          **{n:s for n,s in result["services"].items() if s["driver"] in {"router","gateway"}}}
    disabled = {n for n,s in result["services"].items() if s.get("enabled", True) is False}
    result["services"] = {n:s for n,s in result["services"].items() if n not in disabled}
    result["backends"] = {n:s for n,s in result["backends"].items() if n not in disabled}
    return result


def generation_workers(config):
    return config.get("workers") or {k:v for k,v in config["backends"].items() if v.get("task", "generate") == "generate"}


def native_spec(service):
    spec = json.loads((ROOT / service["config"]).read_text())
    target = spec["vendors"][service["vendor"]]
    # A verified vendor exception can retain eager while other vendors use graphs.
    spec["enforce_eager"] = target.get("enforce_eager", spec.get("enforce_eager", False))
    for key, value in service.get("overrides", {}).items():
        if key == "env":
            target["env"].update(value)
        elif key in {"host", "port", "devices", "engine_args"}:
            target[key] = value
        else:
            spec[key] = value
    return spec, target


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
    result.update(learned_routing={"enabled": False}, pools=copy.deepcopy(config["backends"]))
    routers = router_configs(config)
    if config["router"].get("pools"):
        for spec in routers.values():
            result["pools"][spec["backend"]].update(
                base_url=f"http://{spec['host']}:{spec['port']}/v1", health_path="/readiness")
        result["random_routes"] = {"awq": [r["backend"] for r in routers.values()]}
        result.setdefault("random_route_health_seconds", 5)
        result["worker_registry"] = {"urls": [f"http://{r['host']}:{r['port']}/workers" for r in routers.values()]}
    else:
        router = routers["router"]
        result["pools"]["awq"] = dict(model="Qwen3-14B-AWQ",
            base_url=f"http://{router['host']}:{router['port']}/v1", health_path="/readiness")
        result["worker_registry"] = {"url": f"http://{router['host']}:{router['port']}/workers"}
    result["worker_registry"]["expected_urls"] = [w["base_url"].removesuffix("/v1") for w in generation_workers(config).values()]
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
    elif config["services"][name]["driver"] == "router":
        identity["router"] = router_configs(config)[name]
        identity["workers"] = router_workers(config, name)
    elif config["services"][name]["driver"] == "native-awq":
        spec, target = native_spec(config["services"][name])
        identity["native"] = {k: v for k, v in spec.items() if k != "benchmark" and k != "vendors"}
        identity["native"].update(target=target, seed=spec["benchmark"]["seed"])
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def commands(config, name):
    service = config["services"][name]
    if service["driver"] == "native-awq":
        spec, target = native_spec(service)
        argv = [shutil.which("vllm") or "vllm", "serve", spec["model_path"], "--served-model-name", spec["model"],
                "--quantization", spec["quantization"], "--dtype", spec["dtype"],
                "--tensor-parallel-size", str(spec["tp"]), "--pipeline-parallel-size", str(spec["pp"]),
                "--distributed-executor-backend", "mp", "--host", target["host"], "--port", str(target["port"]),
                "--max-model-len", str(spec["max_model_len"]), "--max-num-seqs", str(spec["max_num_seqs"]),
                "--max-num-batched-tokens", str(spec["max_num_batched_tokens"]),
                "--gpu-memory-utilization", str(spec["gpu_memory_utilization"])]
        if type(spec.get("enforce_eager", False)) is not bool:
            raise ValueError("enforce_eager must be boolean")
        if spec.get("enforce_eager", False):
            argv += ["--enforce-eager"]
        argv += ["--enable-prefix-caching" if spec.get("prefix_caching", False) else "--no-enable-prefix-caching",
                 "--seed", str(spec["benchmark"]["seed"]), *target.get("engine_args", [])]
        return argv, target["env"]
    if service["driver"] == "embedding":
        if not (Path(service["model_path"]) / "DOWNLOAD_COMPLETE").exists():
            raise RuntimeError("Embedding download is not verified complete yet")
        return service["argv"], service.get("env", {})
    if service["driver"] == "gateway":
        target = ROOT / config["result_dir"] / "gateway.json"
        save(target, gateway_config(config))
        return [sys.executable, "-m", "heteroserve.gateway", "--config", str(target)], {}
    if service["driver"] == "router":
        router = router_configs(config)[name]
        workers = [v["base_url"].removesuffix("/v1") for v in router_workers(config, name).values()]
        # Preserve the vendor's PYTHONPATH; the router target contains only its wheels.
        bootstrap = "import sys,runpy; sys.path.insert(0," + repr(router["runtime"]) + "); sys.argv="
        argv = ["launch_router", "--host", router["host"], "--port", str(router["port"]),
                "--worker-urls", *workers, "--policy", router["policy"], "--disable-retries",
                "--request-timeout-secs", "180", "--health-check-interval-secs", "10",
                "--worker-startup-check-interval", "5", "--worker-startup-timeout-secs", "600",
                "--prometheus-host", "127.0.0.1", "--prometheus-port", str(router["metrics_port"]), "--max-payload-size", "2097152"]
        bootstrap += repr(argv) + "; runpy.run_module('vllm_router.launch_router',run_name='__main__')"
        return [sys.executable, "-c", bootstrap], {}
    raise ValueError("Unknown service driver")


def serve_signature(argv):
    """Normalize only native vLLM serve commands, including executable shebangs."""
    if "serve" not in argv:
        return None
    index = argv.index("serve")
    if not any(Path(x).name == "vllm" or x == "vllm.entrypoints.cli.main" for x in argv[:index]):
        return None
    args = argv[index + 1:]
    if not args or args[0].startswith("--"):
        return None
    result = {"model_path": args[0]}
    index = 1
    while index < len(args):
        key = args[index]
        if not key.startswith("--") or key in result:
            return None
        value = True
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            value = args[index + 1]
            if key == "--dtype":
                value = {"half": "float16"}.get(value, value)
            else:
                try:
                    value = str(Decimal(value).normalize())
                except InvalidOperation:
                    pass
            index += 1
        result[key] = value
        index += 1
    return result


def legacy_processes():
    roots = {"/workspace/heterogeneous_compute/pipeline_execution.py", "/workspace/src/heteroserve/pipeline_execution.py",
             "/workspace/homogeneous/experiments/pp_baseline/run_node.py"}
    found = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            if stamp(int(path.name)) is None:
                continue
            argv = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
            if not argv:
                continue
            old_script = any(x in roots for x in argv)
            old_model = "serve" in argv and any(x in argv for x in (
                "/models/Qwen3-14B", "/workspace/deploy/baselines/vllm.yaml", "/workspace/deploy/iluvatar/vllm.yaml"))
            if old_script or old_model:
                found.append({"pid": int(path.name), "argv": argv})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return found


def adopt_native(config, name, argv, extra, marker):
    expected = serve_signature(argv)
    if expected is None:
        raise ValueError("Native AWQ requires a complete vLLM serve command")
    candidates = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(path.name)
            identity = stamp(pid)
            if identity is None:
                continue
            actual = [x.decode(errors="replace") for x in (path / "cmdline").read_bytes().split(b"\0") if x]
            if serve_signature(actual) != expected:
                continue
            env = dict(x.split("=", 1) for x in (path / "environ").read_text().split("\0") if "=" in x)
            differing = [key for key, value in extra.items() if env.get(key) != value]
            if differing:
                raise RuntimeError("Existing AWQ environment differs: " + ", ".join(differing))
            if os.getpgid(pid) != pid:
                raise RuntimeError("Existing AWQ process does not own a dedicated process group")
            candidates.append({"pid": pid, "stamp": identity, "argv": argv, "adopted_argv": actual,
                               "adopted": True, "config_sha256": fingerprint(config, name)})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if len(candidates) > 1:
        raise RuntimeError("Multiple matching AWQ processes; refusing ambiguous adoption")
    if candidates:
        if stamp(candidates[0]["pid"]) != candidates[0]["stamp"]:
            raise RuntimeError("Existing AWQ exited during adoption")
        save(marker, candidates[0])
        print(json.dumps({"service": name, "adopted": True, "pid": candidates[0]["pid"]}))
        return True
    return False


def local(config, action, name):
    directory = ROOT / config["result_dir"] / name
    marker = directory / "process.json"
    service = config["services"][name]
    if action == "start" and config.get("legacy_baselines_enabled") is False:
        old = legacy_processes()
        if old:
            raise RuntimeError("Retired baseline processes are still running: " + str([r["pid"] for r in old]))
    if service["driver"] == "ascend-int4":
        from heterogeneous_compute.quantization.deploy import start_server, stop_server
        spec = json.loads((ROOT / service["config"]).read_text())
        overrides = service.get("overrides", {})
        spec.update({k:v for k,v in overrides.items() if k not in {"env","port","model_path"}})
        spec["env"].update(overrides.get("env", {}))
        if "port" in overrides:
            spec["profiles"]["awq"]["port"] = overrides["port"]
        if "model_path" in overrides:
            spec["profiles"]["awq"]["model"] = overrides["model_path"]
        spec["result_dir"] = config["result_dir"] + "/" + name
        spec["cudagraph_metrics"] = service.get("cudagraph_metrics", False)
        if action == "start":
            require_checkpoint(spec["profiles"]["awq"]["model"])
            start_server(spec, "awq")
        elif action == "stop":
            ascend_marker = directory / "awq/server.json"
            if ascend_marker.exists() and stamp(json.loads(ascend_marker.read_text())["pid"]) is None:
                print(json.dumps({"service": name, "stopped": True, "already_exited": True}))
                return
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
    if service["driver"] == "native-awq":
        require_checkpoint(native_spec(service)[0]["model_path"])
    if service["driver"] == "native-awq" and adopt_native(config, name, argv, extra, marker):
        return
    env = dict(os.environ, **extra)
    env["PYTHONPATH"] = str(ROOT / "homogeneous/src") + ":" + str(ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    spec = config.get("workers", {}).get(name) or config["backends"].get(name)
    endpoint = urlsplit(spec["base_url"]) if spec else None
    listen = router_configs(config)[name] if service["driver"] == "router" else config.get(name, {})
    port = endpoint.port if endpoint else listen["port"]
    host = endpoint.hostname if endpoint else listen["host"]
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


def require_checkpoint(model_path):
    path = Path(model_path)
    if not any((path / name).exists() for name in ["QUANTIZATION_COMPLETE", "DOWNLOAD_COMPLETE"]):
        raise RuntimeError("AWQ checkpoint has not completed verified distribution: " + str(path))
    quant = json.loads((path / "config.json").read_text()).get("quantization_config", {})
    if quant.get("quant_method") != "awq" or quant.get("bits") != 4 or quant.get("group_size") != 128:
        raise ValueError("Expected the existing standard AWQ/group128 checkpoint")


def status(config):
    endpoints = {name: spec["base_url"].removesuffix("/v1") for name, spec in config["backends"].items()}
    endpoints.update({name: spec["base_url"].removesuffix("/v1") for name, spec in generation_workers(config).items()})
    endpoints["gateway"] = f"http://{config['gateway']['address']}:{config['gateway']['port']}"
    endpoints.update({n:f"http://{s['host']}:{s['port']}" for n,s in router_configs(config).items()})
    result = {}
    for name, url in endpoints.items():
        try:
            path = "/ready" if name == "gateway" else "/readiness" if name in router_configs(config) else "/health"
            response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url + path, timeout=4)
            with response:
                detail = response.read().decode()
            result[name] = {"healthy": True, "detail": detail[:2000]}
        except Exception as exc:
            result[name] = {"healthy": False, "error": str(exc)}
    save(ROOT / config["result_dir"] / "status.json", result)
    print(json.dumps(result, ensure_ascii=False))


def verify_workers(config):
    """Keep every worker's outcome, including unavailable devices, for acceptance."""
    def direct(item):
        name, worker = item
        result = {"worker": name, "devices": worker.get("devices"), "passed": False}
        try:
            response = http(worker["base_url"] + "/chat/completions", {"model": worker["model"],
                "messages": [{"role": "user", "content": "What is 2+3? Reply with only the digit."}],
                "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
            answer = response["choices"][0]["message"]["content"].strip()
            result.update(passed=answer == "5", answer=answer)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            result["error"] = str(exc)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(direct, generation_workers(config).items()))


def verify(config):
    base = f"http://{config['gateway']['address']}:{config['gateway']['port']}"
    ready = http(base + "/ready")
    routers = router_configs(config)
    workers = [worker for spec in routers.values()
               for worker in http(f"http://{spec['host']}:{spec['port']}/workers")["workers"]]
    expected = {v["base_url"].removesuffix("/v1") for v in generation_workers(config).values()}
    registered_healthy = {w["url"] for w in workers if w["is_healthy"]}
    coverage_complete = ready["expert_coverage_complete"] and registered_healthy == expected
    tests = []
    direct_results = verify_workers(config)
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
        result = {}
        for spec in routers.values():
            with opener.open(f"http://{spec['host']}:{spec['metrics_port']}/metrics", timeout=10) as response:
                raw = response.read().decode()
            result.update({worker: float(value) for worker, value in re.findall(
                r'^vllm_router_processed_requests_total\{worker="([^"]+)"\}\s+([0-9.eE+\-]+)', raw, re.M)})
        return result
    before = counters()
    def pooled(_, model="awq"):
        response = http(base + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "What is 2+3? Reply with only the digit."}],
            "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=120)
        assert response["choices"][0]["message"]["content"].strip() == "5"
    pooled_count = max(24, 2 * len(expected))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(pooled, range(pooled_count)))
    after = counters()
    dispatched = {worker: after.get(worker, 0) - before.get(worker, 0) for worker in expected}
    assert sum(n > 0 for n in dispatched.values()) >= min(3, len(expected)), dispatched
    assert sum(dispatched.values()) >= pooled_count, dispatched
    explicit_coverage_requests = 0
    if config["router"].get("pools"):
        # A finite random L1 sample need not visit every L2 instance. Verify
        # each pool's round-robin separately without changing the L1 policy.
        for name, spec in routers.items():
            count = len(router_workers(config, name))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda i: pooled(i, spec["backend"]), range(count)))
            explicit_coverage_requests += count
        after = counters()
        dispatched = {worker: after.get(worker, 0) - before.get(worker, 0) for worker in expected}
    if all(spec["policy"] == "round_robin" for spec in routers.values()):
        assert all(dispatched[url] > 0 for url in registered_healthy), dispatched
    embedding_result = {"skipped": True, "reason": "Embedding disabled in the active configuration"}
    if "embedding" in config["backends"]:
        embedding = http(base + "/v1/embeddings", {"model": "embedding", "input": ["The capital of China is Beijing.", "The capital of China is Beijing.", "An apple is a fruit."]}, timeout=120)
        vectors = [row["embedding"] for row in embedding["data"]]
        import math
        assert embedding["model"] == "embedding" and len(vectors) == 3
        assert all(len(v) == 1024 and all(math.isfinite(x) for x in v) for v in vectors)
        assert max(abs(x-y) for x, y in zip(vectors[0], vectors[1])) < .001
        assert sum((x-y)**2 for x,y in zip(vectors[0], vectors[2])) > .01
        embedding_result = {"count": 3, "dimensions": 1024, "finite": True, "duplicate_consistency": True}
    report = {"passed": bool(coverage_complete and all(r["passed"] for r in direct_results)),
              "ready": ready, "generation": tests, "direct_workers": direct_results,
              "missing_healthy_workers": sorted(expected - registered_healthy),
              "unexpected_healthy_workers": sorted(registered_healthy - expected),
              "official_router": {"policy": config["router"]["policy"], "routers": list(routers),
                  "l1_random_routes": gateway_config(config).get("random_routes", {}),
                  "route_embeddings": gateway_config(config).get("route_embeddings", {}),
                  "pooled_requests": pooled_count, "explicit_coverage_requests": explicit_coverage_requests,
                  "concurrency": 4, "worker_dispatch_deltas": dispatched},
              "embedding": embedding_result,
              "scope": "Functional acceptance only; no throughput/stability/Kubernetes or cross-vendor PP claim",
              "completed_at": datetime.now(timezone.utc).isoformat()}
    save(ROOT / config["result_dir"] / "acceptance.json", report)
    print(json.dumps(report, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit("Full device coverage failed; see acceptance.json for individual outcomes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["sync", "start", "stop", "status", "verify"])
    parser.add_argument("--config", type=Path, default=DEFAULT)
    parser.add_argument("--ssh-config", type=Path)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    config = expand_config(json.loads(args.config.read_text(encoding="utf-8")))
    if config["resource_limits_enabled"] is not False:
        raise ValueError("This deployment keeps resource limits disabled")
    names = args.only or list(config["services"])
    if args.only and "router" not in config["services"]:
        names = [expanded for name in names for expanded in (list(router_configs(config)) if name == "router" else [name])]
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
                   ROOT / "heterogeneous_compute/quantization/deployment.json", ROOT / "heterogeneous_compute/configs/native_awq.json",
                   ROOT / "heterogeneous_compute/tests/test_quantization_deployment.py",
                   ROOT / "heterogeneous_compute/tests/test_model_download.py", ROOT / "heterogeneous_compute/tests/test_int4_compilation.py",
                   ROOT / "heterogeneous_compute/tests/test_serving.py", ROOT / "shared/download_model.py", ROOT / "shared/ascend_int4/vllm_adapter.py",
                   ROOT / "heterogeneous_compute/probes/communication_probe.py",
                   ROOT / "shared/ascend_int4/layout.py", ROOT / "shared/ascend_int4/sitecustomize.py"]
        sources += [ROOT / "heterogeneous_compute/benchmark.py",
                    ROOT / "heterogeneous_compute/system_benchmark.py", ROOT / "heterogeneous_compute/benchmark_telemetry.py",
                    ROOT / "heterogeneous_compute/configs/system_benchmark.json", ROOT / "heterogeneous_compute/tests/test_system_benchmark.py"]
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
