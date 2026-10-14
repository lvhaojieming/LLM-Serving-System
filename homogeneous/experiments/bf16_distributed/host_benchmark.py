"""Measure host dispatch on retained model Pods and restore paused templates."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import time
import traceback

import manage as manager

ROOT = manager.ROOT / "artifacts/bf16_distributed/system/diagnostics"
PROFILE = "/scratch/host-execution-profile"


def inventory(cluster, kind, expert=None):
    args = ["get", kind, "-n", "heteroserve", "-o", "json"]
    if expert:
        args += ["-l", "heteroserve.io/distributed-group=" + expert + "-distributed"]
    return json.loads(manager.kubectl(cluster, args).stdout)["items"]


def prepared(expert, settings, profile=True, local_bf16=False):
    cluster, pool, runtime = manager.configuration(expert)
    if local_bf16:
        assert expert == "bf16"
        _, awq, _ = manager.configuration("awq")
        pool.update(nodes=awq["nodes"], model_nodes=awq["nodes"], parallelism=awq["parallelism"],
                    host_execution=awq["host_execution"])
        # Original BF16 weights fill the 50% budget on two 32 GiB cards.
        # This isolated diagnostic needs room for KV cache; the PP2 baseline
        # and AWQ settings are restored by the existing cleanup path.
        runtime["model_parameters"] = {**runtime["model_parameters"], "gpu_memory_utilization": .8}
    if settings is not None:
        pool["host_execution"] = settings
    manifest = manager.objects(cluster, pool, runtime)
    if profile:
        for obj in manifest["items"]:
            if obj["kind"] == "ConfigMap" and "worker.json" in obj["data"]:
                obj["data"] = manager.profiler_data(obj["data"])
            if obj["kind"] == "Deployment":
                container = obj["spec"]["template"]["spec"]["containers"][0]
                container["env"].append({"name": "VLLM_TORCH_PROFILER_DIR", "value": PROFILE})
    return cluster, pool, runtime, manifest


def wait_ready(cluster, expert, expected=None, restart=None, count=None):
    count = count or (len(expected) if expected else (2 if expert == "bf16" else 1))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        pods = [p for p in inventory(cluster, "pods", expert) if not p["metadata"].get("deletionTimestamp")]
        if expected:
            assert {p["metadata"]["uid"] for p in pods} == set(expected), "Pod allocation changed"
        if len(pods) == count and all(any(c["type"] == "Ready" and c["status"] == "True"
                            for c in p["status"].get("conditions", [])) for p in pods):
            if restart is None or all(p["status"]["containerStatuses"][0]["restartCount"] >
                                      restart[p["metadata"]["uid"]] for p in pods):
                return pods
        time.sleep(4)
    raise TimeoutError(expert + " did not become ready")


def execute(cluster, pod, source, data):
    shell = "source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 -u -c " + shlex.quote(source)
    result = manager.kubectl(cluster, ["exec", "-i", "-n", "heteroserve", pod, "--", "bash", "-ec", shell],
                             stdin=json.dumps(data), timeout=600)
    return json.loads(result.stdout)


def restart_in_place(cluster, expert, pods, manifest):
    cms = {x["metadata"]["name"]: x for x in manifest["items"] if x["kind"] == "ConfigMap"}
    restart = {p["metadata"]["uid"]: p["status"]["containerStatuses"][0]["restartCount"] for p in pods}
    for pod in pods:
        name = expert + "-distributed-stage-" + pod["metadata"]["labels"]["heteroserve.io/stage"] + "-runtime"
        desired = cms[name]["data"]
        manager.kubectl(cluster, ["patch", "cm", name, "-n", "heteroserve", "--type=merge", "-p", json.dumps({"data": desired})])
        expected = hashlib.sha256(desired["worker.json"].encode()).hexdigest()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            source = "import hashlib;print(hashlib.sha256(open('/opt/heteroserve/worker.json','rb').read()).hexdigest())"
            actual = manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", source]).stdout.strip()
            if actual == expected:
                break
            time.sleep(3)
        else:
            raise TimeoutError("ConfigMap did not project")
        manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", "import os,signal;os.kill(1,signal.SIGTERM)"])
    return wait_ready(cluster, expert, restart.keys(), restart)


def measure(cluster, expert, name, pods, speed=True):
    head = next(p for p in pods if p["metadata"]["labels"]["heteroserve.io/stage"] == "0")["metadata"]["name"]
    bench = (manager.HERE / "benchmark.py").read_text()
    sample = json.loads((ROOT / "sample.json").read_text())
    identities, ranks = [], []
    for pod in pods:
        source = "import urllib.request,json;print(urllib.request.urlopen('http://127.0.0.1:8000/identity').read().decode())"
        identities.append(json.loads(manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", source]).stdout))
        source = "from pathlib import Path;[p.unlink() for p in Path(" + repr(PROFILE) + ").glob('*-timing.json')]"
        manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", source])
    profile = execute(cluster, head, (ROOT / "model-speed.py").read_text(),
                      {"sample": sample, "benchmark": bench, "profile": True, "output_tokens": 64, "concurrency": 8})
    for pod in pods:
        source = "from pathlib import Path;import json;print(json.dumps([json.loads(p.read_text()) for p in Path(" + repr(PROFILE) + ").glob('*-timing.json')]))"
        ranks += json.loads(manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", source]).stdout)
        (ROOT / ("host-" + name + "-" + pod["metadata"]["labels"]["heteroserve.io/stage"] + ".log")).write_text(
            manager.kubectl(cluster, ["logs", "-n", "heteroserve", pod["metadata"]["name"]]).stdout)
    assert len(ranks) == 2 * len(pods)
    labels = sorted({key for rank in ranks for key in rank["measurements"]})
    summary = {key: statistics.mean(statistics.median(rank["measurements"][key]) for rank in ranks if rank["measurements"].get(key)) for key in labels}
    result = {"identities": identities, "profile": profile, "ranks": ranks,
              "cpu_mean_of_rank_medians_us": summary, "scope": "Host wall intervals include waits; nested labels overlap"}
    if speed:
        rows = [json.loads(x) for x in (manager.ROOT / "artifacts/bf16_distributed/requests.jsonl").read_text().splitlines()][3:203]
        result["speed"] = execute(cluster, head, (ROOT / "execution-compare.py").read_text(),
            {"rows": rows, "benchmark": bench, "sample": sample, "quality_rounds": 0, "repeats": 1,
             "output_tokens": 64, "concurrency": 8, "scope": "Same Pod/NPU allocation; profiling disabled during 200 requests; timing wrappers retained"})
    (ROOT / ("host-" + name + ".json")).write_text(json.dumps(result, indent=2))
    return {"file": "host-" + name + ".json", "cpu": summary, "identities": identities,
            "dispatch_env": [rank["dispatch_env"] for rank in ranks],
            "speed": result.get("speed", {}).get("cases", [{}])[0].get("summary")}


def measure_ttft(cluster, expert, pods, name=None):
    name = name or expert
    head = next(p for p in pods if p["metadata"]["labels"]["heteroserve.io/stage"] == "0")["metadata"]["name"]
    rows = [json.loads(x) for x in (manager.ROOT / "artifacts/bf16_distributed/requests.jsonl").read_text().splitlines()][3:203]
    source = """import sys,json,types,asyncio,contextlib
args=json.load(sys.stdin)
bench=types.ModuleType('bench');exec(args['benchmark'],bench.__dict__)
config=json.load(open('/opt/heteroserve/worker.json'))
with contextlib.redirect_stdout(sys.stderr):
 result=asyncio.run(bench.diagnose_ttft(config,args['rows'],args['sample']))
print(json.dumps(result))
"""
    result = execute(cluster, head, source, {"rows": rows,
        "sample": json.loads((ROOT / "sample.json").read_text()),
        "benchmark": (manager.HERE / "benchmark.py").read_text()})
    result["pods"] = [{"uid": pod["metadata"]["uid"], "node": pod["spec"]["nodeName"],
                       "name": pod["metadata"]["name"]} for pod in pods]
    result["identities"] = []
    for pod in pods:
        identity = manager.kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--",
            "python3", "-c", "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/identity').read().decode())"]).stdout
        result["identities"].append(json.loads(identity))
        (ROOT / ("ttft-" + name + "-" + pod["metadata"]["labels"]["heteroserve.io/stage"] + ".log")).write_text(
            manager.kubectl(cluster, ["logs", "-n", "heteroserve", pod["metadata"]["name"]]).stdout)
    (ROOT / ("ttft-" + name + ".json")).write_text(json.dumps(result, indent=2))
    return {"file": "ttft-" + name + ".json", "identities": result["identities"],
            "cases": [{key: case[key] for key in ("name", "summary", "components")} for case in result["cases"]]}


def system_profiles():
    from manage_pool import load_pool
    config = json.loads((manager.HERE / "config.json").read_text(encoding="utf-8"))
    settings = config["system_benchmark"]
    if set(settings) != {"instances", "parallelism"} or set(settings["instances"]) != {"awq", "gptq"}:
        raise ValueError("System benchmark requires one configured instance per expert")
    awq = load_pool(manager.ROOT / "deploy/pools/ascend-awq.json")
    profiles = {}
    for expert, ident in settings["instances"].items():
        pool = load_pool(manager.ROOT / ("deploy/pools/ascend-" + expert + ".json"))
        if ident not in pool["instances"]:
            raise ValueError("Unknown system benchmark instance: " + ident)
        for name, spec in pool["instances"].items():
            spec["enabled"] = name == ident
        pool["instances"][ident]["parallelism"] = copy.deepcopy(settings["parallelism"])
        # Both converted checkpoints use the same native W4A16 method. Share
        # the tested execution path while retaining distinct weights/identity.
        pool["host_execution"] = copy.deepcopy(awq["host_execution"])
        pool["execution_optimizations"] = copy.deepcopy(awq["execution_optimizations"])
        pool.setdefault("runtime", {}).setdefault("model_parameters", {}).update(
            max_num_seqs=16, max_num_batched_tokens=4096)
        pool["requests_per_worker"] = 16
        pool["router_replicas"] = 1
        profiles[expert] = (pool, ident)
    return config, profiles


def run_system(output_prefix="system-speed"):
    from manage_instances import instance_objects, deployment_name, validate_instances, preflight, plan, ready
    from manage_pool import resolve_runtime, router_objects, verify_pods
    from manage_lab import operation_lock
    config, profiles = system_profiles()
    cluster = manager.configuration("awq")[0]
    ROOT.mkdir(parents=True, exist_ok=True)
    if not output_prefix.replace('-', '').isalnum() or '/' in output_prefix:
        raise ValueError("Invalid result prefix")
    path = ROOT / (output_prefix + "-state.json")
    if path.exists():
        raise FileExistsError("Preserve or inspect the previous system speed result before rerunning")
    state = {"phase": "preparing", "profiles": {k: v[0] for k, v in profiles.items()}, "before": {}, "results": {}}
    def save():
        path.write_text(json.dumps(state, indent=2))
    def get(kind, name):
        return json.loads(manager.kubectl(cluster, ["get", kind, name, "-n", "heteroserve", "-o", "json"]).stdout)
    def restore(obj):
        name = obj["metadata"]["name"]
        operations = [{"op": "test", "path": "/metadata/uid", "value": obj["metadata"]["uid"]}]
        if obj["kind"] == "ConfigMap":
            operations.append({"op": "replace", "path": "/data", "value": obj["data"]})
        else:
            operations.extend([{"op": "replace", "path": "/spec/replicas", "value": obj["spec"]["replicas"]},
                               {"op": "replace", "path": "/spec/template", "value": obj["spec"]["template"]}])
        manager.kubectl(cluster, ["patch", obj["kind"], name, "-n", "heteroserve", "--type=json", "-p", json.dumps(operations)])
    before = {x["metadata"]["name"]: {"uid": x["metadata"]["uid"], "replicas": x["spec"]["replicas"]}
              for x in inventory(cluster, "deploy")}
    state["deployment_identities"] = before
    desired, models = [], []
    for expert, (pool, ident) in profiles.items():
        validate_instances(pool, cluster)
        model = deployment_name(pool, ident); models.append(model)
        assert before[model]["replicas"] == 0 and before[pool["pool"] + "-router"]["replicas"] == 0
        preflight(cluster, pool, plan(cluster, pool, ident))
        desired += instance_objects(cluster, pool, ident)
        desired += [x for x in router_objects(cluster, pool, resolve_runtime(pool)) if x["kind"] in {"ConfigMap", "Deployment"}]
    gateway = get("configmap", "heteroserve-gateway")
    gwdeploy = get("deployment", "heteroserve-gateway")
    state["before"]["ConfigMap/heteroserve-gateway"] = copy.deepcopy(gateway)
    state["before"]["Deployment/heteroserve-gateway"] = copy.deepcopy(gwdeploy)
    sources = [manager.HERE / name for name in ("host_benchmark.py", "benchmark.py", "config.json")]
    sources += [manager.ROOT / ("src/heteroserve/" + name) for name in
                ("worker.py", "router.py", "native_parallel.py", "engine_sitecustomize.py", "execution_worker.py", "ascend_execution.py", "gateway.py", "routing/service.py", "routing/client.py", "routing/ascend.py", "routing/runtime.py")]
    state["sources"] = {str(p.relative_to(manager.ROOT)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "source": p.read_text()} for p in sources}
    save()
    client = None
    try:
        with operation_lock(manager.ROOT):
            for obj in desired:
                original = get(obj["kind"], obj["metadata"]["name"])
                assert original["metadata"]["labels"].get(manager.OWNER) == cluster["name"]
                state["before"][obj["kind"] + "/" + obj["metadata"]["name"]] = original
            save()
            payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": desired})
            manager.kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
            manager.kubectl(cluster, ["apply", "-f", "-"], stdin=payload)
            cfg = json.loads(gateway["data"]["gateway.json"])
            for expert, (pool, _) in profiles.items():
                cfg["pools"][expert]["base_url"] = "http://" + pool["pool"] + "-router.heteroserve.svc.cluster.local:8000/v1"
            gateway["data"]["gateway.json"] = json.dumps(cfg, indent=2)
            gwdeploy["spec"]["template"]["metadata"]["annotations"]["heteroserve.io/runtime-sha256"] = hashlib.sha256(json.dumps(gateway["data"], sort_keys=True).encode()).hexdigest()
            # The saved snapshots must remain immutable while this test runs.
            for obj, field in ((gateway, "data"), (gwdeploy, "spec/template")):
                value = obj["data"] if field == "data" else obj["spec"]["template"]
                patch = [{"op": "test", "path": "/metadata/uid", "value": obj["metadata"]["uid"]},
                         {"op": "replace", "path": "/" + field, "value": value}]
                manager.kubectl(cluster, ["patch", obj["kind"], obj["metadata"]["name"], "-n", "heteroserve", "--type=json", "-p", json.dumps(patch)])
        state["phase"] = "waiting_models"; save()
        deadline = time.monotonic() + 900
        active = []
        while time.monotonic() < deadline:
            pods = inventory(cluster, "pods")
            active = [p for p in pods if p["metadata"].get("labels", {}).get("heteroserve.io/pool") in {v[0]["pool"] for v in profiles.values()}
                      and p["metadata"].get("labels", {}).get("app.kubernetes.io/name") == "expert" and not p["metadata"].get("deletionTimestamp")]
            status = [{"name": p["metadata"]["name"], "ready": ready(p), "restarts": sum(x.get("restartCount", 0) for x in p["status"].get("containerStatuses", []))} for p in active]
            state["startup"] = status; save(); print("SYSTEM_STARTUP", json.dumps(status), flush=True)
            if len(active) == 2 and all(ready(p) for p in active): break
            time.sleep(10)
        else: raise TimeoutError("The two expert models did not become Ready")
        state["acceptance"] = {}
        for expert, (pool, _) in profiles.items():
            selected = [p for p in active if p["metadata"]["labels"]["heteroserve.io/pool"] == pool["pool"]]
            state["acceptance"][expert] = verify_pods(cluster, pool, resolve_runtime(pool), selected)
        state["phase"] = "waiting_gateway"; save()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            pods = inventory(cluster, "pods")
            candidates = [p for p in pods if p["metadata"].get("labels", {}).get("app.kubernetes.io/name") == "gateway" and ready(p)]
            if len(candidates) == 1:
                client = candidates[0]; break
            time.sleep(5)
        else: raise TimeoutError("Gateway did not report healthy expert coverage")
        benchcfg = {"runtime": {"served_model": "auto"}, "expert_deployments": state["acceptance"],
                    "benchmark": {**config["benchmark"], "entry": "gateway", "scope": "Formal Gateway -> learned L1 -> official vLLM Router -> one AWQ TP2/PP1 and one GPTQ TP2/PP1; warm closed-loop; 4 expert NPUs plus 2 existing L1 NPUs"}}
        state["client"] = {"name": client["metadata"]["name"], "uid": client["metadata"]["uid"]}
        source = """import json,sys,subprocess
from pathlib import Path
a=json.load(sys.stdin);root=Path('/tmp/bf16_distributed_system');root.mkdir(exist_ok=True)
owner=root/'.owner'
if owner.exists() and owner.read_text()!='heteroserve-bf16-benchmark':raise RuntimeError('Unowned benchmark directory')
owner.write_text('heteroserve-bf16-benchmark')
pidfile=root/'system-speed.pid'
if pidfile.exists():
 proc=Path('/proc')/pidfile.read_text().strip()
 if proc.exists() and ') Z ' not in (proc/'stat').read_text():raise RuntimeError('Previous benchmark still running')
(root/'requests.jsonl').write_text(a['requests'])
(root/'benchmark.py').write_text(a['source'])
(root/'system-speed-config.json').write_text(json.dumps(a['config']))
with (root/'system-speed-client.log').open('w') as log:
 process=subprocess.Popen([sys.executable,str(root/'benchmark.py'),'--config',str(root/'system-speed-config.json'),'--url','http://heteroserve-gateway.heteroserve.svc.cluster.local:8000','--requests',str(root/'requests.jsonl'),'--output',str(root/'system-speed.json')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pidfile.write_text(str(process.pid));print(json.dumps({'pid':process.pid}))
"""
        state["phase"] = "measuring"; save(); print("SYSTEM_MEASURING", flush=True)
        requests = (manager.ROOT / "artifacts/bf16_distributed/system/requests.jsonl").read_text()
        state["client_process"] = execute(cluster, client["metadata"]["name"], source,
                         {"requests": requests, "source": (manager.HERE / "benchmark.py").read_text(), "config": benchcfg})
        save(); deadline = time.monotonic() + 600
        poll = """import json
from pathlib import Path
root=Path('/tmp/bf16_distributed_system');p=root/'system-speed.json'
r=json.loads(p.read_text()) if p.exists() else {}
proc=Path('/proc')/(root/'system-speed.pid').read_text().strip()
alive=proc.exists() and ') Z ' not in (proc/'stat').read_text()
print(json.dumps({'success':r.get('success',False),'progress':r.get('progress'), 'error':r.get('error'),'alive':alive}))
"""
        previous = None
        while time.monotonic() < deadline:
            progress = execute(cluster, client["metadata"]["name"], poll, {})
            state["progress"] = progress; save()
            if progress != previous: print("SYSTEM_PROGRESS", json.dumps(progress), flush=True); previous = progress
            if progress["success"]: break
            if not progress["alive"]: raise RuntimeError("System benchmark stopped: " + repr(progress))
            time.sleep(5)
        else: raise TimeoutError("System benchmark exceeded its budget")
        result = execute(cluster, client["metadata"]["name"],
                         "from pathlib import Path;print(Path('/tmp/bf16_distributed_system/system-speed.json').read_text())", {})
        assert result["success"]
        samples = result["cases"][0]["samples"]
        identities = {r["identity"]["pod_uid"] for v in state["acceptance"].values() for r in v["replicas"]}
        assert all(x["pod_uid"] in identities for x in samples)
        result["selection_counts"] = dict(Counter(x["expert"] for x in samples))
        result["worker_counts"] = dict(Counter(x["pod_uid"] for x in samples))
        (ROOT / (output_prefix + ".json")).write_text(json.dumps(result, indent=2))
        state["results"] = {"summary": result["cases"][0]["summary"], "selection_counts": result["selection_counts"], "worker_counts": result["worker_counts"]}
        state["phase"] = "completed"; save()
    except BaseException as exc:
        state.update(phase="failed", error=repr(exc), traceback=traceback.format_exc()); print(state["traceback"], flush=True)
    finally:
        state["cleanup_errors"] = []
        if client and state.get("client_process"):
            try:
                cleanup = """import json,os,signal,sys
from pathlib import Path
a=json.load(sys.stdin);pid=a['pid'];proc=Path('/proc')/str(pid)
if proc.exists() and ') Z ' not in (proc/'stat').read_text():
 cmd=(proc/'cmdline').read_bytes()
 assert b'benchmark.py' in cmd and b'system-speed-config.json' in cmd
 os.killpg(pid,signal.SIGTERM)
root=Path('/tmp/bf16_distributed_system');print(json.dumps({'log':(root/'system-speed-client.log').read_text()}))
"""
                log = execute(cluster, client["metadata"]["name"], cleanup, state["client_process"])
                (ROOT / (output_prefix + "-client.log")).write_text(log["log"])
            except BaseException as exc: state["cleanup_errors"].append(repr(exc))
        for name in models:
            try: manager.kubectl(cluster, ["scale", "deployment", name, "-n", "heteroserve", "--replicas=0"])
            except BaseException as exc: state["cleanup_errors"].append(repr(exc))
        for obj in reversed(list(state["before"].values())):
            try: restore(obj)
            except BaseException as exc: state["cleanup_errors"].append(repr(exc))
        current = {x["metadata"]["name"]: x for x in inventory(cluster, "deploy")}
        state["restored"] = all(current[name]["metadata"]["uid"] == old["uid"] and current[name]["spec"]["replicas"] == old["replicas"] for name, old in before.items())
        save()
    print("SYSTEM_SPEED_FINISHED", state["phase"], json.dumps(state.get("results")), flush=True)


def run_l1():
    cluster = manager.configuration("awq")[0]
    path = ROOT / "l1-optimization.json"
    if path.exists(): raise FileExistsError("Inspect the preserved L1 optimization result before rerunning")
    before_path = ROOT / "l1-optimization-before.json"
    live_pods = [p for p in inventory(cluster, "pods")
                 if p["metadata"]["labels"].get("app.kubernetes.io/name") == "l1-router" and not p["metadata"].get("deletionTimestamp")]
    if before_path.exists():
        before = json.loads(before_path.read_text())
        if {p['metadata']['uid'] for p in before['pods']} != {p['metadata']['uid'] for p in live_pods}:
            raise FileExistsError("Archive the previous L1 before-snapshot before rerunning on new Pods")
    else:
        before = {"pods": live_pods, "deployments": inventory(cluster, "deploy"),
                  "gateway_config_source": (manager.ROOT / "deploy/gateway.json").read_text(), "configmaps": {}}
        for instance in ("heteroserve-l1-01", "heteroserve-l1-02"):
            before['configmaps'][instance] = json.loads(manager.kubectl(cluster, ["get", "configmap", instance, "-n", "heteroserve", "-o", "json"]).stdout)
        before_path.write_text(json.dumps(before, indent=2))
    pod = next(p for p in before["pods"] if p["metadata"]["labels"]["heteroserve.io/instance"] == "l1-01")
    name, uid = pod["metadata"]["name"], pod["metadata"]["uid"]
    url = "http://" + pod["status"]["podIP"] + ":8000"
    client = next(p for p in inventory(cluster, "pods") if p["metadata"]["labels"].get("app.kubernetes.io/name") == "gateway" and p["status"]["phase"] == "Running")
    rows = [json.loads(x) for x in (manager.ROOT / "artifacts/bf16_distributed/system/requests.jsonl").read_text().splitlines()][3:203]
    state = {"phase": "reference", "pod_uid": uid, "cases": {}, "before_file": "l1-optimization-before.json"}
    def save(): path.write_text(json.dumps(state, indent=2))
    def measure(label, inputs=rows, concurrency=1):
        state["phase"] = label; save(); print("L1_MEASURING", label, flush=True)
        source = """import asyncio,json,sys,types,contextlib
a=json.load(sys.stdin);b=types.ModuleType('bench');exec(a['source'],b.__dict__)
result=asyncio.run(b.diagnose_l1(a['rows'],a['url'],a['concurrency']))
print(json.dumps(result))
"""
        result = execute(cluster, client["metadata"]["name"], source,
                         {"source": (manager.HERE / "benchmark.py").read_text(), "rows": inputs, "url": url, "concurrency": concurrency})
        state["cases"][label] = result; save(); print("L1_RESULT", label, json.dumps(result["summary"]), flush=True)
        return result
    original = copy.deepcopy(before["configmaps"]["heteroserve-l1-01"])
    def configure(queue, profile=False):
        current = json.loads(manager.kubectl(cluster, ["get", "pod", name, "-n", "heteroserve", "-o", "json"]).stdout)
        assert current["metadata"]["uid"] == uid
        restart = current["status"]["containerStatuses"][0]["restartCount"]
        data = copy.deepcopy(original["data"])
        for module in ("routing/runtime.py", "routing/ascend.py", "routing/service.py"):
            data["heteroserve--" + module.replace('/', '--')] = (manager.ROOT / "src/heteroserve" / module).read_text()
        cfg = json.loads(data["gateway.json"])
        cfg["router"].update(task_queue_enable=queue, cpu_affinity_conf=0, profile_stages=profile)
        data["gateway.json"] = json.dumps(cfg, indent=2)
        patch = [{"op": "test", "path": "/metadata/uid", "value": original["metadata"]["uid"]},
                 {"op": "replace", "path": "/data", "value": data}]
        manager.kubectl(cluster, ["patch", "configmap", "heteroserve-l1-01", "-n", "heteroserve", "--type=json", "-p", json.dumps(patch)])
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        manager.kubectl(cluster, ["annotate", "pod", name, "-n", "heteroserve", "heteroserve.io/l1-diagnostic=" + digest, "--overwrite"])
        expected = {"/opt/runtime/" + (key.replace('--', '/') if key != 'gateway.json' else key): hashlib.sha256(value.encode()).hexdigest()
                    for key, value in data.items() if key in {'gateway.json', 'heteroserve--routing--runtime.py', 'heteroserve--routing--ascend.py', 'heteroserve--routing--service.py'}}
        check = "import json,sys,hashlib;from pathlib import Path;a=json.load(sys.stdin);print(json.dumps({'ready':all(Path(k).exists() and hashlib.sha256(Path(k).read_text().encode()).hexdigest()==v for k,v in a.items())}))"
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if execute(cluster, name, check, expected)["ready"]: break
            time.sleep(3)
        else: raise TimeoutError("Projected L1 source/config update did not arrive")
        snapshot = "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/identity',timeout=3).read().decode())"
        try:
            live = execute(cluster, name, snapshot, {})
            if live.get('task_queue_enable') == queue and live.get('profile_stages') == profile:
                return
        except RuntimeError:
            pass
        try:
            manager.kubectl(cluster, ["exec", "-n", "heteroserve", name, "--", "python3", "-c", "import os,signal;os.kill(1,signal.SIGTERM)"])
        except RuntimeError:
            pass  # Exec can disconnect when PID1 stops; readiness below verifies restart.
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            current = json.loads(manager.kubectl(cluster, ["get", "pod", name, "-n", "heteroserve", "-o", "json"]).stdout)
            assert current["metadata"]["uid"] == uid
            if current["status"]["containerStatuses"][0]["restartCount"] > restart and any(x['type']=='Ready' and x['status']=='True' for x in current['status'].get('conditions',[])): break
            time.sleep(3)
        else: raise TimeoutError("L1 did not become Ready after the configured restart")
    save()
    try:
        reference = measure("reference")
        configure(1, True)
        measure("stage_profile", [rows[i] for i in range(0, 200, 10)])
        configure(1, False)
        paired = measure("paired_reference")
        def parity(result):
            assert result["identity"]["physical_devices"] == reference["identity"]["physical_devices"]
            flips = sum(a["decision"]["expert"] != b["decision"]["expert"] for a, b in zip(reference["samples"], result["samples"]))
            difference = max(abs(a["decision"]["probabilities"][key] - b["decision"]["probabilities"][key])
                             for a, b in zip(reference["samples"], result["samples"]) for key in a["decision"]["probabilities"])
            result["parity"] = {"expert_changes": flips, "max_probability_difference": difference}
            assert flips == 0 and difference == 0, result["parity"]
        parity(paired)
        configure(2)
        winner = measure("queue2"); parity(winner)
        assert winner["summary"]["compute_ms"]["mean"] < paired["summary"]["compute_ms"]["mean"]
        state["winner"] = {"task_queue_enable": 2, "summary": winner["summary"]}
        final = measure("winner_concurrency4", concurrency=4); parity(final)
        state["phase"] = "completed"; save()
    except BaseException as exc:
        state.update(phase="failed", error=repr(exc), traceback=traceback.format_exc()); save(); print(state["traceback"], flush=True)
        patch = [{"op":"test","path":"/metadata/uid","value":original["metadata"]["uid"]},
                 {"op":"replace","path":"/data","value":original["data"]}]
        manager.kubectl(cluster, ["patch", "configmap", "heteroserve-l1-01", "-n", "heteroserve", "--type=json", "-p", json.dumps(patch)])
        raise
    print("L1_OPTIMIZATION_FINISHED", state["phase"], json.dumps(state["winner"]), flush=True)


def run(mode="host", output_prefix="system-speed"):
    if mode == "system":
        return run_system(output_prefix)
    if mode == "l1":
        return run_l1()
    cluster = manager.configuration("awq")[0]
    state_path = ROOT / {"host": "host-execution-state.json", "ttft": "ttft-state.json",
                         "ttft-local-bf16": "ttft-local-bf16-state.json"}[mode]
    if state_path.exists():
        raise FileExistsError("Inspect existing host experiment before rerunning")
    all_deploys = inventory(cluster, "deploy")
    targets = {x["metadata"]["name"]: x["metadata"]["uid"] for x in all_deploys
               if x["metadata"]["name"].startswith(("awq-distributed-stage-", "bf16-distributed-stage-"))}
    assert len(targets) == 3
    paused = {x["metadata"]["name"]: x["metadata"]["uid"] for x in all_deploys if x["spec"]["replicas"] == 0 and x["metadata"]["name"] not in targets}
    assert all(x["spec"]["replicas"] == 0 for x in all_deploys if x["metadata"]["name"] in targets)
    state = {"phase": "starting", "results": {}, "target_uids": targets, "original_paused": paused}
    def save():
        state_path.write_text(json.dumps(state, indent=2))
    def check_paused():
        current = {x["metadata"]["name"]: x for x in inventory(cluster, "deploy")}
        assert all(current[name]["metadata"]["uid"] == uid and current[name]["spec"]["replicas"] == 0 for name, uid in paused.items())
    def stop(expert):
        for deploy in inventory(cluster, "deploy", expert):
            assert targets[deploy["metadata"]["name"]] == deploy["metadata"]["uid"]
            manager.kubectl(cluster, ["scale", "deploy", deploy["metadata"]["name"], "-n", "heteroserve", "--replicas=0"])
        deadline = time.monotonic() + 180
        while inventory(cluster, "pods", expert):
            if time.monotonic() > deadline:
                raise TimeoutError("Model did not release devices")
            time.sleep(3)
    def launch(expert, settings):
        c, pool, runtime, manifest = prepared(expert, settings, profile=mode == "host",
                                             local_bf16=mode == "ttft-local-bf16")
        original = manager.objects
        manager.objects = lambda *args: manifest
        try:
            desired_uids = {obj["metadata"]["name"]: targets[obj["metadata"]["name"]]
                            for obj in manifest["items"] if obj["kind"] == "Deployment"}
            manager.apply_experiment(c, pool, runtime, desired_uids)
        finally:
            manager.objects = original
        return wait_ready(c, expert, count=pool["parallelism"]["pp"])
    save()
    try:
        check_paused()
        if mode in {"ttft", "ttft-local-bf16"}:
            state["sources"] = {str(path.relative_to(manager.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (manager.HERE / "host_benchmark.py", manager.HERE / "benchmark.py",
                             manager.HERE / "config.json", manager.HERE / "manage.py")}
            for expert in (("bf16",) if mode == "ttft-local-bf16" else ("bf16", "awq")):
                state["phase"] = expert + "_ttft"; save(); print("MEASURING", state["phase"], flush=True)
                pods = launch(expert, None)
                state["results"][expert] = measure_ttft(cluster, expert, pods,
                    name="bf16-local" if mode == "ttft-local-bf16" else expert)
                save(); stop(expert); check_paused()
            state["phase"] = "completed"
            return
        state["phase"] = "bf16_reference"; save()
        pods = launch("bf16", {"task_queue_enable": 1, "cpu_affinity_conf": 0})
        state["results"]["bf16"] = measure(cluster, "bf16", "bf16", pods, speed=False); save(); stop("bf16")
        state["phase"] = "awq_reference"; save()
        pods = launch("awq", {"task_queue_enable": 1, "cpu_affinity_conf": 0})
        state["results"]["awq_reference"] = measure(cluster, "awq", "awq-reference", pods); save()
        for name, settings in [("awq_queue2", {"task_queue_enable": 2, "cpu_affinity_conf": 0}),
                               ("awq_mp_queue2", {"task_queue_enable": 2, "cpu_affinity_conf": 0, "executor": "mp"})]:
            state["phase"] = name; save(); print("MEASURING", name, flush=True)
            _, _, _, manifest = prepared("awq", settings)
            pods = restart_in_place(cluster, "awq", pods, manifest)
            state["results"][name] = measure(cluster, "awq", name.replace("_", "-"), pods); save(); check_paused()
        state["phase"] = "completed"
    except BaseException as exc:
        state.update(phase="failed", error=repr(exc), traceback=traceback.format_exc()); print(state["traceback"], flush=True)
    finally:
        save()
        for expert in ("awq", "bf16"):
            try:
                stop(expert)
                c, pool, runtime, manifest = prepared(expert, None, profile=False)
                for obj in manifest["items"]:
                    if obj["kind"] == "Deployment": obj["spec"]["replicas"] = 0
                manager.kubectl(c, ["apply", "-f", "-"], stdin=json.dumps(manifest))
            except BaseException as exc:
                state.setdefault("cleanup_errors", []).append(repr(exc))
        check_paused(); state["models_stopped"] = True; save()
    print("HOST_EXPERIMENT_FINISHED", state["phase"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("host", "ttft", "ttft-local-bf16", "system", "l1"), default="host")
    parser.add_argument("--output-prefix", default="system-speed")
    args = parser.parse_args()
    run(args.mode, args.output_prefix)
