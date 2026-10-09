"""Stable vLLM instances, shared workload templates, and safe per-instance operations."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from manage_lab import ROOT, OWNER, active_nodes, kubectl, load_config, names, operation_lock
from manage_pool import artifact_path, load_pool, resolve_runtime, pool_objects, verify_pods

INSTANCE_LABEL = "heteroserve.io/instance"
ENGINE_FIELDS = {"max_model_len", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization", "enforce_eager"}


def normalized_instances(pool):
    """Normalize legacy replica configuration into a desired stable-instance list."""
    if "instances" in pool:
        return copy.deepcopy(pool["instances"])
    return {pool["expert"] + "-" + str(i + 1).zfill(2):
            {"node": pool["model_nodes"][i % len(pool["model_nodes"])], "enabled": True,
             "parallelism": {"tp": 1, "pp": 1}, "engine": {}}
            for i in range(pool["replicas"])}


def validate_instances(pool, cluster):
    specs = pool.get("instances")
    if not isinstance(specs, dict):
        raise ValueError("instances must be an ID -> configuration mapping")
    runtime = resolve_runtime(pool)
    allowed = {names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in runtime["device_hosts"]}
    registered = {names(cluster, n)[0] for n in cluster["nodes"] if n["host"] in runtime["device_hosts"]}
    contexts = set()
    for ident, spec in specs.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,23}", ident) or len(pool["pool"] + "-" + ident + "-runtime") > 63:
            raise ValueError("Invalid or too long instance ID: " + ident)
        if set(spec) - {"node", "enabled", "parallelism", "engine", "traffic", "lifecycle"}:
            raise ValueError("Unsupported instance fields: " + ident)
        eligible = allowed if spec.get("enabled", True) else registered
        if spec.get("node") not in eligible or type(spec.get("enabled", True)) is not bool:
            raise ValueError("Instance needs an eligible NPU node and boolean enabled state: " + ident)
        parallel = spec.get("parallelism", {"tp": 1, "pp": 1})
        if set(parallel) != {"tp", "pp"} or any(type(v) is not int or v < 1 for v in parallel.values()):
            raise ValueError("TP and PP must be positive integers")
        if parallel["tp"] * parallel["pp"] > 8:
            raise ValueError("Single-node parallelism cannot require more than eight cards")
        engine = spec.get("engine", {})
        if set(engine) - ENGINE_FIELDS:
            raise ValueError("Unsupported or identity-changing engine override")
        for key, value in engine.items():
            if key == "enforce_eager":
                valid = type(value) is bool
            elif key == "gpu_memory_utilization":
                valid = type(value) in {int, float} and math.isfinite(value) and 0 < value < 1
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError("Invalid engine parameter: " + key)
        traffic = spec.get("traffic", {})
        if set(traffic) - {"max_inflight", "request_timeout_seconds"} or any(type(v) is not int or v < 1 for v in traffic.values()):
            raise ValueError("Instance traffic only supports positive max_inflight")
        lifecycle = spec.get("lifecycle", {})
        if set(lifecycle) - {"startup_seconds", "drain_seconds", "termination_seconds"} or any(type(v) is not int or v < 1 for v in lifecycle.values()):
            raise ValueError("Invalid instance lifecycle budget")
        merged = {**pool, **lifecycle}
        if merged["startup_seconds"] < 5 or merged["termination_seconds"] < merged["drain_seconds"] + 35:
            raise ValueError("Instance termination must cover drain +35s")
        params = {**runtime["model_parameters"], **engine}
        if spec.get("enabled", True):
            contexts.add(params["max_model_len"])
        if parallel["pp"] > 1 and params["max_num_batched_tokens"] < params["max_model_len"]:
            raise ValueError("PP uses the existing V0 engine: max_num_batched_tokens must cover max_model_len")
    if len(contexts) > 1:
        raise ValueError("Enabled instances in one pool must share max_model_len for interchangeable routing; change pool defaults or all instance overrides together")
    return specs


def deployment_name(pool, ident):
    return pool["pool"] + "-" + ident


def resize_specs(pool, count):
    if type(count) is not int or count < 0:
        raise ValueError("Instance count must be a nonnegative integer")
    result = normalized_instances(pool)
    active = [i for i, v in result.items() if v.get("enabled", True)]
    if count < len(active):
        for ident in active[count:]:
            result[ident]["enabled"] = False
    else:
        for ident, spec in result.items():
            if len(active) == count:
                break
            if not spec.get("enabled", True):
                spec["enabled"] = True
                active.append(ident)
        nodes = pool.get("model_nodes", []) or [v["node"] for v in result.values()]
        if not nodes and count:
            raise ValueError("Choose an eligible node before creating an instance")
        seq = 1
        while len(active) < count:
            ident = pool["expert"] + "-" + str(seq).zfill(2)
            seq += 1
            if ident in result:
                continue
            node = min(nodes, key=lambda n: sum(v["node"] == n and v.get("enabled", True) for v in result.values()))
            result[ident] = {"node": node, "enabled": True, "parallelism": {"tp": 1, "pp": 1}, "engine": {}}
            active.append(ident)
    return result


def effective(pool, ident, root=ROOT):
    spec = normalized_instances(pool)[ident]
    p = copy.deepcopy(pool)
    p.pop("instances", None)
    p.update(spec.get("lifecycle", {}))
    p.update(instance_id=ident, replicas=int(spec.get("enabled", True)), model_nodes=[spec["node"]],
             parallelism=spec.get("parallelism", {"tp": 1, "pp": 1}),
             requests_per_worker=spec.get("traffic", {}).get("max_inflight", pool["requests_per_worker"]),
             request_timeout_seconds=spec.get("traffic", {}).get("request_timeout_seconds", pool.get("request_timeout_seconds", 180)))
    n = resolve_runtime(pool, root)
    n["model_parameters"].update(spec.get("engine", {}))
    return p, n


def instance_objects(cluster, pool, ident):
    p, n = effective(pool, ident)
    cm, deployment, _, _ = pool_objects(cluster, p, n)
    name = deployment_name(pool, ident)
    cm["metadata"]["name"] = name + "-runtime"
    deployment["metadata"]["name"] = name
    for obj in (cm, deployment):
        obj["metadata"]["labels"][INSTANCE_LABEL] = ident
    deployment["spec"]["selector"]["matchLabels"][INSTANCE_LABEL] = ident
    template = deployment["spec"]["template"]
    template["metadata"]["labels"][INSTANCE_LABEL] = ident
    template["metadata"]["annotations"]["heteroserve.io/max-inflight"] = str(p["requests_per_worker"])
    next(v for v in template["spec"]["volumes"] if v["name"] == "runtime")["configMap"]["name"] = name + "-runtime"
    deployment["spec"]["strategy"]["rollingUpdate"] = {"maxSurge": 1, "maxUnavailable": 0}
    return [cm, deployment]


def objects(cluster, pool):
    validate_instances(pool, cluster)
    base = copy.deepcopy(pool)
    base.pop("instances", None)
    base["replicas"] = 1
    _, _, service, pdb = pool_objects(cluster, base, resolve_runtime(pool))
    result = [service, pdb]
    for ident in pool["instances"]:
        result.extend(instance_objects(cluster, pool, ident))
    return result


def get(cluster, kind, name=None, namespace="heteroserve", selector=None):
    args = ["get", kind]
    if name:
        args += [name, "--ignore-not-found=true"]
    if namespace:
        args += ["-n", namespace]
    if selector:
        args += ["-l", selector]
    args += ["-o", "json"]
    raw = kubectl(cluster, args).stdout
    return json.loads(raw) if raw.strip() else None


def ready(pod):
    return (not pod["metadata"].get("deletionTimestamp") and any(
        v["type"] == "Ready" and v["status"] == "True" for v in pod.get("status", {}).get("conditions", [])))


def matches(actual, desired):
    """Ignore Kubernetes-injected defaults while comparing declared template fields."""
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(k in actual and matches(actual[k], v) for k, v in desired.items())
    if isinstance(desired, list):
        return isinstance(actual, list) and len(actual) == len(desired) and all(matches(a, b) for a, b in zip(actual, desired))
    return actual == desired


def owned(obj, cluster):
    if obj["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
        raise RuntimeError("Refusing unowned workload: " + obj["metadata"]["name"])


def wait_instance(cluster, pool, ident):
    p, _ = effective(pool, ident)
    kubectl(cluster, ["rollout", "status", "deployment/" + deployment_name(pool, ident), "-n", pool["namespace"],
                      "--timeout=" + str(p["startup_seconds"]) + "s"], timeout=p["startup_seconds"] + 60)
    deadline = time.monotonic() + p["startup_seconds"]
    while True:
        pods = get(cluster, "pods", namespace=pool["namespace"], selector=INSTANCE_LABEL + "=" + ident + ",heteroserve.io/pool=" + pool["pool"])["items"]
        active = [v for v in pods if ready(v)]
        expected = int(pool["instances"][ident].get("enabled", True))
        if len(active) == expected and (expected or not pods):
            return active
        if time.monotonic() > deadline:
            raise TimeoutError("Instance failed to reach declared readiness: " + ident)
        time.sleep(3)


def fingerprint(pool, ident):
    p, n = effective(pool, ident)
    data = {"model": n["model_source"], "weights": pool["weights_manifest_sha256"], "image": n["model_image"],
            "parallelism": p["parallelism"], "engine": n["model_parameters"],
            "adapter": {name: hashlib.sha256((ROOT / "src/heteroserve" / name).read_bytes()).hexdigest()
                        for name in ("native_parallel.py", "engine_sitecustomize.py")}}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_parallel(cluster, pool, ident):
    p, n = effective(pool, ident)
    if p["parallelism"] == {"tp": 1, "pp": 1}:
        return
    path = ROOT / "state/capabilities" / (fingerprint(pool, ident) + ".json")
    if path.exists() and json.loads(path.read_text()).get("passed"):
        return
    from manage_nodes import validate_candidate
    result = validate_candidate(cluster, p, n, p["model_nodes"][0])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def plan(cluster, pool, selected=None):
    validate_instances(pool, cluster)
    deployments = get(cluster, "deployments", namespace=pool["namespace"], selector="heteroserve.io/pool=" + pool["pool"])["items"]
    present = {v["metadata"]["name"]: v for v in deployments}
    changes = []
    for ident, spec in pool["instances"].items():
        if selected and ident != selected:
            continue
        obj = instance_objects(cluster, pool, ident)[1]
        current = present.get(obj["metadata"]["name"])
        if current:
            owned(current, cluster)
        old_hash = current["spec"]["template"]["metadata"].get("annotations", {}).get("heteroserve.io/runtime-sha256") if current else None
        new_hash = obj["spec"]["template"]["metadata"]["annotations"]["heteroserve.io/runtime-sha256"]
        changed = current is None or not matches(current["spec"]["template"], obj["spec"]["template"]) or current["spec"].get("replicas") != obj["spec"]["replicas"]
        changes.append({"instance": ident, "node": spec["node"], "enabled": spec.get("enabled", True),
                        "parallelism": spec.get("parallelism", {"tp": 1, "pp": 1}),
                        "cards": spec.get("parallelism", {"tp": 1, "pp": 1})["tp"] * spec.get("parallelism", {"tp": 1, "pp": 1})["pp"],
                        "action": "create" if current is None else "update" if changed else "unchanged",
                        "runtime_changed": old_hash != new_hash, "deployment": obj["metadata"]["name"]})
    if not selected:
        for obj in deployments:
            labels = obj["metadata"].get("labels", {})
            ident = labels.get(INSTANCE_LABEL)
            if ident and ident not in pool["instances"]:
                owned(obj, cluster)
                changes.append({"instance": ident, "node": None, "enabled": False, "cards": 0,
                                "action": "remove", "runtime_changed": False, "deployment": obj["metadata"]["name"]})
    return changes


def preflight(cluster, pool, changes):
    """Check desired and rolling capacity before altering any running workload."""
    from manage_nodes import request, physical_free
    from manage_lab import remote
    npu = resolve_runtime(pool)
    nodes = {v["metadata"]["name"]: v for v in get(cluster, "nodes", namespace=None)["items"]}
    pods = json.loads(kubectl(cluster, ["get", "pods", "-A", "-o", "json"]).stdout)["items"]
    total = {}
    for ident, spec in pool["instances"].items():
        if spec.get("enabled", True):
            p = spec.get("parallelism", {"tp": 1, "pp": 1})
            total[spec["node"]] = total.get(spec["node"], 0) + p["tp"] * p["pp"]
    for change in changes:
        if change["action"] == "unchanged" or not change["enabled"]:
            continue
        name = change["node"]
        node = nodes.get(name)
        if not node or not ready(node) or node.get("spec", {}).get("unschedulable"):
            raise RuntimeError("Target node is not Ready/schedulable: " + name)
        required = {"heteroserve.io/npu-ready": "true", "heteroserve.io/hardware": pool["hardware"], "heteroserve.io/backend": pool["backend"]}
        if any(node["metadata"].get("labels", {}).get(k) != v for k, v in required.items()):
            raise RuntimeError("Target hardware/backend labels do not match")
        if any(v["effect"] in {"NoSchedule", "NoExecute"} for v in node.get("spec", {}).get("taints", [])):
            raise RuntimeError("Target node has an incompatible scheduling taint")
        local = [v for v in pods if v["spec"].get("nodeName") == name and v["status"].get("phase") not in {"Succeeded", "Failed"}]
        allocated = int(node["status"].get("allocatable", {}).get(npu["resource"], 0))
        used = sum(request(v, npu["resource"]) for v in local)
        others = sum(request(v, npu["resource"]) for v in local if v["metadata"].get("labels", {}).get("heteroserve.io/pool") != pool["pool"])
        host = next(n for n in active_nodes(cluster) if names(cluster, n)[0] == name)
        free = min(allocated - used, len(physical_free(host["host"])))
        from manage_nodes import quantity
        for resource, needed in (("cpu", pool["worker_request_cpu"]), ("memory", pool["worker_request_memory"])):
            remaining = quantity(node["status"]["allocatable"].get(resource, 0)) - sum(request(v, resource) for v in local)
            if remaining < quantity(needed):
                raise RuntimeError("Not enough scheduling " + resource + " for safe startup on " + name)
        if total[name] + others > allocated or free < change["cards"]:
            raise RuntimeError("Not enough cards for safe startup/rolling update on " + name +
                               "; free=" + str(free) + ", needed=" + str(change["cards"]) + ". Pause or move an instance first.")
        volume = names(cluster, host)[2]
        code = """import json,sys,hashlib,subprocess
from pathlib import Path
v=json.loads(subprocess.check_output(['docker','volume','inspect',sys.argv[1]]))[0]
assert (v.get('Labels') or {}).get('io.heteroserve.lab')==sys.argv[2]
p=Path(v['Mountpoint'])/'heteroserve-assets'/sys.argv[3]/'conversion_manifest.json'
assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[4],'Model weights have not been prepared on this node'
"""
        remote(host["host"], ["python3", "-c", code, volume, cluster["name"], pool.get("model_asset", "model"), pool["weights_manifest_sha256"]])


def apply(cluster, pool, selected=None):
    changes = plan(cluster, pool, selected)
    preflight(cluster, pool, changes)
    path = artifact_path(pool, "instances-apply.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "plan": changes, "completed": []}
    legacy = get(cluster, "deployment", pool["pool"], pool["namespace"])
    if legacy:
        owned(legacy, cluster)
        if selected:
            raise RuntimeError("First instance migration must apply the whole pool")
    # Service/PDB selectors remain pool-wide while workers move to independent Deployments.
    for obj in objects(cluster, pool)[:2]:
        kubectl(cluster, ["apply", "-f", "-"], stdin=json.dumps(obj))
    try:
        for change in changes:
            ident = change["instance"]
            if change["action"] == "remove":
                name = change["deployment"]
                obj = get(cluster, "deployment", name, pool["namespace"])
                if obj:
                    owned(obj, cluster)
                    kubectl(cluster, ["delete", "deployment", name, "-n", pool["namespace"], "--timeout=360s"], timeout=380)
                cm = get(cluster, "configmap", name + "-runtime", pool["namespace"])
                if cm:
                    owned(cm, cluster)
                    kubectl(cluster, ["delete", "configmap", cm["metadata"]["name"], "-n", pool["namespace"]])
                report["completed"].append(ident)
                continue
            if change["action"] == "unchanged":
                wait_instance(cluster, pool, ident)
                report["completed"].append(ident)
                continue
            if change["enabled"]:
                validate_parallel(cluster, pool, ident)
            payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": instance_objects(cluster, pool, ident)})
            kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
            kubectl(cluster, ["apply", "-f", "-"], stdin=payload)
            pods = wait_instance(cluster, pool, ident)
            if pods:
                p, n = effective(pool, ident)
                verify_pods(cluster, p, n, pods)
            report["completed"].append(ident)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if legacy:
                remaining = max(0, pool["replicas"] - len(report["completed"]))
                kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(remaining)])
        if legacy:
            kubectl(cluster, ["delete", "deployment", pool["pool"], "-n", pool["namespace"], "--wait=true", "--timeout=360s"], timeout=380)
            cm = get(cluster, "configmap", pool["pool"] + "-runtime", pool["namespace"])
            if cm:
                owned(cm, cluster)
                kubectl(cluster, ["delete", "configmap", cm["metadata"]["name"], "-n", pool["namespace"]])
        if not selected:
            for deployment in get(cluster, "deployments", namespace=pool["namespace"], selector=INSTANCE_LABEL)["items"]:
                labels = deployment["metadata"].get("labels", {})
                if labels.get("heteroserve.io/pool") != pool["pool"] or labels.get(INSTANCE_LABEL) in pool["instances"]:
                    continue
                owned(deployment, cluster)
                name = deployment["metadata"]["name"]
                kubectl(cluster, ["delete", "deployment", name, "-n", pool["namespace"], "--timeout=360s"], timeout=380)
                cm = get(cluster, "configmap", name + "-runtime", pool["namespace"])
                if cm:
                    owned(cm, cluster)
                    kubectl(cluster, ["delete", "configmap", name + "-runtime", "-n", pool["namespace"]])
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def verify(cluster, pool, selected=None):
    reports = []
    for ident, spec in pool["instances"].items():
        if selected and ident != selected:
            continue
        pods = wait_instance(cluster, pool, ident)
        if pods:
            p, n = effective(pool, ident)
            reports.extend(verify_pods(cluster, p, n, pods)["replicas"])
    result = {"passed": True, "pool": pool["pool"], "replicas": reports,
              "scope": "selected_instance" if selected else "independent_instances", "paused": not reports}
    path = artifact_path(pool, "instances/" + selected + ".json" if selected else "pool-verification.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def engine_values(command):
    """Read explicit flags from the live vLLM API process, never from a ConfigMap."""
    result = {}
    for key in ENGINE_FIELDS:
        flag = "--" + key.replace("_", "-")
        matching = [v for v in command if v == flag or v.startswith(flag + "=")]
        if key == "enforce_eager":
            result[key] = bool(matching)
        elif matching:
            token = matching[-1]
            raw = token.split("=", 1)[1] if "=" in token else command[command.index(token) + 1]
            result[key] = float(raw) if key == "gpu_memory_utilization" else int(raw)
    return result


def inventory(cluster, pool=None, runtime=False, selected=None):
    selector = OWNER + "=" + cluster["name"]
    if pool:
        selector += ",heteroserve.io/pool=" + pool["pool"]
    if selected:
        selector += "," + INSTANCE_LABEL + "=" + selected
    pods = get(cluster, "pods", selector=selector)["items"]
    rows = []
    for pod in pods:
        labels = pod["metadata"].get("labels", {})
        role = labels.get("app.kubernetes.io/name")
        if role not in {"expert", "gateway", "l1-router"} or (pool and labels.get("heteroserve.io/pool") != pool["pool"]):
            continue
        row = {"instance": labels.get(INSTANCE_LABEL, "gateway" if role == "gateway" else "legacy"),
               "pool": labels.get("heteroserve.io/pool", "gateway"), "node": pod["spec"].get("nodeName", "Pending"),
               "pod": pod["metadata"]["name"], "uid": pod["metadata"]["uid"], "ready": ready(pod),
               "restarts": sum(v.get("restartCount", 0) for v in pod.get("status", {}).get("containerStatuses", [])),
               "physical_devices": None, "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if pod.get("status", {}).get("phase") == "Running":
            try:
                code = ("import json,urllib.request;print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/identity',timeout=5))))" if role in {"expert", "l1-router"} else
                        "import json;from pathlib import Path;print(json.dumps({'physical_devices':[int(p.name[7:]) for p in Path('/dev').glob('davinci*') if p.name[7:].isdigit()]}))")
                info = json.loads(kubectl(cluster, ["exec", "-n", "heteroserve", pod["metadata"]["name"], "--", "python3", "-c", code]).stdout)
                if role in {"expert", "l1-router"} and (info["pod_uid"] != row["uid"] or info["node"] != row["node"]):
                    raise RuntimeError("Identity does not match the observed Pod")
                if role in {"expert", "l1-router"} and INSTANCE_LABEL in labels and info.get("instance_id") != labels[INSTANCE_LABEL]:
                    raise RuntimeError("Instance identity does not match the workload label")
                row.update(physical_devices=info["physical_devices"], parallelism=info.get("parallelism", {"tp": 1, "pp": 1}))
                if role == "l1-router":
                    row.update(runtime_values={"node": row["node"], "enabled": not bool(pod["metadata"].get("deletionTimestamp")),
                                               "max_inflight": info["max_inflight"]}, encoder_device=info["encoder_device"], head_device=info["head_device"])
                if runtime and role == "expert":
                    code = """import json,urllib.request,urllib.error
from pathlib import Path
try:
 result=json.load(urllib.request.urlopen('http://127.0.0.1:8000/configuration',timeout=5))
 result['source']='process_snapshot'
except urllib.error.HTTPError as exc:
 if exc.code!=404:raise
 commands=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   argv=p.read_bytes().decode().strip('\\0').split('\\0')
   if 'vllm.entrypoints.openai.api_server' in argv:commands.append(argv)
  except (OSError,UnicodeError):pass
 if len(commands)!=1:raise RuntimeError('Cannot identify exactly one live vLLM API process')
 identity=json.load(urllib.request.urlopen('http://127.0.0.1:8000/identity',timeout=5))
 result={**identity,'source':'live_process_arguments','engine_command':commands[0]}
print(json.dumps(result))
"""
                    snapshot = json.loads(kubectl(cluster, ["exec", "-n", pod["metadata"].get("namespace", "heteroserve"),
                                                          pod["metadata"]["name"], "--", "python3", "-c", code], timeout=25).stdout)
                    if snapshot.get("pod_uid") != row["uid"] or snapshot.get("node") != row["node"] or snapshot.get("instance_id") != row["instance"]:
                        raise RuntimeError("Runtime configuration identity does not match the observed Pod")
                    values = {"node": row["node"], "enabled": not bool(pod["metadata"].get("deletionTimestamp")),
                              **{"parallelism." + k: v for k, v in snapshot["parallelism"].items()}}
                    params = snapshot.get("engine", {}) if snapshot["source"] == "process_snapshot" else engine_values(snapshot["engine_command"])
                    values.update({"engine." + k: v for k, v in params.items() if k in ENGINE_FIELDS})
                    for group in ("traffic", "lifecycle"):
                        values.update({group + "." + k: v for k, v in snapshot.get(group, {}).items()})
                    container = next(v for v in pod["spec"]["containers"] if v["name"] == "expert")
                    probe = container.get("startupProbe", {})
                    if probe:
                        values["lifecycle.startup_seconds"] = probe.get("periodSeconds", 10) * probe.get("failureThreshold", 3)
                    values["lifecycle.termination_seconds"] = pod["spec"].get("terminationGracePeriodSeconds", 30)
                    row.update(runtime_values=values, runtime_source=snapshot["source"],
                               draining=snapshot.get("draining", bool(pod["metadata"].get("deletionTimestamp"))))
            except Exception as error:
                row["runtime_error" if runtime else "identity_error"] = str(error)[:300]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "apply", "verify", "inventory"])
    parser.add_argument("--config", default=str(ROOT / "deploy/pools/ascend-awq.json"))
    parser.add_argument("--instance")
    args = parser.parse_args()
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    pool = load_pool(Path(args.config))
    if args.action == "inventory":
        result = inventory(cluster)
    elif args.action == "plan":
        result = plan(cluster, pool, args.instance)
    elif args.action == "apply":
        with operation_lock(ROOT):
            result = apply(cluster, pool, args.instance)
    else:
        result = globals()[args.action](cluster, pool, args.instance)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
