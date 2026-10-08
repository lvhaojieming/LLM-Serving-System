"""Reversible worker membership, invoked by control.py; never uninstall or delete host data."""
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from manage_lab import (ROOT, OWNER, PROBE_IMAGE, active_nodes, apply_network_tuning,
                        ensure_node, kubectl, load_config, names, node_args, preflight, remote, operation_lock, validate_config)
from manage_npu import exclusive_reservations
from manage_pool import pool_objects, verify_pods

CLUSTER = "deploy/lab/cluster.json"
POOL = "deploy/pools/ascend-awq.json"
NPU = "deploy/lab/npu.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".node-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def get(cluster, *args):
    return json.loads(kubectl(cluster, ["get", *args, "-o", "json"]).stdout)


def ready(node):
    return any(c["type"] == "Ready" and c["status"] == "True" for c in node.get("status", {}).get("conditions", []))


def inspect_container(cluster, node, allow_missing=False):
    name, network, _ = names(cluster, node)
    result = remote(node["host"], ["docker", "inspect", name], check=False)
    if result.returncode:
        # An SSH/daemon failure must never be interpreted as permission to create.
        if allow_missing and "No such" in result.stderr and node["host"] not in cluster["locked_container_ids"]:
            return None
        raise RuntimeError("Cannot inspect fixed node container: " + name)
    info = json.loads(result.stdout)[0]
    expected = next(x.split("=", 1)[1] for x in node_args(cluster, node) if x.startswith(OWNER + ".config="))
    labels = info["Config"].get("Labels") or {}
    if labels.get(OWNER) != cluster["name"] or labels.get(OWNER + ".config") != expected:
        raise RuntimeError("Container ownership/configuration mismatch: " + name)
    if info["HostConfig"]["NetworkMode"] != network:
        raise RuntimeError("Refusing non-private node container")
    locked = cluster["locked_container_ids"].get(node["host"])
    if locked and info["Id"] != locked:
        raise RuntimeError("Fixed container identity changed")
    return info


def find_node(cluster, host, allow_new=False):
    if host not in cluster["available_hosts"] + cluster.get("authorized_additional_hosts", []):
        raise ValueError("Host is outside the explicitly authorized inventory")
    for node in cluster["nodes"]:
        if node["host"] == host:
            if node["role"] != "agent":
                raise ValueError("This workflow manages workers, never the control plane")
            return node
    if not allow_new:
        raise ValueError("Node is not registered")
    suffix = host.rsplit(".", 1)[1]
    return {"host": host, "role": "agent", "subnet": "172.29." + suffix + ".0/24", "address": "172.29." + suffix + ".2"}


def physical_free(host):
    table = remote(host, ["cat", "/proc/uda/namespace_node"]).stdout
    occupied = {d for r in exclusive_reservations(table) for d in r["physical_npus"]}
    return sorted(set(range(8)) - occupied)


def quantity(value):
    """Exact scheduler units used by these manifests, including node Ki and CPU m."""
    suffixes = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
                "K": 1000, "M": 1000**2, "G": 1000**3, "m": .001}
    for suffix, factor in suffixes.items():
        if str(value).endswith(suffix):
            return float(str(value)[:-len(suffix)]) * factor
    return float(value)


def request(pod, resource):
    containers = pod["spec"]["containers"]
    if pod["spec"].get("initContainers"):
        # Sidecar/init scheduling semantics are outside this initial controller.
        raise RuntimeError("Capacity calculation requires review of init containers")
    return sum(quantity(c.get("resources", {}).get("requests", {}).get(resource, "0")) for c in containers) + quantity(pod["spec"].get("overhead", {}).get(resource, "0"))


def removable_workloads(cluster, pool, node_name, pods):
    victims = []
    for pod in pods:
        if pod["spec"].get("nodeName") != node_name:
            continue
        m = pod["metadata"]
        labels = m.get("labels", {})
        owners = m.get("ownerReferences", [])
        owned = labels.get(OWNER) == cluster["name"]
        daemon = any(o["kind"] == "DaemonSet" and o["name"] in {
            "ascend-device-plugin-daemonset", cluster["name"] + "-network"} for o in owners)
        expert = (m["namespace"] == pool["namespace"] and labels.get("app.kubernetes.io/name") == "expert"
                  and labels.get("heteroserve.io/pool") == pool["pool"]
                  and any(o["kind"] == "ReplicaSet" for o in owners))
        if not owned or not (daemon or expert):
            raise RuntimeError("Migrate/review workload before retiring node: " + m["namespace"] + "/" + m["name"])
        if expert:
            victims.append(m["uid"])
    return victims


def removal_capacity(cluster, pool, npu, name, nodes, pods, physical):
    allowed = set(pool["model_nodes"]) - {name}
    if not allowed:
        raise RuntimeError("Cannot remove the last model candidate node")
    slots = 0
    details = []
    for node in nodes:
        node_name = node["metadata"]["name"]
        if node_name not in allowed or not ready(node) or node.get("spec", {}).get("unschedulable"):
            continue
        labels = node["metadata"].get("labels", {})
        if any(labels.get(k) != v for k, v in {"heteroserve.io/npu-ready": "true", "heteroserve.io/hardware": pool["hardware"], "heteroserve.io/backend": pool["backend"]}.items()):
            continue
        if any(t["effect"] in {"NoSchedule", "NoExecute"} for t in node.get("spec", {}).get("taints", [])):
            continue
        local = [p for p in pods if p["spec"].get("nodeName") == node_name and p["status"].get("phase") not in {"Succeeded", "Failed"}]
        alloc = node["status"]["allocatable"]
        free = {key: max(0, quantity(alloc.get(key, 0)) - sum(request(p, key) for p in local)) for key in (npu["resource"], "cpu", "memory")}
        available = min(int(free[npu["resource"]]), physical[node_name],
                        int(free["cpu"] / quantity(pool["worker_request_cpu"])),
                        int(free["memory"] / quantity(pool["worker_request_memory"])),
                        max(0, int(alloc.get("pods", 0)) - len(local)))
        existing = sum(p["metadata"].get("labels", {}).get("heteroserve.io/pool") == pool["pool"] and
                       p["metadata"].get("labels", {}).get("app.kubernetes.io/name") == "expert" and
                       not p["metadata"].get("deletionTimestamp") and ready(p) for p in local)
        slots += available + existing
        details.append({"node": node_name, "additional_slots": available, "existing_ready_workers": existing})
    if slots < pool["replicas"]:
        raise RuntimeError("Insufficient compatible remaining capacity: " + str(slots) + " < " + str(pool["replicas"]))
    return details


def plan(action, host):
    cluster = load_config(ROOT / CLUSTER)
    pool, npu, gateway = read(POOL), read(NPU), read("deploy/gateway.json")
    node = find_node(cluster, host, action == "add")
    if node not in cluster["nodes"]:
        proposed = copy.deepcopy(cluster)
        proposed["nodes"].append(node)
        proposed.setdefault("new_node_hosts", []).append(host)
        validate_config(proposed)
    name = names(cluster, node)[0]
    info = inspect_container(cluster, node, allow_missing=action == "add")
    result = {"action": action, "host": host, "node": name, "container_id": info["Id"] if info else None,
              "replicas_unchanged": pool["replicas"], "retains_container_volume_models": True}
    if action == "recover":
        journal = "artifacts/kubernetes/nodes/" + host.rsplit(".", 1)[1] + "/remove.json"
        previous = read(journal)
        if previous.get("passed") or previous["plan"]["container_id"] != info["Id"]:
            raise RuntimeError("Recovery requires an interrupted removal of this exact container")
        result["rejoin_required"] = host in cluster.get("retired_node_hosts", [])
        result["previous_stages"] = previous["stages"]
    elif action == "remove":
        if host in cluster.get("retired_node_hosts", []):
            raise ValueError("Node already retired; use add to rejoin explicitly")
        if name == gateway["deployment"]["node"] or name in pool["router_nodes"]:
            raise RuntimeError("Migrate Gateway/Router configuration before retiring this worker")
        pods = get(cluster, "pods", "-A")["items"]
        result["worker_uids"] = removable_workloads(cluster, pool, name, pods)
        nodes = get(cluster, "nodes")["items"]
        free = {names(cluster, n)[0]: len(physical_free(n["host"])) for n in active_nodes(cluster)
                if names(cluster, n)[0] in pool["model_nodes"] and n["host"] != host}
        result["capacity"] = removal_capacity(cluster, pool, npu, name, nodes, pods, free)
    else:
        active = name in pool["model_nodes"] and host not in cluster.get("retired_node_hosts", [])
        result["already_active"] = bool(active and info and info["State"]["Running"])
        if result["already_active"]:
            actual = get(cluster, "node", name)
            if not ready(actual) or actual.get("spec", {}).get("unschedulable"):
                raise RuntimeError("Existing worker is not Ready/schedulable; use node recover for interrupted removal")
        if not result["already_active"]:
            if len(set(pool["model_nodes"]) | {name}) > pool["replicas"]:
                raise RuntimeError("Increase pool replicas first to exercise every selected inference node")
            free = physical_free(host)
            result["physical_free_cards"] = free
            if len(free) != 8:
                raise RuntimeError("Exclusive device reservations remain; no automatic stop/reset of existing services")
            if info and not info["State"]["Running"] and host not in cluster.get("retired_node_hosts", []):
                raise RuntimeError("Stopped container has no controller retirement record; intentional pause preserved")
    return result


def stream(source, producer, target, consumer):
    """Stream existing assets over SSH without archives or a new environment."""
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
    with tempfile.TemporaryFile() as errors:
        p = subprocess.Popen([*ssh, "root@" + source, shlex.join(producer)], stdout=subprocess.PIPE, stderr=errors)
        q = subprocess.Popen([*ssh, "root@" + target, shlex.join(consumer)], stdin=p.stdout, stdout=subprocess.DEVNULL, stderr=errors)
        p.stdout.close()
        try:
            q.wait(timeout=1800)
            p.wait(timeout=30)
            if p.returncode or q.returncode:
                errors.seek(0)
                raise RuntimeError("Asset stream failed: " + errors.read().decode(errors="replace")[-1000:])
        finally:
            for process in (p, q):
                if process.poll() is None:
                    process.kill()
                process.wait()


def asset_manifest(host, directory):
    code = """import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]); result={}
if not p.is_dir(): raise RuntimeError('Asset directory missing: '+str(p))
for f in sorted(p.rglob('*')):
 if '__pycache__' in f.parts or f.suffix=='.pyc':continue
 if f.is_symlink():raise RuntimeError('Review symlink before asset transfer: '+str(f))
 if f.is_file():
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  result[str(f.relative_to(p))]=h.hexdigest()
print(json.dumps(result))
"""
    return json.loads(remote(host, ["python3", "-c", code, directory], timeout=600).stdout)


def canonical_image(reference):
    """Match Docker Hub shorthand to containerd's fully qualified references."""
    head = reference.split("/", 1)[0]
    if "/" not in reference:
        return "docker.io/library/" + reference
    if "." not in head and ":" not in head and head != "localhost":
        return "docker.io/" + reference
    return reference


def missing_images(required, source, target):
    available = {canonical_image(v): v for v in source}
    present = {canonical_image(v) for v in target}
    absent = [canonical_image(v) for v in required if canonical_image(v) not in present]
    if any(v not in available for v in absent):
        raise RuntimeError("Pinned image missing from source runtime: " + str([v for v in absent if v not in available]))
    return [available[v] for v in dict.fromkeys(absent)]


def prepare_assets(cluster, node, npu, pool):
    source, target = npu["host"], node["host"]
    results = {}
    for key in ("model_source", "adapter_source"):
        directory = npu[key]
        expected = asset_manifest(source, directory)
        if key == "model_source" and expected.get("conversion_manifest.json") != pool["weights_manifest_sha256"]:
            raise RuntimeError("Source weights do not match pinned manifest")
        exists = remote(target, ["test", "-e", directory], check=False)
        if exists.returncode:
            if exists.returncode != 1:
                raise RuntimeError("Cannot inspect target asset path")
            remote(target, ["mkdir", "-p", str(Path(directory).parent)])
            stream(source, ["tar", "--exclude=__pycache__", "--exclude=*.pyc", "-C", str(Path(directory).parent), "-cf", "-", Path(directory).name],
                   target, ["tar", "--keep-old-files", "-C", str(Path(directory).parent), "-xf", "-"])
        actual = asset_manifest(target, directory)
        if actual != expected:
            raise RuntimeError("Existing target assets differ; refusing overwrite: " + directory)
        results[key] = {"sha256": expected, "reused": exists.returncode == 0}
    source_node = next(n for n in active_nodes(cluster) if n["host"] == source)
    src_name, dst_name = names(cluster, source_node)[0], names(cluster, node)[0]
    images = [cluster["image"], PROBE_IMAGE, npu["plugin_image"], npu["model_image"]]
    # Pause reference comes from the source runtime, never an unpinned registry guess.
    source_images = remote(source, ["docker", "exec", src_name, "ctr", "-n", "k8s.io", "images", "ls", "-q"]).stdout.splitlines()
    images += [x for x in source_images if "pause:" in x]
    present = remote(target, ["docker", "exec", dst_name, "ctr", "-n", "k8s.io", "images", "ls", "-q"]).stdout.splitlines()
    missing = missing_images(images, source_images, present)
    if missing:
        stream(source, ["docker", "exec", src_name, "ctr", "-n", "k8s.io", "images", "export", "--platform", "linux/arm64", "-", *missing],
               target, ["docker", "exec", "-i", dst_name, "ctr", "-n", "k8s.io", "images", "import", "--platform", "linux/arm64", "--digests", "-"])
    results["imported_images"] = missing
    run("manage_npu.py", "publish", "--host", target)
    return results


def candidate_objects(cluster, pool, npu, name):
    objects = pool_objects(cluster, pool, npu)
    config = copy.deepcopy(objects[0])
    candidate = pool["pool"] + "-candidate-" + name.rsplit("-", 1)[1]
    config["metadata"]["name"] = candidate
    template = copy.deepcopy(objects[1]["spec"]["template"])
    template["metadata"]["name"] = candidate
    template["metadata"]["namespace"] = pool["namespace"]
    # Neither the Service nor native Router discovery matches this staging Pod.
    template["metadata"]["labels"]["app.kubernetes.io/name"] = "expert-candidate"
    spec = template["spec"]
    spec["restartPolicy"] = "Never"
    spec.pop("affinity", None)
    spec.pop("topologySpreadConstraints", None)
    spec["nodeSelector"] = {"kubernetes.io/hostname": name}
    next(v for v in spec["volumes"] if v["name"] == "runtime")["configMap"]["name"] = candidate
    return config, {"apiVersion": "v1", "kind": "Pod", **template}


def validate_candidate(cluster, pool, npu, name):
    objects = candidate_objects(cluster, pool, npu, name)
    created = []
    try:
        for obj in objects:
            # create refuses collisions; never adopt or delete somebody else's Pod.
            data = json.loads(kubectl(cluster, ["create", "-f", "-", "-o", "json"], stdin=json.dumps(obj)).stdout)
            created.append((obj["kind"], obj["metadata"]["name"], data["metadata"]["uid"]))
        podname = objects[1]["metadata"]["name"]
        kubectl(cluster, ["wait", "pod/" + podname, "-n", pool["namespace"], "--for=condition=Ready",
                          "--timeout=" + str(pool["startup_seconds"]) + "s"], timeout=pool["startup_seconds"] + 60)
        pod = get(cluster, "pod", podname, "-n", pool["namespace"])
        if pod["spec"]["nodeName"] != name:
            raise RuntimeError("Candidate scheduled on wrong node")
        from verify_lifecycle import discovery, inside_router
        view = discovery(cluster, pool)
        if pod["metadata"]["uid"] in {e["uid"] for e in view["endpoints"]}:
            raise RuntimeError("Unapproved candidate leaked into production discovery")
        code = "import urllib.request,sys;print(urllib.request.urlopen('http://'+sys.argv[1]+':8000/ready',timeout=10).status)"
        if inside_router(cluster, pool, code, pod["status"]["podIP"]).strip() != "200":
            raise RuntimeError("Router cannot reach candidate across the Pod network")
        result = verify_pods(cluster, pool, npu, [pod])
        result["excluded_from_production_discovery"] = True
        result["router_to_candidate_network"] = True
        return result
    finally:
        for kind, objname, uid in reversed(created):
            current = get(cluster, kind, objname, "-n", pool["namespace"])
            if current["metadata"]["uid"] != uid or current["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Candidate identity changed; refusing cleanup")
            kubectl(cluster, ["delete", kind, objname, "-n", pool["namespace"], "--wait=true", "--timeout=360s"], timeout=380)


def verify_service():
    run("control.py", "verify", "all")
    return {"pool": read("artifacts/kubernetes/pool-verification.json"),
            "routing": read("artifacts/kubernetes/lifecycle/routing.json"),
            "gateway": read("artifacts/kubernetes/learned-gateway.json")}


def remove(cluster, node, report, checkpoint):
    pool = read(POOL)
    name = names(cluster, node)[0]
    # Re-evaluate live capacity/ownership immediately before any mutation.
    report["plan"] = plan("remove", node["host"])
    kubectl(cluster, ["cordon", name])
    checkpoint("cordoned")
    # Eviction honors PDB and existing worker preStop/grace period. No force deletion.
    kubectl(cluster, ["drain", name, "--ignore-daemonsets", "--delete-emptydir-data", "--timeout=1500s"], timeout=1560)
    checkpoint("drained")
    report["migration"] = verify_service()
    remaining = get(cluster, "pods", "-A")["items"]
    if removable_workloads(cluster, pool, name, remaining):
        raise RuntimeError("Expert Pods still present after drain")
    # Persist exclusion before stopping, so general up/verify never resumes it.
    pool["model_nodes"] = [v for v in pool["model_nodes"] if v != name]
    write(POOL, pool)
    cluster.setdefault("retired_node_hosts", []).append(node["host"])
    write(CLUSTER, cluster)
    checkpoint("retirement_recorded")
    info = inspect_container(cluster, node)
    remote(node["host"], ["docker", "stop", "--time", "60", info["Id"]], timeout=90)
    kubectl(cluster, ["delete", "node", name, "--wait=true", "--timeout=60s"])
    checkpoint("node_removed_container_retained")
    run("manage_pool.py", "apply")
    report["acceptance"] = verify_service()
    report["container_preserved"] = inspect_container(cluster, node)["Id"] == info["Id"]
    report["container_stopped"] = not inspect_container(cluster, node)["State"]["Running"]
    report["node_absent"] = name not in {n["metadata"]["name"] for n in get(cluster, "nodes")["items"]}
    if not all(report[k] for k in ("container_preserved", "container_stopped", "node_absent")):
        raise RuntimeError("Retirement postconditions failed")


def add(cluster, node, report, checkpoint):
    pool, npu = read(POOL), read(NPU)
    name = names(cluster, node)[0]
    if report["plan"]["already_active"]:
        report["acceptance"] = verify_service()
        return
    if node not in cluster["nodes"]:
        cluster["nodes"].append(node)
        cluster.setdefault("new_node_hosts", []).append(node["host"])
        validate_config(cluster)
        write(CLUSTER, cluster)
        cluster = load_config(ROOT / CLUSTER)
    preflight(cluster)
    info = inspect_container(cluster, node, allow_missing=True)
    if info is None:
        source = npu["host"]
        image = remote(node["host"], ["docker", "image", "inspect", cluster["image_config_id"]], check=False)
        if image.returncode:
            stream(source, ["docker", "save", cluster["image_config_id"]], node["host"], ["docker", "load"])
        server = next(n for n in cluster["nodes"] if n["role"] == "server")
        token = remote(server["host"], ["docker", "exec", names(cluster, server)[0], "cat", "/var/lib/rancher/k3s/server/node-token"]).stdout
        writer = "import os,sys;os.makedirs(sys.argv[1],mode=0o700,exist_ok=True);f=os.open(sys.argv[1]+'/join-token',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600);os.fchmod(f,0o600);os.write(f,sys.stdin.buffer.read());os.close(f)"
        remote(node["host"], ["python3", "-c", writer, cluster["state_directory"]], stdin=token)
        ensure_node(cluster, node)
        info = inspect_container(cluster, node)
        cluster["locked_container_ids"][node["host"]] = info["Id"]
        cluster["new_node_hosts"].remove(node["host"])
        write(CLUSTER, cluster)
    elif not info["State"]["Running"]:
        # Only explicit node add may restart a controller-retired container.
        if node["host"] not in cluster.get("retired_node_hosts", []):
            raise RuntimeError("Intentional pause preserved")
        remote(node["host"], ["docker", "start", info["Id"]])
    checkpoint("container_started")
    report["assets"] = prepare_assets(cluster, node, npu, pool)
    deadline = time.monotonic() + 180
    while name not in {n["metadata"]["name"] for n in get(cluster, "nodes")["items"]}:
        if time.monotonic() > deadline:
            raise TimeoutError("Node did not register")
        time.sleep(3)
    kubectl(cluster, ["cordon", name])
    kubectl(cluster, ["label", "node", name, "heteroserve.io/npu-ready=false", "--overwrite"])
    if node["host"] in cluster.get("retired_node_hosts", []):
        cluster["retired_node_hosts"].remove(node["host"])
        if not cluster["retired_node_hosts"]:
            del cluster["retired_node_hosts"]
    if node["host"] not in npu["device_hosts"]:
        npu["device_hosts"].append(node["host"])
    write(CLUSTER, cluster)
    write(NPU, npu)
    run("manage_npu.py", "plugin", "--host", node["host"])
    kubectl(cluster, ["wait", "node/" + name, "--for=condition=Ready", "--timeout=180s"], timeout=200)
    apply_network_tuning(cluster)
    deadline = time.monotonic() + 180
    while int(get(cluster, "node", name)["status"].get("allocatable", {}).get(npu["resource"], 0)) != 8:
        if time.monotonic() > deadline:
            raise TimeoutError("Expected eight allocatable physical Ascend devices")
        time.sleep(3)
    # Pool admission remains blocked by npu-ready=false and candidate labels.
    kubectl(cluster, ["uncordon", name])
    checkpoint("devices_registered_pool_excluded")
    try:
        namespaces = get(cluster, "namespaces")["items"]
        validation = next((v for v in namespaces if v["metadata"]["name"] == "heteroserve-npu-validation"), None)
        if validation:
            if validation["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Foreign isolation namespace")
            if get(cluster, "pods", "-n", "heteroserve-npu-validation")["items"]:
                raise RuntimeError("Existing isolation experiment must finish before node onboarding")
        try:
            run("manage_npu.py", "isolation", "--host", node["host"])
            run("manage_npu.py", "isolation-results", "--host", node["host"])
            report["isolation"] = read("artifacts/kubernetes/npu/" + node["host"].rsplit(".", 1)[1] + "/isolation.json")
        finally:
            run("manage_npu.py", "cleanup", "--host", node["host"])
        report["candidate"] = validate_candidate(cluster, pool, npu, name)
    except BaseException:
        kubectl(cluster, ["cordon", name])
        raise
    checkpoint("isolated_model_verified")
    kubectl(cluster, ["label", "node", name, "heteroserve.io/npu-ready=true", "heteroserve.io/hardware=" + pool["hardware"],
                      "heteroserve.io/backend=" + pool["backend"], "--overwrite"])
    if name not in pool["model_nodes"]:
        pool["model_nodes"].append(name)
    write(POOL, pool)
    run("manage_pool.py", "apply")
    report["acceptance"] = verify_service()
    if not any(r["identity"]["node"] == name for r in report["acceptance"]["pool"]["replicas"]):
        raise RuntimeError("New node has no production worker; inspect scheduling/spread before claiming admission")
    report["container_id"] = inspect_container(cluster, node)["Id"]


def execute(action, host, apply=False):
    result = plan(action, host)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if not apply:
        return result
    with operation_lock(ROOT):
        cluster = load_config(ROOT / CLUSTER)
        node = find_node(cluster, host, action == "add")
        result = plan(action, host)
        report = {"passed": False, "plan": result, "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "configuration_before": {p: read(p) for p in (CLUSTER, POOL, NPU)}, "stages": [],
                  "python": sys.version,
                  "code_sha256": {p: hashlib.sha256((ROOT / "scripts" / p).read_bytes()).hexdigest()
                                  for p in ("control.py", "manage_nodes.py", "manage_lab.py", "manage_npu.py", "manage_pool.py")}}
        path = "artifacts/kubernetes/nodes/" + host.rsplit(".", 1)[1] + "/" + action + ".json"
        if (ROOT / path).exists():
            previous = read(path)
            report["previous_attempts"] = previous.get("previous_attempts", []) + [
                {k: previous[k] for k in ("started_at", "passed", "stages", "error", "code_sha256") if k in previous}]
        def checkpoint(stage):
            report["stages"].append({"stage": stage, "at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
            write(path, report)
            print("NODE_STAGE " + stage, flush=True)
        try:
            checkpoint("preflight_passed")
            {"add": add, "remove": remove, "recover": recover}[action](cluster, node, report, checkpoint)
            report["passed"] = True
        except BaseException as error:
            report["error"] = str(error)
            report["recovery"] = "Inspect last stage and actual state. Preserve cordon on failure; node add explicitly resumes a controller-retired node. No force drain or automatic data deletion."
            raise
        finally:
            report["configuration_after"] = {p: read(p) for p in (CLUSTER, POOL, NPU)}
            checkpoint("finished" if report["passed"] else "needs_attention")
        return report


def recover(cluster, node, report, checkpoint):
    """Explicitly recover this controller's interrupted removal, never a generic pause."""
    if report["plan"]["rejoin_required"]:
        report["plan"] = plan("add", node["host"])
        add(cluster, node, report, checkpoint)
        return
    info = inspect_container(cluster, node)
    if not info["State"]["Running"]:
        raise RuntimeError("Unrecorded stopped state; refusing restart")
    kubectl(cluster, ["uncordon", names(cluster, node)[0]])
    checkpoint("interrupted_removal_uncordoned")
    report["acceptance"] = verify_service()
