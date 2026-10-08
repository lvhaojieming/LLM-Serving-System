"""Render and manage the Ascend expert pool through the private Kubernetes API."""
import argparse
import copy
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from manage_lab import ROOT, OWNER, kubectl, load_config, names, remote

POOL_CONFIG = ROOT / "deploy/pools/ascend-awq.json"


def load_pool(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "defaults" not in raw:
        return raw
    base = (path.parent / raw["defaults"]).resolve()
    if base.parent != path.parent.resolve() or base.name != "defaults.json":
        raise ValueError("Pool defaults must be the shared defaults.json")
    return {**json.loads(base.read_text(encoding="utf-8")), **raw}


def resolve_runtime(pool, root=ROOT):
    runtime = (root / pool.get("runtime_config", "deploy/lab/npu.json")).resolve()
    if not runtime.is_relative_to(root.resolve()):
        raise ValueError("Runtime configuration must remain in the formal project")
    npu = json.loads(runtime.read_text(encoding="utf-8"))
    overrides = copy.deepcopy(pool.get("runtime", {}))
    if set(overrides) - {"model_source", "served_model", "model_parameters"}:
        raise ValueError("Pool runtime overrides are limited to model and engine parameters")
    params = overrides.pop("model_parameters", {})
    npu.update(overrides)
    npu["model_parameters"].update(params)
    asset = Path(pool.get("model_asset", "model"))
    if asset.is_absolute() or ".." in asset.parts:
        raise ValueError("Model asset must stay inside the node asset directory")
    return npu


def artifact_path(pool, filename):
    # Preserve the original AWQ report location; additional pools have separate records.
    base = ROOT / "artifacts/kubernetes"
    if pool["expert"] != "awq":
        base = base / "pools" / pool["expert"]
    return base / filename


def configuration():
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    pool = load_pool(POOL_CONFIG)
    npu = resolve_runtime(pool)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", pool["pool"]) or pool["namespace"] != "heteroserve":
        raise ValueError("Unexpected pool identity")
    if not re.fullmatch(r"[a-f0-9]{64}", pool["weights_manifest_sha256"]):
        raise ValueError("Model weights must have a pinned manifest")
    if not set(pool["model_nodes"]) <= {names(cluster, n)[0] for n in cluster["nodes"]}:
        raise ValueError("Model nodes must belong to the fixed lab")
    if pool["termination_seconds"] <= pool["drain_seconds"]:
        raise ValueError("Termination budget must cover draining and engine shutdown")
    if type(pool["replicas"]) is not int or pool["replicas"] < 1:
        raise ValueError("The initial pool requires a positive manual replica count")
    return cluster, pool, npu


def worker_config(pool, npu):
    command = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", npu["model_source"].replace("/root/", "/workspace/"),
               "--served-model-name", npu["served_model"], "--host", "127.0.0.1", "--port", "8001", "--tensor-parallel-size", "1", "--seed", "0"]
    for key, value in npu["model_parameters"].items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command += [flag, str(value)]
    return {"pool": pool["pool"], "model": npu["served_model"], "device_type": pool["device_type"],
            "weights_manifest_sha256": pool["weights_manifest_sha256"], "image": npu["model_image"],
            "port": 8000, "engine_url": "http://127.0.0.1:8001", "engine_command": command,
            "engine_env": {"MOQE_ASCEND_INT4_ADAPTER": "1"},
            "health_interval_seconds": 3, "health_timeout_seconds": 10, "readiness_failures": 3,
            "liveness_failures": 10, "admission_timeout_seconds": 120,
            "request_timeout_seconds": 180, "drain_seconds": pool["drain_seconds"], "engine_stop_seconds": 30,
            "max_inflight": pool["requests_per_worker"]}


def pool_objects(cluster, pool, npu):
    namespace = pool["namespace"]
    selector = {"app.kubernetes.io/part-of": "heteroserve", "heteroserve.io/pool": pool["pool"], "app.kubernetes.io/name": "expert"}
    labels = {**selector, OWNER: cluster["name"], "heteroserve.io/expert": pool["expert"],
              "heteroserve.io/hardware": pool["hardware"], "heteroserve.io/backend": pool["backend"],
              "heteroserve.io/model-version": pool["weights_manifest_sha256"][:16]}
    data = {"worker.py": (ROOT / "src/heteroserve/worker.py").read_text(), "worker.json": json.dumps(worker_config(pool, npu), indent=2)}
    config_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    env = {"HOME": "/scratch/home", "XDG_CACHE_HOME": "/scratch/cache", "VLLM_CACHE_ROOT": "/scratch/vllm",
           "VLLM_USE_V1": "1", "VLLM_NO_USAGE_STATS": "1", "VLLM_VERSION": "0.8.4", "OMP_NUM_THREADS": "4",
           "TE_PARALLEL_COMPILER": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "MOQE_ASCEND_INT4_ADAPTER": "0",
           "PYTHONPATH": "/workspace/zhangjinhao/moqe-runtime/adapter:/workspace/zhangjinhao/moqe-runtime/python-deps:/workspace/vllm:/workspace/vllm-ascend"}
    container = {"name": "expert", "image": npu["model_image"], "imagePullPolicy": "Never",
                 "command": ["/bin/bash", "-ec"],
                 "args": ["mkdir -p /scratch/home /scratch/cache /scratch/vllm; source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 /opt/heteroserve/worker.py --config /opt/heteroserve/worker.json"],
                 "env": [{"name": k, "value": v} for k, v in env.items()] + [
                     {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
                     {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
                 "ports": [{"name": "http", "containerPort": 8000}],
                 "resources": {"requests": {"cpu": pool["worker_request_cpu"], "memory": pool["worker_request_memory"], npu["resource"]: "1"}, "limits": {npu["resource"]: "1"}},
                 "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
                 "startupProbe": {"httpGet": {"path": "/ready", "port": "http"}, "periodSeconds": 5, "timeoutSeconds": 3,
                                  "failureThreshold": pool["startup_seconds"] // 5},
                 "readinessProbe": {"httpGet": {"path": "/ready", "port": "http"}, "periodSeconds": 3, "timeoutSeconds": 3, "failureThreshold": 1},
                 "livenessProbe": {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 10, "timeoutSeconds": 3, "failureThreshold": 6},
                 "lifecycle": {"preStop": {"exec": {"command": ["python3", "-c",
                     "import urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/drain',method='POST'),timeout=" + str(pool["drain_seconds"] + 5) + ").read()"]}}},
                 "volumeMounts": [{"name": "runtime", "mountPath": "/opt/heteroserve", "readOnly": True},
                                  {"name": "driver", "mountPath": "/usr/local/Ascend/driver", "readOnly": True},
                                  {"name": "weights", "mountPath": npu["model_source"].replace("/root/", "/workspace/"), "readOnly": True},
                                  {"name": "adapter", "mountPath": npu["adapter_source"].replace("/root/", "/workspace/"), "readOnly": True},
                                  {"name": "scratch", "mountPath": "/scratch"}, {"name": "tmp", "mountPath": "/tmp"}]}
    pod = {"automountServiceAccountToken": False, "terminationGracePeriodSeconds": pool["termination_seconds"],
           "nodeSelector": {"heteroserve.io/hardware": pool["hardware"], "heteroserve.io/backend": pool["backend"], "heteroserve.io/npu-ready": "true"},
           "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [
               {"key": "kubernetes.io/hostname", "operator": "In", "values": pool["model_nodes"]}]}]}}},
           "topologySpreadConstraints": [{"maxSkew": 1, "topologyKey": "kubernetes.io/hostname", "whenUnsatisfiable": "ScheduleAnyway", "labelSelector": {"matchLabels": selector}}],
           "containers": [container], "volumes": [
               {"name": "runtime", "configMap": {"name": pool["pool"] + "-runtime"}},
               {"name": "driver", "hostPath": {"path": "/usr/local/Ascend/driver", "type": "Directory"}},
                 {"name": "weights", "hostPath": {"path": "/var/lib/rancher/k3s/heteroserve-assets/" + pool.get("model_asset", "model"), "type": "Directory"}},
               {"name": "adapter", "hostPath": {"path": "/var/lib/rancher/k3s/heteroserve-assets/adapter", "type": "Directory"}},
               {"name": "scratch", "emptyDir": {"sizeLimit": pool["scratch_limit"]}},
               {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}]}
    return [{"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": pool["pool"] + "-runtime", "namespace": namespace, "labels": labels}, "data": data},
            {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": pool["pool"], "namespace": namespace, "labels": labels},
             "spec": {"replicas": pool["replicas"], "revisionHistoryLimit": 3, "progressDeadlineSeconds": pool["startup_seconds"] + 120,
                      "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}},
                      "selector": {"matchLabels": selector}, "template": {"metadata": {"labels": labels, "annotations": {
                          "heteroserve.io/runtime-sha256": config_hash, "prometheus.io/scrape": "true", "prometheus.io/port": "8000", "prometheus.io/path": "/metrics"}}, "spec": pod}}},
            {"apiVersion": "v1", "kind": "Service", "metadata": {"name": pool["pool"], "namespace": namespace, "labels": labels},
             "spec": {"selector": selector, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}},
            {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": {"name": pool["pool"], "namespace": namespace, "labels": labels},
             "spec": {"minAvailable": 1, "selector": {"matchLabels": selector}}}]


def render():
    import yaml
    cluster, pool, npu = configuration()
    objects = pool_objects(cluster, pool, npu) + router_objects(cluster, pool, npu)
    path = ROOT / "deploy/k8s/pools" / (POOL_CONFIG.stem + ".yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump_all(objects, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("Rendered " + str(path))


def prepare_model():
    """Reuse existing publication and transfer logic for the selected expert."""
    from manage_nodes import prepare_assets, inspect_container
    cluster, pool, npu = configuration()
    source_runtime = copy.deepcopy(npu)
    source_runtime["host"] = pool.get("asset_source_host", npu["host"])
    reports = {}
    for node in cluster["nodes"]:
        if names(cluster, node)[0] in pool["model_nodes"]:
            inspect_container(cluster, node)
            reports[node["host"]] = prepare_assets(cluster, node, source_runtime, pool, POOL_CONFIG)
    path = artifact_path(pool, "asset-publication.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reports, indent=2), encoding="utf-8")


def router_objects(cluster, pool, npu):
    name, namespace = pool["pool"] + "-router", pool["namespace"]
    selector = {"app.kubernetes.io/part-of": "heteroserve", "app.kubernetes.io/name": "router", "heteroserve.io/pool": pool["pool"]}
    labels = {**selector, OWNER: cluster["name"]}
    config = {"namespace": namespace, "pool": pool["pool"], "expert": pool["expert"], "model": npu["served_model"],
              "core_url": "http://127.0.0.1:8002", "discovery_cache_seconds": pool["discovery_cache_seconds"],
              "discovery_interval_seconds": 1, "requests_per_worker": pool["requests_per_worker"], "max_request_bytes": 2 * 1024 * 1024,
              "drain_seconds": pool["drain_seconds"], "core_command": ["python3", "-m", "vllm_router.launch_router", "--host", "127.0.0.1", "--port", "8002",
                  "--service-discovery", "--selector", "app.kubernetes.io/name=expert", "heteroserve.io/pool=" + pool["pool"],
                  "--service-discovery-namespace", namespace, "--service-discovery-port", "8000", "--policy", "power_of_two",
                  "--health-check-interval-secs", "2", "--health-check-timeout-secs", "5", "--disable-retries",
                  "--max-concurrent-requests", "64", "--queue-timeout-secs", "2"]}
    data = {"router.py": (ROOT / "src/heteroserve/router.py").read_text(), "router.json": json.dumps(config, indent=2)}
    container = {"name": "router", "image": npu["model_image"], "imagePullPolicy": "Never", "command": ["python3", "/opt/heteroserve/router.py", "--config", "/opt/heteroserve/router.json"],
                 "env": [{"name": "PYTHONPATH", "value": "/router-deps"}, {"name": "HOME", "value": "/tmp"}, {"name": "MOQE_ASCEND_INT4_ADAPTER", "value": "0"}],
                 "ports": [{"name": "http", "containerPort": 8000}], "resources": {"requests": {"cpu": "500m", "memory": "512Mi"}},
                 "securityContext": {"runAsUser": 10001, "runAsGroup": 10001, "runAsNonRoot": True, "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
                 "startupProbe": {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 3, "failureThreshold": 60},
                 "readinessProbe": {"httpGet": {"path": "/ready", "port": "http"}, "periodSeconds": 3, "timeoutSeconds": 5},
                 "livenessProbe": {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 10, "failureThreshold": 6},
                 "lifecycle": {"preStop": {"exec": {"command": ["python3", "-c", "import urllib.request;urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/drain',method='POST'),timeout=" + str(pool["drain_seconds"] + 5) + ").read()"]}}},
                 "volumeMounts": [{"name": "runtime", "mountPath": "/opt/heteroserve", "readOnly": True}, {"name": "deps", "mountPath": "/router-deps", "readOnly": True}, {"name": "tmp", "mountPath": "/tmp"}]}
    server = next(n for n in cluster["nodes"] if n["role"] == "server")
    api_destinations = [(str(ipaddress.ip_network(cluster["service_cidr"]).network_address + 1), 443),
                        (server["address"], 6443), (server["host"], cluster["api_port"])]
    return [{"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name + "-runtime", "namespace": namespace, "labels": labels}, "data": data},
            {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "namespace": namespace, "labels": labels}, "spec": {
                "replicas": pool["router_replicas"], "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}},
                "selector": {"matchLabels": selector}, "template": {"metadata": {"labels": labels, "annotations": {
                    "heteroserve.io/runtime-sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}}, "spec": {
                    "serviceAccountName": "heteroserve-discovery", "automountServiceAccountToken": True,
                    "terminationGracePeriodSeconds": pool["termination_seconds"],
                    "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": pool["router_nodes"]}]}]}},
                                 "podAntiAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [{"labelSelector": {"matchLabels": selector}, "topologyKey": "kubernetes.io/hostname"}]}},
                    "topologySpreadConstraints": [{"maxSkew": 1, "topologyKey": "kubernetes.io/hostname", "whenUnsatisfiable": "ScheduleAnyway", "labelSelector": {"matchLabels": selector}}],
                    "containers": [container], "volumes": [{"name": "runtime", "configMap": {"name": name + "-runtime"}},
                        {"name": "deps", "hostPath": {"path": "/var/lib/rancher/k3s/heteroserve-assets/router", "type": "Directory"}},
                        {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}]}}}},
            {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name, "namespace": namespace, "labels": labels},
             "spec": {"selector": selector, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}},
            {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy", "metadata": {"name": name + "-api", "namespace": namespace}, "spec": {
                "podSelector": {"matchLabels": selector}, "policyTypes": ["Egress"], "egress": [
                    {"to": [{"ipBlock": {"cidr": ip + "/32"}}], "ports": [{"protocol": "TCP", "port": port}]} for ip, port in api_destinations]}}]


def router():
    cluster, pool, npu = configuration()
    objects = router_objects(cluster, pool, npu)
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": objects})
    kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)


def prepare_router():
    cluster, pool, npu = configuration()
    packages = pool["router_dependencies"]
    download = """import json,sys,urllib.request,hashlib
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
for package in json.loads(sys.argv[2]):
 p=root/package['filename']
 if not p.exists():urllib.request.urlretrieve(package['url'],p)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=package['sha256']:raise RuntimeError('Router dependency checksum mismatch')
print('Pinned dependency cache verified; no global environment installation')
"""
    print(remote(npu["host"], ["python3", "-c", download, cluster["state_directory"] + "/router", json.dumps(packages)]).stdout)
    publish = """import json,sys,subprocess,tarfile,zipfile,hashlib
from pathlib import Path
state,volume,owner,packages=sys.argv[1],sys.argv[2],sys.argv[3],json.loads(sys.argv[4])
cache=Path(state)/'router';cache.mkdir(parents=True,exist_ok=True)
allowed={p['filename'] for p in packages}
archive=tarfile.open(fileobj=sys.stdin.buffer,mode='r|')
for member in archive:
 if not member.isfile() or member.name not in allowed:raise RuntimeError('Unexpected router archive member')
 data=archive.extractfile(member).read();package=next(p for p in packages if p['filename']==member.name)
 if hashlib.sha256(data).hexdigest()!=package['sha256']:raise RuntimeError('Dependency transfer checksum mismatch')
 destination=cache/member.name
 if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()!=package['sha256']:raise RuntimeError('Existing cache differs')
 if not destination.exists():destination.write_bytes(data)
info=json.loads(subprocess.check_output(['docker','volume','inspect',volume]))[0]
if (info.get('Labels') or {}).get('io.heteroserve.lab')!=owner:raise RuntimeError('Unowned node volume')
root=Path(info['Mountpoint'])/'heteroserve-assets/router';root.mkdir(parents=True,exist_ok=True)
for package in packages:
 with zipfile.ZipFile(cache/package['filename']) as wheel:
  for item in wheel.infolist():
   relative=Path(item.filename)
   if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe wheel path')
   target=root/relative
   if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
   data=wheel.read(item);target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.read_bytes()!=data:raise RuntimeError('Published dependency differs: '+str(relative))
   if not target.exists():target.write_bytes(data)
(root/'publication.json').write_text(json.dumps(packages,indent=2))
print('Verified router dependencies published into existing node volume')
"""
    for node in cluster["nodes"]:
        if names(cluster, node)[0] not in pool["router_nodes"]:
            continue
        identity = remote(node["host"], ["docker", "inspect", names(cluster, node)[0], "--format", "{{.Id}}"]).stdout.strip()
        if identity != cluster["locked_container_ids"][node["host"]]:
            raise RuntimeError("Fixed node identity changed")
        if node["host"] == npu["host"]:
            sender = ["tar", "-C", cluster["state_directory"] + "/router", "-cf", "-", *[p["filename"] for p in packages]]
        else:
            sender = ["ssh", "-o", "BatchMode=yes", "root@" + npu["host"], shlex.join(["tar", "-C", cluster["state_directory"] + "/router", "-cf", "-", *[p["filename"] for p in packages]])]
        pipe = subprocess.Popen(sender, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        destination = ["python3", "-c", publish, cluster["state_directory"], names(cluster, node)[2], cluster["name"], json.dumps(packages)]
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@" + node["host"], shlex.join(destination)], stdin=pipe.stdout)
        pipe.stdout.close()
        if pipe.wait() or result.returncode:
            raise RuntimeError("Router dependency publication failed")


def apply():
    cluster, pool, npu = configuration()
    base = ROOT / "deploy/k8s/base"
    for file in ["namespace.yaml", "rbac.yaml", "network.yaml"]:
        kubectl(cluster, ["apply", "-f", "-"], stdin=(base / file).read_text())
    # Driver and node-local weights need exact, read-only hostPath mounts. This
    # namespace exception does not grant workloads privileged containers or host PID/network.
    kubectl(cluster, ["label", "namespace", pool["namespace"], "pod-security.kubernetes.io/enforce=privileged", "--overwrite"])
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": pool_objects(cluster, pool, npu)})
    kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)


def status():
    cluster, pool, npu = configuration()
    print(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "heteroserve.io/pool=" + pool["pool"], "-o", "wide"]).stdout)
    print(kubectl(cluster, ["get", "events", "-n", pool["namespace"], "--sort-by=.lastTimestamp"]).stdout)


def verify():
    cluster, pool, npu = configuration()
    # Scaling while changing node affinity can briefly leave the old revision
    # at the desired Ready count. Wait for the declared revision before probing.
    kubectl(cluster, ["rollout", "status", "deployment/" + pool["pool"], "-n", pool["namespace"],
                      "--timeout=" + str(pool["startup_seconds"]) + "s"], timeout=pool["startup_seconds"] + 60)
    deadline = time.monotonic() + pool["startup_seconds"]
    while True:
        data = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "heteroserve.io/pool=" + pool["pool"] + ",app.kubernetes.io/name=expert", "-o", "json"]).stdout)
        pods = [p for p in data["items"] if not p["metadata"].get("deletionTimestamp") and
                any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if len(pods) == pool["replicas"]:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Expected replica count is not Ready; inspect startup and scheduling events")
        time.sleep(3)
    result = verify_pods(cluster, pool, npu, pods)
    path = artifact_path(pool, "pool-verification.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def verify_pods(cluster, pool, npu, pods):
    """Shared real-model check for production workers and isolated onboarding Pods."""
    checker = """import json,sys,time,urllib.request
model=sys.argv[1]
def get(path):return json.load(urllib.request.urlopen('http://127.0.0.1:8000'+path,timeout=10))
identity=get('/identity');get('/ready');get('/live')
models=get('/v1/models');assert model in [m['id'] for m in models['data']]
body={'model':model,'messages':[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
def send(body):return urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120)
with send(body) as response:answer=json.load(response)['choices'][0]['message']['content'].strip()
assert answer=='42',answer
body['stream']=True;chunks=[];done=False;started=time.monotonic();first=None
with send(body) as response:
 for line in response:
  text=line.decode().strip()
  if not text.startswith('data:'):continue
  value=text[5:].strip()
  if value=='[DONE]':done=True;break
  item=json.loads(value)
  for choice in item.get('choices',[]):
   content=choice.get('delta',{}).get('content')
   if content:
    if first is None:first=time.monotonic()-started
    chunks.append(content)
assert done and ''.join(chunks).strip()=='42',(done,chunks)
print(json.dumps({'passed':True,'identity':identity,'ordinary_answer':answer,'stream_answer':''.join(chunks).strip(),'stream_done':done,'ttft_seconds':first}))
"""
    reports = []
    for pod in pods:
        raw = kubectl(cluster, ["exec", "-n", pool["namespace"], pod["metadata"]["name"], "--", "python3", "-c", checker, npu["served_model"]]).stdout
        report = json.loads(raw)
        if report["identity"]["pod_uid"] != pod["metadata"]["uid"] or report["identity"]["node"] != pod["spec"]["nodeName"] or len(report["identity"]["physical_devices"]) != 1:
            raise RuntimeError("Actual device/Pod identity does not match Kubernetes")
        reports.append(report)
    result = {"passed": True, "pool": pool["pool"], "replicas": reports, "scope": "direct_worker_model_acceptance"}
    return result


def main():
    global POOL_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "apply", "status", "verify", "router", "prepare-router", "prepare-model"])
    parser.add_argument("--config", default=str(POOL_CONFIG), help="Reusable expert-pool configuration")
    args = parser.parse_args()
    POOL_CONFIG = Path(args.config)
    {"render": render, "apply": apply, "status": status, "verify": verify, "router": router, "prepare-router": prepare_router, "prepare-model": prepare_model}[args.action]()


if __name__ == "__main__":
    main()
