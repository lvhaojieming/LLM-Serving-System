"""Ascend validation in the fixed lab containers; reuse host assets without changing their bytes."""
import argparse
import json
from pathlib import Path
import re
import time
import os

from manage_lab import ROOT, OWNER, kubectl, load_config, names, remote, resource_policy, active_nodes


def exclusive_reservations(table):
    """Decode the driver's exclusive namespace mappings, not npu-smi utilization."""
    reservations = []
    if not re.search(r"ns_id\s+\d+\s+identify\s+\S+\s+root_tgid\s+\d+\s+dev_num\s+\d+", table):
        raise RuntimeError("Unrecognized or empty driver namespace mapping")
    for block in re.split(r"(?=ns_id\s+\d+)", table):
        header = re.search(r"ns_id\s+(\d+)\s+identify\s+(\S+)\s+root_tgid\s+(\d+)\s+dev_num\s+(\d+)", block)
        if not header:
            continue
        namespace, identity, pid, count = header.groups()
        devices = [int(m.group(2)) for m in re.finditer(r"^\s*(\d+)\s+(\d+)\s*$", block, re.MULTILINE)]
        if len(devices) != int(count):
            raise RuntimeError("Driver namespace device count is inconsistent")
        if int(namespace) < 128 and devices:
            reservations.append({"namespace_id": int(namespace), "container_identity": identity,
                                 "root_pid": int(pid), "physical_npus": devices})
    return reservations


def preflight():
    cluster, npu, node, name, volume = configuration()
    table = remote(node["host"], ["cat", "/proc/uda/namespace_node"]).stdout
    reservations = exclusive_reservations(table)
    result = {"host": node["host"], "exclusive_reservations": reservations,
              "passed": not reservations,
              "note": "No automatic container stop or device reset; Kubernetes capacity does not imply free physical devices."}
    path = ROOT / "artifacts/kubernetes/npu-preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if reservations:
        raise RuntimeError("Exclusive driver reservations remain; confirm ownership before releasing them")


def configuration():
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    npu = json.loads((ROOT / "deploy/lab/npu.json").read_text(encoding="utf-8"))
    npu["host"] = os.environ.get("HETEROSERVE_NPU_HOST", npu["host"])
    node = next(n for n in cluster["nodes"] if n["host"] == npu["host"])
    name, _, volume = names(cluster, node)
    for fixed_node in active_nodes(cluster):
        identity = remote(fixed_node["host"], ["docker", "inspect", names(cluster, fixed_node)[0], "--format", "{{.Id}}"])
        if identity.stdout.strip() != cluster["locked_container_ids"][fixed_node["host"]]:
            raise RuntimeError("Fixed Docker container identity changed on " + fixed_node["host"])
    return cluster, npu, node, name, volume


def publish(driver_only=False):
    cluster, npu, node, name, volume = configuration()
    mapping = {"driver": npu["driver_source"], "model": npu["model_source"], "adapter": npu["adapter_source"]}
    pool_config = os.environ.get("HETEROSERVE_POOL_CONFIG")
    if pool_config and not driver_only:
        from manage_pool import load_pool, resolve_runtime
        pool = load_pool(Path(pool_config))
        runtime = resolve_runtime(pool)
        mapping.pop("model")
        mapping[pool.get("model_asset", "model")] = runtime["model_source"]
    if driver_only:
        mapping = {"driver": mapping["driver"]}
    publisher = """import json,os,sys,subprocess,shutil
from pathlib import Path
volume,owner,mapping=sys.argv[1],sys.argv[2],json.loads(sys.argv[3])
info=json.loads(subprocess.check_output(['docker','volume','inspect',volume]))[0]
if (info.get('Labels') or {}).get('io.heteroserve.lab')!=owner: raise RuntimeError('Unowned volume')
root=Path(info['Mountpoint'])/'heteroserve-assets'; root.mkdir(exist_ok=True)
results=[]
for asset,source in mapping.items():
 src=Path(source); dst=root/asset
 if not src.is_dir(): raise RuntimeError('Missing existing asset: '+source)
 if src.stat().st_dev!=root.stat().st_dev: raise RuntimeError('Hardlink reuse requires the same filesystem')
 dst.mkdir(parents=True,exist_ok=True)
 count=0; size=0; linked=0; copied=0
 for base,dirs,files in os.walk(src,followlinks=False):
  relative=Path(base).relative_to(src); out=dst/relative; out.mkdir(exist_ok=True)
  for d in list(dirs):
   s=Path(base)/d; t=out/d
   if s.is_symlink():
    if not t.exists() and not t.is_symlink(): t.symlink_to(os.readlink(s),target_is_directory=True)
    dirs.remove(d)
  for f in files:
   s=Path(base)/f; t=out/f
   if s.is_symlink() and asset=='driver':
    if not t.exists() and not t.is_symlink(): t.symlink_to(os.readlink(s))
    continue
   s=s.resolve()
   if t.exists():
    if not os.path.samefile(s,t) and (asset!='driver' or s.read_bytes()!=t.read_bytes()): raise RuntimeError('Existing published file differs: '+str(t))
    if os.path.samefile(s,t): linked+=1
    else: copied+=1
   else:
    try: os.link(s,t); linked+=1
    except PermissionError:
     if asset!='driver': raise
     shutil.copy2(s,t); copied+=1
   count+=1; size+=s.stat().st_size
 results.append({'asset':asset,'source':source,'files':count,'logical_bytes':size,'hardlinked_files':linked,'protected_metadata_copies':copied,'method':'hardlink with protected driver metadata copy' if copied else 'hardlink'})
(root/'publication.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
"""
    print(remote(node["host"], ["python3", "-c", publisher, volume, cluster["name"], json.dumps(mapping)]).stdout)
    link = "mkdir -p /usr/local/Ascend; if [ -e /usr/local/Ascend/driver ] || [ -L /usr/local/Ascend/driver ]; then test \"$(readlink /usr/local/Ascend/driver)\" = /var/lib/rancher/k3s/heteroserve-assets/driver; else ln -s /var/lib/rancher/k3s/heteroserve-assets/driver /usr/local/Ascend/driver; fi"
    remote(node["host"], ["docker", "exec", name, "sh", "-ec", link])
    print(json.dumps({"event": "assets_published", "container_id_preserved": True}), flush=True)


def plugin_objects(cluster, npu, node):
    service_account = "heteroserve-ascend-plugin"
    permissions = [
        {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "patch", "update"]},
        {"apiGroups": [""], "resources": ["nodes", "nodes/status", "nodes/proxy"], "verbs": ["get", "patch", "update"]},
        {"apiGroups": [""], "resources": ["configmaps"], "verbs": ["get", "create", "update", "list", "watch"]},
        {"apiGroups": [""], "resources": ["events"], "verbs": ["create"]},
    ]
    objects = [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": service_account, "namespace": "kube-system"}},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": {"name": service_account}, "rules": permissions},
        {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding", "metadata": {"name": service_account},
         "subjects": [{"kind": "ServiceAccount", "name": service_account, "namespace": "kube-system"}],
         "roleRef": {"kind": "ClusterRole", "name": service_account, "apiGroup": "rbac.authorization.k8s.io"}},
    ]
    volumes = [
        ("device-plugin", "/var/lib/kubelet/device-plugins", "/var/lib/kubelet/device-plugins", False),
        ("pod-resource", "/var/lib/kubelet/pod-resources", "/var/lib/kubelet/pod-resources", True),
        ("driver", "/usr/local/Ascend/driver", "/usr/local/Ascend/driver", True),
        ("logs", "/var/log/mindx-dl/devicePlugin", "/var/log/mindx-dl/devicePlugin", False),
        ("containerd", "/run/k3s/containerd", "/run/containerd", True),
    ]
    labels = {OWNER: cluster["name"], "app": "ascend-device-plugin"}
    target_nodes = [names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in npu.get("device_hosts", [npu["host"]])]
    pod = {"serviceAccountName": service_account,
           "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{
               "matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": target_nodes}]}]}}},
           "hostPID": True, "containers": [{"name": "device-plugin", "image": npu["plugin_image"], "imagePullPolicy": "Never",
               "command": ["/bin/bash", "-ec"],
               "args": ["mkdir -p /var/log/mindx-dl/devicePlugin; exec device-plugin -volcanoType=false -presetVirtualDevice=true -hotReset=-1 -logFile=/var/log/mindx-dl/devicePlugin/devicePlugin.log -logLevel=0"],
               "env": [{"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
                       {"name": "HOST_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}},
                       {"name": "LD_LIBRARY_PATH", "value": "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common"}],
               "securityContext": {"privileged": True, "readOnlyRootFilesystem": True},
               "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}, "limits": {"cpu": "500m", "memory": "512Mi"}},
               "volumeMounts": [{"name": a, "mountPath": c, "readOnly": d} for a, b, c, d in volumes]}],
           "volumes": [{"name": a, "hostPath": {"path": b, "type": "DirectoryOrCreate" if a == "logs" else "Directory"}} for a, b, c, d in volumes]}
    objects.append({"apiVersion": "apps/v1", "kind": "DaemonSet", "metadata": {"name": "ascend-device-plugin-daemonset", "namespace": "kube-system", "labels": labels},
                    "spec": {"selector": {"matchLabels": labels}, "template": {"metadata": {"labels": labels}, "spec": pod}}})
    return resource_policy(cluster, {"apiVersion": "v1", "kind": "List", "items": objects})


def plugin():
    cluster, npu, node, name, volume = configuration()
    payload = json.dumps(plugin_objects(cluster, npu, node))
    kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)


def status():
    cluster, npu, node, name, volume = configuration()
    data = json.loads(kubectl(cluster, ["get", "node", names(cluster, node)[0], "-o", "json"]).stdout)
    print(json.dumps({"container_id": cluster["locked_container_ids"][node["host"]],
                      "npu_capacity": data["status"]["capacity"].get(npu["resource"], "0"),
                      "npu_allocatable": data["status"]["allocatable"].get(npu["resource"], "0")}, indent=2))


def npu_volumes():
    return [{"name": "driver", "hostPath": {"path": "/usr/local/Ascend/driver", "type": "Directory"}},
            {"name": "checker", "configMap": {"name": "npu-checker"}},
            {"name": "tmp", "emptyDir": {"sizeLimit": "256Mi"}}]


def isolation_objects(cluster, npu, node):
    namespace = "heteroserve-npu-validation"
    objects = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace, "labels": {OWNER: cluster["name"]}}},
               {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "npu-checker", "namespace": namespace},
                "data": {"check_npu.py": (ROOT / "scripts/check_npu.py").read_text(encoding="utf-8")}}]
    for index in range(2):
        name = "npu-isolation-" + str(index)
        pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": namespace, "labels": {OWNER: cluster["name"], "app": "npu-isolation"}},
               "spec": {"restartPolicy": "Never", "automountServiceAccountToken": False,
                   "nodeSelector": {"kubernetes.io/hostname": names(cluster, node)[0]},
                   "containers": [{"name": "check", "image": npu["model_image"], "imagePullPolicy": "Never",
                       "command": ["/bin/bash", "-ec"],
                       "args": ["source /usr/local/Ascend/ascend-toolkit/set_env.sh; while [ ! -e /tmp/start-check ]; do sleep 1; done; python3 /validation/check_npu.py --assigned-device-files --runtime-device-map automatic; sleep infinity"],
                       "env": [{"name": "HOME", "value": "/tmp"},
                               {"name": "XDG_CACHE_HOME", "value": "/tmp/cache"},
                               {"name": "TE_PARALLEL_COMPILER", "value": str(npu["validation"]["compiler_processes"])},
                               {"name": "OMP_NUM_THREADS", "value": str(npu["validation"]["cpu_threads"])},
                               {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
                               {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
                       "resources": {"requests": {"cpu": "500m", "memory": "1Gi", npu["resource"]: "1"},
                                     "limits": {"cpu": "2", "memory": npu["validation"]["memory_limit"], npu["resource"]: "1"}},
                       "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                       "volumeMounts": [{"name": "driver", "mountPath": "/usr/local/Ascend/driver", "readOnly": True},
                                        {"name": "checker", "mountPath": "/validation", "readOnly": True},
                                        {"name": "tmp", "mountPath": "/tmp"}]}], "volumes": npu_volumes()}}
        objects.append(pod)
    return resource_policy(cluster, objects)


def isolation():
    preflight()
    cluster, npu, node, name, volume = configuration()
    objects = isolation_objects(cluster, npu, node)
    kubectl(cluster, ["apply", "-f", "-"], stdin=json.dumps(objects[0]))
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": objects[1:]})
    kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)


def isolation_results():
    cluster, npu, node, name, volume = configuration()
    reports = []
    if npu["validation"]["compute_execution"] != "sequential":
        raise RuntimeError("Validation is configured for sequential computation")
    namespace = "heteroserve-npu-validation"
    for index in range(2):
        kubectl(cluster, ["wait", "pod/npu-isolation-" + str(index), "-n", namespace,
                          "--for=condition=Ready", "--timeout=90s"])
    snapshot = json.loads(kubectl(cluster, ["get", "pods", "-n", namespace, "-l", "app=npu-isolation", "-o", "json"]).stdout)
    concurrent_allocation = len(snapshot["items"]) == 2 and all(p["status"]["phase"] == "Running" for p in snapshot["items"])
    if not concurrent_allocation:
        raise RuntimeError("Both validation Pods must hold their device allocations before computation")
    for index in range(2):
        pod_name = "npu-isolation-" + str(index)
        # Each checker exits before the next starts; both Pods retain their NPU requests.
        kubectl(cluster, ["exec", "-n", namespace, pod_name, "--", "touch", "/tmp/start-check"])
        deadline = time.monotonic() + npu["validation"]["result_timeout_seconds"]
        while True:
            pod = json.loads(kubectl(cluster, ["get", "pod", "-n", "heteroserve-npu-validation", pod_name, "-o", "json"]).stdout)
            spec = pod["spec"]
            workload = spec["containers"][0]
            if spec.get("hostPID") or spec.get("hostNetwork") or workload.get("securityContext", {}).get("privileged"):
                raise RuntimeError("Diagnostic or privileged Pod cannot pass normal isolation acceptance")
            if workload["resources"]["limits"].get(npu["resource"]) != "1":
                raise RuntimeError("Isolation acceptance requires exactly one assigned NPU")
            raw = kubectl(cluster, ["logs", "-n", "heteroserve-npu-validation", pod_name]).stdout
            start = raw.find('{\n  "passed"')
            if start >= 0:
                break
            if time.monotonic() >= deadline or pod["status"]["phase"] in {"Failed", "Succeeded"}:
                raise RuntimeError("NPU checker did not produce a result before completion or timeout")
            time.sleep(2)
        report, _ = json.JSONDecoder().raw_decode(raw[start:])
        if report["passed"] and report.get("pod_uid") != pod["metadata"]["uid"]:
            raise RuntimeError("Result does not match the current Pod identity")
        reports.append(report)
    assigned = [r.get("allocated_physical_npus") for r in reports]
    passed = all(r["passed"] and len(r.get("denied_unallocated_npus", [])) == 7 for r in reports) and assigned[0] != assigned[1]
    result = {"passed": passed, "pods": reports, "distinct_assignments": assigned[0] != assigned[1],
              "concurrent_allocation": concurrent_allocation, "compute_execution": "sequential",
              "fixed_containers_preserved": True}
    path = ROOT / "artifacts/kubernetes/npu" / npu["host"].rsplit(".", 1)[1] / "isolation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("Kubernetes NPU isolation checks failed")


def cleanup():
    """Remove only this lab's two validation Pods, retaining reusable infrastructure."""
    cluster, npu, node, name, volume = configuration()
    namespace = "heteroserve-npu-validation"
    data = json.loads(kubectl(cluster, ["get", "pods", "-n", namespace, "-o", "json"]).stdout)
    for pod in data["items"]:
        metadata = pod["metadata"]
        if metadata["name"] not in {"npu-isolation-0", "npu-isolation-1"}:
            continue
        if metadata.get("labels", {}).get(OWNER) != cluster["name"]:
            raise RuntimeError("Cannot delete a validation Pod without matching ownership")
        print(kubectl(cluster, ["delete", "pod", metadata["name"], "-n", namespace]).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "publish", "publish-driver", "plugin", "status", "isolation", "isolation-results", "cleanup"])
    parser.add_argument("--host", help="Authorized lab host to inspect or validate")
    parser.add_argument("--pool-config", help="Publish this pool's weights into its own asset directory")
    args = parser.parse_args()
    if args.host:
        os.environ["HETEROSERVE_NPU_HOST"] = args.host
    if args.pool_config:
        os.environ["HETEROSERVE_POOL_CONFIG"] = args.pool_config
    {"preflight": preflight, "publish": publish, "publish-driver": lambda: publish(driver_only=True), "plugin": plugin, "status": status, "isolation": isolation,
     "isolation-results": isolation_results, "cleanup": cleanup}[args.action]()


if __name__ == "__main__":
    main()
