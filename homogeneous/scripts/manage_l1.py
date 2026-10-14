"""Stable learned-router replicas; reuse Gateway image, publication and Pod template."""
import argparse
import copy
import json
from pathlib import Path
import re
import time

from manage_lab import ROOT, OWNER, load_config, names, active_nodes, kubectl, remote
from manage_pool import configuration
import manage_gateway as gateway
from manage_instances import matches, owned, ready

CONFIG = ROOT / "deploy/l1-router.json"
SERVICE = "heteroserve-l1-router"


def config(root=ROOT):
    return json.loads((root / "deploy/l1-router.json").read_text(encoding="utf-8"))


def validate(cfg, cluster, npu):
    eligible = {names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in npu["device_hosts"]}
    if cfg.get("namespace") != "heteroserve" or not isinstance(cfg.get("instances"), dict):
        raise ValueError("L1 requires the project namespace and stable-instance dictionary")
    for key in ("max_inflight", "startup_seconds", "drain_seconds", "termination_seconds"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError("Invalid L1 budget: " + key)
    if cfg["termination_seconds"] < cfg["drain_seconds"] + 10:
        raise ValueError("L1 termination must cover draining and server shutdown")
    for ident, spec in cfg["instances"].items():
        if not re.fullmatch(r"l1-[a-z0-9][a-z0-9-]{0,20}", ident) or set(spec) - {"node", "enabled", "max_inflight"}:
            raise ValueError("Invalid L1 instance: " + ident)
        if spec.get("node") not in eligible or type(spec.get("enabled", True)) is not bool or type(spec.get("max_inflight", cfg["max_inflight"])) is not int or spec.get("max_inflight", cfg["max_inflight"]) < 1:
            raise ValueError("L1 needs an eligible node, enabled flag and positive budget")


def instance_config(cfg, base, ident):
    result = copy.deepcopy(base)
    result.pop("router_service", None)
    spec = cfg["instances"][ident]
    result["deployment"]["node"] = spec["node"]
    result.update(instance_id=ident, max_inflight=spec.get("max_inflight", cfg["max_inflight"]),
                  drain_seconds=cfg["drain_seconds"], chat_template_sha256=base["router_service"]["chat_template_sha256"])
    return result


def objects(cluster, pool, npu, cfg, base, selected=None):
    validate(cfg, cluster, npu)
    selector = {"app.kubernetes.io/name": "l1-router", OWNER: cluster["name"], "heteroserve.io/pool": "l1-router"}
    result = [{"apiVersion": "v1", "kind": "Service", "metadata": {"name": SERVICE, "namespace": cfg["namespace"], "labels": selector},
               "spec": {"clusterIP": "None", "selector": selector, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}},
              {"apiVersion": "policy/v1", "kind": "PodDisruptionBudget", "metadata": {"name": SERVICE, "namespace": cfg["namespace"], "labels": selector},
               "spec": {"minAvailable": 1, "selector": {"matchLabels": selector}}}]
    for ident, spec in cfg["instances"].items():
        if selected and selected != ident:
            continue
        generated = gateway.objects(cluster, pool, npu, instance_config(cfg, base, ident), role="l1-router", name="heteroserve-" + ident)
        deployment = generated[1]
        deployment["spec"].update(replicas=int(spec.get("enabled", True)),
                                   strategy={"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0}})
        pod = deployment["spec"]["template"]["spec"]
        pod["terminationGracePeriodSeconds"] = cfg["termination_seconds"]
        pod["containers"][0]["startupProbe"]["failureThreshold"] = (cfg["startup_seconds"] + 4) // 5
        result.extend(generated)
    return result


def get_pods(cluster, cfg, ident=None):
    selector = "app.kubernetes.io/name=l1-router," + OWNER + "=" + cluster["name"]
    if ident:
        selector += ",heteroserve.io/instance=" + ident
    return json.loads(kubectl(cluster, ["get", "pods", "-n", cfg["namespace"], "-l", selector, "-o", "json"]).stdout)["items"]


def plan(cluster, pool, npu, cfg, base, selected=None):
    if selected and selected not in cfg["instances"]:
        raise ValueError("Unknown L1 instance")
    existing = json.loads(kubectl(cluster, ["get", "deployments", "-n", cfg["namespace"], "-l", "app.kubernetes.io/name=l1-router", "-o", "json"]).stdout)["items"]
    current = {v["metadata"]["name"]: v for v in existing}
    result = []
    for obj in objects(cluster, pool, npu, cfg, base, selected):
        if obj["kind"] != "Deployment":
            continue
        old = current.get(obj["metadata"]["name"])
        if old:
            owned(old, cluster)
        unchanged = old and matches(old["spec"]["template"], obj["spec"]["template"]) and old["spec"]["replicas"] == obj["spec"]["replicas"]
        ident = obj["metadata"]["labels"]["heteroserve.io/instance"]
        result.append({"instance": ident, "node": cfg["instances"][ident]["node"], "enabled": bool(obj["spec"]["replicas"]),
                       "action": "unchanged" if unchanged else "update" if old else "create"})
    if not selected:
        for obj in existing:
            ident = obj["metadata"].get("labels", {}).get("heteroserve.io/instance")
            if ident not in cfg["instances"]:
                owned(obj, cluster)
                result.append({"instance": ident, "node": None, "enabled": False, "action": "remove"})
    return result


def prepare(cluster, pool, cfg, base):
    for node_name in sorted({v["node"] for v in cfg["instances"].values() if v.get("enabled", True)}):
        candidate = copy.deepcopy(base)
        candidate["deployment"]["node"] = node_name
        node=next(n for n in cluster['nodes'] if names(cluster,n)[0]==node_name)
        code="""import json,sys,subprocess,hashlib
from pathlib import Path
volume,owner,checkpoint,template=sys.argv[1:]
info=json.loads(subprocess.check_output(['docker','volume','inspect',volume]))[0]
assert info.get('Labels',{}).get('io.heteroserve.lab')==owner
root=Path(info['Mountpoint'])/'heteroserve-assets/router-model';manifest=root/'publication.json'
valid=manifest.exists()
if valid:
 for name,digest in json.loads(manifest.read_text()).items():
  p=(root/name).resolve();assert root.resolve() in p.parents
  h=hashlib.sha256()
  if not p.exists():valid=False;break
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  if h.hexdigest()!=digest:valid=False;break
 if valid:
  valid=hashlib.sha256((root/'checkpoint_best.pt').read_bytes()).hexdigest()==checkpoint
  value=json.loads((root/'tokenizer/tokenizer_config.json').read_text())['chat_template']
  valid=valid and isinstance(value,str) and hashlib.sha256(value.encode()).hexdigest()==template
print(json.dumps(valid))
"""
        reused=json.loads(remote(node['host'],['python3','-c',code,names(cluster,node)[2],cluster['name'],base['router']['checkpoint_sha256'],base['router_service']['chat_template_sha256']]).stdout)
        if reused:
            print('Reused verified router assets on '+node_name);continue
        gateway.prepare(cluster, pool, candidate)


def verify(cluster, cfg, base, selected=None):
    specs = {k:v for k,v in cfg["instances"].items() if not selected or k == selected}
    if selected and not specs:raise ValueError("Unknown L1 instance")
    for ident in specs:
        kubectl(cluster,['rollout','status','deployment/heteroserve-'+ident,'-n',cfg['namespace'],'--timeout='+str(cfg['startup_seconds'])+'s'],timeout=cfg['startup_seconds']+60)
    deadline = time.monotonic() + cfg["startup_seconds"]
    while True:
        pods = get_pods(cluster, cfg, selected)
        if len(pods)==sum(int(v.get('enabled',True)) for v in specs.values()) and all(sum(ready(p) for p in pods if p["metadata"]["labels"]["heteroserve.io/instance"] == ident) == int(spec.get("enabled", True)) for ident,spec in specs.items()) and all(not p["metadata"].get("deletionTimestamp") for p in pods):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("L1 did not reach configured readiness")
        time.sleep(3)
    reports = []
    code = """import json,urllib.request
identity=json.load(urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=10))
body={'messages':[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],'max_new_tokens':32,'template_kwargs':{'enable_thinking':False}}
decision=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/route',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=60))
print(json.dumps({'identity':identity,'route':decision}))
"""
    for pod in pods:
        if not ready(pod):
            continue
        result = json.loads(kubectl(cluster, ["exec", "-n", cfg["namespace"], pod["metadata"]["name"], "--", "python3", "-c", code]).stdout)
        identity = result["identity"]
        if identity["pod_uid"] != pod["metadata"]["uid"] or identity["node"] != pod["spec"]["nodeName"] or identity["node"] != specs[identity['instance_id']]['node'] or len(identity["physical_devices"]) != 1 or identity["checkpoint_sha256"] != base["router"]["checkpoint_sha256"] or not all(identity[k].startswith("npu:") for k in ("head_device", "encoder_device")):
            raise RuntimeError("L1 actual device/identity/version verification failed")
        reports.append(result)
    report = {"passed": True, "replicas": reports, "selected": selected}
    path = ROOT / "artifacts/kubernetes/l1-router-verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def apply(cluster, pool, npu, cfg, base, selected=None):
    from manage_nodes import physical_free
    changes = plan(cluster, pool, npu, cfg, base, selected)
    nodes = {names(cluster,n)[0]:n for n in active_nodes(cluster)}
    required = {}
    for change in changes:
        if change["enabled"] and change["action"] in {"create", "update"}:
            required[change["node"]] = required.get(change["node"], 0) + 1
    observed = {v["metadata"]["name"]:v for v in json.loads(kubectl(cluster, ["get", "nodes", "-o", "json"]).stdout)["items"]}
    for node_name, count in required.items():
        node = observed[node_name]
        if node["spec"].get("unschedulable") or not any(c["type"] == "Ready" and c["status"] == "True" for c in node["status"].get("conditions", [])) or int(node["status"].get("allocatable", {}).get(npu["resource"], 0)) < count:
            raise RuntimeError("L1 node is not Ready/schedulable or lacks advertised NPU capacity: " + node_name)
        if len(physical_free(nodes[node_name]["host"])) < count:
            raise RuntimeError("No free NPU for L1 create/rolling update on " + node_name)
    existing=json.loads(kubectl(cluster,['get','configmaps,services,deployments,poddisruptionbudgets','-n',cfg['namespace'],'-o','json']).stdout)['items']
    named={(v['kind'],v['metadata']['name']):v for v in existing}
    for obj in objects(cluster, pool, npu, cfg, base, selected):
        old=named.get((obj['kind'],obj['metadata']['name']))
        if old:owned(old,cluster)
        payload = json.dumps(obj)
        kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
        kubectl(cluster, ["apply", "-f", "-"], stdin=payload)
    if not selected:
        for change in changes:
            if change["action"] == "remove":
                name = "heteroserve-" + change["instance"]
                for kind in ("deployment", "service", "configmap"):
                    data = json.loads(kubectl(cluster, ["get", kind, name, "-n", cfg["namespace"], "-o", "json"]).stdout)
                    owned(data, cluster)
                    kubectl(cluster, ["delete", kind, name, "-n", cfg["namespace"]])
    return verify(cluster, cfg, base, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "plan", "apply", "verify", "status", "render"])
    parser.add_argument("--instance")
    args = parser.parse_args()
    cluster, pool, npu = configuration()
    cfg, base = config(), gateway.config()
    validate(cfg, cluster, npu)
    if args.action == "prepare":
        prepare(cluster, pool, cfg, base)
    elif args.action == "apply":
        print(json.dumps(apply(cluster, pool, npu, cfg, base, args.instance), indent=2))
    elif args.action == "verify":
        print(json.dumps(verify(cluster, cfg, base, args.instance), indent=2))
    elif args.action == "plan":
        print(json.dumps(plan(cluster, pool, npu, cfg, base, args.instance), indent=2))
    elif args.action == "render":
        import yaml
        (ROOT / "deploy/k8s/l1-router.yaml").write_text(yaml.safe_dump_all(objects(cluster, pool, npu, cfg, base), sort_keys=False), encoding="utf-8")
    else:
        print(json.dumps(get_pods(cluster, cfg, args.instance), indent=2))


if __name__ == "__main__":
    main()
