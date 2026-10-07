"""Install one pinned monitoring stack; reuse it and keep credentials outside Git."""
import argparse
import json
from pathlib import Path

from manage_lab import ROOT, OWNER, kubectl, load_config, names, remote

NAMESPACE = "heteroserve-monitoring"


def objects(cluster):
    images = {x["name"]: x["image"] for x in json.loads((ROOT / "deploy/monitoring/images.json").read_text())}
    labels = {OWNER: cluster["name"], "app.kubernetes.io/part-of": "heteroserve-monitoring"}
    result = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": NAMESPACE, "labels": {**labels, "pod-security.kubernetes.io/enforce": "restricted"}}}]
    for name in ["prometheus", "kube-state-metrics"]:
        result.append({"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": name, "namespace": NAMESPACE}})
    for name, rules in [("prometheus", [{"apiGroups": [""], "resources": ["pods", "services", "endpoints"], "verbs": ["get", "list", "watch"]}]),
                        ("kube-state-metrics", [{"apiGroups": [""], "resources": ["nodes", "pods"], "verbs": ["get", "list", "watch"]},
                            {"apiGroups": ["apps"], "resources": ["deployments", "replicasets", "daemonsets"], "verbs": ["get", "list", "watch"]},
                            {"apiGroups": ["policy"], "resources": ["poddisruptionbudgets"], "verbs": ["get", "list", "watch"]}])]:
        role = "heteroserve-" + name
        result.extend([{"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRole", "metadata": {"name": role}, "rules": rules},
            {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "ClusterRoleBinding", "metadata": {"name": role},
             "roleRef": {"kind": "ClusterRole", "name": role, "apiGroup": "rbac.authorization.k8s.io"},
             "subjects": [{"kind": "ServiceAccount", "name": name, "namespace": NAMESPACE}]}])
    for name, size in [("prometheus", "8Gi"), ("grafana", "1Gi")]:
        result.extend([{"apiVersion": "v1", "kind": "PersistentVolume", "metadata": {"name": "heteroserve-" + name, "labels": labels},
            "spec": {"capacity": {"storage": size}, "accessModes": ["ReadWriteOnce"], "persistentVolumeReclaimPolicy": "Retain", "storageClassName": "",
                     "local": {"path": "/var/lib/rancher/k3s/heteroserve-monitoring/" + name},
                     "nodeAffinity": {"required": {"nodeSelectorTerms": [{"matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In", "values": ["heteroserve-lab-217"]}]}]}}}},
            {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": name, "namespace": NAMESPACE}, "spec": {
                "volumeName": "heteroserve-" + name, "storageClassName": "", "accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": size}}}}])
    prometheus = {"global": {"scrape_interval": "15s", "evaluation_interval": "15s"}, "rule_files": ["/etc/prometheus/alerts.yaml"], "scrape_configs": [
        {"job_name": "kube-state-metrics", "static_configs": [{"targets": ["kube-state-metrics." + NAMESPACE + ".svc:8080"]}]},
        {"job_name": "inference", "kubernetes_sd_configs": [{"role": "pod", "namespaces": {"names": ["heteroserve"]}}], "relabel_configs": [
            {"source_labels": ["__meta_kubernetes_pod_label_app_kubernetes_io_part_of"], "action": "keep", "regex": "heteroserve"},
            {"source_labels": ["__meta_kubernetes_pod_label_app_kubernetes_io_name"], "action": "keep", "regex": "expert|router"},
            {"source_labels": ["__meta_kubernetes_pod_ready"], "action": "keep", "regex": "true"},
            {"source_labels": ["__address__"], "target_label": "__address__", "regex": "([^:]+)(?::\\d+)?", "replacement": "$1:8000"},
            *[{"source_labels": [source], "target_label": target} for source, target in [
                ("__meta_kubernetes_pod_node_name", "node"), ("__meta_kubernetes_pod_uid", "pod_uid"),
                ("__meta_kubernetes_pod_label_heteroserve_io_pool", "pool"), ("__meta_kubernetes_pod_label_app_kubernetes_io_name", "role")]]]}]}
    alerts = {"groups": [{"name": "heteroserve", "rules": [
        {"alert": "NodeNotReady", "expr": 'kube_node_status_condition{condition="Ready",status="true"} == 0', "for": "1m", "labels": {"severity": "critical"}},
        {"alert": "ExpertCapacityUnavailable", "expr": 'kube_deployment_status_replicas_unavailable{namespace="heteroserve"} > 0', "for": "3m", "labels": {"severity": "warning"}},
        {"alert": "DiscoveryExpired", "expr": "heteroserve_discovery_fresh == 0", "for": "30s", "labels": {"severity": "critical"}},
        {"alert": "InferenceRestartLoop", "expr": 'increase(kube_pod_container_status_restarts_total{namespace="heteroserve"}[10m]) > 2', "for": "1m", "labels": {"severity": "warning"}}]}]}
    queries = [("Ready nodes", 'sum(kube_node_status_condition{condition="Ready",status="true"})'),
               ("NPU allocatable", 'sum by (node) (kube_node_status_allocatable{resource=~"huawei.*"})'),
               ("Ready experts", "sum(heteroserve_worker_ready)"), ("Inference inflight", "sum(heteroserve_worker_inflight)"),
               ("Discovery freshness", "heteroserve_discovery_fresh"),
               ("Pod restarts", 'sum by (pod) (kube_pod_container_status_restarts_total{namespace="heteroserve"})'),
               ("TTFT p95", 'histogram_quantile(0.95,sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))'),
               ("Pending Pods", 'sum(kube_pod_status_phase{namespace="heteroserve",phase="Pending"})')]
    dashboard = {"uid": "heteroserve", "title": "HeteroServe: nodes, devices and inference", "schemaVersion": 39, "version": 1,
        "refresh": "15s", "time": {"from": "now-1h", "to": "now"}, "panels": [
            {"id": i + 1, "type": "timeseries", "title": title, "datasource": {"type": "prometheus", "uid": "prometheus"},
             "gridPos": {"x": (i % 2) * 12, "y": (i // 2) * 8, "w": 12, "h": 8}, "targets": [{"refId": "A", "expr": expr}]} for i, (title, expr) in enumerate(queries)]}
    data = {"prometheus.json": json.dumps(prometheus), "alerts.yaml": json.dumps(alerts),
            "datasources.yaml": json.dumps({"apiVersion": 1, "datasources": [{"name": "Prometheus", "uid": "prometheus", "type": "prometheus", "url": "http://prometheus:9090", "access": "proxy", "isDefault": True}]}),
            "dashboards.yaml": json.dumps({"apiVersion": 1, "providers": [{"name": "heteroserve", "type": "file", "options": {"path": "/etc/grafana/dashboards"}}]}),
            "dashboard.json": json.dumps(dashboard)}
    result.append({"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": "monitoring-config", "namespace": NAMESPACE}, "data": data})
    for name, port, uid, args, mounts, volumes, env in [
        ("prometheus", 9090, 65534, ["--config.file=/etc/prometheus/prometheus.json", "--storage.tsdb.path=/prometheus", "--storage.tsdb.retention.time=7d", "--storage.tsdb.retention.size=5GB"],
         [{"name": "config", "mountPath": "/etc/prometheus", "readOnly": True}, {"name": "data", "mountPath": "/prometheus"}],
         [{"name": "data", "persistentVolumeClaim": {"claimName": "prometheus"}}], []),
        ("grafana", 3000, 472, [], [{"name": "data", "mountPath": "/var/lib/grafana"},
            {"name": "config", "mountPath": "/etc/grafana/provisioning/datasources/datasources.yaml", "subPath": "datasources.yaml", "readOnly": True},
            {"name": "config", "mountPath": "/etc/grafana/provisioning/dashboards/dashboards.yaml", "subPath": "dashboards.yaml", "readOnly": True},
            {"name": "config", "mountPath": "/etc/grafana/dashboards/dashboard.json", "subPath": "dashboard.json", "readOnly": True}],
         [{"name": "data", "persistentVolumeClaim": {"claimName": "grafana"}}],
         [{"name": "GF_SECURITY_ADMIN_PASSWORD", "valueFrom": {"secretKeyRef": {"name": "grafana-admin", "key": "password"}}},
          {"name": "GF_ANALYTICS_REPORTING_ENABLED", "value": "false"}, {"name": "GF_ANALYTICS_CHECK_FOR_UPDATES", "value": "false"}]),
        ("kube-state-metrics", 8080, 65532, ["--resources=nodes,pods,deployments,replicasets,daemonsets,poddisruptionbudgets", "--namespaces=heteroserve,kube-system,heteroserve-monitoring"], [], [], [])]:
        select = {"app": name, **labels}
        probe = "/api/health" if name == "grafana" else "/-/ready" if name == "prometheus" else "/healthz"
        pod = {"nodeSelector": {"kubernetes.io/hostname": "heteroserve-lab-217"}, "automountServiceAccountToken": name != "grafana",
            "securityContext": {"runAsNonRoot": True, "runAsUser": uid, "runAsGroup": uid, "fsGroup": uid, "seccompProfile": {"type": "RuntimeDefault"}},
            "containers": [{"name": name, "image": images[name], "imagePullPolicy": "IfNotPresent", "args": args, "env": env,
                "ports": [{"name": "http", "containerPort": port}], "resources": {"requests": {"cpu": "100m", "memory": "256Mi"}},
                "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                "startupProbe": {"httpGet": {"path": probe, "port": "http"}, "periodSeconds": 5, "failureThreshold": 60},
                "readinessProbe": {"httpGet": {"path": probe, "port": "http"}, "periodSeconds": 5},
                "volumeMounts": mounts}], "volumes": [{"name": "config", "configMap": {"name": "monitoring-config"}}, *volumes]}
        if name != "grafana":
            pod["serviceAccountName"] = name
        result.extend([{"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "namespace": NAMESPACE, "labels": labels},
                       "spec": {"replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": select}, "template": {"metadata": {"labels": select}, "spec": pod}}},
                      {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name, "namespace": NAMESPACE}, "spec": {"selector": select, "ports": [{"port": port, "targetPort": "http"}]}}])
    # Existing project policy permits only same-namespace traffic. Allow the monitor's scrape path explicitly.
    result.append({"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy", "metadata": {"name": "monitoring-scrape", "namespace": "heteroserve"},
        "spec": {"podSelector": {"matchLabels": {"app.kubernetes.io/part-of": "heteroserve"}}, "policyTypes": ["Ingress"],
                 "ingress": [{"from": [{"namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": NAMESPACE}}, "podSelector": {"matchLabels": {"app": "prometheus"}}}], "ports": [{"protocol": "TCP", "port": 8000}]}]}})
    return result


def apply():
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    server = next(n for n in cluster["nodes"] if n["role"] == "server")
    prepare = """import json,os,sys,subprocess,secrets
from pathlib import Path
volume,owner,state=sys.argv[1:]
info=json.loads(subprocess.check_output(['docker','volume','inspect',volume]))[0]
if (info.get('Labels') or {}).get('io.heteroserve.lab')!=owner:raise RuntimeError('Unowned monitor volume')
root=Path(info['Mountpoint'])/'heteroserve-monitoring';root.mkdir(exist_ok=True)
for name,uid in [('prometheus',65534),('grafana',472)]:
 p=root/name;p.mkdir(exist_ok=True);os.chown(p,uid,uid)
p=Path(state)/'monitoring';p.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(Path(state),0o700);os.chmod(p,0o700)
password=p/'grafana-admin-password'
if not password.exists():password.write_text(secrets.token_urlsafe(24))
os.chmod(password,0o600)
"""
    remote(server["host"], ["python3", "-c", prepare, names(cluster, server)[2], cluster["name"], cluster["state_directory"]])
    items = objects(cluster)
    kubectl(cluster, ["apply", "-f", "-"], stdin=json.dumps(items[0]))
    password = remote(server["host"], ["cat", cluster["state_directory"] + "/monitoring/grafana-admin-password"]).stdout.strip()
    secret = {"apiVersion": "v1", "kind": "Secret", "metadata": {"name": "grafana-admin", "namespace": NAMESPACE}, "type": "Opaque", "stringData": {"password": password}}
    kubectl(cluster, ["apply", "-f", "-"], stdin=json.dumps(secret))
    print(kubectl(cluster, ["apply", "-f", "-"], stdin=json.dumps({"apiVersion": "v1", "kind": "List", "items": items[1:]})).stdout)


def render():
    import yaml
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    (ROOT / "deploy/monitoring/stack.yaml").write_text(yaml.safe_dump_all(objects(cluster), sort_keys=False), encoding="utf-8")


def verify():
    import time
    import urllib.parse
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        pods = json.loads(kubectl(cluster, ["get", "pods", "-n", NAMESPACE, "-o", "json"]).stdout)["items"]
        ready = [p for p in pods if not p["metadata"].get("deletionTimestamp") and any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if len(ready) == 3:
            break
        time.sleep(3)
    else:
        raise TimeoutError("Monitoring components are not all Ready")
    prom = next(p["metadata"]["name"] for p in ready if p["metadata"]["labels"]["app"] == "prometheus")
    observations = {}
    for query in ['sum(kube_node_status_condition{condition="Ready",status="true"})', 'sum(kube_node_status_allocatable{resource=~"huawei.*"})',
                  'sum(up{job="inference"})', 'sum(heteroserve_worker_ready)', 'count(vllm:time_to_first_token_seconds_count)']:
        url = "http://127.0.0.1:9090/api/v1/query?" + urllib.parse.urlencode({"query": query})
        response = json.loads(kubectl(cluster, ["exec", "-n", NAMESPACE, prom, "--", "wget", "-qO-", url]).stdout)
        observations[query] = response["data"]["result"]
        if not observations[query]:
            raise RuntimeError("Expected monitoring series is absent: " + query)
    targets = json.loads(kubectl(cluster, ["exec", "-n", NAMESPACE, prom, "--", "wget", "-qO-", "http://127.0.0.1:9090/api/v1/targets"]).stdout)
    if any(t["health"] != "up" for t in targets["data"]["activeTargets"]):
        raise RuntimeError("A monitoring target is not being scraped successfully")
    server = next(n for n in cluster["nodes"] if n["role"] == "server")
    grafana = next(p for p in ready if p["metadata"]["labels"]["app"] == "grafana")
    pid = remote(server["host"], ["docker", "inspect", names(cluster, server)[0], "--format", "{{.State.Pid}}"]).stdout.strip()
    checker = """import json,sys,base64,urllib.request
from pathlib import Path
password=Path(sys.argv[1]).read_text().strip()
headers={'Authorization':'Basic '+base64.b64encode(('admin:'+password).encode()).decode()}
request=urllib.request.Request('http://'+sys.argv[2]+':3000/api/dashboards/uid/heteroserve',headers=headers)
data=json.load(urllib.request.urlopen(request,timeout=10))
assert data['dashboard']['uid']=='heteroserve' and data['meta']['provisioned']
print(json.dumps({'dashboard_uid':data['dashboard']['uid'],'provisioned':True,'panels':len(data['dashboard']['panels'])}))
"""
    dashboard = json.loads(remote(server["host"], ["nsenter", "-t", pid, "-n", "python3", "-c", checker,
        cluster["state_directory"] + "/monitoring/grafana-admin-password", grafana["status"]["podIP"]]).stdout)
    result = {"passed": True, "queries": observations, "scrape_targets": len(targets["data"]["activeTargets"]), "dashboard": dashboard}
    path = ROOT / "artifacts/kubernetes/monitoring-verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "apply", "verify"])
    args = parser.parse_args()
    {"render": render, "apply": apply, "verify": verify}[args.action]()


if __name__ == "__main__":
    main()
