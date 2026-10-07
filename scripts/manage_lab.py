"""Manage only labelled lab containers; never touch the shared host Kubernetes."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OWNER = "io.heteroserve.lab"
PROBE_IMAGE = "busybox@sha256:d82c2ab94640ded77cf76514ce6a84870761105058a4a9e51b05a8a79be97a6c"


def resource_policy(config, value):
    """Remove experimental hard limits while retaining scheduling requests and NPU allocation."""
    if config.get("resource_limits_enabled", True):
        return value
    if isinstance(value, list):
        return [resource_policy(config, x) for x in value if not isinstance(x, dict) or x.get("kind") != "ResourceQuota"]
    if isinstance(value, dict):
        result = {k: resource_policy(config, v) for k, v in value.items()}
        if "resources" in result and "limits" in result["resources"]:
            limits = result["resources"]["limits"]
            result["resources"]["limits"] = {k: v for k, v in limits.items() if k.startswith("huawei.com/")}
            if not result["resources"]["limits"]:
                del result["resources"]["limits"]
        if "emptyDir" in result:
            result["emptyDir"].pop("sizeLimit", None)
        return result
    return value


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["name"] != "heteroserve-lab":
        raise ValueError("Unexpected lab identity")
    if not re.fullmatch(r"rancher/k3s@sha256:[0-9a-f]{64}", config["image"]):
        raise ValueError("Use a pinned official K3s image digest")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", config["image_config_id"]):
        raise ValueError("The platform image configuration digest must be pinned")
    if config["state_directory"] != "/root/zhangjinhao/LLM-Serving-System/state":
        raise ValueError("Unexpected lab data directory")
    if len([n for n in config["nodes"] if n["role"] == "server"]) != 1:
        raise ValueError("The pilot requires exactly one server")
    locked = set(config.get("locked_container_ids", {}))
    joining = set(config.get("new_node_hosts", []))
    if locked & joining or locked | joining != {n["host"] for n in config["nodes"]}:
        raise ValueError("Every node must have a fixed identity or explicit join authorization")
    if any(n["role"] != "agent" for n in config["nodes"] if n["host"] in joining):
        raise ValueError("Only additional agents may join the fixed control plane")
    if any(not re.fullmatch(r"[0-9a-f]{64}", ident) for ident in config["locked_container_ids"].values()):
        raise ValueError("Locked container identities must be 64 hexadecimal characters")
    if config["cpus_per_node"] != 4 or config["memory_per_node"] != "8g":
        raise ValueError("Original creation budget must remain recorded")
    if not isinstance(config.get("resource_limits_enabled", True), bool):
        raise ValueError("Resource limit policy must be boolean")
    if config.get("network_tuning") != {"flannel_tx_checksum_offload": False}:
        raise ValueError("This nested lab requires the verified Flannel checksum workaround")
    networks = [ipaddress.ip_network(config[k]) for k in ("pod_cidr", "service_cidr")]
    hosts = set()
    for node in config["nodes"]:
        ipaddress.ip_address(node["host"])
        if node["host"] not in config["available_hosts"] or node["host"] in hosts:
            raise ValueError("Nodes must be unique and user-authorized")
        hosts.add(node["host"])
        if node["role"] not in {"server", "agent"}:
            raise ValueError("Invalid node role")
        subnet = ipaddress.ip_network(node["subnet"])
        if ipaddress.ip_address(node["address"]) not in subnet:
            raise ValueError("Container IP is outside its network")
        networks.append(subnet)
    if any(a.overlaps(b) for i, a in enumerate(networks) for b in networks[i + 1:]):
        raise ValueError("Lab networks overlap")
    return config


def remote(host, argv, stdin=None, timeout=90, check=True):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                             "root@" + host, shlex.join(argv)], input=stdin,
                            stdin=subprocess.DEVNULL if stdin is None else None, text=True,
                            capture_output=True, timeout=timeout)
    if check and result.returncode:
        # Never print stdout from commands that may read a join token or kubeconfig.
        raise RuntimeError(f"Remote operation failed on {host}: {result.stderr[-1000:]}")
    return result


def names(config, node):
    prefix = config["name"] + "-" + node["host"].rsplit(".", 1)[1]
    return prefix, prefix + "-network", prefix + "-data"


def node_args(config, node):
    name, network, volume = names(config, node)
    identity_config = {k: v for k, v in config.items() if k not in {"image_config_id", "network_tuning", "locked_container_ids", "resource_limits_enabled", "fingerprint_node_hosts", "new_node_hosts"}}
    if config.get("fingerprint_node_hosts"):
        identity_config["nodes"] = [n for n in config["nodes"] if n["host"] in config["fingerprint_node_hosts"]]
    fingerprint = hashlib.sha256(json.dumps({"cluster": identity_config, "node": node}, sort_keys=True).encode()).hexdigest()
    argv = ["docker", "run", "-d", "--name", name, "--hostname", name,
            "--label", OWNER + "=" + config["name"], "--label", OWNER + ".config=" + fingerprint,
            "--privileged", "--restart", "unless-stopped", "--network", network, "--ip", node["address"],
            "-v", volume + ":/var/lib/rancher/k3s", "-p", node["host"] + ":8472:8472/udp"]
    if node["role"] == "server":
        argv += ["-p", f"{node['host']}:{config['api_port']}:6443"]
    else:
        argv += ["-v", config["state_directory"] + "/join-token:/run/secrets/join-token:ro"]
    if config.get("resource_limits_enabled", True):
        argv += ["--cpus", str(config["cpus_per_node"]), "--memory", config["memory_per_node"],
                 "--memory-swap", config["memory_per_node"], "--pids-limit", "16384"]
    argv += [config["image"], node["role"], "--node-name", name,
             "--node-ip", node["address"], "--node-external-ip", node["host"],
             "--kubelet-arg", "max-pods=32", "--kubelet-arg", "protect-kernel-defaults=true"]
    if node["role"] == "server":
        argv += ["--tls-san", node["host"], "--advertise-address", node["host"],
                 "--advertise-port", str(config["api_port"]), "--cluster-cidr", config["pod_cidr"],
                 "--service-cidr", config["service_cidr"], "--flannel-external-ip",
                 "--disable", "traefik", "--disable", "servicelb", "--disable", "metrics-server",
                 "--disable", "local-storage"]
    else:
        server = next(n for n in config["nodes"] if n["role"] == "server")
        argv += ["--server", f"https://{server['host']}:{config['api_port']}",
                 "--token-file", "/run/secrets/join-token"]
    return argv


def owned_resource(config, node, kind, name):
    result = remote(node["host"], ["docker", kind, "inspect", name, "--format", "{{json .Labels}}"], check=False)
    if result.returncode:
        return False
    if (json.loads(result.stdout or "{}") or {}).get(OWNER) != config["name"]:
        raise RuntimeError(f"Refusing to use an unowned {kind}: {name}")
    return True


def ensure_node(config, node):
    name, network, volume = names(config, node)
    locked_id = config.get("locked_container_ids", {}).get(node["host"])
    if locked_id:
        identity = remote(node["host"], ["docker", "inspect", name, "--format", "{{.Id}}"], check=False)
        if identity.returncode or identity.stdout.strip() != locked_id:
            raise RuntimeError("The user-fixed lab container is missing or changed; refusing to recreate it")
    existing = remote(node["host"], ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"], check=False)
    if not existing.returncode:
        labels = json.loads(existing.stdout or "{}") or {}
        expected = node_args(config, node)
        fingerprint = expected[expected.index(OWNER + "=" + config["name"]) + 2].split("=", 1)[1]
        if labels.get(OWNER) != config["name"] or labels.get(OWNER + ".config") != fingerprint:
            raise RuntimeError(f"Existing container ownership/config mismatch: {name}")
        status = remote(node["host"], ["docker", "inspect", name, "--format", "{{.State.Status}}"])
        if status.stdout.strip() != "running":
            raise RuntimeError(f"{name} is stopped; up will not undo an intentional pause")
        return
    if not owned_resource(config, node, "network", network):
        remote(node["host"], ["docker", "network", "create", "--subnet", node["subnet"],
                              "--label", OWNER + "=" + config["name"], network])
    if not owned_resource(config, node, "volume", volume):
        remote(node["host"], ["docker", "volume", "create", "--label", OWNER + "=" + config["name"], volume])
    # Reuse a verified Docker archive if registry downloading is slow or unavailable.
    image = remote(node["host"], ["docker", "image", "inspect", config["image_config_id"], "--format", "{{.Id}}"], check=False)
    if image.returncode:
        remote(node["host"], ["docker", "pull", config["image"]], timeout=300)
        image = remote(node["host"], ["docker", "image", "inspect", config["image"], "--format", "{{.Id}}"])
    if image.stdout.strip() != config["image_config_id"]:
        raise RuntimeError("Image config digest mismatch; refusing to launch")
    launch = node_args(config, node)
    launch[launch.index(config["image"])] = config["image_config_id"]
    remote(node["host"], launch)
    print(json.dumps({"event": "node_created", "host": node["host"], "container": name}), flush=True)


def kubectl(config, argv, stdin=None, timeout=90):
    server = next(n for n in config["nodes"] if n["role"] == "server")
    name, _, _ = names(config, server)
    labels = remote(server["host"], ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"])
    if (json.loads(labels.stdout) or {}).get(OWNER) != config["name"]:
        raise RuntimeError("Refusing to use an unowned control plane")
    return remote(server["host"], ["docker", "exec", "-i", name, "kubectl",
                                   "--kubeconfig", "/etc/rancher/k3s/k3s.yaml",
                                   "--request-timeout=10s", *argv], stdin=stdin, timeout=timeout)


def up(config, config_path=None):
    preflight(config)
    server = next(n for n in config["nodes"] if n["role"] == "server")
    ensure_node(config, server)
    name, _, _ = names(config, server)
    for _ in range(90):
        result = remote(server["host"], ["docker", "exec", name, "cat", "/var/lib/rancher/k3s/server/node-token"], check=False)
        if result.returncode == 0 and result.stdout.strip():
            token = result.stdout
            break
        time.sleep(2)
    else:
        raise TimeoutError("Lab server did not create its join token")
    for node in config["nodes"]:
        if node["role"] != "agent":
            continue
        # Token is carried on stdin, persisted privately, and is never logged.
        writer = "import os,sys; p=sys.argv[1]; os.makedirs(p,mode=0o700,exist_ok=True); os.chmod(p,0o700); f=os.open(p+'/join-token',os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); os.fchmod(f,0o600); os.write(f,sys.stdin.buffer.read()); os.close(f)"
        remote(node["host"], ["python3", "-c", writer, config["state_directory"]], stdin=token)
        ensure_node(config, node)
        if node["host"] in config.get("new_node_hosts", []):
            identity = remote(node["host"], ["docker", "inspect", names(config, node)[0], "--format", "{{.Id}}"]).stdout.strip()
            if not re.fullmatch(r"[0-9a-f]{64}", identity):
                raise RuntimeError("New node did not return a complete Docker identity")
            config["locked_container_ids"][node["host"]] = identity
            config["new_node_hosts"].remove(node["host"])
            Path(config_path or ROOT / "deploy/lab/cluster.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            print(json.dumps({"event": "new_node_identity_locked", "host": node["host"], "container_id": identity}), flush=True)
    apply_network_tuning(config)
    print(kubectl(config, ["get", "nodes", "-o", "wide"]).stdout, flush=True)


def preflight(config):
    """Inspect real host networks and available memory before any container creation."""
    reader = """import json,subprocess,socket,sys
ports=json.loads(sys.argv[1]); routes=json.loads(subprocess.check_output(['ip','-j','route','show','table','main']))
netids=subprocess.check_output(['docker','network','ls','-q'],text=True).split()
nets=json.loads(subprocess.check_output(['docker','network','inspect',*netids])) if netids else []
named=[n for n in nets if n['Name']==sys.argv[2]]
foreign=any((n.get('Labels') or {}).get('io.heteroserve.lab')!=sys.argv[3] for n in named)
owned=[] if foreign else named
owned_subnets=[p['Subnet'] for n in owned for p in n.get('IPAM',{}).get('Config',[]) if 'Subnet' in p]
owned_bridges=[n.get('Options',{}).get('com.docker.network.bridge.name','br-'+n['Id'][:12]) for n in owned]
memory={line.split(':')[0]:line.split(':')[1].strip() for line in open('/proc/meminfo')}
conflicts=[]
for kind,port in ports:
 s=socket.socket(socket.AF_INET,socket.SOCK_STREAM if kind=='tcp' else socket.SOCK_DGRAM)
 try:s.bind(('0.0.0.0',port))
 except OSError:conflicts.append([kind,port])
 finally:s.close()
prefixes=[r['dst'] for r in routes if r.get('dst') not in [None,'default'] and not (r['dst'] in owned_subnets and r.get('dev') in owned_bridges)]
prefixes += [p['Subnet'] for n in nets if n not in owned for p in n.get('IPAM',{}).get('Config',[]) if 'Subnet' in p]
print(json.dumps({'prefixes':prefixes,'port_conflicts':conflicts,'memory_available_kib':int(memory['MemAvailable'].split()[0]),'foreign_named_network':foreign,'owned_subnets':owned_subnets}))
"""
    planned = [ipaddress.ip_network(config[k]) for k in ("pod_cidr", "service_cidr")]
    for node in config["nodes"]:
        name, network, _ = names(config, node)
        existing = remote(node["host"], ["docker", "inspect", name, "--format", "{{json .Config.Labels}}"], check=False)
        if existing.returncode == 0:
            if (json.loads(existing.stdout) or {}).get(OWNER) != config["name"]:
                raise RuntimeError("Lab name collides with a foreign container")
            continue  # ensure_node will validate config and intentional pause state.
        ports = [["udp", 8472]] + ([["tcp", config["api_port"]]] if node["role"] == "server" else [])
        info = json.loads(remote(node["host"], ["python3", "-c", reader, json.dumps(ports), network, config["name"]]).stdout)
        if info["foreign_named_network"] or (info["owned_subnets"] and info["owned_subnets"] != [node["subnet"]]):
            raise RuntimeError("Network ownership or configured subnet mismatch")
        candidate = [*planned, ipaddress.ip_network(node["subnet"])]
        overlaps = [p for p in info["prefixes"] if any(ipaddress.ip_network(p, strict=False).overlaps(n) for n in candidate)]
        if overlaps or info["port_conflicts"] or info["memory_available_kib"] < 10 * 1024 * 1024:
            raise RuntimeError(f"Host preflight failed on {node['host']}: overlapping routes={overlaps}, busy ports={info['port_conflicts']}, memory_kib={info['memory_available_kib']}")
        print(json.dumps({"event": "preflight_passed", "host": node["host"],
                          "network_overlap": False, "port_conflicts": False,
                          "memory_available_gib": round(info["memory_available_kib"] / 1024 / 1024, 1)}), flush=True)


def network_tuning_object(config):
    """hostNetwork refers to the lab node's private network namespace, not the physical host."""
    labels = {OWNER: config["name"], "app": "network-tuning"}
    tune = "while true; do if ethtool -k flannel.1 2>/dev/null | grep -q 'tx-checksum-ip-generic: on'; then ethtool -K flannel.1 tx-checksum-ip-generic off; fi; sleep 5; done"
    return resource_policy(config, {"apiVersion": "apps/v1", "kind": "DaemonSet",
            "metadata": {"name": config["name"] + "-network", "namespace": "kube-system", "labels": labels},
            "spec": {"selector": {"matchLabels": labels}, "template": {"metadata": {"labels": labels}, "spec": {
                "automountServiceAccountToken": False, "hostNetwork": True, "dnsPolicy": "ClusterFirstWithHostNet",
                "securityContext": {"seccompProfile": {"type": "RuntimeDefault"}},
                "affinity": {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [{
                    "matchExpressions": [{"key": "kubernetes.io/hostname", "operator": "In",
                                          "values": [names(config, n)[0] for n in config["nodes"]]}]}]}}},
                "tolerations": [{"key": "node-role.kubernetes.io/control-plane", "operator": "Exists", "effect": "NoSchedule"}],
                "containers": [{"name": "network-tuning", "image": config["image"], "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/sh", "-ec"], "args": [tune],
                    "securityContext": {"runAsUser": 0, "allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                        "capabilities": {"drop": ["ALL"], "add": ["NET_ADMIN"]}},
                    "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"cpu": "100m", "memory": "64Mi"}},
                    "readinessProbe": {"exec": {"command": ["/bin/sh", "-ec", "ethtool -k flannel.1 | grep -q 'tx-checksum-ip-generic: off'"]},
                                       "periodSeconds": 5, "timeoutSeconds": 2}}]
            }}}})


def apply_network_tuning(config):
    # Only private bridge-backed lab node containers may host this network helper.
    for node in config["nodes"]:
        name, network, _ = names(config, node)
        state = remote(node["host"], ["docker", "inspect", name, "--format", '{"labels":{{json .Config.Labels}},"network":{{json .HostConfig.NetworkMode}}}'])
        info = json.loads(state.stdout)
        if (info["labels"] or {}).get(OWNER) != config["name"] or info["network"] != network:
            raise RuntimeError("Network tuning is restricted to owned private lab namespaces")
    payload = json.dumps(network_tuning_object(config))
    kubectl(config, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    kubectl(config, ["apply", "-f", "-"], stdin=payload)
    deadline = time.monotonic() + 240
    while True:
        daemon = json.loads(kubectl(config, ["get", "daemonset", config["name"] + "-network", "-n", "kube-system", "-o", "json"]).stdout)
        status = daemon.get("status", {})
        if status.get("numberReady") == len(config["nodes"]) and status.get("desiredNumberScheduled") == len(config["nodes"]):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("All node network helpers must be Ready before cross-node validation")
        time.sleep(3)
    print(json.dumps({"event": "network_tuning_applied", "scope": "private_lab_node_namespaces"}), flush=True)


def validation_objects(config):
    """One reusable probe definition, expanded per physical node by configuration."""
    namespace = config["name"] + "-validation"
    objects = [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": namespace,
                "labels": {OWNER: config["name"], "pod-security.kubernetes.io/enforce": "restricted"}}},
               {"apiVersion": "v1", "kind": "ResourceQuota", "metadata": {"name": "validation-budget", "namespace": namespace},
                "spec": {"hard": {"pods": "8", "requests.cpu": "2", "limits.cpu": "4", "requests.memory": "1Gi", "limits.memory": "2Gi"}}}]
    for node in config["nodes"]:
        node_name, _, _ = names(config, node)
        probe = "probe-" + node["host"].rsplit(".", 1)[1]
        labels = {OWNER: config["name"], "app": probe}
        pod = {"automountServiceAccountToken": False,
               "nodeSelector": {"kubernetes.io/hostname": node_name},
               "securityContext": {"runAsNonRoot": True, "runAsUser": 10001,
                                   "runAsGroup": 10001, "seccompProfile": {"type": "RuntimeDefault"}},
               "containers": [{"name": "probe", "image": PROBE_IMAGE, "imagePullPolicy": "IfNotPresent",
                  "command": ["/bin/sh", "-ec"], "args": ["mkdir -p /tmp/www; printf '{\"pod_uid\":\"%s\",\"node\":\"%s\",\"pod_ip\":\"%s\"}\\n' \"$POD_UID\" \"$NODE_NAME\" \"$POD_IP\" > /tmp/www/index.html; exec httpd -f -p 8000 -h /tmp/www"],
                  "env": [{"name": key, "valueFrom": {"fieldRef": {"fieldPath": path}}} for key, path in
                          [("POD_UID", "metadata.uid"), ("NODE_NAME", "spec.nodeName"), ("POD_IP", "status.podIP")]],
                  "ports": [{"name": "http", "containerPort": 8000}],
                  "readinessProbe": {"httpGet": {"path": "/", "port": "http"}, "periodSeconds": 2, "timeoutSeconds": 2},
                  "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}, "limits": {"cpu": "500m", "memory": "128Mi"}},
                  "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
                  "volumeMounts": [{"name": "tmp", "mountPath": "/tmp"}]}],
               "volumes": [{"name": "tmp", "emptyDir": {"sizeLimit": "16Mi"}}]}
        objects.extend([
            {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": probe, "namespace": namespace, "labels": labels},
             "spec": {"replicas": 1, "selector": {"matchLabels": labels}, "template": {"metadata": {"labels": labels}, "spec": pod}}},
            {"apiVersion": "v1", "kind": "Service", "metadata": {"name": probe, "namespace": namespace},
             "spec": {"selector": labels, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}}
        ])
    return resource_policy(config, {"apiVersion": "v1", "kind": "List", "items": objects})


def unlimit(config):
    if config.get("resource_limits_enabled", True):
        raise RuntimeError("Set resource_limits_enabled=false before removing limits")
    results = []
    for node in config["nodes"]:
        ensure_node(config, node)
        name, _, _ = names(config, node)
        identity = config["locked_container_ids"][node["host"]]
        # Docker 20.10 ignores zero updates. Use cgroup v1's unlimited memory
        # sentinel and CPUQuota=-1, which also overrides legacy NanoCPUs on restart.
        unlimited_memory = remote(node["host"], ["cat", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]).stdout.strip()
        if int(unlimited_memory) < 2 ** 60:
            raise RuntimeError("Host cgroup root does not expose an unlimited memory sentinel")
        remote(node["host"], ["docker", "update", "--cpu-quota=-1", "--memory=" + unlimited_memory,
                               "--memory-swap=" + unlimited_memory, "--pids-limit=-1", identity])
        data = json.loads(remote(node["host"], ["docker", "inspect", name]).stdout)[0]
        caps = data["HostConfig"]
        if data["Id"] != identity or caps["CpuQuota"] != -1 or str(caps["Memory"]) != unlimited_memory or caps["PidsLimit"] not in {-1, 0}:
            raise RuntimeError("Docker resource changes did not match the requested policy")
        results.append({"host": node["host"], "container_id": identity,
                        "cpu_quota": caps["CpuQuota"], "memory_unlimited_sentinel": caps["Memory"],
                        "memory_swap": caps["MemorySwap"], "pids": caps["PidsLimit"]})
    namespace = config["name"] + "-validation"
    ns = json.loads(kubectl(config, ["get", "namespace", namespace, "-o", "json"]).stdout)
    if ns["metadata"].get("labels", {}).get(OWNER) != config["name"]:
        raise RuntimeError("Refusing to alter an unowned validation namespace")
    kubectl(config, ["delete", "resourcequota", "validation-budget", "-n", namespace, "--ignore-not-found=true"])
    path = ROOT / "artifacts/kubernetes/resource-policy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"resource_limits_enabled": False, "containers": results}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


def wait_ready_probes(config, namespace, old_uid=None, timeout=240):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pods = json.loads(kubectl(config, ["get", "pods", "-n", namespace, "-l", OWNER + "=" + config["name"], "-o", "json"]).stdout)["items"]
        ready = [p for p in pods if not p["metadata"].get("deletionTimestamp") and
                 any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if len(ready) == len(config["nodes"]) and (not old_uid or all(p["metadata"]["uid"] != old_uid for p in ready)):
            return ready
        time.sleep(3)
    raise TimeoutError("Validation Pods were not Ready before the deadline; inspect events and image pulls")


def verify(config):
    namespace = config["name"] + "-validation"
    snapshot = json.loads(kubectl(config, ["get", "nodes", "-o", "json"]).stdout)["items"]
    expected = {names(config, n)[0] for n in config["nodes"]}
    ready = {n["metadata"]["name"] for n in snapshot if any(c["type"] == "Ready" and c["status"] == "True" for c in n["status"]["conditions"])}
    if not expected <= ready:
        raise RuntimeError("All configured physical nodes must be Ready before validation")
    apply_network_tuning(config)
    objects = validation_objects(config)
    # A dry-run Namespace is not persisted for the following namespaced objects.
    # Bootstrap the validated project Namespace before server-validating workloads.
    ns_payload = json.dumps(objects["items"][0])
    kubectl(config, ["apply", "--dry-run=server", "-f", "-"], stdin=ns_payload)
    kubectl(config, ["apply", "-f", "-"], stdin=ns_payload)
    payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": objects["items"][1:]})
    kubectl(config, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
    kubectl(config, ["apply", "-f", "-"], stdin=payload)
    pods = wait_ready_probes(config, namespace)
    calls = []
    for source in pods:
        for target in pods:
            if source["spec"]["nodeName"] == target["spec"]["nodeName"]:
                continue
            host_suffix = target["spec"]["nodeName"].rsplit("-", 1)[1]
            for mode, address in [("pod_ip", target["status"]["podIP"]),
                                  ("service_dns", f"probe-{host_suffix}.{namespace}.svc.cluster.local")]:
                response = json.loads(kubectl(config, ["exec", "-n", namespace, source["metadata"]["name"], "--",
                                                      "wget", "-T", "5", "-qO-", f"http://{address}:8000/"]).stdout)
                if (response["pod_uid"] != target["metadata"]["uid"] or
                    response["node"] != target["spec"]["nodeName"] or response["pod_ip"] != target["status"]["podIP"]):
                    raise RuntimeError("Cross-host response identity mismatch")
                calls.append({"source": source["spec"]["nodeName"], "target": response["node"], "mode": mode, "passed": True})
    victim = pods[0]
    old_uid = victim["metadata"]["uid"]
    kubectl(config, ["delete", "pod", "-n", namespace, victim["metadata"]["name"], "--wait=false"])
    replacements = wait_ready_probes(config, namespace, old_uid=old_uid)
    report = {"cluster": config["name"], "physical_nodes": sorted(expected), "api_dry_run": "passed",
              "cross_host_checks": calls, "deleted_pod_uid": old_uid,
              "replacement_pod_uids": [p["metadata"]["uid"] for p in replacements],
              "deployment_recovery": "passed", "npu_control": "not_tested"}
    output = ROOT / "artifacts/kubernetes/lab-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "preflight", "up", "status", "verify", "unlimit"])
    parser.add_argument("--config", default=str(ROOT / "deploy/lab/cluster.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "plan":
        print(json.dumps({"nodes": [{"host": n["host"], "command": node_args(config, n)} for n in config["nodes"]],
                          "npu_managed": False}, indent=2))
    elif args.action == "preflight":
        preflight(config)
    elif args.action == "up":
        up(config, args.config)
    elif args.action == "verify":
        verify(config)
    elif args.action == "unlimit":
        unlimit(config)
    else:
        print(kubectl(config, ["get", "nodes", "-o", "wide"]).stdout)


if __name__ == "__main__":
    main()
