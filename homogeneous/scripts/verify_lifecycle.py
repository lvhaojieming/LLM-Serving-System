"""Exercise only owned expert deployments and record recovery, discovery and drain behavior."""
import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from manage_lab import ROOT, OWNER, kubectl, names, remote
import manage_pool
from manage_pool import configuration, artifact_path


def pods(cluster, pool, role="expert"):
    data = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l",
        "heteroserve.io/pool=" + pool["pool"] + ",app.kubernetes.io/name=" + role, "-o", "json"]).stdout)
    return [p for p in data["items"] if not p["metadata"].get("deletionTimestamp")]


def wait_ready(cluster, pool, count, role="expert", timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = pods(cluster, pool, role)
        if len(current) == count and all(any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", [])) for p in current):
            return current
        time.sleep(3)
    raise TimeoutError("Expected ready replicas did not recover")


def inside_router(cluster, pool, code, *args):
    ready = wait_ready(cluster, pool, pool["router_replicas"], "router")
    return kubectl(cluster, ["exec", "-n", pool["namespace"], ready[0]["metadata"]["name"], "--", "python3", "-c", code, *args]).stdout


def discovery(cluster, pool):
    code = "import json,urllib.request;print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8000/discovery',timeout=10))))"
    return json.loads(inside_router(cluster, pool, code))


def wait_discovered(cluster, pool, expected_uids, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = discovery(cluster, pool)
        if state["fresh"] and {e["uid"] for e in state["endpoints"]} == set(expected_uids):
            return state
        time.sleep(2)
    raise TimeoutError("Ready worker identities were not discovered")


def routing(cluster, pool, npu):
    ready = wait_ready(cluster, pool, pool["replicas"])
    view = wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in ready])
    expected = {p["metadata"]["uid"] for p in ready}
    code = """import json,sys,urllib.request
body={'model':sys.argv[1],'messages':[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
records=[];expected=set(json.loads(sys.argv[2]));seen=set()
for i in range(max(8,20*len(expected))):
 request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=120) as response:
  answer=json.load(response)['choices'][0]['message']['content'].strip();assert answer=='42'
  records.append({'uid':response.headers['x-heteroserve-pod-uid'],'node':response.headers['x-heteroserve-node'],'answer':answer})
  seen.add(records[-1]['uid'])
 if len(records)>=8 and seen==expected:break
print(json.dumps(records))
"""
    records = json.loads(inside_router(cluster, pool, code, pool["expert"], json.dumps(sorted(expected))))
    if {r["uid"] for r in records} != expected:
        raise RuntimeError("Routing acceptance requires real requests to every Ready worker and no unexpected identity")
    return {"passed": True, "discovered": view["endpoints"], "routed_requests": records}


def scale(cluster, pool, npu):
    initial = {p["metadata"]["uid"] for p in wait_ready(cluster, pool, pool["router_replicas"], "router")}
    target = pool["replicas"] + 1
    try:
        kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(target)])
        ready = wait_ready(cluster, pool, target)
        view = wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in ready])
        if {p["metadata"]["uid"] for p in pods(cluster, pool, "router")} != initial:
            raise RuntimeError("Router restarted during expert scale-up")
        result = {"passed": True, "scaled_replicas": target, "router_uids_unchanged": sorted(initial), "workers": view["endpoints"]}
    finally:
        kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(pool["replicas"])])
    final = wait_ready(cluster, pool, pool["replicas"])
    wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in final])
    return result


def delete(cluster, pool, npu):
    ready = wait_ready(cluster, pool, pool["replicas"])
    victim = ready[0]
    if victim["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
        raise RuntimeError("Cannot delete an unowned worker")
    started = time.monotonic()
    kubectl(cluster, ["delete", "pod", victim["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
    replacement = wait_ready(cluster, pool, pool["replicas"])
    current = {p["metadata"]["uid"] for p in replacement}
    if victim["metadata"]["uid"] in current:
        raise RuntimeError("Deleted identity remained a worker")
    view = wait_discovered(cluster, pool, current)
    return {"passed": True, "deleted_uid": victim["metadata"]["uid"], "replacement_uids": sorted(current),
            "recovery_seconds": time.monotonic() - started, "discovered": view["endpoints"]}


def cache(cluster, pool, npu):
    routers = wait_ready(cluster, pool, pool["router_replicas"], "router")
    identities = {p["metadata"]["uid"] for p in routers}
    role = json.loads(kubectl(cluster, ["get", "role", "heteroserve-discovery", "-n", pool["namespace"], "-o", "json"]).stdout)
    altered = [{**r, "verbs": [v for v in r["verbs"] if v != "list"]} for r in role["rules"]]
    started = time.monotonic()
    checker = """import json,urllib.request,urllib.error
view=json.load(urllib.request.urlopen('http://127.0.0.1:8000/discovery',timeout=10))
try:
 response=urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=10);status=response.status
except urllib.error.HTTPError as error:status=error.code
print(json.dumps({'view':view,'status':status}))
"""
    try:
        kubectl(cluster, ["patch", "role", "heteroserve-discovery", "-n", pool["namespace"], "--type=merge", "-p", json.dumps({"rules": altered})])
        time.sleep(pool["discovery_cache_seconds"] + 7)
        states = []
        for pod in routers:
            state = json.loads(kubectl(cluster, ["exec", "-n", pool["namespace"], pod["metadata"]["name"], "--", "python3", "-c", checker]).stdout)
            if state["view"]["fresh"] or state["status"] != 503:
                raise RuntimeError("API authorization failure did not expire routing discovery")
            states.append(state)
    finally:
        kubectl(cluster, ["patch", "role", "heteroserve-discovery", "-n", pool["namespace"], "--type=merge", "-p", json.dumps({"rules": role["rules"]})])
    restored = wait_ready(cluster, pool, pool["router_replicas"], "router", timeout=120)
    if {p["metadata"]["uid"] for p in restored} != identities:
        raise RuntimeError("Router restarted instead of recovering discovery")
    return {"passed": True, "fault": "temporary discovery list authorization denial", "observed_seconds": time.monotonic() - started,
            "expired_states": states, "router_uids_unchanged": sorted(identities)}


def shortage(cluster, pool, npu):
    name = "resource-shortage-check"
    obj = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": name, "namespace": pool["namespace"], "labels": {OWNER: cluster["name"], "app": name}},
           "spec": {"restartPolicy": "Never", "automountServiceAccountToken": False, "nodeSelector": {"heteroserve.io/npu-ready": "true"},
               "containers": [{"name": "check", "image": npu["model_image"], "imagePullPolicy": "Never", "command": ["sleep", "60"],
                   "resources": {"requests": {npu["resource"]: "9"}, "limits": {npu["resource"]: "9"}},
                   "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}}]}}
    kubectl(cluster, ["create", "-f", "-"], stdin=json.dumps(obj))
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            pod = json.loads(kubectl(cluster, ["get", "pod", name, "-n", pool["namespace"], "-o", "json"]).stdout)
            reasons = [c for c in pod["status"].get("conditions", []) if c["type"] == "PodScheduled" and c["status"] == "False"]
            if reasons:
                if pod["status"]["phase"] != "Pending" or pod["spec"].get("nodeName") or npu["resource"] not in reasons[0].get("message", ""):
                    raise RuntimeError("Expected clear device shortage without assigning a device")
                return {"passed": True, "scope": "synthetic scheduling probe, not an expert", "condition": reasons[0]}
            time.sleep(2)
        raise TimeoutError("Resource shortage was not reported")
    finally:
        kubectl(cluster, ["delete", "pod", name, "-n", pool["namespace"], "--wait=false"])


def drain(cluster, pool, npu):
    workers = wait_ready(cluster, pool, pool["replicas"])
    gateway = wait_ready(cluster, pool, pool["router_replicas"], "router")[0]
    checker = """import json,sys,urllib.request,time
body={'model':sys.argv[1],'messages':[{'role':'user','content':'List the numbers from 1 to 200, one per line. Do not summarize.'}],'max_tokens':1024,'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=180) as response:
 uid=response.headers['x-heteroserve-pod-uid'];started=False;done=False;content=[];begin=time.monotonic()
 for line in response:
  text=line.decode().strip()
  if not text.startswith('data:'):continue
  value=text[5:].strip()
  if value=='[DONE]':done=True;break
  chunk=json.loads(value)
  for choice in chunk.get('choices',[]):
   token=choice.get('delta',{}).get('content')
   if token:
    content.append(token)
    if not started:print(json.dumps({'phase':'stream_started','uid':uid}),flush=True);started=True
 assert done and content
 print(json.dumps({'phase':'stream_finished','uid':uid,'done':done,'output_characters':len(''.join(content)),'duration_seconds':time.monotonic()-begin}),flush=True)
"""
    server = next(n for n in cluster["nodes"] if n["role"] == "server")
    command = ["docker", "exec", names(cluster, server)[0], "kubectl", "--kubeconfig", "/etc/rancher/k3s/k3s.yaml",
        "exec", "-n", pool["namespace"], gateway["metadata"]["name"], "--", "python3", "-c", checker, npu["served_model"]]
    process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "root@" + server["host"], shlex.join(command)],
        stdout=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
    try:
        first = json.loads(process.stdout.readline())
        victim = next(p for p in workers if p["metadata"]["uid"] == first["uid"])
        if victim["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
            raise RuntimeError("Streaming worker ownership does not match")
        started = time.monotonic()
        operation = pool.get("verification_operation", "delete")
        if operation in {"scale", "update"}:
            kubectl(cluster, ["annotate", "pod", victim["metadata"]["name"], "-n", pool["namespace"], "controller.kubernetes.io/pod-deletion-cost=-1000", "--overwrite"])
        if operation == "scale":
            kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(pool["replicas"] - 1)])
        elif operation == "update":
            kubectl(cluster, ["rollout", "restart", "deployment/" + pool["pool"], "-n", pool["namespace"]])
        else:
            kubectl(cluster, ["delete", "pod", victim["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
        final = json.loads(process.stdout.readline())
        if process.wait(timeout=240) or not final["done"]:
            raise RuntimeError("SSE was truncated during graceful Pod termination")
        if operation == "update":
            kubectl(cluster, ["rollout", "status", "deployment/" + pool["pool"], "-n", pool["namespace"], "--timeout=1200s"], timeout=1260)
        final_count = pool["replicas"] - 1 if operation == "scale" else pool["replicas"]
        recovered = wait_ready(cluster, pool, final_count)
        if first["uid"] in {p["metadata"]["uid"] for p in recovered}:
            raise RuntimeError("Streaming worker was not actually retired by the requested operation")
        wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in recovered])
        return {"passed": True, "operation": operation, "retired_uid": first["uid"], "stream": final,
                "total_recovery_seconds": time.monotonic() - started, "replacement_uids": [p["metadata"]["uid"] for p in recovered]}
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def scale_drain(cluster, pool, npu):
    original = pool["replicas"]
    try:
        kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(original + 1)])
        changed = {**pool, "replicas": original + 1, "verification_operation": "scale"}
        ready = wait_ready(cluster, changed, original + 1)
        wait_discovered(cluster, changed, [p["metadata"]["uid"] for p in ready])
        return drain(cluster, changed, npu)
    finally:
        kubectl(cluster, ["scale", "deployment", pool["pool"], "-n", pool["namespace"], "--replicas=" + str(original)])
        wait_ready(cluster, pool, original)


def update_drain(cluster, pool, npu):
    return drain(cluster, {**pool, "verification_operation": "update"}, npu)


def rollback(cluster, pool, npu):
    wait_ready(cluster, pool, pool["replicas"])
    started = time.monotonic()
    try:
        missing = npu["model_image"].split("@")[0] + "@sha256:" + "0" * 64
        kubectl(cluster, ["set", "image", "deployment/" + pool["pool"], "expert=" + missing, "-n", pool["namespace"]])
        deadline = time.monotonic() + 90
        failure = None
        while time.monotonic() < deadline:
            for pod in pods(cluster, pool):
                for container in pod["status"].get("containerStatuses", []):
                    waiting = container.get("state", {}).get("waiting", {})
                    if waiting.get("reason") in {"ErrImageNeverPull", "ImagePullBackOff", "ErrImagePull"}:
                        failure = waiting
            if failure:
                break
            time.sleep(2)
        if not failure:
            raise RuntimeError("Deliberately unavailable image was not reported")
    finally:
        kubectl(cluster, ["rollout", "undo", "deployment/" + pool["pool"], "-n", pool["namespace"]])
        kubectl(cluster, ["rollout", "status", "deployment/" + pool["pool"], "-n", pool["namespace"], "--timeout=1200s"], timeout=1260)
        ready = wait_ready(cluster, pool, pool["replicas"])
        wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in ready])
    return {"passed": True, "detected_failure": failure, "recovered_seconds": time.monotonic() - started, "restored_image": npu["model_image"]}


def maintenance(cluster, pool, npu):
    node = "heteroserve-lab-209"
    started = time.monotonic()
    initial = wait_ready(cluster, pool, pool["replicas"])
    try:
        kubectl(cluster, ["cordon", node])
        data = json.loads(kubectl(cluster, ["get", "node", node, "-o", "json"]).stdout)
        if not data["spec"].get("unschedulable"):
            raise RuntimeError("Node did not become unschedulable")
        kubectl(cluster, ["drain", node, "--ignore-daemonsets", "--delete-emptydir-data", "--timeout=600s" ], timeout=660)
        ready = wait_ready(cluster, pool, pool["replicas"])
        if any(p["spec"]["nodeName"] == node for p in ready):
            raise RuntimeError("Drained node still owns an active expert")
        # One gateway is expected during voluntary maintenance because gateways are spread.
        gateways = pods(cluster, pool, "router")
        healthy = [p for p in gateways if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
        if not healthy:
            raise RuntimeError("Maintenance removed all routing capacity")
        return {"passed": True, "cordoned_node": node, "original_worker_uids": [p["metadata"]["uid"] for p in initial],
                "remaining_workers": [{"uid": p["metadata"]["uid"], "node": p["spec"]["nodeName"]} for p in ready],
                "remaining_ready_gateways": len(healthy), "duration_seconds": time.monotonic() - started}
    finally:
        kubectl(cluster, ["uncordon", node])
        wait_ready(cluster, pool, pool["router_replicas"], "router")
        current = wait_ready(cluster, pool, pool["replicas"])
        if len({p["spec"]["nodeName"] for p in current}) == 1:
            kubectl(cluster, ["delete", "pod", current[0]["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
            wait_ready(cluster, pool, pool["replicas"])


def engine(cluster, pool, npu):
    victim = wait_ready(cluster, pool, pool["replicas"])[0]
    uid = victim["metadata"]["uid"]
    initial_restarts = victim["status"]["containerStatuses"][0]["restartCount"]
    freeze = """import os,signal,json
from pathlib import Path
found=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().decode().split('\\0')
 except (OSError,UnicodeDecodeError):continue
 if 'vllm.entrypoints.openai.api_server' in args and '8001' in args:found.append(int(p.name))
assert len(found)==1,found
os.killpg(found[0],signal.SIGSTOP)
print(json.dumps({'suspended_owned_engine_pid':found[0]}))
"""
    started = time.monotonic()
    print(kubectl(cluster, ["exec", "-n", pool["namespace"], victim["metadata"]["name"], "--", "python3", "-c", freeze]).stdout, flush=True)
    unready_at = restarted_at = None
    deadline = started + 900
    while time.monotonic() < deadline:
        pod = json.loads(kubectl(cluster, ["get", "pod", victim["metadata"]["name"], "-n", pool["namespace"], "-o", "json"]).stdout)
        if pod["metadata"]["uid"] != uid:
            raise RuntimeError("Expected liveness container restart, not a replacement Pod")
        healthy = any(c["type"] == "Ready" and c["status"] == "True" for c in pod["status"].get("conditions", []))
        restarts = pod["status"].get("containerStatuses", [{}])[0].get("restartCount", initial_restarts)
        if not healthy and unready_at is None:
            unready_at = time.monotonic() - started
            print(json.dumps({"phase": "engine_removed_from_ready", "seconds": unready_at}), flush=True)
        if restarts > initial_restarts and restarted_at is None:
            restarted_at = time.monotonic() - started
            print(json.dumps({"phase": "container_restarted", "seconds": restarted_at}), flush=True)
        if restarted_at is not None and healthy:
            wait_discovered(cluster, pool, [p["metadata"]["uid"] for p in wait_ready(cluster, pool, pool["replicas"])])
            return {"passed": True, "fault": "SIGSTOP of owned model engine process group", "pod_uid_unchanged": uid,
                    "unready_seconds": unready_at, "restart_seconds": restarted_at, "recovered_seconds": time.monotonic() - started,
                    "restart_count_before": initial_restarts, "restart_count_after": restarts}
        time.sleep(5)
    # Recover the suspended process through normal Pod replacement if probe recovery failed.
    kubectl(cluster, ["delete", "pod", victim["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
    wait_ready(cluster, pool, pool["replicas"])
    raise TimeoutError("Liveness did not recover the suspended engine within its test budget")


def node_failure(cluster, pool, npu):
    node = next(n for n in cluster["nodes"] if n["host"] == "10.107.206.209")
    name = names(cluster, node)[0]
    identity = cluster["locked_container_ids"][node["host"]]
    actual = remote(node["host"], ["docker", "inspect", name, "--format", "{{.Id}}"]).stdout.strip()
    if actual != identity:
        raise RuntimeError("Cannot inject a fault into an unrecognized node container")
    before = wait_ready(cluster, pool, pool["replicas"])
    wait_ready(cluster, pool, pool["router_replicas"], "router")
    started = time.monotonic()
    unready = None
    minimum_capacity = pool["replicas"]
    try:
        remote(node["host"], ["docker", "stop", "--time", "15", identity])
        print(json.dumps({"phase": "owned_lab_node_stopped", "node": name, "container_id": identity}), flush=True)
        deadline = started + 900
        while time.monotonic() < deadline:
            data = json.loads(kubectl(cluster, ["get", "node", name, "-o", "json"]).stdout)
            node_ready = any(c["type"] == "Ready" and c["status"] == "True" for c in data["status"]["conditions"])
            if not node_ready and unready is None:
                unready = time.monotonic() - started
                print(json.dumps({"phase": "node_detected_unready", "seconds": unready}), flush=True)
            current = pods(cluster, pool)
            ready = [p for p in current if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
            minimum_capacity = min(minimum_capacity, len(ready))
            if unready is not None and len(ready) == pool["replicas"] and all(p["spec"]["nodeName"] != name for p in ready):
                healthy_gateways = [p for p in pods(cluster, pool, "router") if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]
                if not healthy_gateways:
                    raise RuntimeError("No gateway survived node loss")
                checker = """import urllib.request,json,sys
body={'model':sys.argv[1],'messages':[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],'max_tokens':32,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=120) as r:assert json.load(r)['choices'][0]['message']['content'].strip()=='42'
print('SURVIVING_GATEWAY_INFERENCE_PASSED')
"""
                kubectl(cluster, ["exec", "-n", pool["namespace"], healthy_gateways[0]["metadata"]["name"], "--", "python3", "-c", checker, npu["served_model"]])
                return {"passed": True, "fault": "stop one owned nested K3s node container; not physical host power loss", "container_id_preserved": identity,
                        "node_unready_seconds": unready, "replica_recovery_seconds": time.monotonic() - started,
                        "minimum_ready_experts": minimum_capacity, "surviving_ready_gateways": len(healthy_gateways),
                        "replacement_workers": [{"uid": p["metadata"]["uid"], "node": p["spec"]["nodeName"]} for p in ready],
                        "initial_worker_uids": [p["metadata"]["uid"] for p in before]}
            time.sleep(5)
        raise TimeoutError("Surviving node did not recover target expert replicas")
    finally:
        remote(node["host"], ["docker", "start", identity])
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            data = json.loads(kubectl(cluster, ["get", "node", name, "-o", "json"]).stdout)
            if any(c["type"] == "Ready" and c["status"] == "True" for c in data["status"]["conditions"]):
                break
            time.sleep(3)
        wait_ready(cluster, pool, pool["router_replicas"], "router")
        current = wait_ready(cluster, pool, pool["replicas"])
        if len({p["spec"]["nodeName"] for p in current}) == 1:
            kubectl(cluster, ["delete", "pod", current[0]["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
            wait_ready(cluster, pool, pool["replicas"])


def remove_rejoin(cluster, pool, npu):
    """Drain and unregister one owned agent, then rejoin using its existing container."""
    node = next(n for n in cluster["nodes"] if n["host"] == "10.107.206.210")
    name = names(cluster, node)[0]
    identity = cluster["locked_container_ids"][node["host"]]
    if remote(node["host"], ["docker", "inspect", name, "--format", "{{.Id}}"]).stdout.strip() != identity:
        raise RuntimeError("Cannot remove a changed node container")
    started = time.monotonic()
    previous = json.loads(kubectl(cluster, ["get", "node", name, "-o", "json"]).stdout)["metadata"]["uid"]
    removed = False
    try:
        kubectl(cluster, ["cordon", name])
        kubectl(cluster, ["drain", name, "--ignore-daemonsets", "--delete-emptydir-data", "--timeout=600s"], timeout=660)
        ready = wait_ready(cluster, pool, pool["replicas"])
        if any(p["spec"]["nodeName"] == name for p in ready):
            raise RuntimeError("Cannot unregister a node with active expert replicas")
        remote(node["host"], ["docker", "stop", "--time", "15", identity])
        kubectl(cluster, ["delete", "node", name])
        removed = not kubectl(cluster, ["get", "node", name, "--ignore-not-found=true"]).stdout.strip()
        if not removed:
            raise RuntimeError("Node remained registered")
    finally:
        remote(node["host"], ["docker", "start", identity])
        deadline = time.monotonic() + 300
        joined = None
        while time.monotonic() < deadline:
            response = kubectl(cluster, ["get", "node", name, "--ignore-not-found=true", "-o", "json"]).stdout
            if response.strip():
                joined = json.loads(response)
                if any(c["type"] == "Ready" and c["status"] == "True" for c in joined["status"]["conditions"]) and joined["status"].get("allocatable", {}).get(npu["resource"]) == "8":
                    break
            time.sleep(3)
        else:
            raise TimeoutError("Retained agent container failed to rejoin Ready with expected devices")
        kubectl(cluster, ["label", "node", name, "heteroserve.io/hardware=" + pool["hardware"], "heteroserve.io/backend=" + pool["backend"],
            "heteroserve.io/npu-ready=true", "heteroserve.io/purpose=inference", "--overwrite"])
        kubectl(cluster, ["uncordon", name])
        wait_ready(cluster, pool, pool["router_replicas"], "router")
        current = wait_ready(cluster, pool, pool["replicas"])
        if len({p["spec"]["nodeName"] for p in current}) == 1:
            kubectl(cluster, ["delete", "pod", current[0]["metadata"]["name"], "-n", pool["namespace"], "--wait=false"])
            wait_ready(cluster, pool, pool["replicas"])
    if joined["metadata"]["uid"] == previous:
        raise RuntimeError("Expected a new Kubernetes Node identity after unregister/rejoin")
    return {"passed": True, "node_removed": removed, "old_node_uid": previous, "new_node_uid": joined["metadata"]["uid"],
            "docker_id_preserved": identity, "total_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["routing", "scale", "delete", "cache", "shortage", "drain", "scale_drain", "update_drain", "rollback", "maintenance", "engine", "node_failure", "remove_rejoin"])
    parser.add_argument("--config", default=str(manage_pool.POOL_CONFIG))
    args = parser.parse_args()
    manage_pool.POOL_CONFIG = Path(args.config)
    cluster, pool, npu = configuration()
    result = {"action": args.action, **globals()[args.action](cluster, pool, npu)}
    path = artifact_path(pool, "lifecycle/" + args.action + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
