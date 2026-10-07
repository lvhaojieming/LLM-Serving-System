"""Reuse the existing Ascend runtime and publish learned-router assets once."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from manage_lab import ROOT, OWNER, kubectl, names, remote
from manage_pool import configuration


def config():
    return json.loads((ROOT / "deploy/gateway.json").read_text())


def objects(cluster, pool, npu, cfg):
    name = "heteroserve-gateway"
    namespace = pool["namespace"]
    labels = {"app.kubernetes.io/part-of": "heteroserve", "app.kubernetes.io/name": "gateway", OWNER: cluster["name"]}
    files = [ROOT / "src/heteroserve" / p for p in ["__init__.py", "gateway.py", "routing/__init__.py", "routing/runtime.py", "routing/ascend.py", "routing/embedding_graph.py"]]
    data, items = {}, []
    for path in files:
        relative = path.relative_to(ROOT / "src").as_posix()
        key = relative.replace("/", "--")
        data[key] = path.read_text()
        items.append({"key": key, "path": relative})
    data["gateway.json"] = json.dumps(cfg, indent=2)
    items.append({"key": "gateway.json", "path": "gateway.json"})
    container = {"name": "gateway", "image": npu["model_image"], "imagePullPolicy": "Never",
        "command": ["bash", "-ec"], "args": ["source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 -m heteroserve.gateway --config /opt/runtime/gateway.json"],
        "env": [{"name": "PYTHONPATH", "value": "/opt/runtime:/workspace/zhangjinhao/moqe-runtime/python-deps"},
                {"name": "HOME", "value": "/scratch"}, {"name": "HF_HUB_OFFLINE", "value": "1"}, {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                {"name": "MOQE_ASCEND_INT4_ADAPTER", "value": "0"}, {"name": "TE_PARALLEL_COMPILER", "value": "1"}, {"name": "OMP_NUM_THREADS", "value": "4"}],
        "ports": [{"name": "http", "containerPort": 8000}],
        "resources": {"requests": {"cpu": "2", "memory": "4Gi", npu["resource"]: "1"}, "limits": {npu["resource"]: "1"}},
        "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True, "capabilities": {"drop": ["ALL"]}},
        "startupProbe": {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 5, "failureThreshold": 240},
        "readinessProbe": {"httpGet": {"path": "/ready", "port": "http"}, "periodSeconds": 3, "timeoutSeconds": 5},
        "livenessProbe": {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 10, "timeoutSeconds": 3, "failureThreshold": 6},
        "lifecycle": {"preStop": {"exec": {"command": ["python3", "-c", "import urllib.request;urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/drain',method='POST'),timeout=245).read()"]}}},
        "volumeMounts": [{"name": key, "mountPath": mount, "readOnly": True} for key, mount in [
            ("runtime", "/opt/runtime"), ("assets", "/assets/router"), ("driver", "/usr/local/Ascend/driver"), ("deps", "/workspace/zhangjinhao/moqe-runtime/python-deps")]] + [{"name": "scratch", "mountPath": "/scratch"}, {"name": "tmp", "mountPath": "/tmp"}]}
    volumes = [{"name": "runtime", "configMap": {"name": name, "items": items}},
        {"name": "scratch", "emptyDir": {"sizeLimit": "2Gi"}}, {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}]
    for key, path in [("assets", cfg["deployment"]["asset_path"]), ("driver", "/usr/local/Ascend/driver"), ("deps", "/var/lib/rancher/k3s/heteroserve-assets/adapter/python-deps")]:
        volumes.append({"name": key, "hostPath": {"path": path, "type": "Directory"}})
    return [{"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": name, "namespace": namespace, "labels": labels}, "data": data},
        {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "namespace": namespace, "labels": labels}, "spec": {
            "replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": labels}, "template": {"metadata": {"labels": labels,
                "annotations": {"heteroserve.io/runtime-sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()}}, "spec": {
                "automountServiceAccountToken": False, "terminationGracePeriodSeconds": 300,
                "nodeSelector": {"kubernetes.io/hostname": cfg["deployment"]["node"], "heteroserve.io/npu-ready": "true"},
                "containers": [container], "volumes": volumes}}}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": name, "namespace": namespace, "labels": labels},
            "spec": {"selector": labels, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}}]


def prepare(cluster, pool, cfg):
    deployment = cfg["deployment"]
    node = next(n for n in cluster["nodes"] if names(cluster, n)[0] == deployment["node"])
    current = remote(node["host"], ["docker", "inspect", names(cluster, node)[0], "--format", "{{.Id}}"] ).stdout.strip()
    if current != cluster["locked_container_ids"][node["host"]]:
        raise RuntimeError("Fixed experiment node container identity changed")
    sender_code = """import sys,json,tarfile
from pathlib import Path
c=json.loads(sys.argv[1]);r=Path(c['asset_source_root'])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|',dereference=True) as t:
 for source,destination in [(c['source_checkpoint'],'checkpoint_best.pt'),(c['source_embedding'],'embedding'),(c['source_tokenizer'],'tokenizer'),(c['source_architecture'],'architecture/src/moqe_router')]:
  t.add(r/source,arcname=destination,filter=lambda x:None if '__pycache__' in x.name or x.name.endswith('.pyc') else x)
"""
    receiver_code = """import sys,json,tarfile,hashlib,shutil
from pathlib import Path
volume,owner,expected=sys.argv[1:]
import subprocess
v=json.loads(subprocess.check_output(['docker','volume','inspect',volume]))[0]
if v.get('Labels',{}).get('io.heteroserve.lab')!=owner:raise RuntimeError('Unowned node volume')
root=Path(v['Mountpoint'])/'heteroserve-assets/router-model';root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for item in tar:
  rel=Path(item.name)
  if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Unsafe router asset path')
  path=root/rel
  if item.isdir():path.mkdir(parents=True,exist_ok=True);continue
  if not item.isfile():raise RuntimeError('Only regular router assets are allowed')
  path.parent.mkdir(parents=True,exist_ok=True)
  source=tar.extractfile(item)
  if path.exists():
   digest=hashlib.sha256()
   while True:
    chunk=source.read(1024*1024)
    if not chunk:break
    digest.update(chunk)
   existing=hashlib.sha256()
   with path.open('rb') as f:
    while True:
     chunk=f.read(1024*1024)
     if not chunk:break
     existing.update(chunk)
   if digest.digest()!=existing.digest():raise RuntimeError('Existing router asset differs: '+str(rel))
  else:
   with path.open('wb') as f:shutil.copyfileobj(source,f,1024*1024)
if hashlib.sha256((root/'checkpoint_best.pt').read_bytes()).hexdigest()!=expected:raise RuntimeError('Checkpoint hash mismatch')
manifest={}
for path in root.rglob('*'):
 if path.is_file() and path.name!='publication.json':
  h=hashlib.sha256()
  with path.open('rb') as f:
   while True:
    data=f.read(1024*1024)
    if not data:break
    h.update(data)
  manifest[str(path.relative_to(root))]=h.hexdigest()
(root/'publication.json').write_text(json.dumps(manifest,indent=2))
print('Router assets verified in existing node volume; files='+str(len(manifest)))
"""
    sender = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "root@" + deployment["asset_source_host"],
        shlex.join(["python3", "-c", sender_code, json.dumps(deployment)])], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "root@" + node["host"], shlex.join(["python3", "-c", receiver_code,
        names(cluster, node)[2], cluster["name"], cfg["router"]["checkpoint_sha256"]])], stdin=sender.stdout)
    sender.stdout.close()
    if sender.wait() or result.returncode:
        raise RuntimeError("Router asset publication failed")


def verify(cluster, pool):
    kubectl(cluster, ["rollout", "status", "deployment/heteroserve-gateway", "-n", pool["namespace"], "--timeout=120s"], timeout=150)
    data = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "app.kubernetes.io/name=gateway", "-o", "json"]).stdout)
    pod = next(p for p in data["items"] if not p["metadata"].get("deletionTimestamp"))
    code = """import json,urllib.request,urllib.error
base='http://127.0.0.1:8000'
ready=json.load(urllib.request.urlopen(base+'/ready',timeout=10))
records=[];successful=[]
prompts=['What is 17 + 25? Reply with the number only.','Hello!','What is the capital of France?','Translate hello into Chinese.','Write a Python function to add two integers.','What is 2 + 2?']
def send(prompt,stream,model='auto'):
 body={'model':model,'messages':[{'role':'user','content':prompt}],'max_tokens':32,'temperature':0,'stream':stream,'chat_template_kwargs':{'enable_thinking':False}}
 request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=180) as r:
   text=r.read().decode();expert=r.headers['x-moqe-expert']
   if stream:
    assert 'data: [DONE]' in text
    answer=''.join(json.loads(line[6:])['choices'][0]['delta'].get('content','') or '' for line in text.splitlines() if line.startswith('data: {') and json.loads(line[6:]).get('choices'))
   else:
    value=json.loads(text);assert value['model']==model;answer=value['choices'][0]['message']['content']
   if prompt==prompts[0]:assert answer.strip()=='42',answer
   return {'status':r.status,'expert':expert,'worker_uid':r.headers['x-heteroserve-pod-uid'],'router_ms':r.headers.get('x-moqe-router-ms'),'stream':stream,'answer':answer,'stream_done':('data: [DONE]' in text) if stream else None}
 except urllib.error.HTTPError as e:
  error=e.read().decode();assert e.code==503 and 'undeployed expert' in error,error
  return {'status':503,'error':error,'stream':stream}
for prompt in prompts:
 r=send(prompt,False);records.append(r)
 if r['status']==200:successful.append(prompt)
if successful:records.append(send(successful[0],True))
else:records.append(send(prompts[0],False,'awq'));records.append(send(prompts[0],True,'awq'))
metrics=urllib.request.urlopen(base+'/metrics',timeout=10).read().decode()
assert 'heteroserve_gateway_inflight 0' in metrics
print(json.dumps({'ready':ready,'requests':records,'auto_generation_verified':bool(successful),'final_metrics':metrics}))
"""
    result = json.loads(kubectl(cluster, ["exec", "-n", pool["namespace"], pod["metadata"]["name"], "--", "python3", "-c", code], timeout=600).stdout)
    logs = kubectl(cluster, ["logs", "-n", pool["namespace"], pod["metadata"]["name"], "--tail=300"]).stdout
    decisions = []
    for line in logs.splitlines():
        if '"event": "gateway_request_finished"' in line:
            item = json.loads(line[line.index("{"):])
            if item.get("decision"):
                decisions.append(item["decision"])
    if not decisions:
        raise RuntimeError("No real learned-router decisions were recorded")
    result.update(passed=True, pod_uid=pod["metadata"]["uid"], node=pod["spec"]["nodeName"], decisions=decisions,
                  source_commit=config()["migration_source"]["commit"], checkpoint_sha256=config()["router"]["checkpoint_sha256"],
                  scope="real checkpoint and AWQ chain; GPTQ is explicitly undeployed")
    devices = json.loads(kubectl(cluster, ["exec", "-n", pool["namespace"], pod["metadata"]["name"], "--", "python3", "-c",
        "import json;from pathlib import Path;print(json.dumps([p.name for p in Path('/dev').glob('davinci*') if p.name[7:].isdigit()]))"]).stdout)
    if len(devices) != 1:
        raise RuntimeError("Learned Router did not receive exactly one NPU")
    result["physical_devices"] = devices
    path = ROOT / "artifacts/kubernetes/learned-gateway.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "prepare", "apply", "status", "verify"])
    args = parser.parse_args()
    cluster, pool, npu = configuration()
    cfg = config()
    if args.action == "prepare":
        prepare(cluster, pool, cfg)
    elif args.action == "verify":
        verify(cluster, pool)
    elif args.action == "render":
        import yaml
        (ROOT / "deploy/k8s/gateway.yaml").write_text(yaml.safe_dump_all(objects(cluster, pool, npu, cfg), sort_keys=False), encoding="utf-8")
    elif args.action == "apply":
        payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": objects(cluster, pool, npu, cfg)})
        kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
        print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)
    else:
        print(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "app.kubernetes.io/name=gateway", "-o", "wide"]).stdout)


if __name__ == "__main__":
    main()
