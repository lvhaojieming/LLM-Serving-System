"""Manage the separate BF16 experiment using the existing private-cluster templates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import base64
import re
import random

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from manage_lab import OWNER, kubectl, load_config, names, resource_policy
from manage_pool import resolve_runtime, pool_objects, load_pool, router_objects, host_execution as shared_host_execution
from manage_nodes import physical_free
from manage_lab import operation_lock


def configuration(expert="bf16"):
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    if config.get("engine_backend", "vllm") != "vllm":
        raise ValueError("Only the configured vLLM backend is supported")
    overrides = config.pop("model_overrides", {})
    if (not isinstance(overrides, dict) or set(overrides) - {"bf16", "awq", "gptq"} or
            any(not isinstance(v, dict) or set(v) - {"nodes", "parallelism", "host_execution"} for v in overrides.values())):
        raise ValueError("model_overrides supports nodes, parallelism and host_execution per model")
    config.update(copy.deepcopy(overrides.get(expert, {})))
    if type(config.get("native_int4_fast_path", False)) is not bool:
        raise ValueError("native_int4_fast_path must be a boolean")
    if type(config.get("native_int4_script_mlp", False)) is not bool:
        raise ValueError("native_int4_script_mlp must be a boolean")
    if type(config.get("qwen3_fused_qknorm", False)) is not bool:
        raise ValueError("qwen3_fused_qknorm must be a boolean")
    if type(config.get("ascend_pp_cache_views", False)) is not bool:
        raise ValueError("ascend_pp_cache_views must be a boolean")
    if config.get("qknorm_mode", "native") not in {"native", "preserve_rounding"}:
        raise ValueError("Unknown Q/K normalization mode")
    checks = config.get("qknorm_verify_calls", 0)
    if type(checks) is not int or not 0 <= checks <= 8:
        raise ValueError("qknorm_verify_calls must be an integer from 0 to 8")
    if expert not in {"bf16", "awq", "gptq"}:
        raise ValueError("Unknown model profile")
    if expert != "bf16":
        existing = load_pool(ROOT / ("deploy/pools/ascend-" + expert + ".json"))
        quantized = resolve_runtime(existing)
        config.update(name=expert + "-distributed", pool=expert + "-distributed-ascend910b-vllm", expert=expert,
            model_asset=existing.get("model_asset", "model"), weights_manifest_file="conversion_manifest.json",
            weights_manifest_sha256=existing["weights_manifest_sha256"],
            asset_source_host=existing.get("asset_source_host", quantized["host"]),
            runtime={"model_source": quantized["model_source"], "served_model": quantized["served_model"],
                "model_parameters": {**config["runtime"]["model_parameters"],
                    "dtype": "float16", "quantization": "moqe_ascend_int4"}})
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    pool = {**json.loads((ROOT / "deploy/pools/defaults.json").read_text(encoding="utf-8")),
            **config, "replicas": 1, "model_nodes": config["nodes"]}
    tp, pp = config["parallelism"]["tp"], config["parallelism"]["pp"]
    if type(tp) is not int or tp not in {1, 2, 4, 8} or type(pp) is not int or pp != len(config["nodes"]) or pp < 1:
        raise ValueError("One PP stage per node; use supported per-node TP sizes")
    if len(set(config["nodes"])) != pp:
        raise ValueError("PP stages must occupy different physical nodes")
    available = {names(cluster, n)[0] for n in cluster["nodes"]}
    if not set(config["nodes"]) <= available:
        raise ValueError("All stages must use existing registered nodes")
    runtime = resolve_runtime(pool)
    runtime["host"] = pool.get("asset_source_host", runtime["host"])
    device_nodes = {names(cluster, node)[0] for node in cluster["nodes"] if node["host"] in runtime["device_hosts"]}
    if not set(config["nodes"]) <= device_nodes:
        raise ValueError("Choose existing NPU-enabled inference nodes, not just registered nodes")
    if expert == "bf16":
        if runtime["model_parameters"].get("quantization") is not None or runtime["model_parameters"].get("dtype") != "bfloat16":
            raise ValueError("The BF16 profile requires original weights")
        runtime["model_parameters"].pop("quantization", None)
    params = runtime["model_parameters"]
    if params["max_num_batched_tokens"] < params["max_model_len"]:
        raise ValueError("The V0 PP engine needs a batch-token budget covering max_model_len")
    profile = config["benchmark"]
    if profile["dataset"]["max_prompt_tokens"] + profile["output_tokens"] > params["max_model_len"]:
        raise ValueError("Real prompts plus output must fit the configured context")
    if any(type(n) is not int or n < 1 or n > pool["requests_per_worker"] for n in profile["concurrency"]):
        raise ValueError("Baseline benchmark concurrency must fit the admission budget")
    return cluster, pool, runtime


def host_execution(pool):
    parallel = pool["parallelism"]
    default = "mp" if parallel["tp"] * parallel["pp"] == 1 else "ray"
    return shared_host_execution(pool, default=default)


def result_directory(pool):
    base = ROOT / "artifacts/bf16_distributed"
    return base if pool["expert"] == "bf16" else base / pool["expert"]


def quantization_objects(cluster, pool, runtime, export_awq=False):
    """Use the baseline asset and runtime image, with one isolated NPU for calibration."""
    if pool["expert"] not in {"awq", "gptq"}:
        raise ValueError("Select AWQ or GPTQ for requantization")
    if export_awq and pool["expert"] != "awq":
        raise ValueError("Select AWQ for standard storage export")
    name = "qwen3-export-awq" if export_awq else "qwen3-requantize-" + pool["expert"]
    labels = {OWNER: cluster["name"], "app.kubernetes.io/part-of": "heteroserve",
              "heteroserve.io/task": "awq-export" if export_awq else "requantization", "heteroserve.io/algorithm": pool["expert"]}
    settings = json.loads((HERE / "config.json").read_text())
    source = settings["requantization"]["source"]
    if source != settings["runtime"]["model_source"].replace("/root/", "/workspace/"):
        raise ValueError("Requantization must use the baseline original checkpoint")
    if len(pool["nodes"]) != 1:
        raise ValueError("Select a single existing quantization node")
    data = {"quantize.py": (HERE / "quantize.py").read_text(), "config.json": json.dumps(settings)}
    metadata = {"name": name, "namespace": pool["namespace"], "labels": labels}
    cm = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": metadata, "data": data}
    container = {"name": "quantize", "image": runtime["model_image"], "imagePullPolicy": "Never", "workingDir": "/scratch",
        "command": ["/bin/bash", "-ec"], "args": [
            "source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
            "source /usr/local/Ascend/nnal/atb/set_env.sh; "
            "exec python3 /opt/heteroserve/quantize.py --config /opt/heteroserve/config.json --algorithm " + pool["expert"]],
        "env": [{"name": name, "value": value} for name, value in {
            "PYTHONPATH": "/quantization/dependencies", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "HOME": "/scratch", "XDG_CACHE_HOME": "/scratch/cache", "OMP_NUM_THREADS": "8",
            "TE_PARALLEL_COMPILER": "1", "MOQE_ASCEND_INT4_ADAPTER": "0"}.items()],
        "resources": {"requests": {"cpu": "4", "memory": "48Gi", runtime["resource"]: "1"},
                      "limits": {runtime["resource"]: "1"}},
        "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                            "capabilities": {"drop": ["ALL"]}},
        "volumeMounts": [{"name": "runtime", "mountPath": "/opt/heteroserve", "readOnly": True},
            {"name": "driver", "mountPath": "/usr/local/Ascend/driver", "readOnly": True},
            {"name": "weights", "mountPath": source, "readOnly": True},
            {"name": "quantization", "mountPath": "/quantization"},
            {"name": "scratch", "mountPath": "/scratch"}, {"name": "tmp", "mountPath": "/tmp"}]}
    if export_awq:
        container["args"][0] += " --export-awq"
        container["env"].append({"name": "TORCH_DEVICE_BACKEND_AUTOLOAD", "value": "0"})
        container["resources"] = {"requests": {"cpu": "4", "memory": "32Gi"}, "limits": {}}
    job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": metadata, "spec": {
        "backoffLimit": 0, "template": {"metadata": {"labels": labels}, "spec": {
            "restartPolicy": "Never", "automountServiceAccountToken": False,
            "nodeSelector": {"kubernetes.io/hostname": pool["nodes"][0]}, "containers": [container],
            "volumes": [{"name": "runtime", "configMap": {"name": name}},
                {"name": "driver", "hostPath": {"path": "/usr/local/Ascend/driver", "type": "Directory"}},
                {"name": "weights", "hostPath": {"path": "/var/lib/rancher/k3s/heteroserve-assets/models/bf16", "type": "Directory"}},
                {"name": "quantization", "hostPath": {"path": "/var/lib/rancher/k3s/heteroserve-assets/quantization", "type": "Directory"}},
                {"name": "scratch", "emptyDir": {}}, {"name": "tmp", "emptyDir": {}}]}}}}
    return resource_policy(cluster, {"apiVersion": "v1", "kind": "List", "items": [cm, job]})


def start_quantization(cluster, pool, runtime, export_awq=False):
    manifest = quantization_objects(cluster, pool, runtime, export_awq=export_awq)
    filename = "awq-export-job.json" if export_awq else "requantization-" + pool["expert"] + "-job.json"
    return start_task_job(cluster, pool, manifest, filename)


def dispatch_objects(cluster, pool, runtime, mode="dispatch"):
    """Reuse the calibration Job and adapter assets for a two-NPU dispatch probe."""
    if pool["expert"] != "awq" or pool["parallelism"] != {"tp": 2, "pp": 1}:
        raise ValueError("Dispatch comparison requires the existing AWQ TP2/PP1 configuration")
    if mode not in {"dispatch", "prefill"}:
        raise ValueError("Unknown dispatch benchmark mode")
    manifest = quantization_objects(cluster, pool, runtime)
    cm, job = manifest["items"]
    name = "qwen3-dispatch-benchmark"
    for obj in (cm, job):
        obj["metadata"]["name"] = name
        obj["metadata"]["labels"]["heteroserve.io/task"] = "dispatch-benchmark"
    job["spec"]["template"]["metadata"]["labels"]["heteroserve.io/task"] = "dispatch-benchmark"
    cm["data"] = {"dispatch_benchmark.py": (HERE / "dispatch_benchmark.py").read_text(),
                  "native_parallel.py": (ROOT / "src/heteroserve/native_parallel.py").read_text()}
    pod = job["spec"]["template"]["spec"]
    pod["volumes"] = [v for v in pod["volumes"] if v["name"] != "weights"]
    next(v for v in pod["volumes"] if v["name"] == "runtime")["configMap"]["name"] = name
    pod["volumes"].append({"name": "adapter", "hostPath": {
        "path": "/var/lib/rancher/k3s/heteroserve-assets/adapter", "type": "Directory"}})
    container = pod["containers"][0]
    container["volumeMounts"] = [v for v in container["volumeMounts"] if v["name"] != "weights"]
    container["volumeMounts"].append({"name": "adapter", "mountPath":
        runtime["adapter_source"].replace("/root/", "/workspace/"), "readOnly": True})
    _, deployment, _, _ = pool_objects(cluster, pool, runtime)
    env = deployment["spec"]["template"]["spec"]["containers"][0]["env"]
    container["env"] = [v for v in env if v["name"] not in {"POD_UID", "NODE_NAME"}]
    pythonpath = next(v for v in container["env"] if v["name"] == "PYTHONPATH")
    pythonpath["value"] = "/opt/heteroserve:" + pythonpath["value"]
    container["resources"] = {"requests": {runtime["resource"]: "2"}, "limits": {runtime["resource"]: "2"}}
    container["args"] = ["source /usr/local/Ascend/ascend-toolkit/set_env.sh; "
        "source /usr/local/Ascend/nnal/atb/set_env.sh; "
        "exec python3 -u /opt/heteroserve/dispatch_benchmark.py "
        "--output /quantization/" + ("dispatch-prefill.json" if mode == "prefill" else "dispatch-benchmark.json") +
        " --mode " + mode + " --seed " + str(pool["benchmark"]["seed"])]
    return resource_policy(cluster, manifest)


def start_task_job(cluster, pool, manifest, filename):
    """Apply one task-owned Job only once, preserving the exact launch manifest."""
    with operation_lock():
        for obj in manifest["items"]:
            present = kubectl(cluster, ["get", obj["kind"], obj["metadata"]["name"], "-n", pool["namespace"],
                "--ignore-not-found=true", "-o", "json"]).stdout
            if present.strip():
                raise RuntimeError("Quantization resource already exists; inspect it before rerunning")
        payload = json.dumps(manifest)
        kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
        result = kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout
    directory = ROOT / "artifacts/bf16_distributed/system/diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(manifest, indent=2))
    print(result)




def profiler_data(data, mode="timing"):
    """Use the same worker instrumentation for initial launch and live rollouts."""
    data = copy.deepcopy(data)
    data["profiling_worker.py"] = (HERE / "profiling_worker.py").read_text()
    data["profiling.json"] = json.dumps({"mode": mode})
    data["execution_worker.py"] = (ROOT / "src/heteroserve/execution_worker.py").read_text()
    data["ascend_execution.py"] = (ROOT / "src/heteroserve/ascend_execution.py").read_text()
    worker = json.loads(data["worker.json"])
    if "quantized_worker.py" in data:
        data["quantized_worker.py"] = data["quantized_worker.py"].replace(
            "from vllm_ascend.worker.worker import NPUWorker", "from profiling_worker import NPUWorker").replace(
            "from execution_worker import NPUWorker", "from profiling_worker import NPUWorker")
    elif "--worker-cls" in worker["engine_command"]:
        worker["engine_command"][worker["engine_command"].index("--worker-cls") + 1] = "profiling_worker.NPUWorker"
    else:
        worker["engine_command"].extend(["--worker-cls", "profiling_worker.NPUWorker"])
    data["worker.json"] = json.dumps(worker, indent=2)
    return data


def configure_profiler(cluster, pool, directory=None, mode="operators"):
    """Toggle the existing vLLM profiler on active experiment stages only."""
    from pathlib import PurePosixPath
    key = "VLLM_TORCH_PROFILER_DIR"
    if mode not in {"operators", "timing"}:
        raise ValueError("Unknown profiler mode")
    if directory and (not directory.startswith("/scratch/") or ".." in PurePosixPath(directory).parts):
        raise ValueError("Profiler output must be inside the experiment Pod's /scratch directory")
    state_path = result_directory(pool) / "profiling-state.json"
    with operation_lock():
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        stages = []
        originals = state.get("runtime_original", {}) if state and state.get("enabled") else {}
        for stage in range(pool["parallelism"]["pp"]):
            name = pool["name"] + "-stage-" + str(stage)
            deployment = json.loads(kubectl(cluster, ["get", "deployment", name,
                "-n", pool["namespace"], "-o", "json"]).stdout)
            if deployment["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Profiler cannot modify an unowned Deployment")
            if deployment["spec"].get("replicas", 1) != 1:
                raise RuntimeError("Profiler cannot start a paused Deployment")
            containers = deployment["spec"]["template"]["spec"]["containers"]
            if len(containers) != 1:
                raise RuntimeError("Expected the existing single model container")
            container = containers[0]
            existing = [v for v in container.get("env", []) if v["name"] == key]
            if existing and (not state or not state.get("enabled") or
                    existing != [{"name": key, "value": state["directory"]}]):
                raise RuntimeError("An existing profiler setting belongs to another operation")
            config_name = name + "-runtime"
            configmap = json.loads(kubectl(cluster, ["get", "configmap", config_name,
                "-n", pool["namespace"], "-o", "json"]).stdout)
            if configmap["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Profiler cannot modify an unowned ConfigMap")
            originals.setdefault(config_name, configmap["data"])
            stages.append({"name": name, "container": container["name"], "existing": existing})
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"enabled": bool(directory), "directory": directory, "mode": mode,
                 "namespace": pool["namespace"], "stages": stages,
                 "runtime_original": originals,
                 "note": "Collect traces before disabling: /scratch is Pod-local and rollout replaces Pods"}
        state_path.write_text(json.dumps(state, indent=2))
        for stage in stages:
            config_name = stage["name"] + "-runtime"
            data = copy.deepcopy(originals[config_name])
            if directory:
                data = profiler_data(data, mode)
            else:
                data["profiling_worker.py"] = None
                data["profiling.json"] = None
                for extra in ("execution_worker.py", "ascend_execution.py"):
                    if extra not in originals[config_name]: data[extra] = None
            kubectl(cluster, ["patch", "configmap", config_name, "-n", pool["namespace"],
                "--type=merge", "-p", json.dumps({"data": data})])
            change = {"name": key, "value": directory} if directory else {"name": key, "$patch": "delete"}
            signature = hashlib.sha256((data["profiling_worker.py"] + data["profiling.json"]).encode()).hexdigest() if directory else None
            patch = {"spec": {"template": {"metadata": {"annotations": {
                "heteroserve.io/profiler-sha256": signature}}, "spec": {"containers": [
                {"name": stage["container"], "env": [change]}]}}}}
            kubectl(cluster, ["patch", "deployment", stage["name"], "-n", pool["namespace"],
                "--type=strategic", "-p", json.dumps(patch)])
        return state


def system_pool(pool):
    result = copy.deepcopy(pool)
    result.update(name="heteroserve-gateway", expert="system")
    result["runtime"] = {"served_model": "auto"}
    result["expert_deployments"] = {}
    for expert in ("awq", "gptq"):
        _, expert_pool, runtime = configuration(expert)
        result["expert_deployments"][expert] = {"nodes": expert_pool["nodes"], "parallelism": expert_pool["parallelism"],
            "model_parameters": runtime["model_parameters"], "model_source": runtime["model_source"],
            "served_model": runtime["served_model"], "weights_manifest_sha256": expert_pool["weights_manifest_sha256"]}
    import manage_l1
    result["resource_budget"] = {"model_npus": sum(v["parallelism"]["tp"] * v["parallelism"]["pp"] for v in result["expert_deployments"].values()),
        "l1_npus": sum(v.get("enabled", True) for v in manage_l1.config()["instances"].values()),
        "baseline_bf16_model_npus": pool["parallelism"]["tp"] * pool["parallelism"]["pp"],
        "note": "Devices reserved for the system configuration; record actual expert selection separately. Idle retained BF16 Pods are additional allocated devices. Not an equal-device-count comparison."}
    result["benchmark"].update(entry="gateway", scope="Gateway -> learned probability L1 -> official pool vLLM Router -> AWQ/GPTQ at configured TP/PP; warm closed-loop ShareGPT; baseline BF16 used direct Service")
    return result


def prepare_chat_data(pool):
    """Recover the selected public first-turn messages without changing the token contract."""
    base = ROOT / "artifacts/bf16_distributed"
    selected = [json.loads(line) for line in (base / "requests.jsonl").read_text().splitlines()]
    dataset = pool["benchmark"]["dataset"]
    rows = json.loads((Path(dataset["directory"]) / dataset["filename"]).read_text())
    wanted = {row["id"] for row in selected}
    messages = {}
    for index, row in enumerate(rows):
        ident = str(row.get("id", index))
        conversation = row.get("conversations", [])
        if ident in wanted and ident not in messages and len(conversation) >= 2 and conversation[0].get("from") in {"human", "user"}:
            text = conversation[0].get("value")
            if isinstance(text, str) and text.strip():
                messages[ident] = [{"role": "user", "content": text}]
    if set(messages) != wanted:
        raise ValueError("Selected request IDs are missing from the pinned ShareGPT corpus")
    cluster, expert_pool, runtime = configuration("awq")
    pods = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l",
        "heteroserve.io/distributed-group=awq-distributed,heteroserve.io/stage=0", "-o", "json"]).stdout)["items"]
    candidates = [p for p in pods if not p["metadata"].get("deletionTimestamp") and p["status"]["phase"] == "Running"]
    if len(candidates) != 1:
        raise ValueError("Reuse one existing AWQ tokenizer environment for the input-contract check")
    from manage_lab import remote
    template = json.loads(remote("10.107.206.209", ["cat", "/root/zhangjinhao/model/Qwen3-14B/tokenizer_config.json"]).stdout)["chat_template"]
    code = """import json,sys
from transformers import AutoTokenizer
a=json.load(sys.stdin);t=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True)
for row in a['rows']:
 ids=t.apply_chat_template(row['messages'],chat_template=a['template'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if ids!=row['input_ids']:raise ValueError('Chat input differs from original token contract: '+row['id'])
print('CHAT_TOKEN_CONTRACT_VERIFIED',len(a['rows']))
"""
    import shlex
    shell = "source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 -c " + shlex.quote(code) + " " + shlex.quote(runtime["model_source"].replace("/root/", "/workspace/"))
    prepared = [{**row, "messages": messages[row["id"]]} for row in selected]
    print(kubectl(cluster, ["exec", "-i", "-n", pool["namespace"], candidates[0]["metadata"]["name"], "--", "bash", "-ec", shell],
        stdin=json.dumps({"rows": prepared, "template": template}), timeout=180).stdout)
    content = "".join(json.dumps(row) + "\n" for row in prepared)
    directory = base / "system"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "requests.jsonl"
    if target.exists() and target.read_text() != content:
        raise ValueError("Existing full-system request set differs")
    target.write_text(content)
    metadata = json.loads((base / "dataset.json").read_text())
    metadata.update(chat_requests_sha256=hashlib.sha256(content.encode()).hexdigest(),
        baseline_token_contract_sha256=hashlib.sha256((base / "requests.jsonl").read_bytes()).hexdigest())
    (directory / "dataset.json").write_text(json.dumps(metadata, indent=2))


def routing_objects(cluster):
    """Reuse the two existing experts and official Router; keep production expert replicas paused."""
    import manage_gateway
    import manage_l1
    manifest = []
    gateway_config = manage_gateway.config()
    for expert in ("awq", "gptq"):
        _, pool, runtime = configuration(expert)
        manifest.extend(objects(cluster, pool, runtime)["items"])
        original_service = next(obj for obj in manifest if obj["kind"] == "Service" and obj["metadata"]["name"] == pool["name"])
        alias = copy.deepcopy(original_service)
        alias["metadata"]["name"] = pool["pool"]
        manifest.append(alias)
        pool.update(router_replicas=1, router_nodes=["heteroserve-lab-209"])
        manifest.extend(router_objects(cluster, pool, runtime))
        gateway_config["pools"][expert].update(enabled=True,
            base_url="http://" + pool["pool"] + "-router.heteroserve.svc.cluster.local:8000/v1", model=expert)
    _, pool, runtime = configuration()
    manifest.extend(manage_l1.objects(cluster, pool, runtime, manage_l1.config(), manage_gateway.config()))
    manifest.extend(manage_gateway.objects(cluster, pool, runtime, gateway_config))
    for obj in manifest:
        obj["metadata"].setdefault("labels", {})[OWNER] = cluster["name"]
    return resource_policy(cluster, {"apiVersion": "v1", "kind": "List", "items": manifest}), gateway_config


def objects(cluster, pool, runtime):
    result = []
    tp, pp = pool["parallelism"]["tp"], pool["parallelism"]["pp"]
    executor, host_env = host_execution(pool)
    address = pool["name"] + "-ray." + pool["namespace"] + ".svc.cluster.local:6379"
    serving_selector = None
    for stage, node in enumerate(pool["nodes"]):
        local = {**pool, "parallelism": {"tp": tp, "pp": 1}, "model_nodes": [node],
                 "instance_id": pool["name"] + "-stage-" + str(stage)}
        cm, deployment, _, _ = pool_objects(cluster, local, runtime)
        name = pool["name"] + "-stage-" + str(stage)
        cm["metadata"]["name"] = name + "-runtime"
        deployment["metadata"]["name"] = name
        labels = {"heteroserve.io/distributed-group": pool["name"], "heteroserve.io/stage": str(stage)}
        for obj in (cm, deployment):
            obj["metadata"]["labels"].update(labels)
        pod = deployment["spec"]["template"]
        pod["metadata"]["labels"].update(labels)
        deployment["spec"]["selector"]["matchLabels"].update(labels)
        if stage != 0:
            for labelset in (cm["metadata"]["labels"], deployment["metadata"]["labels"],
                             deployment["spec"]["selector"]["matchLabels"], pod["metadata"]["labels"]):
                labelset["app.kubernetes.io/name"] = "expert-stage"
        pod["spec"]["nodeSelector"]["kubernetes.io/hostname"] = node
        pod["spec"].pop("affinity", None)
        pod["spec"].pop("topologySpreadConstraints", None)
        deployment["spec"]["strategy"] = {"type": "Recreate"}
        worker = json.loads(cm["data"]["worker.json"])
        worker.update(parallelism={"tp": tp, "pp": pp}, device_count=tp, distributed_executor_backend=executor,
            host_env=host_env,
            cluster={"stage": stage, "tp": tp, "pp": pp, "address": address}, cluster_startup_seconds=300)
        command = worker["engine_command"]
        if "--distributed-executor-backend" in command:
            command[command.index("--distributed-executor-backend") + 1] = executor
        else:
            command.extend(["--distributed-executor-backend", executor])
        command.extend(["--pipeline-parallel-size", str(pp)])
        if pool["expert"] == "bf16":
            worker["engine_env"].update(MOQE_ASCEND_INT4_ADAPTER="0", MOQE_ASCEND_INT4_PARALLEL="0")
            cm["data"] = {"worker.py": cm["data"]["worker.py"]}
        else:
            # Ray actors start before vLLM copies driver environment variables. Install the
            # existing adapter explicitly when each actor imports its accelerator worker.
            cm["data"]["quantized_worker.py"] = ("from adapter_patch import install\ninstall()\n"
                "from native_parallel import install\n" +
                ("install(fast_path=True)\n" if pool.get("native_int4_fast_path", False) else "install()\n") +
                "from vllm_ascend.worker.worker import NPUWorker\n")
            cm["data"].setdefault("native_parallel.py", (ROOT / "src/heteroserve/native_parallel.py").read_text())
            cm["data"].setdefault("sitecustomize.py", (ROOT / "src/heteroserve/engine_sitecustomize.py").read_text())
            command.extend(["--worker-cls", "quantized_worker.NPUWorker"])
        execution = {}
        if pool["expert"] != "bf16" and pool.get("native_int4_script_mlp", False):
            execution["native_int4_script_mlp"] = True
        if pool.get("ascend_pp_cache_views", False):
            execution["ascend_pp_cache_views"] = True
        if pool.get("qwen3_fused_qknorm", False):
            execution.update(qwen3_fused_qknorm=True, qknorm_mode=pool.get("qknorm_mode", "native"))
            if pool.get("qknorm_verify_calls", 0):
                execution["qknorm_verify_calls"] = pool["qknorm_verify_calls"]
        if execution:
            worker["execution_optimizations"] = execution
            cm["data"]["execution_worker.py"] = (ROOT / "src/heteroserve/execution_worker.py").read_text()
            cm["data"]["ascend_execution.py"] = (ROOT / "src/heteroserve/ascend_execution.py").read_text()
            if pool["expert"] == "bf16":
                command.extend(["--worker-cls", "execution_worker.NPUWorker"])
            else:
                cm["data"]["quantized_worker.py"] = cm["data"]["quantized_worker.py"].replace(
                    "from vllm_ascend.worker.worker import NPUWorker", "from execution_worker import NPUWorker")
        cm["data"].update({"runtime.py": (HERE / "runtime.py").read_text(encoding="utf-8"),
                           "worker.json": json.dumps(worker, indent=2)})
        pod["metadata"]["annotations"]["heteroserve.io/runtime-sha256"] = hashlib.sha256(json.dumps(cm["data"], sort_keys=True).encode()).hexdigest()
        next(v for v in pod["spec"]["volumes"] if v["name"] == "runtime")["configMap"]["name"] = name + "-runtime"
        container = pod["spec"]["containers"][0]
        container["env"] = [v for v in container["env"] if v["name"] not in host_env]
        container["env"].extend({"name": name, "value": value} for name, value in host_env.items())
        pod["spec"]["volumes"].append({"name": "ray-shm", "emptyDir": {"medium": "Memory", "sizeLimit": "512Mi"}})
        container["volumeMounts"].append({"name": "ray-shm", "mountPath": "/dev/shm"})
        container["args"] = ["mkdir -p /scratch/home /scratch/cache /scratch/vllm; source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 /opt/heteroserve/runtime.py --config /opt/heteroserve/worker.json"]
        container["env"] = [v for v in container["env"] if v["name"] != "VLLM_USE_V1"] + [
            {"name": "VLLM_USE_V1", "value": "0"}, {"name": "RAY_USAGE_STATS_ENABLED", "value": "0"},
            {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}}]
        # The child interpreter must find the configured worker module even when
        # single-card execution no longer triggers the pool's TP path setup.
        pythonpath = next(v for v in container["env"] if v["name"] == "PYTHONPATH")
        if "/opt/heteroserve" not in pythonpath["value"].split(":"):
            pythonpath["value"] = "/opt/heteroserve:" + pythonpath["value"]
        if stage != 0:
            container.pop("lifecycle", None)
        result += [cm, deployment]
        if stage == 0:
            serving_selector = copy.deepcopy(pod["metadata"]["labels"])
    labels = {OWNER: cluster["name"], "app.kubernetes.io/part-of": "heteroserve", "heteroserve.io/distributed-group": pool["name"]}
    result += [
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": pool["name"], "namespace": pool["namespace"], "labels": labels},
         "spec": {"selector": serving_selector, "ports": [{"name": "http", "port": 8000, "targetPort": "http"}]}},
        {"apiVersion": "v1", "kind": "Service", "metadata": {"name": pool["name"] + "-ray", "namespace": pool["namespace"], "labels": labels},
         "spec": {"selector": serving_selector, "publishNotReadyAddresses": True, "ports": [{"name": "gcs", "port": 6379, "targetPort": 6379}]}},
        {"apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy", "metadata": {"name": pool["name"], "namespace": pool["namespace"], "labels": labels},
         "spec": {"podSelector": {"matchLabels": {"heteroserve.io/distributed-group": pool["name"]}}, "policyTypes": ["Ingress", "Egress"],
             "ingress": [{"from": [{"podSelector": {"matchLabels": {"heteroserve.io/distributed-group": pool["name"]}}}]}],
             "egress": [{"to": [{"podSelector": {"matchLabels": {"heteroserve.io/distributed-group": pool["name"]}}}]}]}}
    ]
    if executor == "mp":
        result = [obj for obj in result if not (obj["kind"] == "Service" and obj["metadata"]["name"] == pool["name"] + "-ray")]
    return resource_policy(cluster, {"apiVersion": "v1", "kind": "List", "items": result})






def apply_experiment(cluster, pool, runtime, resume_deployment_uids=None):
    """Apply configured stages and remove only obsolete, active experiment stages."""
    manifest = objects(cluster, pool, runtime)
    namespace = pool["namespace"]
    resume_deployment_uids = resume_deployment_uids or {}
    with operation_lock():
        inventory = json.loads(kubectl(cluster, ["get", "deployments,configmaps,services,networkpolicies",
            "-n", namespace, "-o", "json"]).stdout)["items"]
        existing = {(obj["kind"], obj["metadata"]["name"]): obj for obj in inventory}
        desired = {(obj["kind"], obj["metadata"]["name"]) for obj in manifest["items"]}
        previous = []
        for key in desired:
            obj = existing.get(key)
            if obj is None:
                continue
            if obj["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Existing object is not owned by this project")
            if obj["kind"] == "Deployment" and obj["spec"].get("replicas", 1) == 0:
                if (not obj["metadata"].get("uid") or
                        resume_deployment_uids.get(obj["metadata"]["name"]) != obj["metadata"]["uid"]):
                    raise RuntimeError("Refusing to restart a paused experiment Deployment")
            previous.append(obj)
        obsolete = []
        for obj in inventory:
            key = obj["kind"], obj["metadata"]["name"]
            labels = obj["metadata"].get("labels", {})
            if key in desired or labels.get("heteroserve.io/distributed-group") != pool["name"]:
                continue
            stage = re.fullmatch(re.escape(pool["name"]) + r"-stage-(\d+)(-runtime)?", key[1])
            retired = (stage and int(stage[1]) >= pool["parallelism"]["pp"] and
                       ((key[0] == "Deployment" and not stage[2]) or (key[0] == "ConfigMap" and stage[2])))
            unused_ray = key == ("Service", pool["name"] + "-ray") and (
                host_execution(pool)[0] == "mp")
            if not (retired or unused_ray):
                continue
            if labels.get(OWNER) != cluster["name"]:
                raise RuntimeError("Obsolete stage is not owned by this project")
            # Keep paused stages and their configuration intact.
            if retired:
                deployment = existing.get(("Deployment", pool["name"] + "-stage-" + stage[1]))
                if deployment and deployment["spec"].get("replicas", 1) == 0:
                    if (not deployment["metadata"].get("uid") or
                            resume_deployment_uids.get(deployment["metadata"]["name"]) != deployment["metadata"]["uid"]):
                        continue
            obsolete.append(obj)
        payload = json.dumps(manifest)
        kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
        directory = result_directory(pool)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "deployment-transition.json").write_text(json.dumps({"previous": previous,
            "obsolete": obsolete, "desired": manifest}, indent=2))
        result = kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout
        for obj in sorted(obsolete, key=lambda x: x["kind"] != "Deployment"):
            kubectl(cluster, ["delete", obj["kind"], obj["metadata"]["name"], "-n", namespace, "--wait=false"])
        return {"applied": result, "removed": [obj["metadata"]["name"] for obj in obsolete]}


def prepare(cluster, pool, runtime):
    # Reuse checksums, SSH streaming and hardlink publication; leave images untouched.
    from manage_nodes import asset_manifest, stream
    from manage_lab import remote
    expected = asset_manifest(runtime["host"], runtime["model_source"])
    if expected.get(pool["weights_manifest_file"]) != pool["weights_manifest_sha256"]:
        raise ValueError("Source manifest does not match the pinned configuration")
    pinned = json.loads(remote(runtime["host"], ["cat", str(Path(runtime["model_source"]) / pool["weights_manifest_file"])]).stdout)
    recorded = pinned.get("files", {x["file"]: x["sha256"] for x in pinned.get("shards", [])})
    if not recorded or any(expected.get(name) != checksum for name, checksum in recorded.items()):
        raise ValueError("Source file differs from its recorded SHA-256")
    if pool["expert"] != "bf16":
        if pinned.get("source_quantization") != pool["expert"] or pinned.get("requantized") is not False:
            raise ValueError("Quantized profile must reuse the matching existing checkpoint")
    directory = result_directory(pool)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory / "pool.json"
    resolved.write_text(json.dumps(pool, indent=2))
    if pool["expert"] != "bf16":
        reference = json.loads((ROOT / "artifacts/bf16_distributed/dataset.json").read_text())
        base = json.loads(remote("10.107.206.209", ["cat", "/root/zhangjinhao/model/Qwen3-14B/weights_manifest.json"]).stdout)["files"]
        tokenizer = {name: expected.get(name) for name in ("tokenizer.json", "vocab.json", "merges.txt")}
        if any(not checksum or checksum != base.get(name) for name, checksum in tokenizer.items()):
            raise ValueError("Quantized vocabulary differs from the baseline input-token contract")
        requests = ROOT / "artifacts/bf16_distributed/requests.jsonl"
        if hashlib.sha256(requests.read_bytes()).hexdigest() != reference["requests_sha256"]:
            raise ValueError("Baseline request set changed")
        (directory / "dataset.json").write_text(json.dumps({**reference, "expert": pool["expert"],
            "requests_file": str(requests), "tokenizer_file_sha256": tokenizer,
            "template_policy": "Reuse the BF16 canonical template's exact input token IDs"}, indent=2))
    report = {}
    for node in cluster["nodes"]:
        if names(cluster, node)[0] not in pool["nodes"]:
            continue
        host = node["host"]
        if len(physical_free(host)) < pool["parallelism"]["tp"]:
            raise RuntimeError("Insufficient free physical devices on " + host)
        present = remote(host, ["test", "-e", runtime["model_source"]], check=False)
        if present.returncode == 1:
            remote(host, ["mkdir", "-p", str(Path(runtime["model_source"]).parent)])
            stream(runtime["host"], ["tar", "-C", str(Path(runtime["model_source"]).parent), "-cf", "-", Path(runtime["model_source"]).name],
                   host, ["tar", "--keep-old-files", "-C", str(Path(runtime["model_source"]).parent), "-xf", "-"])
        elif present.returncode != 0:
            raise RuntimeError("Target model directory could not be inspected")
        if asset_manifest(host, runtime["model_source"]) != expected:
            raise RuntimeError("Target assets differ; refusing overwrite")
        subprocess.run([sys.executable, str(ROOT / "scripts/manage_npu.py"), "publish",
            "--host", host, "--pool-config", str(resolved)], check=True, timeout=600)
        report[host] = {"reused": present.returncode == 0, "sha256": expected}
    path = directory / "assets.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def client_pod(cluster, pool):
    pods = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "app.kubernetes.io/name=gateway", "-o", "json"]).stdout)["items"]
    candidates = [p for p in pods if p["status"]["phase"] == "Running" and p["metadata"]["labels"].get(OWNER) == cluster["name"]]
    if len(candidates) != 1:
        raise RuntimeError("Reuse exactly one running project Gateway as the CPU benchmark client")
    return candidates[0]


def prepare_data(cluster, pool, runtime):
    """Reuse the live head's tokenizer on CPU; keep the public corpus outside the source tree."""
    if pool["expert"] != "bf16":
        print((result_directory(pool) / "dataset.json").read_text())
        return
    profile = pool["benchmark"]
    dataset = profile["dataset"]
    source = Path(dataset["directory"]) / dataset["filename"]
    from download_model import digest, fetch
    source.parent.mkdir(parents=True, exist_ok=True)
    url = "https://hf-mirror.com/datasets/" + dataset["repository"] + "/resolve/" + dataset["revision"] + "/" + dataset["filename"]
    fetch({"path": source.name, "size": dataset["bytes"], "sha256": dataset["sha256"], "url": url}, source.parent.resolve())
    rows = json.loads(source.read_text(encoding="utf-8"))
    candidates = []
    seen = set()
    for index, row in enumerate(rows):
        conversation = row.get("conversations", [])
        if len(conversation) < 2 or conversation[0].get("from") not in {"human", "user"}:
            continue
        text = conversation[0].get("value")
        ident = str(row.get("id", index))
        if not isinstance(text, str) or not text.strip() or ident in seen:
            continue
        seen.add(ident)
        candidates.append({"id": ident, "text": text})
    random.Random(profile["seed"]).shuffle(candidates)
    batch = candidates[:max(2000, profile["requests_per_case"] * 10)]
    pods = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l",
        "heteroserve.io/distributed-group=" + pool["name"] + ",heteroserve.io/stage=0", "-o", "json"]).stdout)["items"]
    active = [p for p in pods if not p["metadata"].get("deletionTimestamp")]
    if len(active) != 1:
        raise RuntimeError("Reuse the one serving head's existing tokenizer environment")
    code = """import json,sys
from transformers import AutoTokenizer
model,settings=sys.argv[1],json.loads(sys.argv[2]);tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
rows=[];seen=set();filtered=0
for item in json.loads(sys.stdin.read()):
 raw=tokenizer.encode(item['text'],add_special_tokens=False)
 if len(raw)<settings['dataset']['min_prompt_tokens']:filtered+=1;continue
 ids=tokenizer.apply_chat_template([{'role':'user','content':item['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
 if len(ids)>settings['dataset']['max_prompt_tokens'] or len(ids)+settings['output_tokens']>settings['dataset']['max_total_tokens']:filtered+=1;continue
 key=tuple(ids)
 if key in seen:filtered+=1;continue
 seen.add(key);rows.append({'id':item['id'],'input_ids':ids})
 if len(rows)==settings['requests_per_case']+settings['warmup_requests']:break
if len(rows)<settings['requests_per_case']+settings['warmup_requests']:raise RuntimeError('Insufficient unique ShareGPT requests')
print(json.dumps({'requests':rows,'filtered':filtered,'chat_template':tokenizer.get_chat_template()}))
"""
    import shlex
    shell = "source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /usr/local/Ascend/nnal/atb/set_env.sh; exec python3 -c " + shlex.quote(code) + " " + shlex.quote(runtime["model_source"].replace("/root/", "/workspace/")) + " " + shlex.quote(json.dumps(profile))
    prepared = json.loads(kubectl(cluster, ["exec", "-i", "-n", pool["namespace"], active[0]["metadata"]["name"], "--", "/bin/bash", "-ec", shell],
        stdin=json.dumps(batch), timeout=180).stdout)
    directory = ROOT / "artifacts/bf16_distributed"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "requests.jsonl"
    content = "".join(json.dumps(row) + "\n" for row in prepared["requests"])
    if path.exists() and path.read_text() != content:
        raise RuntimeError("Existing prepared dataset differs; do not silently overwrite it")
    path.write_text(content)
    metadata = {"dataset": dataset, "seed": profile["seed"], "source_records": len(rows), "eligible_first_turns": len(candidates),
        "candidate_count": len(batch), "selected": len(prepared["requests"]), "filtered": prepared["filtered"],
        "requests_sha256": digest(path), "chat_template_sha256": hashlib.sha256(prepared["chat_template"].encode()).hexdigest(),
        "input_length_statistics_include_chat_template": True,
        "output_policy": "fixed " + str(profile["output_tokens"]) + "-token controlled budget; not natural answer length"}
    (directory / "dataset.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


def verify(cluster, pool):
    import time
    pods = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "heteroserve.io/distributed-group=" + pool["name"], "-o", "json"]).stdout)["items"]
    active = [p for p in pods if not p["metadata"].get("deletionTimestamp")]
    if len(active) != pool["parallelism"]["pp"]:
        raise RuntimeError("Wrong PP-stage Pod count")
    expected = []
    for p in sorted(active, key=lambda p: int(p["metadata"]["labels"]["heteroserve.io/stage"])):
        stage = int(p["metadata"]["labels"]["heteroserve.io/stage"])
        if p["spec"]["nodeName"] != pool["nodes"][stage]:
            raise RuntimeError("A pipeline stage is on the wrong physical node")
        expected.append({"uid": p["metadata"]["uid"], "ip": p["status"]["podIP"], "node": p["spec"]["nodeName"], "stage": stage})
    client = client_pod(cluster, pool)
    code = """import httpx,json,sys,time
expected,model,tp,base=json.loads(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4]
with httpx.Client(timeout=180,trust_env=False) as c:
 deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  try:
   replies=[c.get('http://'+item['ip']+':8000/ready',timeout=5) for item in expected]
   if all(r.status_code==200 for r in replies):break
  except httpx.HTTPError:pass
  time.sleep(3)
 else:raise TimeoutError('Model and Ray stages did not become ready')
 identities=[r.json() for r in replies]
 for actual,wanted in zip(identities,expected):
  assert actual['pod_uid']==wanted['uid'] and actual['node']==wanted['node']
  assert len(actual['physical_devices'])==tp
 body={'model':model,'messages':[{'role':'user','content':'What is 17 + 25? Reply with the number only.'}],'temperature':0,'max_tokens':32,'chat_template_kwargs':{'enable_thinking':False}}
 response=c.post(base+'/v1/chat/completions',json=body);response.raise_for_status()
 answer=response.json()['choices'][0]['message']['content'].strip();assert answer=='42'
 chunks=[];done=False;first=None;started=time.perf_counter()
 with c.stream('POST',base+'/v1/chat/completions',json={**body,'stream':True}) as response:
  response.raise_for_status()
  for line in response.iter_lines():
   if not line.startswith('data:'):continue
   value=line[5:].strip()
   if value=='[DONE]':done=True;break
   event=json.loads(value)
   for choice in event.get('choices',[]):
    content=choice.get('delta',{}).get('content')
    if content:
     if first is None:first=time.perf_counter()-started
     chunks.append(content)
 assert done and ''.join(chunks).strip()=='42'
 topology=c.get(base+'/cluster').json();assert len(topology['nodes_at_startup'])==len(expected)
 assert {x['ip'] for x in topology['nodes_at_startup']}=={x['ip'] for x in expected}
 print(json.dumps({'passed':True,'identities':identities,'ordinary_answer':answer,'stream_answer':''.join(chunks).strip(),'stream_done':done,'stream_ttft_seconds':first,'ray_topology':topology['nodes_at_startup']}))
"""
    raw = kubectl(cluster, ["exec", "-n", pool["namespace"], client["metadata"]["name"], "--", "python3", "-c", code,
        json.dumps(expected), pool["runtime"]["served_model"], str(pool["parallelism"]["tp"]),
        "http://" + pool["name"] + "." + pool["namespace"] + ".svc.cluster.local:8000"], timeout=900).stdout
    result = json.loads(raw)
    rank_lines = []
    for p in active:
        log = kubectl(cluster, ["logs", "-n", pool["namespace"], p["metadata"]["name"]]).stdout
        for line in log.splitlines():
            if re.search(r"rank \d+ in world size \d+ is assigned", line):
                rank_lines.append(line)
        directory = result_directory(pool)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ("stage-" + p["metadata"]["labels"]["heteroserve.io/stage"] + ".log")).write_text(log)
    result.update(config=json.loads((HERE / "config.json").read_text()), rank_lines=rank_lines,
        pods={p["metadata"]["name"]: {"uid": p["metadata"]["uid"], "node": p["spec"]["nodeName"],
            "restarts": [s["restartCount"] for s in p["status"].get("containerStatuses", [])]} for p in active},
        runtime_sha256=hashlib.sha256((HERE / "runtime.py").read_bytes()).hexdigest())
    (result_directory(pool) / "acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return client


def benchmark(cluster, pool, entry="direct"):
    # Metrics-only execution does not add ordinary/SSE functional acceptance requests.
    client = client_pod(cluster, pool)
    for stage_pool in ([configuration(expert)[1] for expert in ("awq", "gptq")] if entry == "gateway" else [pool]):
        stages = json.loads(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l",
            "heteroserve.io/distributed-group=" + stage_pool["name"], "-o", "json"]).stdout)["items"]
        if len(stages) != stage_pool["parallelism"]["pp"] or any(p["metadata"].get("deletionTimestamp") or not any(
                c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", [])) for p in stages):
            raise RuntimeError("Wait for all pipeline stages to become Ready before measuring")
    if entry == "gateway":
        pool = system_pool(pool)
    payload = {"benchmark.py": base64.b64encode((HERE / "benchmark.py").read_bytes()).decode(),
               "config.json": base64.b64encode(json.dumps(pool).encode()).decode()}
    requests = result_directory(pool) / "requests.jsonl" if entry == "gateway" else ROOT / "artifacts/bf16_distributed/requests.jsonl"
    if not requests.exists():
        raise RuntimeError("Run prepare-data before measuring real requests")
    payload["requests.jsonl"] = base64.b64encode(requests.read_bytes()).decode()
    code = """import base64,json,os,subprocess,sys
from pathlib import Path
data=json.loads(sys.stdin.read());root=Path(sys.argv[2]);owner=root/'.owner'
if root.exists() and (not owner.exists() or owner.read_text()!='heteroserve-bf16-benchmark'):raise RuntimeError('Unowned benchmark directory')
root.mkdir(exist_ok=True);owner.write_text('heteroserve-bf16-benchmark')
pid=root/'benchmark.pid'
if pid.exists():
 proc=Path('/proc')/pid.read_text().strip()
 if proc.exists() and ') Z ' not in (proc/'stat').read_text():
  raise RuntimeError('An existing benchmark process must finish first')
 pid.unlink()
for name,value in data.items():
 if name not in {'benchmark.py','config.json','requests.jsonl'}:raise RuntimeError('Unexpected client asset')
 (root/name).write_bytes(base64.b64decode(value))
args=[sys.executable,str(root/'benchmark.py'),'--config',str(root/'config.json'),'--url',sys.argv[1],'--requests',str(root/'requests.jsonl'),'--output',str(root/'benchmark.json')]
with (root/'benchmark.log').open('w') as log:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
pid.write_text(str(p.pid));print(json.dumps({'client_pid':p.pid,'output':str(root/'benchmark.json')}))
"""
    status = json.loads(kubectl(cluster, ["exec", "-i", "-n", pool["namespace"], client["metadata"]["name"], "--", "python3", "-c", code,
        "http://" + pool["name"] + "." + pool["namespace"] + ".svc.cluster.local:8000", client_directory(pool)], stdin=json.dumps(payload)).stdout)
    status.update(client_pod=client["metadata"]["name"], client_uid=client["metadata"]["uid"], client_node=client["spec"]["nodeName"])
    (result_directory(pool) / "benchmark-client.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status))


def client_directory(pool):
    return "/tmp/bf16_distributed" + ("" if pool["expert"] == "bf16" else "_" + pool["expert"])


def collect(cluster, pool):
    """Export completed metrics without sending any inference requests."""
    directory = result_directory(pool)
    identity = json.loads((directory / "benchmark-client.json").read_text())
    pod = json.loads(kubectl(cluster, ["get", "pod", identity["client_pod"], "-n", pool["namespace"], "-o", "json"]).stdout)
    if pod["metadata"]["uid"] != identity["client_uid"]:
        raise RuntimeError("Benchmark client Pod was replaced; refuse unrelated results")
    code = """import base64,json,sys
from pathlib import Path
p=Path(sys.argv[1])
if (p/'.owner').read_text()!='heteroserve-bf16-benchmark':raise RuntimeError('Unowned results')
r=json.loads((p/'benchmark.json').read_text())
print(json.dumps({'report':r,'log':base64.b64encode((p/'benchmark.log').read_bytes()).decode()}))
"""
    data = json.loads(kubectl(cluster, ["exec", "-n", pool["namespace"], identity["client_pod"], "--", "python3", "-c", code, client_directory(pool)]).stdout)
    report = data["report"]
    if not report["success"] and not report.get("error"):
        print(json.dumps({"success": False, "progress": report.get("progress"), "state": "running"}))
        return
    requests = directory / "requests.jsonl" if pool["expert"] == "system" else ROOT / "artifacts/bf16_distributed/requests.jsonl"
    if report["dataset_file_sha256"] != hashlib.sha256(requests.read_bytes()).hexdigest():
        raise RuntimeError("Results belong to a different prepared dataset")
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (directory / "benchmark.log").write_bytes(base64.b64decode(data["log"]))
    print(json.dumps({"success": report["success"], "error": report.get("error"), "cases": [r["summary"] for r in report["cases"]]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "render", "apply", "status", "verify", "prepare-data", "prepare-chat", "apply-routing", "benchmark", "collect", "configure-profiler", "quantize", "export-awq", "benchmark-dispatch"])
    parser.add_argument("--expert", choices=["bf16", "awq", "gptq"], default="bf16")
    parser.add_argument("--entry", choices=["direct", "gateway"], default="direct")
    parser.add_argument("--profile-dir", help="Pod-local /scratch output directory; omit to disable the profiler")
    parser.add_argument("--profile-mode", choices=["operators", "timing"], default="operators")
    parser.add_argument("--dispatch-mode", choices=["dispatch", "prefill"], default="dispatch")
    args = parser.parse_args()
    cluster, pool, runtime = configuration(args.expert)
    if args.action == "benchmark-dispatch":
        start_task_job(cluster, pool, dispatch_objects(cluster, pool, runtime, args.dispatch_mode),
                       "dispatch-prefill-job.json" if args.dispatch_mode == "prefill" else "dispatch-job.json")
        return
    if args.action == "export-awq":
        if args.expert != "awq":
            raise ValueError("Select --expert awq for standard AWQ export")
        start_quantization(cluster, pool, runtime, export_awq=True)
        return
    if args.action == "quantize":
        start_quantization(cluster, pool, runtime)
        return
    if args.action == "configure-profiler":
        state = configure_profiler(cluster, pool, args.profile_dir, args.profile_mode)
        print(json.dumps({key:state[key] for key in ("enabled", "directory", "mode", "stages")}, indent=2))
        return
    if args.action == "prepare-chat":
        prepare_chat_data(pool)
        return
    if args.action == "apply-routing":
        manifest, gateway_config = routing_objects(cluster)
        directory = ROOT / "artifacts/bf16_distributed/system"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "gateway.json").write_text(json.dumps(gateway_config, indent=2))
        (directory / "deployment.json").write_text(json.dumps(manifest, indent=2))
        for obj in manifest["items"]:
            existing = kubectl(cluster, ["get", obj["kind"], obj["metadata"]["name"], "-n", pool["namespace"], "--ignore-not-found=true", "-o", "json"]).stdout
            if existing.strip() and json.loads(existing)["metadata"].get("labels", {}).get(OWNER) != cluster["name"]:
                raise RuntimeError("Unowned routing object")
        payload = json.dumps(manifest)
        kubectl(cluster, ["apply", "--dry-run=server", "-f", "-"], stdin=payload)
        print(kubectl(cluster, ["apply", "-f", "-"], stdin=payload).stdout)
        return
    if args.action == "prepare":
        prepare(cluster, pool, runtime)
        return
    if args.action == "verify":
        verify(cluster, pool)
        return
    if args.action == "prepare-data":
        prepare_data(cluster, pool, runtime)
        return
    if args.action == "benchmark":
        benchmark(cluster, pool, args.entry)
        return
    if args.action == "collect":
        collect(cluster, system_pool(pool) if args.entry == "gateway" else pool)
        return
    manifest = objects(cluster, pool, runtime)
    if args.action == "render":
        import yaml
        target = HERE / "deployment.yaml" if args.expert == "bf16" else result_directory(pool) / "deployment.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump_all(manifest["items"], sort_keys=False), encoding="utf-8")
    elif args.action == "apply":
        print(json.dumps(apply_experiment(cluster, pool, runtime), indent=2))
    else:
        print(kubectl(cluster, ["get", "pods", "-n", pool["namespace"], "-l", "heteroserve.io/distributed-group=" + pool["name"], "-o", "wide"]).stdout)


if __name__ == "__main__":
    main()
