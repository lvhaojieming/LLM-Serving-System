"""Validate stage boundaries and client metric semantics without accelerator mocks."""
import asyncio
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


manager = module("bf16_experiment_manager", "manage.py")
bench = module("bf16_experiment_benchmark", "benchmark.py")
quantizer = module("bf16_experiment_quantizer", "quantize.py")


def test_system_speed_uses_one_configured_tp2_instance_per_expert_with_shared_optimized_execution(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "manage", manager)
    host = module("host_speed", "host_benchmark.py")
    config, profiles = host.system_profiles()
    manifests = set()
    for expert, (pool, ident) in profiles.items():
        from manage_instances import validate_instances, instance_objects
        cluster = manager.configuration(expert)[0]
        validate_instances(pool, cluster)
        assert [k for k, v in pool['instances'].items() if v['enabled']] == [ident]
        assert pool['instances'][ident]['parallelism'] == {'tp': 2, 'pp': 1}
        data, deploy = instance_objects(cluster, pool, ident)
        cfg = json.loads(data['data']['worker.json'])
        assert cfg['device_count'] == 2 and cfg['execution_optimizations']['native_int4_fast_path']
        assert cfg['execution_optimizations']['native_int4_script_mlp']
        assert cfg['engine_env']['TASK_QUEUE_ENABLE'] == '2'
        assert cfg['host_execution']['executor'] == 'mp'
        assert cfg['engine_parameters']['max_num_seqs'] == 16
        manifests.add(cfg['weights_manifest_sha256'])
    assert len(manifests) == 2


def test_engine_metrics_aggregate_labels_and_ignore_histogram_buckets():
    parsed = bench.engine_metrics('''# HELP example
vllm:request_prefill_time_seconds_sum{model_name="qwen",engine="0"} 1.2
vllm:request_prefill_time_seconds_sum{model_name="qwen",engine="1"} 0.8
vllm:request_prefill_time_seconds_count{engine="0"} 4
vllm:request_prefill_time_seconds_count{engine="1"} 3
vllm:request_prefill_time_seconds_bucket{le="1"} 7
vllm:num_preemptions_total{model_name="qwen"} 0
vllm:unrelated_metric 99
''')
    assert parsed == {"request_prefill_time_seconds_sum": 2.0,
                      "request_prefill_time_seconds_count": 7.0, "num_preemptions_total": 0.0}


def test_prefill_operator_probe_reuses_isolated_job_without_cpu_limits():
    cluster, pool, runtime = manager.configuration("awq")
    manifest = manager.dispatch_objects(cluster, pool, runtime, "prefill")
    job = next(obj for obj in manifest["items"] if obj["kind"] == "Job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == runtime["model_image"]
    assert container["resources"]["limits"] == {runtime["resource"]: "2"}
    assert "--mode prefill" in container["args"][0]
    assert "/quantization/dispatch-prefill.json" in container["args"][0]
    assert not any(volume["name"] == "weights" for volume in job["spec"]["template"]["spec"]["volumes"])


def test_deployment_strategy_comparison_keeps_inputs_and_local_quantized_tp_identical():
    _, baseline, _ = manager.configuration("bf16")
    assert baseline["parallelism"] == {"tp": 2, "pp": 2}
    assert baseline["nodes"] == ["heteroserve-lab-210", "heteroserve-lab-211"]
    for expert in ("awq", "gptq"):
        cluster, pool, runtime = manager.configuration(expert)
        assert pool["parallelism"] == {"tp": 2, "pp": 1}
        assert pool["nodes"] == ["heteroserve-lab-211"]
        assert pool["benchmark"] == baseline["benchmark"]
        assert runtime["model_parameters"]["quantization"] == "moqe_ascend_int4"
        manifest = manager.objects(cluster, pool, runtime)
        deployments = [obj for obj in manifest["items"] if obj["kind"] == "Deployment"]
        assert len(deployments) == 1
        container = deployments[0]["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"] == {runtime["resource"]: "2"}


def test_local_tp_tuning_preserves_two_card_isolation_and_does_not_change_bf16():
    cluster, pool, runtime = manager.configuration("awq")
    pool["host_execution"] = {"executor": "mp", "task_queue_enable": 2, "cpu_affinity_conf": 1}
    manifest = manager.objects(cluster, pool, runtime)["items"]
    deployment = next(x for x in manifest if x["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"] == {runtime["resource"]: "2"}
    worker = json.loads(next(x for x in manifest if x["kind"] == "ConfigMap")["data"]["worker.json"])
    assert worker["distributed_executor_backend"] == "mp" and worker["parallelism"] == {"tp": 2, "pp": 1}
    assert worker["engine_command"][worker["engine_command"].index("--distributed-executor-backend") + 1] == "mp"
    assert not any(x["kind"] == "Service" and x["metadata"]["name"].endswith("-ray") for x in manifest)
    env = {x["name"]: x.get("value") for x in container["env"]}
    assert env["TASK_QUEUE_ENABLE"] == "2" and env["CPU_AFFINITY_CONF"] == "1"
    baseline = manager.configuration("bf16")[1]
    assert baseline["parallelism"] == {"tp": 2, "pp": 2}
    baseline["host_execution"] = {"executor": "mp"}
    with pytest.raises(ValueError, match="PP1"):
        manager.objects(cluster, baseline, runtime)


def test_local_two_card_runtime_does_not_import_ray(monkeypatch):
    import sys
    from types import SimpleNamespace
    from fastapi import FastAPI
    app = FastAPI()
    app.state.worker = {"identity": {"physical_devices": [4, 5]}}
    monkeypatch.setitem(sys.modules, "worker", SimpleNamespace(create_app=lambda config: app))
    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setenv("POD_IP", "127.0.0.1")
    runner = module("local_tp_runtime", "runtime.py")
    config = {"distributed_executor_backend": "mp", "cluster": {"tp": 2, "pp": 1, "stage": 0}}
    assert runner.create_app(config) is app
    route = next(r for r in app.routes if r.path == "/cluster")
    assert asyncio.run(route.endpoint())["nodes_at_startup"][0]["npu_resources"] == 2


def test_requantization_uses_original_weights_one_npu_and_no_custom_serving_adapter():
    cluster, pool, runtime = manager.configuration("awq")
    manifest = manager.quantization_objects(cluster, pool, runtime)
    cm, job = manifest["items"]
    spec = job["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["image"] == runtime["model_image"]
    assert container["workingDir"] == "/scratch"
    assert container["resources"]["limits"] == {runtime["resource"]: "1"}
    assert spec["restartPolicy"] == "Never" and job["spec"]["backoffLimit"] == 0
    assert dict((v["name"], v["value"]) for v in container["env"])["MOQE_ASCEND_INT4_ADAPTER"] == "0"
    settings = json.loads(cm["data"]["config.json"])["requantization"]
    source = next(v for v in container["volumeMounts"] if v["name"] == "weights")
    assert source["mountPath"] == settings["source"] and source["readOnly"]
    assert pool["parallelism"] == {"tp": 2, "pp": 1}  # Serving strategy is independent of calibration.


@pytest.mark.parametrize("change", [{"num_bits": 8}, {"group_size": 64}, {"actorder": "group"}])
def test_requantization_rejects_incompatible_quantized_outputs(change):
    group = {"weights": {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 128}}
    config = {"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {"group_0": group}}}
    assert quantizer.validate_quantization(config)
    group["weights"].update(change)
    with pytest.raises(ValueError, match="W4A16"):
        quantizer.validate_quantization(config)


def test_requantization_refuses_to_overwrite_existing_model_before_importing_accelerator(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    settings = {"source": str(tmp_path / "missing"), "outputs": {"awq": str(output)}}
    with pytest.raises(FileExistsError, match="overwrite"):
        quantizer.quantize(settings, "awq")


def test_standard_awq_export_preserves_source_and_refuses_overwrite_without_torch(tmp_path):
    source, output = tmp_path / "compressed", tmp_path / "awq"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        quantizer.export_awq({"outputs": {"awq": str(source)}, "standard_outputs": {"awq": str(output)}})
    assert not source.exists()


def test_standard_awq_export_has_no_npu_allocation_or_cpu_memory_limits():
    cluster, pool, runtime = manager.configuration("awq")
    cm, job = manager.quantization_objects(cluster, pool, runtime, export_awq=True)["items"]
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert cm["metadata"]["name"] == "qwen3-export-awq"
    assert runtime["resource"] not in container["resources"]["requests"]
    assert container["resources"].get("limits", {}) == {}
    assert "--export-awq" in container["args"][0]
    assert {x["name"]: x.get("value") for x in container["env"]}["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"










def test_tokenizer_publication_restores_calibration_settings_without_changing_vocabulary(tmp_path):
    source, saved = tmp_path / "original", tmp_path / "saved"
    source.mkdir(); saved.mkdir()
    original = {"model": {"vocab": {"hello": 1}}, "padding": None, "truncation": None}
    candidate = {**original, "padding": {"length": 512}, "truncation": {"max_length": 512}}
    (source / "tokenizer.json").write_text(json.dumps(original))
    (saved / "tokenizer.json").write_text(json.dumps(candidate))
    (source / "tokenizer_config.json").write_text('{}')
    (saved / "tokenizer_config.json").write_text('{"extra_special_tokens": {}}')
    quantizer.restore_tokenizer(source, saved)
    assert (source / "tokenizer.json").read_bytes() == (saved / "tokenizer.json").read_bytes()
    assert (source / "tokenizer_config.json").read_bytes() == (saved / "tokenizer_config.json").read_bytes()
    candidate["model"] = {"vocab": {"hello": 2}}
    (saved / "tokenizer.json").write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="vocabulary"):
        quantizer.restore_tokenizer(source, saved)


def test_collection_rejects_replaced_client_and_preserves_incomplete_measurement(monkeypatch, tmp_path):
    from types import SimpleNamespace
    directory = tmp_path / "artifacts/bf16_distributed"
    directory.mkdir(parents=True)
    (directory / "benchmark-client.json").write_text(json.dumps({"client_pod": "client", "client_uid": "original"}))
    (directory / "requests.jsonl").write_text("selected requests")
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    pod_uid = "replaced"
    report = {"success": False, "progress": {"completed": 10}, "cases": []}
    calls = []
    def fake_kubectl(cluster, args):
        calls.append(args[0])
        result = {"metadata": {"uid": pod_uid}} if args[0] == "get" else {"report": report, "log": base64.b64encode(b"finished\n").decode()}
        return SimpleNamespace(stdout=json.dumps(result))
    monkeypatch.setattr(manager, "kubectl", fake_kubectl)
    with pytest.raises(RuntimeError, match="replaced"):
        manager.collect({}, {"namespace": "project", "expert": "bf16"})
    assert calls == ["get"]
    pod_uid = "original"
    manager.collect({}, {"namespace": "project", "expert": "bf16"})
    assert not (directory / "benchmark.json").exists()
    report.update(success=True, dataset_file_sha256=hashlib.sha256((directory / "requests.jsonl").read_bytes()).hexdigest())
    manager.collect({}, {"namespace": "project", "expert": "bf16"})
    assert json.loads((directory / "benchmark.json").read_text()) == report
    assert (directory / "benchmark.log").read_bytes() == b"finished\n"


def test_pipeline_stages_have_separate_physical_nodes_and_local_npu_allocations():
    cluster, pool, runtime = manager.configuration()
    pool['nodes'] = ['heteroserve-lab-210','heteroserve-lab-211']
    pool['parallelism'] = {'tp':2,'pp':2}
    objects = manager.objects(cluster, pool, runtime)["items"]
    deployments = [x for x in objects if x["kind"] == "Deployment"]
    assert len(deployments) == 2
    assert [x["spec"]["template"]["spec"]["nodeSelector"]["kubernetes.io/hostname"] for x in deployments] == pool["nodes"]
    for deployment in deployments:
        pod = deployment["spec"]["template"]["spec"]
        assert pod["containers"][0]["resources"]["limits"] == {"huawei.com/Ascend910": "2"}
        assert not pod.get("hostNetwork") and not pod.get("hostPID")
        assert not pod["containers"][0]["securityContext"].get("privileged")
    services = [x for x in objects if x["kind"] == "Service"]
    assert not services[0]["spec"].get("publishNotReadyAddresses", False)
    assert services[1]["spec"]["publishNotReadyAddresses"]
    assert services[0]["spec"]["selector"]["heteroserve.io/stage"] == "0"


@pytest.mark.parametrize("problem", ["paused", "unowned", "foreign-profiler"])
def test_profiler_preflight_never_changes_paused_or_unowned_stages(monkeypatch, tmp_path, problem):
    from types import SimpleNamespace
    from contextlib import nullcontext
    cluster, pool, _ = manager.configuration()
    monkeypatch.setattr(manager, "ROOT", tmp_path)
    monkeypatch.setattr(manager, "operation_lock", nullcontext)
    calls = []
    def fake_kubectl(cluster, args):
        calls.append(args[0])
        deployment = {"metadata": {"labels": {manager.OWNER: cluster["name"]}}, "spec": {
            "replicas": 1, "template": {"spec": {"containers": [{"name": "model", "env": []}]}}}}
        if problem == "paused": deployment["spec"]["replicas"] = 0
        if problem == "unowned": deployment["metadata"]["labels"] = {}
        if problem == "foreign-profiler": deployment["spec"]["template"]["spec"]["containers"][0]["env"] = [
            {"name": "VLLM_TORCH_PROFILER_DIR", "value": "/scratch/someone-else"}]
        return SimpleNamespace(stdout=json.dumps(deployment))
    monkeypatch.setattr(manager, "kubectl", fake_kubectl)
    with pytest.raises(RuntimeError):
        manager.configure_profiler(cluster, pool, "/scratch/heteroserve-profile")
    assert calls == ["get"]
    assert not manager.result_directory(pool).exists()


def test_profiler_rejects_output_outside_owned_scratch_before_remote_operations(monkeypatch):
    cluster, pool, _ = manager.configuration()
    monkeypatch.setattr(manager, "kubectl", lambda *a, **kw: pytest.fail("Unexpected remote operation"))
    for directory in ["/tmp/profile", "/scratch/../shared", "relative"]:
        with pytest.raises(ValueError, match="/scratch"):
            manager.configure_profiler(cluster, pool, directory)


@pytest.mark.parametrize("expert", ["bf16", "awq"])
def test_profiler_restores_original_runtime_and_preserves_other_environment(monkeypatch, tmp_path, expert):
    from types import SimpleNamespace
    from contextlib import nullcontext
    cluster, pool, runtime = manager.configuration(expert)
    resources = {(x['kind'].lower(), x['metadata']['name']): copy.deepcopy(x)
                 for x in manager.objects(cluster, pool, runtime)['items']}
    before = copy.deepcopy(resources)
    execution_source = (manager.ROOT / 'src/heteroserve/ascend_execution.py').read_text()
    worker_source = (manager.ROOT / 'src/heteroserve/execution_worker.py').read_text()
    (tmp_path / 'src/heteroserve').mkdir(parents=True)
    (tmp_path / 'src/heteroserve/ascend_execution.py').write_text(execution_source)
    (tmp_path / 'src/heteroserve/execution_worker.py').write_text(worker_source)
    monkeypatch.setattr(manager, 'ROOT', tmp_path)
    monkeypatch.setattr(manager, 'operation_lock', nullcontext)
    def fake_kubectl(cluster, args):
        kind, name = args[1:3]
        obj = resources[kind, name]
        if args[0] == 'get': return SimpleNamespace(stdout=json.dumps(obj))
        patch = json.loads(args[-1])
        if kind == 'configmap':
            for key, value in patch['data'].items():
                if value is None: obj['data'].pop(key, None)
                else: obj['data'][key] = value
        else:
            env = obj['spec']['template']['spec']['containers'][0]['env']
            for item in patch['spec']['template']['spec']['containers'][0]['env']:
                env[:] = [entry for entry in env if entry['name'] != item['name']]
                if item.get('$patch') != 'delete': env.append(item)
        return SimpleNamespace(stdout='patched')
    monkeypatch.setattr(manager, 'kubectl', fake_kubectl)
    manager.configure_profiler(cluster, pool, '/scratch/heteroserve-profile-light')
    for (kind, name), obj in resources.items():
        if kind == 'configmap': assert 'profiling_worker.py' in obj['data']
    manager.configure_profiler(cluster, pool)
    for (kind, name), obj in resources.items():
        if kind == 'configmap': assert obj['data'] == before[kind, name]['data']
        if kind == 'deployment':
            assert obj['spec']['template']['spec']['containers'][0]['env'] == before[kind, name]['spec']['template']['spec']['containers'][0]['env']
            assert obj['spec']['template']['spec']['containers'][0]['resources'] == before[kind, name]['spec']['template']['spec']['containers'][0]['resources']


@pytest.mark.parametrize('expert', ['bf16','awq','gptq'])
@pytest.mark.parametrize('mode', ['native','preserve_rounding'])
def test_fused_qknorm_is_configured_on_every_stage_without_changing_precision_or_parallelism(expert, mode):
    cluster, pool, runtime = manager.configuration(expert)
    original = copy.deepcopy(runtime['model_parameters'])
    pool['qwen3_fused_qknorm'] = True
    pool['qknorm_mode'] = mode
    pool['qknorm_verify_calls'] = 0
    pool['native_int4_script_mlp'] = False
    for obj in manager.objects(cluster,pool,runtime)['items']:
        if obj['kind'] != 'ConfigMap': continue
        data=obj['data'];worker=json.loads(data['worker.json'])
        assert worker['execution_optimizations'] == {'ascend_pp_cache_views':True,
                                                    'qwen3_fused_qknorm':True,'qknorm_mode':mode}
        assert worker['engine_parameters'] == original
        assert worker['parallelism'] == pool['parallelism']
        assert 'execution_worker.py' in data and 'ascend_execution.py' in data
        if expert == 'bf16':
            assert worker['engine_command'].count('--worker-cls') == 1
            assert 'execution_worker.NPUWorker' in worker['engine_command']
        else: assert 'from execution_worker import NPUWorker' in data['quantized_worker.py']


def test_qknorm_audit_reaches_each_rank_and_invalid_sample_budget_is_rejected(monkeypatch, tmp_path):
    cluster, pool, runtime = manager.configuration('awq')
    pool.update(qwen3_fused_qknorm=True, qknorm_mode='preserve_rounding', qknorm_verify_calls=2)
    for obj in manager.objects(cluster, pool, runtime)['items']:
        if obj['kind'] == 'ConfigMap':
            worker = json.loads(obj['data']['worker.json'])
            assert worker['execution_optimizations']['qknorm_verify_calls'] == 2
    config = json.loads((HERE / 'config.json').read_text())
    monkeypatch.setattr(manager, 'HERE', tmp_path)
    for value in (True, -1, 9, '2'):
        config['qknorm_verify_calls'] = value
        (tmp_path / 'config.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='qknorm_verify_calls'):
            manager.configuration()


@pytest.mark.parametrize('expert', ['bf16','awq','gptq'])
def test_scripted_int4_mlp_keeps_configured_strategy_and_original_engine_parameters(expert):
    cluster, pool, runtime = manager.configuration(expert)
    pool['native_int4_script_mlp'] = True
    original = copy.deepcopy(runtime['model_parameters'])
    manifest = manager.objects(cluster, pool, runtime)['items']
    stages = [obj for obj in manifest if obj['kind']=='Deployment']
    assert len(stages)==pool['parallelism']['pp']
    for obj in manifest:
        if obj['kind']!='ConfigMap':continue
        worker = json.loads(obj['data']['worker.json'])
        assert worker['parallelism']==pool['parallelism']
        assert worker['device_count']==pool['parallelism']['tp']
        assert worker['engine_parameters']==original
        assert worker['execution_optimizations'].get('native_int4_script_mlp',False)==(expert!='bf16')
        assert worker['engine_command'][worker['engine_command'].index('--distributed-executor-backend')+1] == ('mp' if expert == 'awq' else 'ray')


def singlecard_test_configuration(expert, monkeypatch, tmp_path):
    config = json.loads((HERE/'config.json').read_text())
    config['model_overrides'] = {expert:{'nodes':config['nodes'][:1], 'parallelism':{'tp':1,'pp':1}}}
    (tmp_path/'config.json').write_text(json.dumps(config))
    with monkeypatch.context() as local:
        local.setattr(manager, 'HERE', tmp_path)
        return manager.configuration(expert)


@pytest.mark.parametrize('expert', ['awq', 'gptq'])
def test_singlecard_profile_uses_one_npu_without_ray_or_checkpoint_conversion(expert, monkeypatch, tmp_path):
    cluster, pool, runtime = singlecard_test_configuration(expert, monkeypatch, tmp_path)
    assert pool['parallelism'] == {'tp':1, 'pp':1}
    manifest = manager.objects(cluster, pool, runtime)['items']
    deployments = [x for x in manifest if x['kind'] == 'Deployment']
    assert len(deployments) == 1
    assert deployments[0]['spec']['template']['spec']['containers'][0]['resources']['limits'] == {'huawei.com/Ascend910': '1'}
    env = {v['name']:v.get('value') for v in deployments[0]['spec']['template']['spec']['containers'][0]['env']}
    assert '/opt/heteroserve' in env['PYTHONPATH'].split(':')
    assert not any(x['kind']=='Service' and x['metadata']['name'].endswith('-ray') for x in manifest)
    worker = json.loads(next(x for x in manifest if x['kind']=='ConfigMap')['data']['worker.json'])
    assert worker['device_count'] == 1
    command = worker['engine_command']
    assert command[command.index('--distributed-executor-backend')+1] == 'mp'
    assert worker['engine_parameters']['quantization'] == 'moqe_ascend_int4'
    assert '--enforce-eager' in command
    assert manager.configuration()[1]['parallelism'] == {'tp':2, 'pp':2}


@pytest.mark.parametrize('problem', [None, 'unowned', 'paused', 'retired_unowned','valid_resume','wrong_resume'])
def test_parallelism_apply_dryruns_first_and_only_cleans_owned_active_stages(monkeypatch, tmp_path, problem):
    from types import SimpleNamespace
    from contextlib import nullcontext
    cluster, pool, runtime = singlecard_test_configuration('awq', monkeypatch, tmp_path)
    manifest = manager.objects(cluster, pool, runtime)
    inventory = copy.deepcopy(manifest['items'])
    head = next(x for x in inventory if x['kind']=='Deployment')
    head['metadata']['uid'] = 'original-experiment-head'
    retired = copy.deepcopy(head)
    retired['metadata']['name'] = pool['name']+'-stage-1'
    retired['metadata']['labels']['heteroserve.io/stage'] = '1'
    inventory.append(retired)
    unrelated = copy.deepcopy(retired)
    unrelated['metadata']['name'] = 'original-paused-model'
    unrelated['metadata']['labels']['heteroserve.io/distributed-group'] = 'another'
    unrelated['spec']['replicas'] = 0
    inventory.append(unrelated)
    if problem == 'unowned': head['metadata']['labels'][manager.OWNER] = 'someone-else'
    if problem == 'paused': head['spec']['replicas'] = 0
    if problem in {'valid_resume','wrong_resume'}: head['spec']['replicas'] = 0
    if problem == 'retired_unowned': retired['metadata']['labels'][manager.OWNER] = 'someone-else'
    monkeypatch.setattr(manager, 'result_directory', lambda p: tmp_path/'awq')
    monkeypatch.setattr(manager, 'operation_lock', nullcontext)
    calls = []
    def fake(cluster, args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps({'items':inventory}) if args[0]=='get' else 'ok')
    monkeypatch.setattr(manager, 'kubectl', fake)
    resume = {head['metadata']['name']:'original-experiment-head' if problem=='valid_resume' else 'another-object'} if problem in {'valid_resume','wrong_resume'} else None
    if problem and problem!='valid_resume':
        with pytest.raises(RuntimeError): manager.apply_experiment(cluster, pool, runtime, resume)
        assert [x[0] for x in calls] == ['get']
    else:
        result = manager.apply_experiment(cluster, pool, runtime, resume)
        assert result['removed'] == [retired['metadata']['name']]
        assert [x[0] for x in calls] == ['get','apply','apply','delete']
        assert '--dry-run=server' in calls[1]
        snapshot = json.loads((manager.result_directory(pool)/'deployment-transition.json').read_text())
        assert snapshot['obsolete'][0]['metadata']['name'] == retired['metadata']['name']


def test_single_process_runtime_does_not_import_or_start_ray(monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    import sys
    app = FastAPI()
    app.state.worker = {'identity':{'physical_devices':[0]}}
    monkeypatch.setitem(sys.modules, 'worker', SimpleNamespace(create_app=lambda c:app))
    monkeypatch.setitem(sys.modules, 'ray', None)
    monkeypatch.setenv('POD_IP', '127.0.0.1')
    runner = module('single_process_experiment_runtime', 'runtime.py')
    result = runner.create_app({'cluster':{'tp':1,'pp':1,'stage':0}})
    assert result is app
    route = next(r for r in app.routes if r.path=='/cluster')
    assert asyncio.run(route.endpoint())['scope'] == 'Single-process model execution'


def test_ascend_cache_views_follow_virtual_engine_and_are_reused():
    from heteroserve.ascend_execution import install_ascend_pp_cache_views
    from types import SimpleNamespace

    class Cache:
        def __init__(self, bank):
            self.bank, self.slices = bank, 0
        def numel(self):
            return 2
        def __getitem__(self, index):
            self.slices += 1
            return (self.bank, index)

    class Backend:
        __module__ = 'vllm_ascend.attention.attention'
        key_cache = value_cache = None
        def forward(self, layer, query, key, value, kv_cache, metadata, output=None):
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            return self.key_cache, self.value_cache, output

    backend = Backend()
    layer = SimpleNamespace(impl=backend)
    model = SimpleNamespace(named_modules=lambda: [('attention', layer)])
    banks = [Cache(0), Cache(1)]
    assert install_ascend_pp_cache_views(model) == 1
    for bank in [0, 1, 0, 1, 1, 0]:
        result = backend.forward(layer, None, None, None, banks[bank], None, output='sentinel')
        assert result == ((bank, 0), (bank, 1), 'sentinel')
    assert [b.slices for b in banks] == [2, 2]
    assert install_ascend_pp_cache_views(model) == 1
    assert backend.forward(layer, None, None, None, banks[1], None)[0] == (1, 0)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("expert", ["awq", "gptq"])
def test_quantized_profiles_reuse_weights_and_install_adapter_in_each_ray_worker(expert, tp):
    cluster, pool, runtime = manager.configuration(expert)
    # Continue covering the original multi-node TP path independently of the
    # configured deployment, which can now keep an INT4 model on one card.
    pool['nodes'] = ['heteroserve-lab-210','heteroserve-lab-211']
    pool['parallelism'] = {'tp':tp, 'pp':2}
    pool['host_execution'] = {'executor': 'ray'}
    assert runtime["model_parameters"]["quantization"] == "moqe_ascend_int4"
    assert runtime["model_parameters"]["dtype"] == "float16"
    assert pool["benchmark"] == manager.configuration()[1]["benchmark"]
    assert manager.result_directory(pool).name == expert
    assert manager.client_directory(pool).endswith("_" + expert)
    pool["parallelism"]["tp"] = tp
    for obj in manager.objects(cluster, pool, runtime)["items"]:
        if obj["kind"] == "ConfigMap":
            data = obj["data"]
            worker = json.loads(data["worker.json"])
            assert worker["engine_env"]["MOQE_ASCEND_INT4_PARALLEL"] == ("1" if tp > 1 else "0")
            assert "quantized_worker.NPUWorker" in worker["engine_command"]
            assert "native_parallel.py" in data and "sitecustomize.py" in data
            assert "from adapter_patch import install" in data["quantized_worker.py"]
            assert "from native_parallel import install" in data["quantized_worker.py"]


@pytest.mark.parametrize("expert", ["awq", "gptq"])
def test_int4_fast_path_is_explicit_in_every_ray_worker_and_keeps_tensor_contract(expert):
    cluster, pool, runtime = manager.configuration(expert)
    pool["native_int4_fast_path"] = True
    for obj in manager.objects(cluster, pool, runtime)["items"]:
        if obj["kind"] != "ConfigMap": continue
        assert "install(fast_path=True)" in obj["data"]["quantized_worker.py"]
        worker = json.loads(obj["data"]["worker.json"])
        assert worker["parallelism"] == pool['parallelism']
        assert worker["engine_parameters"]["quantization"] == "moqe_ascend_int4"


def test_readiness_failure_exports_terminal_error_without_sending_inference(monkeypatch, tmp_path):
    config = {"benchmark": {"requests_per_case": 1, "warmup_requests": 0, "concurrency": [1],
        "timeout_seconds": 1, "output_tokens": 2}, "runtime": {"served_model": "bf16"}}
    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"id": "one", "input_ids": [1000, 1001]}) + "\n")
    output = tmp_path / "benchmark.json"
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(503)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler)))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(bench.run(config, "http://model", output, requests))
    report = json.loads(output.read_text())
    assert report["success"] is False and report["error"]["phase"] == "readiness"
    assert calls == [("GET", "/ready")]


def test_system_chat_preserves_auto_routing_and_subtracts_worker_wait_from_routing_share():
    headers = {"x-request-id": "trace", "x-moqe-expert": "awq", "x-moqe-l1-instance": "l1-01",
        "x-moqe-l1-node": "node", "x-moqe-l1-pod-uid": "l1-uid", "x-heteroserve-pod-uid": "worker-uid",
        "x-heteroserve-gateway-headers-ms": "30", "x-heteroserve-gateway-pre-dispatch-ms": "1",
        "x-heteroserve-l1-rpc-ms": "15", "x-moqe-router-ms": "10", "x-heteroserve-l1-queue-ms": "1",
        "x-heteroserve-l1-service-ms": "12", "x-heteroserve-l2-pre-dispatch-ms": "1",
        "x-heteroserve-l2-core-headers-ms": "12", "x-heteroserve-worker-headers-ms": "10",
        "x-heteroserve-worker-engine-headers-ms": "8",
        **{"x-heteroserve-l1-" + key: "false" for key in ("graph-enabled", "graph-used", "graph-fallback")}}
    class Events(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(.04)
            yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            await asyncio.sleep(.02)
            yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'
    async def scenario():
        def handler(request):
            body = json.loads(request.content)
            assert request.url.path == '/v1/chat/completions' and body['model'] == 'auto'
            assert body['messages'][0]['content'] == 'hello' and 'prompt' not in body
            return httpx.Response(200, headers=headers, stream=Events())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            row = await bench.request(client, 'http://gateway', 'auto', [1000, 1001], 2, 0, 2,
                'sample', [{'role':'user','content':'hello'}])
        assert row['success'] and row['expert'] == 'awq'
        assert row['routing']['routing_transport_ms'] == 20
        assert row['routing']['l2_routing_transport_ms'] == 3
        assert row['routing']['routing_ttft_percent'] == pytest.approx(2000 / row['ttft_ms'])
        assert row['routing']['graph_enabled'] is False
    asyncio.run(scenario())


def test_routing_shares_are_summarized_per_request_instead_of_dividing_percentiles():
    rows = [{'success':True,'input_tokens':2,'output_tokens':2,'ttft_ms':ttft,'tpot_ms':1,'e2e_ms':500,
        'routing':{'routing_transport_ms':route,'routing_ttft_percent':100*route/ttft}}
        for ttft,route in [(20,10),(300,30)]]
    result = bench.summarize(rows,1)
    assert result['routing_ttft_percent']['mean'] == 30
    assert result['routing_ttft_percent']['p95'] == 48


def test_full_system_manifest_contains_official_routers_and_only_serving_heads_are_discovered():
    cluster, pool, runtime = manager.configuration()
    manifest, gateway = manager.routing_objects(cluster)
    deployments = {obj['metadata']['name']:obj for obj in manifest['items'] if obj['kind']=='Deployment'}
    assert len([name for name in deployments if name.endswith('-router')]) == 2
    for expert in ('awq','gptq'):
        assert gateway['pools'][expert]['model'] == expert
        assert expert+'-distributed' in gateway['pools'][expert]['base_url']
        service = next(obj for obj in manifest['items'] if obj['kind']=='Service' and obj['metadata']['name']==expert+'-distributed-ascend910b-vllm')
        assert service['spec']['selector']['heteroserve.io/stage']=='0'
        assert service['spec']['selector']['app.kubernetes.io/name']=='expert'
    assert gateway['router'] == json.loads((manager.ROOT/'deploy/gateway.json').read_text())['router']


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_original_weights_never_enable_int4_and_ray_world_size_is_not_local_card_count(tp):
    cluster, pool, runtime = manager.configuration()
    pool = copy.deepcopy(pool)
    pool['nodes'] = ['heteroserve-lab-210','heteroserve-lab-211']
    pool['parallelism']['pp'] = 2
    pool["parallelism"]["tp"] = tp
    for obj in manager.objects(cluster, pool, runtime)["items"]:
        if obj["kind"] != "ConfigMap":
            continue
        config = json.loads(obj["data"]["worker.json"])
        assert config["device_count"] == tp
        assert config["parallelism"] == {"tp": tp, "pp": 2}
        assert "--quantization" not in config["engine_command"]
        assert config["engine_env"]["MOQE_ASCEND_INT4_ADAPTER"] == "0"
        command = config["engine_command"]
        assert command[command.index("--distributed-executor-backend") + 1] == "ray"


def test_duplicate_pp_nodes_are_rejected_before_any_remote_operation(monkeypatch, tmp_path):
    data = json.loads((HERE / "config.json").read_text())
    data["nodes"] = [data["nodes"][0]] * 2
    data['parallelism']['pp'] = 2
    (tmp_path / "config.json").write_text(json.dumps(data))
    monkeypatch.setattr(manager, "HERE", tmp_path)
    with pytest.raises(ValueError, match="different physical"):
        manager.configuration()


def test_empty_sse_events_do_not_define_ttft_and_failed_token_counts_are_not_successes():
    class Events(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"text":""}]}\n\n'
            await asyncio.sleep(.02)
            yield b'data: {"choices":[{"text":"hello"}]}\n\n'
            await asyncio.sleep(.01)
            yield b'data: {"choices":[{"text":" world"}]}\n\n'
            yield b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2}}\n\n'
            yield b'data: [DONE]\n\n'
    async def scenario():
        async def transport(request):
            payload = json.loads(request.content)
            assert payload["skip_special_tokens"] is False
            return httpx.Response(200, stream=Events())
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            passed = await bench.request(client, "http://model", "bf16", [1000, 1001], 2, 0, 2)
            assert passed["success"] and passed["ttft_ms"] >= 15 and passed["tpot_ms"] >= 5
            assert 0 <= passed["headers_ms"] <= passed["first_choice_ms"] <= passed["ttft_ms"]
            assert passed["text_after_first_choice_ms"] >= 15
            failed = await bench.request(client, "http://model", "bf16", [1000, 1001], 3, 0, 2)
            assert not failed["success"]
            summary = bench.summarize([passed, failed], 4)
            assert summary["failed"] == 1 and summary["successful"] == 1
            assert summary["output_tokens_per_second"] == .5
            assert summary["total_tokens_per_second"] == 1
    asyncio.run(scenario())


def test_request_deadline_is_total_time_not_only_read_gap_timeout():
    class Hanging(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(2)
            yield b'data: [DONE]\n\n'
    async def scenario():
        async def transport(request):
            return httpx.Response(200, stream=Hanging())
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            row = await bench.request(client, "http://model", "bf16", [1000, 1001], 2, 0, .02)
            assert not row["success"] and "TimeoutError" in row["error"]
    asyncio.run(scenario())
