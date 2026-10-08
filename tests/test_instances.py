"""Independent instance configuration, safe publishing, and real resource contracts."""
import json

import pytest
import control
import manage_instances as manager
from manage_lab import ROOT, OWNER, load_config
from manage_pool import load_pool
from heteroserve.native_parallel import validate_alignment


@pytest.fixture
def settings():
    pool = load_pool(ROOT / "deploy/pools/ascend-awq.json")
    pool["instances"] = manager.normalized_instances(pool)
    return pool


def test_instance_ids_are_stable_and_two_parallel_profiles_have_distinct_resources(settings):
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    settings["instances"]["awq-02"]["parallelism"] = {"tp": 2, "pp": 1}
    a = manager.instance_objects(cluster, settings, "awq-01")
    b = manager.instance_objects(cluster, settings, "awq-02")
    assert a[0]["metadata"]["name"] != b[0]["metadata"]["name"]
    for ident, resources, cards in [("awq-01", a, "1"), ("awq-02", b, "2")]:
        cm, deployment = resources
        spec = deployment["spec"]["template"]["spec"]
        assert spec["containers"][0]["resources"]["limits"] == {"huawei.com/Ascend910": cards}
        assert deployment["spec"]["replicas"] == 1
        assert deployment["spec"]["selector"]["matchLabels"][manager.INSTANCE_LABEL] == ident
        runtime = json.loads(cm["data"]["worker.json"])
        assert runtime["instance_id"] == ident and runtime["pool"] == settings["pool"]
        assert runtime["device_count"] == int(cards)


def test_tp_pp_count_and_v0_batching_are_declared_together(settings):
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    spec = settings["instances"]["awq-01"]
    spec.update(parallelism={"tp": 2, "pp": 2}, engine={"max_num_batched_tokens": 4096})
    manager.validate_instances(settings, cluster)
    cm, deploy = manager.instance_objects(cluster, settings, "awq-01")
    cfg = json.loads(cm["data"]["worker.json"])
    assert cfg["device_count"] == 4
    assert cfg["engine_command"][cfg["engine_command"].index("--tensor-parallel-size") + 1] == "2"
    assert cfg["engine_command"][cfg["engine_command"].index("--pipeline-parallel-size") + 1] == "2"
    env = {v["name"]: v.get("value") for v in deploy["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["VLLM_USE_V1"] == "0"


@pytest.mark.parametrize("parallel", [{"tp": True, "pp": 1}, {"tp": 0, "pp": 1}, {"tp": 4, "pp": 4}])
def test_invalid_parallelism_cannot_be_saved(settings, parallel):
    settings["instances"]["awq-01"]["parallelism"] = parallel
    with pytest.raises(ValueError):
        manager.validate_instances(settings, load_config(ROOT / "deploy/lab/cluster.json"))


def test_pp_rejects_invalid_v0_token_budget(settings):
    settings["instances"]["awq-01"]["parallelism"] = {"tp": 1, "pp": 2}
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        manager.validate_instances(settings, load_config(ROOT / "deploy/lab/cluster.json"))


def test_resize_keeps_existing_ids_and_parameters(settings):
    settings["instances"]["awq-01"]["engine"]["max_num_seqs"] = 3
    smaller = manager.resize_specs(settings, 2)
    assert len(smaller) == 4 and sum(v["enabled"] for v in smaller.values()) == 2
    assert smaller["awq-01"]["engine"]["max_num_seqs"] == 3
    settings["instances"] = smaller
    larger = manager.resize_specs(settings, 5)
    assert sum(v["enabled"] for v in larger.values()) == 5
    assert larger["awq-01"]["engine"]["max_num_seqs"] == 3


def test_kubernetes_defaults_do_not_cause_spurious_rollout():
    assert manager.matches({"spec": {"dnsPolicy": "ClusterFirst", "containers": [{"name": "expert", "image": "pinned", "terminationMessagePolicy": "File"}]}},
                           {"spec": {"containers": [{"name": "expert", "image": "pinned"}]}})
    assert not manager.matches({"a": 1}, {"a": 2})


def test_int4_shards_must_preserve_quantization_and_packing_groups():
    validate_alignment(2560, [2560, 512, 512], 128, 8)
    with pytest.raises(ValueError, match="quantization"):
        validate_alignment(2559, [2560], 128, 8)
    with pytest.raises(ValueError, match="packed"):
        validate_alignment(2560, [2559], 128, 8)


def test_capacity_failure_happens_before_any_workload_mutation(settings, monkeypatch):
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    monkeypatch.setattr(manager, "plan", lambda *a: [])
    monkeypatch.setattr(manager, "preflight", lambda *a: (_ for _ in ()).throw(RuntimeError("No free cards")))
    monkeypatch.setattr(manager, "kubectl", lambda *a, **kw: pytest.fail("No mutation before capacity check"))
    with pytest.raises(RuntimeError, match="No free cards"):
        manager.apply(cluster, settings)


def test_pool_service_discovery_remains_shared_while_deployments_are_independent(settings):
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    objects = manager.objects(cluster, settings)
    services = [v for v in objects if v["kind"] == "Service"]
    assert len(services) == 1 and manager.INSTANCE_LABEL not in services[0]["spec"]["selector"]
    assert len([v for v in objects if v["kind"] == "Deployment"]) == 4


def test_selected_instance_scope_never_updates_other_deployments(settings, monkeypatch):
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    monkeypatch.setattr(manager, "get", lambda *a, **kw: {"items": []})
    result = manager.plan(cluster, settings, "awq-02")
    assert [v["instance"] for v in result] == ["awq-02"]


def test_pool_context_must_remain_interchangeable(settings):
    settings["instances"]["awq-01"]["engine"]["max_model_len"] = 8192
    with pytest.raises(ValueError, match="share max_model_len"):
        manager.validate_instances(settings, load_config(ROOT / "deploy/lab/cluster.json"))


def test_parallelism_cannot_be_overridden_through_a_second_parameter_source(settings):
    from manage_pool import resolve_runtime
    settings["runtime"] = {"model_parameters": {"tensor_parallel_size": 1}}
    with pytest.raises(ValueError, match="Parallelism belongs"):
        resolve_runtime(settings)


def test_removed_instances_are_in_the_reviewable_plan(settings, monkeypatch):
    obsolete = {"metadata": {"name": settings["pool"] + "-awq-old", "labels": {OWNER: "heteroserve-lab", manager.INSTANCE_LABEL: "awq-old"}}}
    monkeypatch.setattr(manager, "get", lambda *a, **kw: {"items": [obsolete]})
    changes = manager.plan(load_config(ROOT / "deploy/lab/cluster.json"), settings)
    assert any(v["instance"] == "awq-old" and v["action"] == "remove" for v in changes)


def test_live_engine_arguments_handle_values_and_false_flags():
    result = manager.engine_values(["python3", "--max-model-len", "4096", "--max-num-seqs=3", "--gpu-memory-utilization", "0.5"])
    assert result == {"max_model_len": 4096, "max_num_seqs": 3, "gpu_memory_utilization": .5, "enforce_eager": False}
    assert "max_num_batched_tokens" not in result


@pytest.mark.parametrize("wrong_identity", [False, True])
def test_runtime_inventory_checks_identity_and_uses_process_values(settings, monkeypatch, wrong_identity):
    from types import SimpleNamespace
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    pod = {"metadata": {"name": "actual-pod", "uid": "actual-uid", "namespace": "heteroserve",
                        "labels": {"app.kubernetes.io/name": "expert", "heteroserve.io/pool": settings["pool"], manager.INSTANCE_LABEL: "awq-01"}},
           "spec": {"nodeName": "heteroserve-lab-209", "containers": [{"name": "expert", "startupProbe": {"periodSeconds": 5, "failureThreshold": 240}}]},
           "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]}}
    identity = {"pod_uid": "actual-uid", "node": "heteroserve-lab-209", "instance_id": "awq-01",
                "physical_devices": [6], "parallelism": {"tp": 1, "pp": 1}}
    snapshot = {**identity, "pod_uid": "wrong" if wrong_identity else "actual-uid", "source": "process_snapshot",
                "engine": {"max_num_seqs": 3}, "traffic": {"max_inflight": 7}, "lifecycle": {"drain_seconds": 240}}
    replies = iter([identity, snapshot])
    monkeypatch.setattr(manager, "get", lambda *a, **kw: {"items": [pod]})
    monkeypatch.setattr(manager, "kubectl", lambda *a, **kw: SimpleNamespace(stdout=json.dumps(next(replies))))
    row = manager.inventory(cluster, settings, runtime=True)[0]
    if wrong_identity:
        assert "runtime_values" not in row and "identity" in row["runtime_error"]
    else:
        assert row["runtime_values"]["engine.max_num_seqs"] == 3
        assert row["runtime_values"]["traffic.max_inflight"] == 7
        assert row["physical_devices"] == [6]
