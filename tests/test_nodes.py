"""Node membership safety: capacity, ownership, staging and failure boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import manage_nodes as nodes
import manage_lab as lab
import control


@pytest.fixture
def config():
    return lab.load_config(lab.ROOT / nodes.CLUSTER)


def pod(name="worker", node="heteroserve-lab-216", owner=True, app="expert"):
    return {"metadata": {"name": name, "namespace": "heteroserve", "uid": name,
                         "labels": {lab.OWNER: "heteroserve-lab" if owner else "someone-else", "app.kubernetes.io/name": app,
                                    "heteroserve.io/pool": "awq-ascend910b-vllm"},
                         "ownerReferences": [{"kind": "ReplicaSet", "name": "workers"}]},
            "spec": {"nodeName": node, "containers": [{"resources": {"requests": {"huawei.com/Ascend910": "1", "cpu": "4", "memory": "16Gi"}}}]},
            "status": {"phase": "Running", "conditions": [{"type": "Ready", "status": "True"}]}}


def machine(name):
    return {"metadata": {"name": name, "labels": {"heteroserve.io/npu-ready": "true", "heteroserve.io/hardware": "ascend910b4", "heteroserve.io/backend": "vllm-ascend"}},
            "spec": {}, "status": {"conditions": [{"type": "Ready", "status": "True"}],
                                     "allocatable": {"huawei.com/Ascend910": "8", "cpu": "64", "memory": "256Gi", "pods": "32"}}}


def test_retirement_does_not_change_fixed_fingerprints(config):
    before = [lab.node_args(config, n) for n in config["nodes"]]
    config["retired_node_hosts"] = ["10.107.206.216"]
    config["authorized_additional_hosts"] = ["10.107.206.208"]
    assert before == [lab.node_args(config, n) for n in config["nodes"]]
    assert "10.107.206.216" not in {n["host"] for n in lab.active_nodes(config)}
    daemon = lab.network_tuning_object(config)
    assert "heteroserve-lab-216" not in str(daemon)


@pytest.mark.parametrize("host", ["10.107.206.200", "10.107.206.217"])
def test_rejects_foreign_host_and_control_plane(config, host):
    with pytest.raises(ValueError):
        nodes.find_node(config, host, True)


@pytest.mark.parametrize("owner,app", [(False, "expert"), (True, "gateway"), (True, "router")])
def test_never_evicts_foreign_or_entry_service(config, owner, app):
    with pytest.raises(RuntimeError, match="Migrate/review"):
        nodes.removable_workloads(config, nodes.read(nodes.POOL), "heteroserve-lab-216", [pod(owner=owner, app=app)])


@pytest.mark.parametrize("fault", ["physical_busy", "memory", "cpu", "taint", "not_ready"])
def test_capacity_must_be_usable_not_just_advertised(config, fault):
    pool, npu = nodes.read(nodes.POOL), nodes.read(nodes.NPU)
    name = "heteroserve-lab-211"
    pool["model_nodes"] = [name, "heteroserve-lab-216"]
    m = machine(name)
    free = 8
    if fault == "physical_busy": free = 0
    if fault == "memory": m["status"]["allocatable"]["memory"] = "1Gi"
    if fault == "cpu": m["status"]["allocatable"]["cpu"] = "500m"
    if fault == "taint": m["spec"]["taints"] = [{"effect": "NoSchedule"}]
    if fault == "not_ready": m["status"]["conditions"][0]["status"] = "False"
    with pytest.raises(RuntimeError, match="Insufficient"):
        nodes.removal_capacity(config, pool, npu, "heteroserve-lab-216", [m], [], {name: free})


def test_capacity_counts_all_other_consumers(config):
    pool, npu = nodes.read(nodes.POOL), nodes.read(nodes.NPU)
    name = "heteroserve-lab-211"
    pool["model_nodes"] = [name, "heteroserve-lab-216"]
    foreign = [pod(str(i), name, False, "other") for i in range(6)]
    with pytest.raises(RuntimeError, match="Insufficient"):
        nodes.removal_capacity(config, pool, npu, "heteroserve-lab-216", [machine(name)], foreign, {name: 8})


def test_candidate_cannot_match_service_or_router(config):
    pool, npu = nodes.read(nodes.POOL), nodes.read(nodes.NPU)
    cm, candidate = nodes.candidate_objects(config, pool, npu, "heteroserve-lab-216")
    assert candidate["metadata"]["labels"]["app.kubernetes.io/name"] == "expert-candidate"
    assert candidate["spec"]["nodeSelector"] == {"kubernetes.io/hostname": "heteroserve-lab-216"}
    assert candidate["spec"]["containers"][0]["resources"]["limits"] == {npu["resource"]: "1"}
    assert next(v for v in candidate["spec"]["volumes"] if v["name"] == "runtime")["configMap"]["name"] == cm["metadata"]["name"]


def test_drain_failure_never_stops_or_deletes_node(config, monkeypatch):
    node = nodes.find_node(config, "10.107.206.216")
    commands = []
    monkeypatch.setattr(nodes, "plan", lambda *a: {})
    def kube(c, argv, **kw):
        commands.append(argv)
        if argv[0] == "drain":
            assert "--force" not in argv and "--disable-eviction" not in argv
            raise RuntimeError("PDB blocks eviction")
    monkeypatch.setattr(nodes, "kubectl", kube)
    monkeypatch.setattr(nodes, "remote", lambda *a, **kw: pytest.fail("No container mutation on drain failure"))
    monkeypatch.setattr(nodes, "write", lambda *a: pytest.fail("No configuration mutation on drain failure"))
    with pytest.raises(RuntimeError, match="PDB"):
        nodes.remove(config, node, {}, lambda s: None)
    assert [c[0] for c in commands] == ["cordon", "drain"]


def test_inspection_transport_failure_is_not_missing_container(config, monkeypatch):
    node = nodes.find_node(config, "10.107.206.212", True)
    monkeypatch.setattr(nodes, "remote", lambda *a, **k: SimpleNamespace(returncode=255, stderr="Connection timed out"))
    with pytest.raises(RuntimeError, match="Cannot inspect"):
        nodes.inspect_container(config, node, True)


def test_candidate_collision_does_not_delete_existing(config, monkeypatch):
    monkeypatch.setattr(nodes, "kubectl", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("AlreadyExists")))
    monkeypatch.setattr(nodes, "get", lambda *a: pytest.fail("Must not delete pre-existing candidate"))
    with pytest.raises(RuntimeError, match="AlreadyExists"):
        nodes.validate_candidate(config, nodes.read(nodes.POOL), nodes.read(nodes.NPU), "heteroserve-lab-216")


def test_node_command_defaults_to_plan(monkeypatch):
    calls = []
    monkeypatch.setattr(nodes, "execute", lambda *a: calls.append(a))
    control.main(["node", "remove", "10.107.206.216"])
    assert calls == [("remove", "10.107.206.216", False)]


def test_os_lock_rejects_concurrent_mutation(tmp_path):
    with lab.operation_lock(tmp_path):
        with pytest.raises(OSError):
            with lab.operation_lock(tmp_path):
                pytest.fail("Overlapping operations acquired the same lock")


def test_retired_nodes_cannot_be_selected_by_regular_configuration(config):
    configs = control.read_configs()
    config["retired_node_hosts"] = ["10.107.206.216"]
    with pytest.raises(ValueError, match="eligible"):
        control.validate(configs, config)


def test_recovery_does_not_restart_an_unrecorded_pause(config, monkeypatch):
    node = nodes.find_node(config, "10.107.206.216")
    monkeypatch.setattr(nodes, "inspect_container", lambda *a: {"State": {"Running": False}})
    monkeypatch.setattr(nodes, "remote", lambda *a, **k: pytest.fail("Cannot restart unrecorded pause"))
    with pytest.raises(RuntimeError, match="Unrecorded"):
        nodes.recover(config, node, {"plan": {"rejoin_required": False}}, lambda s: None)


def test_recovery_before_retirement_uncordons_and_verifies(config, monkeypatch):
    node = nodes.find_node(config, "10.107.206.216")
    calls = []
    monkeypatch.setattr(nodes, "inspect_container", lambda *a: {"State": {"Running": True}})
    monkeypatch.setattr(nodes, "kubectl", lambda c, args: calls.append(args))
    monkeypatch.setattr(nodes, "verify_service", lambda: {"verified": True})
    report = {"plan": {"rejoin_required": False}}
    nodes.recover(config, node, report, lambda s: None)
    assert calls == [["uncordon", "heteroserve-lab-216"]]
    assert report["acceptance"]["verified"]


def test_retired_control_plane_rejected(config):
    config["retired_node_hosts"] = ["10.107.206.217"]
    with pytest.raises(ValueError, match="agents"):
        lab.validate_config(config)


def test_prepared_assets_do_not_query_or_transfer_images(config, monkeypatch):
    pool, npu = nodes.read(nodes.POOL), nodes.read(nodes.NPU)
    node = nodes.find_node(config, "10.107.206.216")
    def manifest(host, directory):
        return {"conversion_manifest.json": pool["weights_manifest_sha256"]} if directory == npu["model_source"] else {"adapter.py": "a" * 64}
    monkeypatch.setattr(nodes, "asset_manifest", manifest)
    def existing(host, argv, **kwargs):
        assert argv[0] == "test", "Image inventory/inspection is forbidden"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(nodes, "remote", existing)
    monkeypatch.setattr(nodes, "stream", lambda *a: pytest.fail("Existing environment must not be transferred"))
    commands = []
    monkeypatch.setattr(nodes, "run", lambda *a: commands.append(a))
    result = nodes.prepare_assets(config, node, npu, pool)
    assert result["image_preflight"] == "skipped_preprovisioned_environment"
    assert result["imported_images"] == []
    assert commands == [("manage_npu.py", "publish", "--host", node["host"])]


def test_preprovisioned_node_launch_never_inspects_or_pulls_images(config, monkeypatch):
    node = nodes.find_node(config, "10.107.206.212", True)
    calls = []
    monkeypatch.setattr(lab, "owned_resource", lambda *a: True)
    def remote(host, argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="")
        assert argv[:2] == ["docker", "run"], "No image inspection, pull or import allowed"
        assert "--pull=never" in argv and config["image_config_id"] in argv
        return SimpleNamespace(returncode=0, stdout="created")
    monkeypatch.setattr(lab, "remote", remote)
    lab.ensure_node(config, node, preprovisioned_images=True)
    assert len(calls) == 2
