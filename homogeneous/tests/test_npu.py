"""NPU admission must account for driver reservations and restrict workload devices."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("manage_npu", ROOT / "scripts/manage_npu.py")
npu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(npu)


def test_exclusive_reservation_survives_idle_model_process():
    table = """ns_id 128 identify 0 root_tgid 1 dev_num 2 namespace abc, udev list:
logic_devid udevid
0 0
1 1

ns_id 0 identify 474713d061e6 root_tgid 2679173 dev_num 2 namespace def, udev list:
logic_devid udevid
0 3
1 6
"""
    assert npu.exclusive_reservations(table) == [{"namespace_id": 0,
        "container_identity": "474713d061e6", "root_pid": 2679173, "physical_npus": [3, 6]}]


def test_incomplete_driver_mapping_cannot_report_free_devices():
    with pytest.raises(RuntimeError, match="inconsistent"):
        npu.exclusive_reservations("ns_id 0 identify abcd root_tgid 45 dev_num 2\n0 1\n")


@pytest.mark.parametrize("table", ["", "unknown format"])
def test_unrecognized_driver_mapping_fails_closed(table):
    with pytest.raises(RuntimeError, match="Unrecognized"):
        npu.exclusive_reservations(table)


def test_workload_does_not_bypass_allocation_and_namespace_isolation():
    cluster = npu.load_config(ROOT / "deploy/lab/cluster.json")
    settings = json.loads((ROOT / "deploy/lab/npu.json").read_text())
    node = next(x for x in cluster["nodes"] if x["host"] == settings["host"])
    objects = npu.isolation_objects(cluster, settings, node)
    pods = [x for x in objects if x["kind"] == "Pod"]
    assert len(pods) == 2
    for pod in pods:
        spec = pod["spec"]
        assert not spec.get("hostPID") and not spec.get("hostNetwork")
        container = spec["containers"][0]
        assert not container["securityContext"].get("privileged")
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert container["resources"]["limits"][settings["resource"]] == "1"
        assert set(container["resources"]["limits"]) == {settings["resource"]}
        assert container["resources"]["requests"][settings["resource"]] == "1"
        assert all(v.get("hostPath", {}).get("path") != "/dev" for v in spec["volumes"])
        assert container["volumeMounts"][0]["readOnly"]
        environment = {x["name"]: x.get("value") for x in container["env"]}
        assert environment["HOME"] == "/tmp"
        assert environment["TE_PARALLEL_COMPILER"] == "1"
        assert "start-check" in container["args"][0]
    plugin = npu.plugin_objects(cluster, settings, node)["items"][-1]
    assert "-hotReset=-1" in plugin["spec"]["template"]["spec"]["containers"][0]["args"][0]


def test_resource_policy_keeps_npu_assignment_but_removes_experiment_caps():
    config = {"resource_limits_enabled": False}
    objects = [{"kind": "ResourceQuota"}, {"kind": "Pod", "spec": {"containers": [{"resources": {
        "requests": {"cpu": "1", "huawei.com/Ascend910": "1"},
        "limits": {"cpu": "2", "memory": "6Gi", "huawei.com/Ascend910": "1"}}}],
        "volumes": [{"emptyDir": {"sizeLimit": "256Mi"}}]}}]
    result = npu.resource_policy(config, objects)
    assert len(result) == 1
    assert result[0]["spec"]["containers"][0]["resources"]["limits"] == {"huawei.com/Ascend910": "1"}
    assert result[0]["spec"]["volumes"][0]["emptyDir"] == {}
    assert result[0]["spec"]["containers"][0]["resources"]["requests"]["cpu"] == "1"
    assert npu.resource_policy({"resource_limits_enabled": True}, objects) == objects


def test_cleanup_refuses_foreign_validation_pod(monkeypatch):
    monkeypatch.setattr(npu, "configuration", lambda: ({"name": "heteroserve-lab"}, {}, {}, "", ""))
    calls = []
    def kubectl(cluster, argv):
        calls.append(argv)
        return SimpleNamespace(stdout=json.dumps({"items": [{"metadata": {
            "name": "npu-isolation-0", "labels": {npu.OWNER: "someone-else"}}}]}))
    monkeypatch.setattr(npu, "kubectl", kubectl)
    with pytest.raises(RuntimeError, match="ownership"):
        npu.cleanup()
    assert all("delete" not in argv for argv in calls)
