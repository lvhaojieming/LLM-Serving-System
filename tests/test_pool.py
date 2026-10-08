"""Expert manifests must preserve allocation, asset locality, and graceful lifecycle."""
import importlib.util
from pathlib import Path
import sys
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("manage_pool", ROOT / "scripts/manage_pool.py")
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def test_pool_is_single_card_and_not_scheduled_where_weights_are_absent():
    cluster, settings, npu = pool.configuration()
    objects = pool.pool_objects(cluster, settings, npu)
    deploy = next(x for x in objects if x["kind"] == "Deployment")
    spec = deploy["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["resources"]["limits"] == {npu["resource"]: "1"}
    assert not spec.get("hostPID") and not spec.get("hostNetwork")
    assert not container["securityContext"].get("privileged")
    assert {"startupProbe", "readinessProbe", "livenessProbe", "lifecycle"} <= container.keys()
    mounts = {x["name"]: x for x in container["volumeMounts"]}
    assert all(mounts[k]["readOnly"] for k in ["driver", "weights", "adapter", "runtime"])
    allowed = spec["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert allowed == settings["model_nodes"]
    assert deploy["spec"]["strategy"]["rollingUpdate"] == {"maxSurge": 0, "maxUnavailable": 1}
    service = next(x for x in objects if x["kind"] == "Service")
    assert service["spec"]["selector"] == deploy["spec"]["selector"]["matchLabels"]
    assert spec["terminationGracePeriodSeconds"] > settings["drain_seconds"]


@pytest.mark.parametrize("seen,passes", [(["a", "b", "c", "d"], True), (["a", "b"], False), (["a", "b", "c", "foreign"], False)])
def test_routing_acceptance_requires_traffic_to_all_replicas(monkeypatch, seen, passes):
    import verify_lifecycle as lifecycle
    expected = ["a", "b", "c", "d"]
    monkeypatch.setattr(lifecycle, "wait_ready", lambda *args: [{"metadata": {"uid": uid}} for uid in expected])
    monkeypatch.setattr(lifecycle, "wait_discovered", lambda *args: {"endpoints": expected})
    def request(cluster, settings, code, expert, uids):
        assert set(json.loads(uids)) == set(expected)
        return json.dumps([{"uid": uid, "answer": "42"} for uid in seen])
    monkeypatch.setattr(lifecycle, "inside_router", request)
    if passes:
        assert lifecycle.routing({}, {"replicas": 4, "expert": "awq"}, {})["passed"]
    else:
        with pytest.raises(RuntimeError, match="every Ready worker"):
            lifecycle.routing({}, {"replicas": 4, "expert": "awq"}, {})


def test_model_acceptance_does_not_probe_old_ready_revision(monkeypatch):
    def incomplete_rollout(cluster, command, **kwargs):
        assert command[:2] == ["rollout", "status"]
        raise RuntimeError("rollout incomplete")
    monkeypatch.setattr(pool, "kubectl", incomplete_rollout)
    with pytest.raises(RuntimeError, match="rollout incomplete"):
        pool.verify()
