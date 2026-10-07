"""Expert manifests must preserve allocation, asset locality, and graceful lifecycle."""
import importlib.util
from pathlib import Path
import sys

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
