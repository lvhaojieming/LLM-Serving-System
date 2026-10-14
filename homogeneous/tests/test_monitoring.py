"""Monitoring must be pinned, durable, bounded, and exclude secret access."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("manage_monitoring", ROOT / "scripts/manage_monitoring.py")
monitoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitoring)


def test_one_stack_has_pinned_images_retained_storage_and_bounded_retention():
    cluster = monitoring.load_config(ROOT / "deploy/lab/cluster.json")
    objects = monitoring.objects(cluster)
    deployments = [o for o in objects if o["kind"] == "Deployment"]
    assert len(deployments) == 3
    for deployment in deployments:
        pod = deployment["spec"]["template"]["spec"]
        container = pod["containers"][0]
        assert "@sha256:" in container["image"]
        assert pod["securityContext"]["runAsNonRoot"]
        assert not container.get("securityContext", {}).get("privileged")
    prometheus = next(o for o in deployments if o["metadata"]["name"] == "prometheus")
    args = prometheus["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--storage.tsdb.retention.time=7d" in args and "--storage.tsdb.retention.size=5GB" in args
    assert all(o["spec"]["persistentVolumeReclaimPolicy"] == "Retain" for o in objects if o["kind"] == "PersistentVolume")
    assert all("secrets" not in rule["resources"] for o in objects if o["kind"] == "ClusterRole" for rule in o["rules"])
    assert all(o["kind"] != "Secret" for o in objects)
    config = next(o for o in objects if o["kind"] == "ConfigMap")
    assert len(json.loads(config["data"]["dashboard.json"])["panels"]) == 8
