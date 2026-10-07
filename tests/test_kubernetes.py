"""Safety regressions for the real deployment baseline, independent of any cluster."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_kubernetes", ROOT / "scripts/check_kubernetes.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.fixture(scope="module")
def rendered():
    text = subprocess.run(["kubectl", "kustomize", str(ROOT / "deploy/k8s/validation")],
                          capture_output=True, text=True, check=True).stdout
    schema_path = ROOT / "artifacts/kubernetes/openapi-v1.28.2.json"
    return list(yaml.safe_load_all(text)), json.loads(schema_path.read_text(encoding="utf-8"))


def test_official_api_schema_and_suspended_single_device(rendered):
    objects, schema = rendered
    assert checks.validate_objects(objects, schema) == 8


@pytest.mark.parametrize("failure", ["active", "privileged", "cordon_bypass", "device_override", "all_devices", "mutable_image", "device_mismatch", "wrong_namespace", "wrong_api"])
def test_rejects_unsafe_or_invalid_deployment(rendered, failure):
    objects, schema = deepcopy(rendered)
    job = next(o for o in objects if o["kind"] == "Job")
    pod = job["spec"]["template"]["spec"]
    container = pod["containers"][0]
    if failure == "active":
        job["spec"]["suspend"] = False
    elif failure == "privileged":
        container["securityContext"]["privileged"] = True
    elif failure == "cordon_bypass":
        pod["nodeName"] = "nh-dc-nm130-h06-20u-server10"
    elif failure == "device_override":
        container["env"].append({"name": "ASCEND_VISIBLE_DEVICES", "value": "0,1,2,3,4,5,6,7"})
    elif failure == "all_devices":
        pod["volumes"] = [{"name": "devices", "hostPath": {"path": "/dev"}}]
    elif failure == "mutable_image":
        container["image"] = "vllm-ascend:latest"
    elif failure == "device_mismatch":
        container["resources"]["limits"]["huawei.com/Ascend910"] = 8
    elif failure == "wrong_namespace":
        job["metadata"]["namespace"] = "kube-system"
    elif failure == "wrong_api":
        job["spec"]["suspend"] = "false"
    with pytest.raises(ValueError):
        checks.validate_objects(objects, schema)


def test_discovery_cannot_read_credentials_or_change_cluster(rendered):
    objects, schema = deepcopy(rendered)
    role = next(o for o in objects if o["kind"] == "Role")
    role["rules"][0]["resources"].append("secrets")
    with pytest.raises(ValueError, match="Discovery"):
        checks.validate_objects(objects, schema)


def test_requires_explicit_context_for_live_check():
    with pytest.raises(ValueError, match="context"):
        checks.cluster_check(Path("private-config"), None, "")
