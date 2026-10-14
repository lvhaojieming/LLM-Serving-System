"""Render and validate deployment files without changing cluster state."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml
from jsonschema import Draft4Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT4

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SHA256 = "cf6e61a06b9603b0332a5228209d708a474399ad4bbd56ff3397f4a0fd61fb92"


def command(argv, stdin=None):
    result = subprocess.run(argv, input=stdin, text=True, encoding="utf-8", capture_output=True, timeout=30)
    if result.returncode:
        # These commands never request Secret values or raw credential configuration.
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def schema_key(obj):
    api, kind = obj["apiVersion"], obj["kind"]
    group, version = api.split("/") if "/" in api else ("core", api)
    group = {"rbac.authorization.k8s.io": "rbac", "networking.k8s.io": "networking"}.get(group, group)
    return f"io.k8s.api.{group}.{version}.{kind}"


def validate_objects(objects, openapi):
    # OpenAPI v2 represents the Kubernetes custom IntOrString scalar as a string.
    # Its Go decoder also accepts integers; JSON Schema does not know this extension.
    openapi = deepcopy(openapi)
    scalar = "io.k8s.apimachinery.pkg.util.intstr.IntOrString"
    openapi["definitions"][scalar] = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    registry = Registry().with_resource("urn:kubernetes:openapi", Resource.from_contents(openapi, default_specification=DRAFT4))
    names = set()
    for obj in objects:
        key = schema_key(obj)
        if key not in openapi["definitions"]:
            raise ValueError(f"Unsupported Kubernetes schema: {key}")
        schema = {"$ref": "urn:kubernetes:openapi#/definitions/" + key}
        errors = list(Draft4Validator(schema, registry=registry).iter_errors(obj))
        if errors:
            raise ValueError(f"{obj.get('kind')}: {errors[0].message}")
        meta = obj["metadata"]
        identity = (obj["apiVersion"], obj["kind"], meta.get("namespace"), meta["name"])
        if identity in names:
            raise ValueError(f"Duplicate object: {identity}")
        names.add(identity)
        if obj["kind"] != "Namespace" and meta.get("namespace") != "heteroserve":
            raise ValueError("Deployment objects must remain in the project namespace")
        if obj["kind"] == "Namespace" and meta["name"] != "heteroserve":
            raise ValueError("Unexpected namespace")
        if obj["kind"] in {"Role", "RoleBinding"}:
            if obj["kind"] == "Role":
                for rule in obj["rules"]:
                    if any("*" in rule.get(field, []) for field in ("resources", "verbs", "apiGroups")):
                        raise ValueError("Wildcard permissions are forbidden")
                    if meta["name"] == "heteroserve-discovery":
                        if set(rule["verbs"]) - {"get", "list", "watch"} or "secrets" in rule["resources"]:
                            raise ValueError("Discovery must be read-only and cannot read secrets")
            continue
        if obj["kind"] != "Job":
            continue
        if obj["spec"].get("suspend") is not True:
            raise ValueError("Validation jobs must remain suspended in the checked-in baseline")
        spec = obj["spec"]["template"]["spec"]
        if spec.get("hostNetwork") or spec.get("hostPID") or spec.get("hostIPC") or spec.get("nodeName"):
            raise ValueError("Do not bypass scheduling or use host namespaces")
        if spec.get("automountServiceAccountToken") is not False:
            raise ValueError("NPU test must not have Kubernetes credentials")
        if spec.get("tolerations"):
            raise ValueError("Baseline must not bypass node taints")
        if spec.get("nodeSelector", {}).get("kubernetes.io/hostname") != "nh-dc-nm130-h06-20u-server10":
            raise ValueError("Validation must target the user-selected .217 node")
        if any("hostPath" in v for v in spec.get("volumes", [])):
            raise ValueError("Device/driver injection must come from a verified runtime, not all-device hostPath")
        for container in spec["containers"]:
            if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", container["image"]):
                raise ValueError("Images must use immutable digests")
            if container.get("securityContext", {}).get("privileged"):
                raise ValueError("Privileged test containers are forbidden")
            if any(e["name"] in {"ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
                   for e in container.get("env", [])):
                raise ValueError("Do not override Device Plugin device assignments")
            resources = container["resources"]
            if str(resources["requests"].get("huawei.com/Ascend910")) != "1" or str(resources["limits"].get("huawei.com/Ascend910")) != "1":
                raise ValueError("NPU requests/limits must be the same integer, exactly one")
    return len(objects)


def cluster_check(kubeconfig, context, rendered):
    if not context:
        raise ValueError("An explicit --context is required for live checks")
    kubectl = ["kubectl", "--kubeconfig", str(kubeconfig), "--context", context, "--request-timeout=5s"]
    version = json.loads(command(kubectl + ["get", "--raw", "/version"]))["gitVersion"]
    node = json.loads(command(kubectl + ["get", "node", "nh-dc-nm130-h06-20u-server10", "-o", "json"]))
    failures = []
    if node["spec"].get("unschedulable"):
        failures.append(".217 is cordoned")
    if not any(c["type"] == "Ready" and c["status"] == "True" for c in node["status"]["conditions"]):
        failures.append(".217 is not Ready")
    if int(node["status"]["allocatable"].get("huawei.com/Ascend910", "0")) < 1:
        failures.append(".217 does not advertise the candidate NPU resource")
    if any(t["effect"] in {"NoSchedule", "NoExecute"} for t in node["spec"].get("taints", [])):
        failures.append(".217 has an unhandled scheduling taint")
    for resource in ("jobs.batch", "deployments.apps", "services", "configmaps", "networkpolicies.networking.k8s.io"):
        if command(kubectl + ["auth", "can-i", "create", resource, "-n", "heteroserve"]).strip() != "yes":
            failures.append(f"Missing project create permission: {resource}")
    if failures:
        raise ValueError("; ".join(failures))
    command(kubectl + ["get", "namespace", "heteroserve", "-o", "name"])
    command(kubectl + ["get", "runtimeclass", "ascend", "-o", "name"])
    # API validation is a separate gate; it creates no persisted workloads.
    command(kubectl + ["apply", "--dry-run=server", "-f", "-"], stdin=rendered)
    return {"server_version": version, "server_dry_run": "passed", "runtime_execution": "not_tested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=ROOT / "artifacts/kubernetes/openapi-v1.28.2.json")
    parser.add_argument("--overlay", type=Path, default=ROOT / "deploy/k8s/validation")
    parser.add_argument("--kubeconfig", type=Path)
    parser.add_argument("--context")
    args = parser.parse_args()
    try:
        schema_bytes = args.schema.read_bytes()
        if hashlib.sha256(schema_bytes).hexdigest() != SCHEMA_SHA256:
            raise ValueError("OpenAPI schema checksum mismatch; use the pinned official v1.28.2 schema")
        rendered = command(["kubectl", "kustomize", str(args.overlay)])
        count = validate_objects(list(yaml.safe_load_all(rendered)), json.loads(schema_bytes))
        report = {"offline_schema": "passed", "objects": count,
                  "schema_version": "v1.28.2", "deployable": "not_confirmed"}
        if args.kubeconfig:
            report.update(cluster_check(args.kubeconfig, args.context, rendered))
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
