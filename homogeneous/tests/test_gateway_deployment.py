from manage_gateway import config, objects
from manage_pool import configuration


def test_gateway_is_cpu_only_no_cluster_management_and_targets_pool_service():
    cluster, pool, npu = configuration()
    cfg = config()
    resources = objects(cluster, pool, npu, cfg)
    spec = next(r for r in resources if r["kind"] == "Deployment")["spec"]
    pod = spec["template"]["spec"]
    assert spec["replicas"] == 1 and spec["strategy"]["type"] == "Recreate"
    assert npu["resource"] not in pod["containers"][0]["resources"]["requests"]
    assert not pod["containers"][0]["resources"].get("limits")
    assert not any(v["name"] in {"driver","assets"} for v in pod["volumes"])
    assert pod["automountServiceAccountToken"] is False
    assert cfg["pools"]["awq"]["base_url"].endswith(".svc.cluster.local:8000/v1")
    assert cfg["pools"]["gptq"]["enabled"] is True
    assert cfg["pools"]["gptq"]["base_url"].endswith(".svc.cluster.local:8000/v1")
    assert cfg["migration_source"]["score_semantics"] == "expert_probabilities"
    assert not any(r["kind"] in {"Role", "ClusterRole"} for r in resources)
