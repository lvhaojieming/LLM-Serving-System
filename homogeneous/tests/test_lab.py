"""Lab creation must be scoped, bounded, and cannot undo paused services."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manage_lab", ROOT / "scripts/manage_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


@pytest.fixture
def config():
    return lab.load_config(ROOT / "deploy/lab/cluster.json")


def test_rejects_mistyped_locked_container_id(config, tmp_path):
    config["locked_container_ids"][config["nodes"][0]["host"]] += "4"
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="64 hexadecimal"):
        lab.load_config(path)


def test_no_host_kubelet_or_device_mounts_and_bounded_resources(config):
    for node in config["nodes"]:
        argv = lab.node_args(config, node)
        assert "--network=host" not in argv and "--pid=host" not in argv
        assert not any(a.startswith(("/etc/kubernetes", "/var/lib/kubelet", "/dev/", "/var/run/docker.sock")) for a in argv)
        assert "--cpus" not in argv and "--memory" not in argv and "--pids-limit" not in argv
        assert "protect-kernel-defaults=true" in argv
        assert "--token" not in argv and "K3S_TOKEN" not in str(argv)


def test_refuses_unowned_container(config, monkeypatch):
    def remote(host, argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout=config["locked_container_ids"][host] if argv[-1] == "{{.Id}}" else '{"other-owner":"someone-else"}')
    monkeypatch.setattr(lab, "remote", remote)
    with pytest.raises(RuntimeError, match="ownership"):
        lab.ensure_node(config, config["nodes"][0])


def test_does_not_restart_a_stopped_lab_node(config, monkeypatch):
    node = config["nodes"][0]
    args = lab.node_args(config, node)
    label = next(a for a in args if a.startswith(lab.OWNER + ".config="))
    labels = {lab.OWNER: config["name"], lab.OWNER + ".config": label.split("=", 1)[1]}
    calls = []
    def remote(host, argv, **kwargs):
        calls.append(argv)
        if argv[-1] == "{{.Id}}":
            return SimpleNamespace(returncode=0, stdout=config["locked_container_ids"][host])
        return SimpleNamespace(returncode=0, stdout=json.dumps(labels) if ".Config.Labels" in argv[-1] else "exited")
    monkeypatch.setattr(lab, "remote", remote)
    with pytest.raises(RuntimeError, match="intentional pause"):
        lab.ensure_node(config, node)
    assert all("start" not in a and "run" not in a for a in calls)


def test_fixed_container_cannot_be_recreated(config, monkeypatch):
    calls = []
    def remote(host, argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr(lab, "remote", remote)
    with pytest.raises(RuntimeError, match="refusing to recreate"):
        lab.ensure_node(config, config["nodes"][0])
    assert all("run" not in a and "start" not in a for a in calls)


def test_network_helper_is_scoped_and_has_no_host_files_or_credentials(config):
    daemon = lab.network_tuning_object(config)
    pod = daemon["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert not pod.get("hostPID") and not pod.get("volumes")
    assert pod["containers"][0]["securityContext"]["capabilities"] == {"drop": ["ALL"], "add": ["NET_ADMIN"]}
    assert not pod["containers"][0]["securityContext"].get("privileged", False)
    nodes = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
    assert set(nodes) == {lab.names(config, node)[0] for node in config["nodes"]}


def test_network_helper_refuses_physical_host_network(config, monkeypatch):
    def remote(*args, **kwargs):
        return SimpleNamespace(stdout=json.dumps({"labels": {lab.OWNER: config["name"]}, "network": "host"}), returncode=0)
    monkeypatch.setattr(lab, "remote", remote)
    with pytest.raises(RuntimeError, match="private lab"):
        lab.apply_network_tuning(config)


@pytest.mark.parametrize("failure", ["unauthorized", "overlap", "mutable_image", "duplicate"])
def test_invalid_inventory_rejected(config, tmp_path, failure):
    edited = deepcopy(config)
    if failure == "unauthorized":
        edited["nodes"][1]["host"] = "10.107.206.200"
    elif failure == "overlap":
        edited["service_cidr"] = edited["pod_cidr"]
    elif failure == "mutable_image":
        edited["image"] = "rancher/k3s:latest"
    elif failure == "duplicate":
        edited["nodes"][1]["host"] = edited["nodes"][0]["host"]
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError):
        lab.load_config(path)
