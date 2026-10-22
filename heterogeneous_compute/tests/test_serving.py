import copy
import json
from pathlib import Path
from heterogeneous_compute.serving import fingerprint, gateway_config, live, local


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/serving.json").read_text())


def test_gateway_aliases_keep_embedding_separate_from_generation_pool():
    cfg = gateway_config(config())
    assert cfg["learned_routing"] == {"enabled": False}
    assert cfg["pools"]["embedding"]["task"] == "embed"
    assert cfg["pools"]["awq"]["base_url"] == "http://127.0.0.1:18601/v1"


def test_unrelated_service_change_does_not_invalidate_gateway_identity():
    cfg = config()
    changed = copy.deepcopy(cfg)
    changed["services"]["ascend-awq"]["cudagraph_metrics"] = False
    assert fingerprint(cfg, "gateway") == fingerprint(changed, "gateway")
    changed["backends"]["ascend-awq"]["base_url"] = "http://192.168.162.24:18428/v1"
    assert fingerprint(cfg, "gateway") != fingerprint(changed, "gateway")
    assert fingerprint(cfg, "router") != fingerprint(changed, "router")


def test_stop_does_not_signal_a_reused_or_foreign_pid(tmp_path, monkeypatch):
    import os
    import heterogeneous_compute.serving as serving
    cfg = config()
    cfg["result_dir"] = str(tmp_path)
    marker = tmp_path / "embedding/process.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps({"pid": os.getpid(), "stamp": ["foreign-boot", "0"]}))
    monkeypatch.setattr(os, "killpg", lambda *args: (_ for _ in ()).throw(AssertionError("foreign PID signalled")))
    assert not live(marker)
    local(cfg, "stop", "embedding")
