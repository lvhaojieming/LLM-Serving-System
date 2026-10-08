import shutil

import pytest

import control


@pytest.fixture
def project(tmp_path):
    for relative in [*control.FILES.values(), "deploy/lab/cluster.json"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(control.ROOT / relative, target)
    return tmp_path


def test_preview_never_writes(project):
    path = project / control.FILES["pool"]
    before = path.read_bytes()
    control.edit(project, "pool", ["replicas=3"])
    assert path.read_bytes() == before


def test_batch_edit_validates_final_state_and_preserves_other_keys(project):
    before = control.read_configs(project)
    control.edit(project, "pool", ["drain_seconds=400", "termination_seconds=450"], True)
    after = control.read_configs(project)
    before["pool"].update(drain_seconds=400, termination_seconds=450)
    assert after == before
    assert not list((project / "deploy/pools").glob(".control-*"))


@pytest.mark.parametrize("section,assignments", [
    ("pool", ["replicas=3", "model_nodes=heteroserve-lab-217"]),
    ("pool", ["router_replicas=2"]),
    ("pool", ["replicas=true"]),
    ("pool", ["drain_seconds=290"]),
    ("gateway", ["deployment.node=unknown"]),
    ("engine", ["model_parameters.gpu_memory_utilization=NaN"]),
    ("engine", ["model_parameters.enforce_eager=1"]),
])
def test_invalid_changes_leave_all_files_untouched(project, section, assignments):
    before = {p: (project / p).read_bytes() for p in control.FILES.values()}
    with pytest.raises(ValueError):
        control.edit(project, section, assignments, True)
    assert before == {p: (project / p).read_bytes() for p in control.FILES.values()}


def test_apply_pool_updates_both_sides_of_admission():
    assert control.commands("apply", "pool") == [
        ("manage_pool.py", "apply"), ("manage_pool.py", "router")]


def test_plan_never_executes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Plan executed a command")
    monkeypatch.setattr(control.subprocess, "run", forbidden)
    control.main(["apply", "all", "--plan"])


def test_apply_stops_on_first_failure(monkeypatch):
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise control.subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(control.subprocess, "run", fail)
    with pytest.raises(control.subprocess.CalledProcessError):
        control.main(["apply", "all"])
    assert len(calls) == 1


def menu_inputs(monkeypatch, project, answers):
    monkeypatch.setattr(control, "ROOT", project)
    values = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(values))


def test_default_menu_save_only_never_deploys(monkeypatch, project):
    menu_inputs(monkeypatch, project, ["2", "1", "3", "s", "y", "0", "0"])
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected deployment"))
    control.main([])
    assert control.read_configs(project)["pool"]["replicas"] == 3


def test_menu_abandon_edit_keeps_config(monkeypatch, project):
    before = control.read_configs(project)
    menu_inputs(monkeypatch, project, ["2", "1", "3", "0", "0"])
    control.main([])
    assert control.read_configs(project) == before


def test_menu_invalid_value_can_be_corrected(monkeypatch, project):
    menu_inputs(monkeypatch, project, ["2", "1", "-2", "s", "1", "3", "s", "y", "0", "0"])
    control.main([])
    assert control.read_configs(project)["pool"]["replicas"] == 3


def test_menu_apply_requires_explicit_execution_and_verifies(monkeypatch):
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    control.run_interactive("apply", "pool")
    assert calls == [["apply", "pool", "--plan"]]
    calls.clear()
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    control.run_interactive("apply", "pool")
    assert calls == [["apply", "pool", "--plan"], ["apply", "pool"], ["verify", "pool"]]


def test_menu_stale_preview_cannot_overwrite_changed_config(project):
    before = control.read_configs(project)
    control.edit(project, "gateway", ["max_inflight=20"], True)
    with pytest.raises(RuntimeError, match="配置已被其他操作修改"):
        control.edit(project, "pool", ["replicas=3"], True, expected=before)
    assert control.read_configs(project)["pool"]["replicas"] == before["pool"]["replicas"]


def test_menu_end_of_input_exits_without_changes(monkeypatch, project):
    before = control.read_configs(project)
    monkeypatch.setattr(control, "ROOT", project)
    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    control.main([])
    assert control.read_configs(project) == before
