import shutil

import pytest

import control


@pytest.fixture
def project(tmp_path):
    for relative in [*control.FILES.values(), "deploy/lab/cluster.json", "deploy/pools/defaults.json", "deploy/pools/ascend-gptq.json"]:
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
    menu_inputs(monkeypatch, project, ["2", "5", "3", "0", "5", "2", "y", "0", "0"])
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected deployment"))
    control.main([])
    assert control.read_configs(project)["pool"]["replicas"] == 3


def test_menu_abandon_edit_keeps_config(monkeypatch, project):
    before = control.read_configs(project)
    menu_inputs(monkeypatch, project, ["2", "5", "3", "0", "0"])
    control.main([])
    assert control.read_configs(project) == before


def test_menu_invalid_value_can_be_corrected(monkeypatch, project):
    menu_inputs(monkeypatch, project, ["2", "5", "-2", "2", "5", "3", "0", "5", "2", "y", "0", "0"])
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


def test_gptq_engine_changes_do_not_modify_awq_or_shared_runtime(monkeypatch, project):
    monkeypatch.setitem(control.FILES, "pool", "deploy/pools/ascend-gptq.json")
    awq = (project / "deploy/pools/ascend-awq.json").read_bytes()
    hardware = (project / "deploy/lab/npu.json").read_bytes()
    control.edit(project, "engine", ["model_parameters.max_num_seqs=3"], True)
    assert control.read_configs(project)["engine"]["model_parameters"]["max_num_seqs"] == 3
    assert (project / "deploy/pools/ascend-awq.json").read_bytes() == awq
    assert (project / "deploy/lab/npu.json").read_bytes() == hardware


def test_selected_pool_commands_use_its_config(monkeypatch):
    monkeypatch.setitem(control.FILES, "pool", "deploy/pools/ascend-gptq.json")
    calls = []
    monkeypatch.setattr(control.subprocess, "run", lambda args, **kw: calls.append(args))
    control.main(["apply", "pool"])
    assert len(calls) == 2
    assert all(args[-2:] == ["--config", str(control.ROOT / control.FILES["pool"])] for args in calls)


def test_pool_logs_do_not_include_other_expert(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setitem(control.FILES, "pool", "deploy/pools/ascend-gptq.json")
    commands = []
    monkeypatch.setattr(control, "kubectl", lambda c, args: (commands.append(args) or SimpleNamespace(stdout="logs")))
    control.main(["logs", "router"])
    assert "app.kubernetes.io/name=router,heteroserve.io/pool=gptq-ascend910b-vllm" in commands[0]


def test_draft_instance_parameters_are_kept_across_pages_without_deployment(monkeypatch, project):
    before = {p: (project / p).read_bytes() for p in control.FILES.values()}
    draft = control.Draft(project)
    ident = next(iter(draft.instances()))
    draft.set("pool", "engine.max_num_seqs", "3", ident)
    draft.set("pool", "traffic.max_inflight", "5", ident)
    assert draft.value("pool", "engine.max_num_seqs", ident) == 3
    assert draft.value("pool", "traffic.max_inflight", ident) == 5
    assert before == {p: (project / p).read_bytes() for p in control.FILES.values()}
    draft.save()
    pool = control.read_configs(project)["pool"]
    assert pool["instances"][ident]["engine"]["max_num_seqs"] == 3
    assert pool["instances"][ident]["traffic"]["max_inflight"] == 5


def test_failed_multi_file_save_restores_every_written_file(monkeypatch, project):
    draft = control.Draft(project)
    draft.set("gateway", "max_inflight", "24")
    ident = next(iter(draft.instances()))
    draft.set("pool", "engine.max_num_seqs", "3", ident)
    before = {p: (project / p).read_bytes() for p in control.FILES.values()}
    original = control.atomic_text
    failures = []
    def fail_second_config(path, text):
        if path == project / control.FILES["pool"] and not failures:
            failures.append(True)
            raise OSError("disk write failed")
        return original(path, text)
    monkeypatch.setattr(control, "atomic_text", fail_second_config)
    with pytest.raises(OSError, match="disk write"):
        draft.save()
    assert before == {p: (project / p).read_bytes() for p in control.FILES.values()}


def test_noop_save_keeps_rollback_record(project):
    draft = control.Draft(project)
    draft.set("gateway", "max_inflight", "24")
    draft.save()
    record = project / "artifacts/kubernetes/control/last-save.json"
    before = record.read_bytes()
    assert control.Draft(project).save() == []
    assert record.read_bytes() == before


def test_paused_pool_verification_does_not_wait_for_ready_router(monkeypatch, project):
    monkeypatch.setattr(control, "ROOT", project)
    draft = control.Draft(project)
    draft.resize(0)
    draft.save()
    commands = []
    monkeypatch.setattr(control.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    control.main(["verify", "pool"])
    assert len(commands) == 1 and commands[0][1].endswith("manage_pool.py")


def confirmed_menu(monkeypatch, project, answers):
    import control_menu
    menu_inputs(monkeypatch, project, answers)
    monkeypatch.setattr(control_menu, "ROOT", project)
    menu = control_menu.Menu()
    monkeypatch.setattr(menu, "publication_plan", lambda selected=None: None)
    return menu


def test_declining_confirmation_preserves_draft_and_running_system(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["n"])
    before = (project / control.FILES["gateway"]).read_bytes()
    menu.draft.set("gateway", "max_inflight", "24")
    monkeypatch.setattr(control, "main", lambda *a: pytest.fail("Unconfirmed update"))
    menu.confirm_changes(update=True)
    assert menu.draft.changes()
    assert (project / control.FILES["gateway"]).read_bytes() == before
    assert not control.publication_record(project).get("pending")


def test_confirm_then_cancel_update_is_recoverable_after_restart(monkeypatch, project):
    import control_menu
    menu = confirmed_menu(monkeypatch, project, ["y", "n"])
    menu.draft.set("gateway", "max_inflight", "24")
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    menu.confirm_changes(update=True)
    assert calls == [["apply", "gateway", "--plan"]]
    assert not menu.draft.changes()
    assert control_menu.Menu().saved == [control.FILES["gateway"]]


def test_sequential_confirmations_keep_all_pending_scopes(project):
    draft = control.Draft(project)
    draft.set("gateway", "max_inflight", "24")
    draft.save()
    draft.set("pool", "engine.max_num_seqs", "3", "awq-01")
    draft.save()
    draft.save()
    assert control.publication_record(project)["pending"] == [control.FILES["gateway"], control.FILES["pool"]]


def test_confirm_update_verifies_before_clearing_pending(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["y", "y"])
    menu.draft.set("gateway", "max_inflight", "24")
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    menu.confirm_changes(update=True)
    assert calls == [["apply", "gateway", "--plan"], ["apply", "gateway"], ["verify", "gateway"]]
    record = control.publication_record(project)
    assert record["pending"] == []
    assert record["publication"]["stage"] == "verified"


def test_update_failure_keeps_failed_and_unattempted_scopes(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["y"])
    menu.draft.set("gateway", "max_inflight", "24")
    menu.draft.set("pool", "engine.max_num_seqs", "3", "awq-01")
    menu.draft.save()
    calls = []
    def execute(args):
        calls.append(args)
        if args == ["verify", "gateway"]:
            raise RuntimeError("ordinary inference failed")
    monkeypatch.setattr(control, "main", execute)
    with pytest.raises(RuntimeError, match="inference failed"):
        menu.update_system()
    record = control.publication_record(project)
    assert record["pending"] == [control.FILES["gateway"], control.FILES["pool"]]
    assert record["publication"]["stage"] == "failed"
    assert "verifying" in record["publication"]["error"]
    assert ["apply", "pool"] not in calls


def test_changed_confirmed_file_cannot_be_published(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.draft.set("gateway", "max_inflight", "24")
    menu.draft.save()
    path = project / control.FILES["gateway"]
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(control, "main", lambda *a: pytest.fail("Stale confirmation executed"))
    with pytest.raises(RuntimeError, match="已确认配置发生变化"):
        menu.update_system()


def test_changed_configuration_after_plan_stops_before_apply(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.draft.set("gateway", "max_inflight", "24")
    menu.draft.save()
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    def confirm(prompt=""):
        path = project / "deploy/pools/defaults.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return "y"
    monkeypatch.setattr("builtins.input", confirm)
    with pytest.raises(RuntimeError, match="计划生成后配置发生变化"):
        menu.update_system()
    assert calls == [["apply", "gateway", "--plan"]]


def test_selected_instance_success_keeps_other_pool_changes_pending(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["y"])
    menu.draft.set("pool", "engine.max_num_seqs", "3", "awq-01")
    menu.draft.set("pool", "engine.max_num_seqs", "3", "awq-02")
    menu.draft.save()
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    menu.update_system([control.FILES["pool"]], "awq-01")
    assert calls == [["instance", "apply", "awq-01"], ["instance", "verify", "awq-01"]]
    assert control.publication_record(project)["pending"] == [control.FILES["pool"]]


def test_confirmation_summary_explains_instance_and_card_changes(project, capsys):
    draft = control.Draft(project)
    draft.resize(5)
    draft.set("pool", "parallelism.tp", "2", "awq-01")
    draft.summary()
    output = capsys.readouterr().out
    assert "启用实例数: 4 → 5" in output
    assert "配置申请卡数: 4 → 6" in output
    assert "instances.awq-01.parallelism.tp: 1 → 2" in output
    assert "instances.awq-05" in output


def test_parameter_panel_separates_running_saved_and_draft(monkeypatch, project, capsys):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.draft.set("pool", "engine.max_num_seqs", "4", "awq-01")
    rows = [{"instance": "awq-01", "node": "heteroserve-lab-209", "uid": "actual-pod",
             "physical_devices": [6], "ready": True, "observed_at": "now",
             "runtime_values": {"engine.max_num_seqs": 3}}]
    menu.parameter_panel({"engine.max_num_seqs": "运行序列数"}, "pool", "awq-01", rows)
    output = capsys.readouterr().out
    assert "运行值 | 已保存值 | 待修改值" in output
    assert "| 3 | 2 | 4 | 实例配置 / 待确认" in output
    assert "实际实例 awq-01" in output and "卡=[6]" in output


def test_unknown_runtime_is_never_replaced_with_saved_value(monkeypatch, project, capsys):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.parameter_panel({"engine.max_num_seqs": "运行序列数"}, "pool", "awq-01", [], "API unavailable")
    output = capsys.readouterr().out
    assert "无运行实例/未采集 | 2 | 2" in output
    assert "未核实" in output and "API unavailable" in output


def test_batch_edit_rolls_back_the_entire_line_on_invalid_assignment(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [])
    fields = {k: v for group in control.INSTANCE_GROUPS.values() for k, v in group.items()}
    menu.edit_fields("tp=2 pp=1 max_num_seqs=4", fields, "pool", "awq-01")
    assert menu.draft.value("pool", "parallelism.tp", "awq-01") == 2
    assert menu.draft.value("pool", "engine.max_num_seqs", "awq-01") == 4
    with pytest.raises(ValueError):
        menu.edit_fields("max_num_seqs=5 bogus=7", fields, "pool", "awq-01")
    assert menu.draft.value("pool", "engine.max_num_seqs", "awq-01") == 4


def test_editor_recovers_in_place_and_supports_batch_edits(monkeypatch, project, capsys):
    menu = confirmed_menu(monkeypatch, project, ["bad=3", "max_num_seqs=4 max_inflight=6", "0"])
    monkeypatch.setattr(menu, "live_instance", lambda ident: ([], "offline"))
    fields = {k: v for group in control.INSTANCE_GROUPS.values() for k, v in group.items()}
    menu.field_editor("instance", fields, "pool", "awq-01")
    assert menu.draft.value("pool", "engine.max_num_seqs", "awq-01") == 4
    assert menu.draft.value("pool", "traffic.max_inflight", "awq-01") == 6
    assert "仍在当前编辑页" in capsys.readouterr().out


def test_instance_override_can_return_to_inherited_default(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [])
    fields = {k: v for group in control.INSTANCE_GROUPS.values() for k, v in group.items()}
    menu.edit_fields("max_num_seqs=4", fields, "pool", "awq-01")
    menu.edit_fields("max_num_seqs=default", fields, "pool", "awq-01")
    assert menu.draft.value("pool", "engine.max_num_seqs", "awq-01") == 2
    assert "engine" not in menu.draft.instances()["awq-01"]
