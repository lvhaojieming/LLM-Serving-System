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
    import control_menu
    monkeypatch.setattr(control, "ROOT", project)
    monkeypatch.setattr(control_menu.Menu, "refresh_resources", lambda self: None)
    monkeypatch.setattr(control_menu.Menu, "command", lambda self, args, log_path: control.main(args))
    values = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(values))


def test_default_menu_save_only_never_deploys(monkeypatch, project):
    menu_inputs(monkeypatch, project, ["2", "5", "3", "0", "5", "2", "y", "0", "0"])
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected deployment"))
    control.main([])
    assert control.read_configs(project)["pool"]["replicas"] == 3


def test_menu_abandon_edit_keeps_config(monkeypatch, project):
    before = control.read_configs(project)
    menu_inputs(monkeypatch, project, ["2", "5", "3", "0", "0", "3"])
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
    import control_menu
    before = control.read_configs(project)
    monkeypatch.setattr(control, "ROOT", project)
    monkeypatch.setattr(control_menu.Menu, "refresh_resources", lambda self: None)
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


@pytest.mark.parametrize("shortcut", ["7", "c"])
def test_main_menu_can_confirm_new_instance_without_publication_submenu(monkeypatch, project, shortcut):
    menu_inputs(monkeypatch, project, ["2", "5", "5", "0", shortcut, "y", "0"])
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **kw: pytest.fail("Save must not deploy"))
    control.main([])
    pool = control.read_configs(project)["pool"]
    assert pool["replicas"] == 5 and "awq-05" in pool["instances"]
    assert control.publication_record(project)["pending"] == [control.FILES["pool"]]


def test_main_menu_update_shortcut_saves_applies_and_verifies(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["2", "5", "5", "0", "u", "y", "y", "0"])
    calls = []
    monkeypatch.setattr(control, "main", lambda args: calls.append(args))
    menu.run()
    assert calls == [["apply", "pool"], ["verify", "pool"]]
    assert control.read_configs(project)["pool"]["replicas"] == 5
    assert control.publication_record(project)["pending"] == []


def test_exit_with_draft_defaults_to_continue_editing(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [""])
    before = (project / control.FILES["pool"]).read_bytes()
    menu.draft.resize(5)
    assert menu.exit_menu() is False
    assert "awq-05" in menu.draft.instances() and menu.draft.changes()
    assert (project / control.FILES["pool"]).read_bytes() == before


def test_exit_can_save_new_instance_without_deploying(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["1", "y"])
    menu.draft.resize(5)
    monkeypatch.setattr(control, "main", lambda *a: pytest.fail("Save must not deploy"))
    assert menu.exit_menu() is True
    assert not menu.draft.changes()
    assert "awq-05" in control.read_configs(project)["pool"]["instances"]


def test_exit_cancelled_confirmation_preserves_draft(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["1", "n"])
    menu.draft.resize(5)
    assert menu.exit_menu() is False
    assert menu.draft.changes() and not control.publication_record(project).get("pending")


def test_exit_cancelled_update_preserves_confirmed_pending_work(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["2", "y", "n"])
    menu.draft.resize(5)
    monkeypatch.setattr(control, "main", lambda *a: pytest.fail("Cancelled update must not deploy"))
    assert menu.exit_menu() is False
    assert not menu.draft.changes()
    assert control.publication_record(project)["pending"] == [control.FILES["pool"]]


def test_persistent_draft_survives_restart_without_changing_saved_config(project):
    path = project / control.FILES["pool"]
    before = path.read_bytes()
    draft = control.Draft(project)
    draft.restore()
    draft.resize(5)
    draft.checkpoint({"pool": control.FILES["pool"], "selected": "awq-05"})
    recovered = control.Draft(project)
    assert recovered.restore()
    assert "awq-05" in recovered.instances()
    assert recovered.restored_context["selected"] == "awq-05"
    assert path.read_bytes() == before
    recovered.save()
    assert not recovered.checkpoint_path.exists()


def test_checkpoint_never_overwrites_another_session(project):
    first, second = control.Draft(project), control.Draft(project)
    first.restore(); second.restore()
    first.resize(5); first.checkpoint()
    second.resize(6)
    with pytest.raises(RuntimeError, match="另一个控制会话"):
        second.checkpoint()
    recovered = control.Draft(project); recovered.restore()
    assert recovered.pool()["replicas"] == 5
    import json
    backup = project / "artifacts/kubernetes/control/conflicts" / (second.session_id + ".json")
    assert len(json.loads(backup.read_text(encoding="utf-8"))["files"][control.FILES["pool"]]["after"]["instances"]) == 6


def test_recovered_draft_rejects_changed_source_and_preserves_both(project):
    draft = control.Draft(project); draft.restore()
    draft.resize(5); draft.checkpoint()
    path = project / control.FILES["pool"]
    changed = path.read_text(encoding="utf-8") + "\n"
    path.write_text(changed, encoding="utf-8")
    recovered = control.Draft(project); recovered.restore()
    with pytest.raises(RuntimeError, match="配置已被其他操作"):
        recovered.save()
    assert path.read_text(encoding="utf-8") == changed
    assert recovered.checkpoint_path.exists()


def test_eof_preserves_working_copy_for_next_launch(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.draft.resize(5)
    monkeypatch.setattr("builtins.input", lambda *a: (_ for _ in ()).throw(EOFError()))
    menu.run()
    recovered = control.Draft(project); recovered.restore()
    assert "awq-05" in recovered.instances()


def test_home_resource_id_opens_correct_expert_without_pool_submenu(monkeypatch, project):
    monkeypatch.setitem(control.FILES, "pool", "deploy/pools/ascend-awq.json")
    menu = confirmed_menu(monkeypatch, project, ["gptq-01", "0"])
    opened = []
    monkeypatch.setattr(menu, "field_editor", lambda title, fields, section, ident: opened.append(ident))
    menu.run()
    assert opened == ["gptq-01"] and control.FILES["pool"].endswith("ascend-gptq.json")


def test_resource_filter_and_api_failure_are_explicit(monkeypatch, project, capsys):
    menu = confirmed_menu(monkeypatch, project, [])
    menu.resource_filter = "gptq"
    menu.resources()
    output = capsys.readouterr().out
    assert "gptq/gptq-01" in output and "awq/awq-01" not in output
    assert "未采集" in output and "采集不可用" in output


def test_new_instance_wizard_sets_parallelism_and_pp_budget_in_one_flow(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["1", "", "1", "2", "2"])
    monkeypatch.setattr(menu, "field_editor", lambda *a: None)
    menu.create_instance()
    spec = menu.draft.instances()["awq-05"]
    assert spec["parallelism"] == {"tp": 2, "pp": 2}
    assert spec["engine"]["max_num_batched_tokens"] == 4096
    recovered = control.Draft(project); recovered.restore()
    assert recovered.instances()["awq-05"] == spec


def test_publish_records_failed_stage_and_keeps_pending_scope(monkeypatch, project):
    menu = confirmed_menu(monkeypatch, project, ["y"])
    menu.draft.resize(5); menu.draft.save()
    monkeypatch.setattr(control, "main", lambda *a: (_ for _ in ()).throw(RuntimeError("backend failed")))
    with pytest.raises(RuntimeError, match="backend failed"):
        menu.update_system()
    import json
    jobs = list((project / "artifacts/kubernetes/control/jobs").glob("*.json"))
    record = json.loads(jobs[0].read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["stage"] == "applying"
    assert control.publication_record(project)["pending"] == [control.FILES["pool"]]


@pytest.mark.parametrize("operation,returncode", [("status", 0), ("fail", 3)])
def test_controller_command_captures_details_in_log(monkeypatch, project, operation, returncode, capsys):
    import control_menu
    original = control_menu.Menu.command
    menu = confirmed_menu(monkeypatch, project, [])
    script = project / "scripts/control.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("import sys\nprint('DETAIL_ONLY_IN_LOG')\nsys.exit(3 if 'fail' in sys.argv else 0)\n", encoding="utf-8")
    log = project / "artifacts/kubernetes/control/jobs/command.log"
    if returncode:
        with pytest.raises(control.subprocess.CalledProcessError):
            original(menu, [operation], log)
    else:
        original(menu, [operation], log)
        assert "DETAIL_ONLY_IN_LOG" not in capsys.readouterr().out
    assert "DETAIL_ONLY_IN_LOG" in log.read_text(encoding="utf-8")
