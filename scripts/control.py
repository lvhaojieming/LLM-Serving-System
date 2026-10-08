"""Unified operator CLI. Existing JSON files and managers remain authoritative."""
import argparse
import copy
import difflib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from contextlib import nullcontext

from manage_lab import ROOT, kubectl, load_config, names, active_nodes, operation_lock
from manage_pool import load_pool, resolve_runtime

FILES = {"gateway": "deploy/gateway.json", "pool": "deploy/pools/ascend-awq.json",
         "engine": "deploy/lab/npu.json"}
# Only expose parameters whose deployment effects are understood. Identity,
# credentials, device allocation and container ownership remain protected.
FIELDS = {
    "gateway": {"max_inflight": "入口并发预算", "max_request_bytes": "请求体字节上限",
                "timeout_seconds": "上游读取超时秒数", "default_max_tokens": "默认输出长度",
                "deployment.node": "Gateway 节点（迁移前 prepare）",
                "pools.awq.enabled": "启用 AWQ 专家入口", "pools.gptq.enabled": "启用 GPTQ 专家入口"},
    "pool": {"replicas": "模型副本数，每副本一张卡", "model_nodes": "模型候选节点",
             "router_nodes": "二级 Router 候选节点（迁移前 prepare）",
             "requests_per_worker": "每实例请求预算，同时更新模型和 Router",
             "startup_seconds": "模型启动预算秒数", "drain_seconds": "排空预算秒数",
             "termination_seconds": "Pod 退出预算秒数",
             "discovery_cache_seconds": "发现缓存有效秒数"},
    "engine": {"model_parameters.max_model_len": "上下文长度",
               "model_parameters.max_num_seqs": "引擎并发序列数",
               "model_parameters.max_num_batched_tokens": "批处理 token 预算",
               "model_parameters.gpu_memory_utilization": "设备内存使用比例",
               "model_parameters.enforce_eager": "启用 eager 模式"},
}


def read_configs(root=None):
    root = ROOT if root is None else root
    pool = load_pool(root / FILES["pool"])
    return {"gateway": json.loads((root / FILES["gateway"]).read_text(encoding="utf-8")),
            "pool": pool, "engine": resolve_runtime(pool, root)}


def available_pools(root=None):
    root = ROOT if root is None else root
    result = {}
    for path in sorted((root / "deploy/pools").glob("ascend-*.json")):
        pool = load_pool(path)
        if pool["expert"] in result:
            raise ValueError("Duplicate expert configuration")
        result[pool["expert"]] = str(path.relative_to(root)).replace("\\", "/")
    return result


def select_pool(expert):
    choices = available_pools()
    if expert not in choices:
        raise ValueError("Unknown expert pool: " + expert)
    FILES["pool"] = choices[expert]


def lookup(obj, key):
    for part in key.split("."):
        obj = obj[part]
    return obj


def validate(configs, cluster):
    gateway, pool, engine = (configs[k] for k in ("gateway", "pool", "engine"))
    for section, fields in FIELDS.items():
        for field in fields:
            value = lookup(configs[section], field)
            if field in {"deployment.node", "model_nodes", "router_nodes"}:
                continue
            if field.endswith(("enforce_eager", ".enabled")):
                if type(value) is not bool:
                    raise ValueError(field + " must be boolean")
            elif field.endswith("gpu_memory_utilization"):
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                    raise ValueError(field + " must be between 0 and 1")
            elif type(value) is not int or value < 1:
                raise ValueError(field + " must be a positive integer")
    nodes = {names(cluster, node)[0]: node["host"] for node in active_nodes(cluster)}
    device_nodes = {name for name, host in nodes.items() if host in engine["device_hosts"]}
    for field, allowed in (("model_nodes", device_nodes), ("router_nodes", set(nodes))):
        selected = pool[field]
        if (not isinstance(selected, list) or not selected or
                any(not isinstance(n, str) or n not in allowed for n in selected) or
                len(set(selected)) != len(selected)):
            raise ValueError(field + " must contain unique eligible node names")
    if gateway["deployment"]["node"] not in device_nodes:
        raise ValueError("Gateway node must have configured NPU support")
    for name, entry in gateway["pools"].items():
        if entry["enabled"] and (not isinstance(entry["base_url"], str) or not entry["base_url"].startswith("http://") or not entry.get("model")):
            raise ValueError("Enabled expert requires its configured Service URL and model: " + name)
    if pool["router_replicas"] != 1:
        raise ValueError("Current admission accounting requires one L2 Router")
    if pool["startup_seconds"] < 5 or pool["termination_seconds"] < pool["drain_seconds"] + 35:
        raise ValueError("Allow startup >=5s and termination >= drain +35s for engine shutdown")
    if cluster.get("resource_limits_enabled") is not False:
        raise ValueError("Existing no-hard-limit policy must remain disabled")
    if pool.get("runtime_config", FILES["engine"]) != FILES["engine"]:
        raise ValueError("Controller currently manages the default Ascend runtime only")


def parse_value(old, raw):
    if isinstance(old, str):
        return raw
    if isinstance(old, list):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return json.loads(raw)


def edit(root, section, assignments, write=False, expected=None):
    with operation_lock(root) if write else nullcontext():
        return edit_locked(root, section, assignments, write, expected)


def edit_locked(root, section, assignments, write=False, expected=None):
    storage = "pool" if section == "engine" else section
    path = root / FILES[storage]
    before = path.read_text(encoding="utf-8")
    raw = json.loads(before)
    configs = read_configs(root)
    if expected is not None and configs != expected:
        raise RuntimeError("配置已被其他操作修改，请重新进入编辑菜单")
    candidate = copy.deepcopy(configs)
    for assignment in assignments:
        key, separator, supplied = assignment.partition("=")
        if not separator or key not in FIELDS[section]:
            raise ValueError("Unknown/edit-protected parameter: " + key)
        old = lookup(candidate[section], key)
        value = parse_value(old, supplied)
        parent = candidate[section]
        parts = key.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = value
        target = raw
        if section == "engine":
            target = raw.setdefault("runtime", {})
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    if section == "engine":
        candidate["pool"]["runtime"] = raw["runtime"]
    validate(candidate, load_config(root / "deploy/lab/cluster.json"))
    after = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    print("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                   fromfile=str(path), tofile=str(path))), end="")
    if write and candidate[section] != configs[section]:
        # Same-directory replacement avoids partial JSON on interrupted writes.
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".control-", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(after)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, path.stat().st_mode)
            if path.read_text(encoding="utf-8") != before:
                raise RuntimeError("Configuration changed during editing; retry")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print("Saved configuration only; apply explicitly to deploy." if write else "Preview only; add --write to save.")


def commands(action, target):
    mapping = {
        "apply": {"pool": [("manage_pool.py", "apply"), ("manage_pool.py", "router")],
                  "gateway": [("manage_gateway.py", "apply")]},
        "status": {"cluster": [("manage_lab.py", "status")], "pool": [("manage_pool.py", "status")],
                   "gateway": [("manage_gateway.py", "status")]},
        "verify": {"pool": [("manage_pool.py", "verify"), ("verify_lifecycle.py", "routing")],
                   "gateway": [("manage_gateway.py", "verify")]},
        "prepare": {"pool": [("manage_pool.py", "prepare-model"), ("manage_pool.py", "prepare-router")], "gateway": [("manage_gateway.py", "prepare")]},
    }
    choices = mapping[action]
    if target != "all" and target not in choices:
        raise ValueError(action + " does not support " + target)
    return [entry for name, entries in choices.items() if target in (name, "all") for entry in entries]


LABELS = {"gateway": "一级路由 / Gateway", "pool": "专家池 / 二级 Router", "engine": "模型推理引擎"}
ERRORS = (ValueError, KeyError, RuntimeError, OSError, subprocess.CalledProcessError)


def choose_target(action):
    targets = ["pool", "gateway", "all"]
    if action == "logs":
        targets = ["pool", "router", "gateway"]
    elif action == "status":
        targets = ["pool", "gateway", "cluster", "all"]
    print("  ".join(f"{i}. {name}" for i, name in enumerate(targets, 1)))
    answer = input("选择对象（回车返回）: ").strip()
    if not answer:
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(targets):
        raise ValueError("请输入列表中的序号")
    return targets[int(answer) - 1]


def run_interactive(action, target):
    if action in {"apply", "prepare"}:
        main([action, target, "--plan"])
        print("应用会按配置恢复副本并可能更新 Pod；准备资产会写入现有节点卷。")
        if input("输入 y 执行，其他输入返回: ").strip().lower() != "y":
            return
    main([action, target])
    if action == "apply":
        print("部署命令完成，开始真实推理验收。")
        main(["verify", target])


def edit_interactive(section):
    original = read_configs(ROOT)
    fields = list(FIELDS[section])
    pending = {}
    target = "gateway" if section == "gateway" else "pool"
    source = FILES["pool"] + " runtime" if section == "engine" else FILES[section]
    print(f"\n{LABELS[section]} | 配置文件: {source}")
    print("修改暂存在内存中；可先调整多个相关参数，再统一保存。")
    print(f"生效操作: apply {target}；可能更新对应 Pod。换节点前需确认权重和运行依赖已准备。")
    while True:
        for index, key in enumerate(fields, 1):
            current = lookup(original[section], key)
            proposed = ""
            if key in pending:
                proposed = " -> " + json.dumps(parse_value(current, pending[key]), ensure_ascii=False)
            print(f"{index:2}. {FIELDS[section][key]} ({key}) = "
                  f"{json.dumps(current, ensure_ascii=False)}{proposed}")
        print("s. 预览并保存配置  a. 预览、保存并应用验收  0. 返回（丢弃未保存修改）")
        answer = input("选择参数或操作: ").strip().lower()
        if answer == "0":
            return
        try:
            if answer in {"s", "a"}:
                if not pending:
                    print("没有待保存的修改；应用已有配置请使用主菜单。")
                    continue
                assignments = [key + "=" + raw for key, raw in pending.items()]
                edit(ROOT, section, assignments, expected=original)
                if input("输入 y 保存以上修改，其他输入继续编辑: ").strip().lower() != "y":
                    continue
                edit(ROOT, section, assignments, write=True, expected=original)
                original = read_configs(ROOT)
                pending.clear()
                if answer == "a":
                    run_interactive("apply", target)
                continue
            if not answer.isdigit() or not 1 <= int(answer) <= len(fields):
                raise ValueError("请选择参数序号、s、a 或 0")
            key = fields[int(answer) - 1]
            old = lookup(original[section], key)
            if key in {"deployment.node", "model_nodes", "router_nodes"}:
                cluster = load_config(ROOT / "deploy/lab/cluster.json")
                eligible = [names(cluster, n)[0] for n in active_nodes(cluster)
                            if key == "router_nodes" or n["host"] in original["engine"]["device_hosts"]]
                print("已配置候选节点: " + ", ".join(eligible))
            print("列表用英文逗号分隔；布尔值填 true/false；回车保留。")
            raw = input("新值: ").strip()
            if raw:
                parse_value(old, raw)
                pending[key] = raw
        except ERRORS as error:
            print("未完成操作: " + str(error))
            print("修改已保存时仍保留在配置文件中；应用或验收失败不表示回滚。")


def interactive():
    print("HeteroServe 推理系统总控")
    print("显示的是本地配置；集群实际状态请选 6。集群操作在 .209 正式项目中执行。")
    try:
        while True:
            cfg = read_configs(ROOT)
            p, g, e = (cfg[k] for k in ("pool", "gateway", "engine"))
            print(f"\n当前配置: 专家池={p['pool']} | 模型副本={p['replicas']} | 二级 Router={p['router_replicas']}")
            print(f"Gateway={g['deployment']['node']} | 入口并发={g['max_inflight']} | "
                  f"实例预算={p['requests_per_worker']} | 引擎并发={e['model_parameters']['max_num_seqs']}")
            print("1. 查看/修改一级路由和 Gateway\n2. 查看/修改专家池和二级 Router\n"
                  "3. 查看/修改模型推理参数\n4. 查看全部原始配置（含固定参数、模型和节点）\n"
                  "5. 离线校验配置\n6. 查询集群实际状态\n7. 应用已有配置并验收\n"
                  "8. 执行推理验收\n9. 查看日志\n10. 准备已有资产\n11. 接入/退出算力节点\n12. 选择专家池\n0. 退出")
            answer = input("请选择: ").strip()
            if answer == "0":
                return
            try:
                if answer in {"1", "2", "3"}:
                    edit_interactive({"1": "gateway", "2": "pool", "3": "engine"}[answer])
                elif answer == "4":
                    print(json.dumps({**cfg, "cluster": load_config(ROOT / "deploy/lab/cluster.json")},
                                     ensure_ascii=False, indent=2))
                elif answer == "5":
                    main(["check"])
                elif answer == "11":
                    node_menu()
                elif answer == "12":
                    choices = list(available_pools())
                    print("  ".join(f"{i}. {name}" for i, name in enumerate(choices, 1)))
                    chosen = input("专家池序号（回车返回）: ").strip()
                    if chosen:
                        if not chosen.isdigit() or not 1 <= int(chosen) <= len(choices):
                            raise ValueError("请选择有效专家池序号")
                        select_pool(choices[int(chosen) - 1])
                elif answer in {"6", "7", "8", "9", "10"}:
                    action = {"6": "status", "7": "apply", "8": "verify", "9": "logs", "10": "prepare"}[answer]
                    target = choose_target(action)
                    if target:
                        run_interactive(action, target)
                else:
                    print("请输入菜单序号。")
            except ERRORS as error:
                print("操作失败: " + str(error))
    except (EOFError, KeyboardInterrupt):
        print("\n已退出；尚未保存的编辑已丢弃，已保存或已执行的操作不会自动撤销。")


def node_menu():
    print("1. 接入/重新接入节点  2. 安全退出节点  3. 恢复中断的退出操作  回车返回")
    choice = input("操作: ").strip()
    if not choice:
        return
    if choice not in {"1", "2", "3"}:
        raise ValueError("请选择 1、2 或 3")
    host = input("已授权服务器的完整 IP: ").strip()
    action = {"1": "add", "2": "remove", "3": "recover"}[choice]
    main(["node", action, host])
    print("将执行节点流程和真实验收；保留外层容器、卷、模型。失败时保留阶段记录，不强制排空。")
    if input("输入 y 执行，其他输入返回: ").strip().lower() == "y":
        main(["node", action, host, "--execute"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", help="Expert pool to manage, e.g. awq or gptq")
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("menu", help="Interactive configuration and deployment menu (default)")
    show = sub.add_parser("show", help="Show editable parameters and their source files")
    show.add_argument("section", choices=[*FILES, "all"], nargs="?", default="all")
    setter = sub.add_parser("set", help="Preview changes; --write saves without deployment")
    setter.add_argument("section", choices=FILES)
    setter.add_argument("assignments", nargs="+", metavar="KEY=VALUE")
    setter.add_argument("--write", action="store_true")
    sub.add_parser("check", help="Offline configuration validation (not live acceptance)")
    node = sub.add_parser("node", help="Plan or execute reversible worker membership")
    node.add_argument("operation", choices=["add", "remove", "recover"])
    node.add_argument("host", help="Authorized physical host IP")
    node.add_argument("--execute", action="store_true", help="Execute the inspected membership change")
    for action in ("apply", "status", "verify", "prepare"):
        command = sub.add_parser(action)
        command.add_argument("target", choices=["pool", "gateway", "cluster", "all"])
        command.add_argument("--plan", action="store_true", help="Print commands without executing")
    logs = sub.add_parser("logs", help="Bounded log snapshot from the private lab")
    logs.add_argument("target", choices=["pool", "router", "gateway"])
    logs.add_argument("--tail", type=int, default=100)
    args = parser.parse_args(argv)
    if args.pool:
        select_pool(args.pool)
    if args.action == "node":
        if read_configs()["pool"]["expert"] != "awq":
            raise ValueError("Physical node membership is global; select awq and migrate other pools before retirement")
        from manage_nodes import execute
        execute(args.operation, args.host, args.execute)
        return
    if args.action in {None, "menu"}:
        interactive()
        return
    if args.action == "set":
        edit(ROOT, args.section, args.assignments, args.write)
        return
    configs = read_configs()
    if args.action == "show":
        for section, fields in FIELDS.items():
            if args.section not in (section, "all"):
                continue
            print(section + " -> " + (FILES["pool"] + " runtime overrides" if section == "engine" else FILES[section]))
            for key, description in fields.items():
                print(f"  {key} = {json.dumps(lookup(configs[section], key), ensure_ascii=False)}  # {description}")
        return
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    if args.action == "logs":
        if not 1 <= args.tail <= 10000:
            raise ValueError("tail must be between 1 and 10000")
        pool = configs["pool"]
        app = {"pool": "expert", "router": "router", "gateway": "gateway"}[args.target]
        selector = "app.kubernetes.io/name=" + app
        if args.target != "gateway":
            selector += ",heteroserve.io/pool=" + pool["pool"]
        print(kubectl(cluster, ["logs", "-n", pool["namespace"], "-l", selector,
                               "--all-containers=true", "--prefix=true", "--tail=" + str(args.tail)]).stdout)
        return
    if args.action in {"check", "apply", "prepare"}:
        validate(configs, cluster)
    if args.action == "check":
        print("Offline configuration checks passed; device availability, assets and inference require live verification.")
        return
    with operation_lock(ROOT) if args.action in {"apply", "prepare"} and not args.plan else nullcontext():
        for script, action in commands(args.action, args.target):
            command = [sys.executable, str(ROOT / "scripts" / script), action]
            if script in {"manage_pool.py", "verify_lifecycle.py"}:
                command += ["--config", str(ROOT / FILES["pool"])]
            print(subprocess.list2cmdline(command), flush=True)
            if not args.plan:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
