"""Unified operator CLI. Existing JSON files and managers remain authoritative."""
import argparse
import copy
import difflib
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import datetime
import uuid
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
             "router_max_concurrent_requests": "官方 Router 并发上限",
             "queue_timeout_seconds": "官方 Router 排队超时秒数",
             "request_timeout_seconds": "实例请求超时默认秒数",
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
                if section == "pool" and field == "replicas" and "instances" in pool and value == 0:
                    continue
                raise ValueError(field + " must be a positive integer")
    nodes = {names(cluster, node)[0]: node["host"] for node in active_nodes(cluster)}
    device_nodes = {name for name, host in nodes.items() if host in engine["device_hosts"]}
    for field, allowed in (("model_nodes", device_nodes), ("router_nodes", set(nodes))):
        selected = pool[field]
        empty_allowed = field == "model_nodes" and "instances" in pool and pool["replicas"] == 0
        if (not isinstance(selected, list) or (not selected and not empty_allowed) or
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
    if "instances" in pool:
        from manage_instances import validate_instances
        validate_instances(pool, cluster)


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
        if section == "pool" and "instances" in configs["pool"] and key in {"replicas", "model_nodes"}:
            if key == "model_nodes":
                raise ValueError("Nodes belong to individual instances; use instance set ID node=...")
            from manage_instances import resize_specs
            raw["instances"] = resize_specs(configs["pool"], value)
            candidate["pool"]["instances"] = raw["instances"]
            candidate["pool"]["replicas"] = value
            candidate["pool"]["model_nodes"] = sorted({v["node"] for v in raw["instances"].values() if v.get("enabled", True)})
            continue
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


ERRORS = (ValueError, KeyError, RuntimeError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired)

INSTANCE_GROUPS = {
    "部署与并行": {"node": "运行节点", "enabled": "启用实例", "parallelism.tp": "TP 张量并行", "parallelism.pp": "PP 流水线并行"},
    "推理引擎": {"engine." + k.split(".", 1)[1]: v for k, v in FIELDS["engine"].items()},
    "健康与退出": {"lifecycle." + k: FIELDS["pool"][k] for k in ("startup_seconds", "drain_seconds", "termination_seconds")},
    "实例流量": {"traffic.max_inflight": "实例请求准入预算", "traffic.request_timeout_seconds": "实例请求超时秒数"},
}


def atomic_text(path, text):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".control-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publication_record(root=None):
    root = ROOT if root is None else root
    path = root / "artifacts/kubernetes/control/last-save.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    allowed = {FILES["gateway"], *available_pools(root).values()}
    if any(name not in allowed for name in record.get("pending", [])):
        raise RuntimeError("发布记录包含不支持的配置文件")
    return record


def record_publication(relative, stage, error=None, root=None):
    root = ROOT if root is None else root
    with operation_lock(root):
        record = publication_record(root)
        record["publication"] = {"file": relative, "stage": stage,
                                 "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if error:
            record["publication"]["error"] = str(error)
        if stage == "verified":
            record["pending"] = [name for name in record.get("pending", []) if name != relative]
        path = root / "artifacts/kubernetes/control/last-save.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, json.dumps(record, ensure_ascii=False, indent=2))


class Draft:
    """One in-memory change set across experts, instances and global traffic."""
    def __init__(self, root=None):
        self.root = ROOT if root is None else root
        self.files = {}
        self.checkpoint_enabled = False
        self.checkpoint_digest = None
        self.restored_context = {}
        self.session_id = uuid.uuid4().hex

    @property
    def checkpoint_path(self):
        return self.root / "artifacts/kubernetes/control/draft.json"

    def restore(self):
        """Recover one workspace without silently rebasing concurrently edited files."""
        self.checkpoint_enabled = True
        if not self.checkpoint_path.exists():
            return False
        payload = self.checkpoint_path.read_bytes()
        record = json.loads(payload)
        allowed = {FILES["gateway"], *available_pools(self.root).values()}
        recovered = record.get("files", {})
        if not isinstance(recovered, dict) or any(name not in allowed for name in recovered):
            raise RuntimeError("草稿恢复文件包含不支持的目标，已保留文件")
        for data in recovered.values():
            if not isinstance(data, dict) or not isinstance(data.get("before"), str) or not isinstance(data.get("after"), dict) or not isinstance(json.loads(data["before"]), dict):
                raise RuntimeError("草稿恢复数据无效，已保留文件")
        context = record.get("context", {})
        if not isinstance(context, dict) or any(context.get(k) is not None and not isinstance(context[k], str) for k in ("pool", "selected")):
            raise RuntimeError("草稿恢复上下文无效，已保留文件")
        self.files = recovered
        self.restored_context = context
        self.checkpoint_digest = hashlib.sha256(payload).hexdigest()
        return bool(recovered)

    def check_checkpoint(self):
        actual = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest() if self.checkpoint_path.exists() else None
        if self.checkpoint_enabled and actual != self.checkpoint_digest:
            changes = self.changes()
            backup = self.root / "artifacts/kubernetes/control/conflicts" / (self.session_id + ".json")
            if changes:
                backup.parent.mkdir(parents=True, exist_ok=True)
                atomic_text(backup, json.dumps({"files": changes, "context": self.restored_context}, ensure_ascii=False, indent=2))
            raise RuntimeError("另一个控制会话修改了草稿；未覆盖他人编辑，本会话修改保留于 " + str(backup))

    def checkpoint(self, context=None):
        self.checkpoint_enabled = True
        with operation_lock(self.root):
            self.check_checkpoint()
            changes = self.changes()
            if not changes:
                if self.checkpoint_path.exists():
                    self.checkpoint_path.unlink()
                self.checkpoint_digest = None
                return
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "files": changes, "context": context or self.restored_context}
            payload = json.dumps(record, ensure_ascii=False, indent=2)
            atomic_text(self.checkpoint_path, payload)
            self.checkpoint_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def discard(self):
        with operation_lock(self.root):
            self.check_checkpoint()
            if self.checkpoint_enabled and self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            self.files.clear()
            self.checkpoint_digest = None

    def file(self, relative):
        if relative not in self.files:
            text = (self.root / relative).read_text(encoding="utf-8")
            self.files[relative] = {"before": text, "after": json.loads(text)}
        return self.files[relative]["after"]

    def pool(self, relative=None):
        relative = relative or FILES["pool"]
        raw = self.file(relative)
        base = json.loads((self.root / "deploy/pools/defaults.json").read_text()) if "defaults" in raw else {}
        result = {**base, **copy.deepcopy(raw)}
        if "instances" in result:
            result["replicas"] = sum(v.get("enabled", True) for v in result["instances"].values())
            result["model_nodes"] = sorted({v["node"] for v in result["instances"].values() if v.get("enabled", True)})
        return result

    def instances(self):
        from manage_instances import normalized_instances
        pool = self.pool()
        raw = self.file(FILES["pool"])
        if "instances" not in raw:
            raw["instances"] = normalized_instances(pool)
            raw.pop("replicas", None)
            raw.pop("model_nodes", None)
        return raw["instances"]

    def value(self, section, key, ident=None):
        if ident:
            from manage_instances import effective
            p, n = effective(self.pool(), ident, self.root)
            spec = self.instances()[ident]
            if key.startswith("engine."):
                return n["model_parameters"][key.split(".", 1)[1]]
            if key.startswith("lifecycle."):
                return p[key.split(".", 1)[1]]
            if key.startswith("traffic."):
                return p["requests_per_worker"] if key == "traffic.max_inflight" else p["request_timeout_seconds"]
            return lookup(spec, key)
        if section == "gateway":
            return lookup(self.file(FILES["gateway"]), key)
        if section == "engine":
            return lookup(resolve_runtime(self.pool(), self.root), key)
        return lookup(self.pool(), key)

    def set(self, section, key, supplied, ident=None):
        if ident:
            allowed = {k for group in INSTANCE_GROUPS.values() for k in group}
            if key not in allowed:
                raise ValueError("Unsupported instance parameter: " + key)
            target = self.instances()[ident]
        else:
            if key not in FIELDS[section]:
                raise ValueError("Unsupported parameter: " + key)
            target = self.file(FILES["pool"] if section == "engine" else FILES[section])
            if section == "engine":
                target = target.setdefault("runtime", {})
        old = self.value(section, key, ident)
        value = parse_value(old, supplied)
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    def resize(self, count):
        from manage_instances import resize_specs
        raw = self.file(FILES["pool"])
        raw["instances"] = resize_specs(self.pool(), count)
        raw.pop("replicas", None)
        raw.pop("model_nodes", None)

    def inherit(self, key, ident):
        group, _, name = key.partition(".")
        if group not in {"engine", "traffic", "lifecycle"} or key not in {k for fields in INSTANCE_GROUPS.values() for k in fields}:
            raise ValueError("只有引擎、请求和退出参数可以恢复默认值")
        spec = self.instances()[ident]
        spec.get(group, {}).pop(name, None)
        if group in spec and not spec[group]:
            del spec[group]

    def changes(self):
        return {name: v for name, v in self.files.items() if json.loads(v["before"]) != v["after"]}

    def preview(self):
        changes = self.changes()
        if not changes:
            print("没有未保存的修改。")
        for name, data in changes.items():
            after = json.dumps(data["after"], ensure_ascii=False, indent=2) + "\n"
            print("".join(difflib.unified_diff(data["before"].splitlines(True), after.splitlines(True), fromfile=name, tofile=name)), end="")
        return changes

    def summary(self):
        """Readable confirmation scope; values here are configuration, not live state."""
        def compare(before, after, prefix=""):
            for key in sorted(set(before) | set(after)):
                old, new = before.get(key), after.get(key)
                field = prefix + key
                if old == new and (key in before) == (key in after):
                    continue
                if isinstance(old, dict) and isinstance(new, dict):
                    compare(old, new, field + ".")
                else:
                    display = lambda v, present: json.dumps(v, ensure_ascii=False) if present else "未设置/继承默认值"
                    print(f"  {field}: {display(old, key in before)} → {display(new, key in after)}")
        for name, data in self.changes().items():
            before, after = json.loads(data["before"]), data["after"]
            print("\n确认范围：" + name)
            if "instances" in before and "instances" in after:
                def capacity(raw):
                    enabled = [v for v in raw["instances"].values() if v.get("enabled", True)]
                    return len(enabled), sum(v.get("parallelism", {}).get("tp", 1) * v.get("parallelism", {}).get("pp", 1) for v in enabled)
                old, new = capacity(before), capacity(after)
                print(f"  启用实例数: {old[0]} → {new[0]}；配置申请卡数: {old[1]} → {new[1]}（更新额外卡另见发布计划）")
            compare(before, after)

    def save(self):
        changes = self.changes()
        if not changes:
            self.files.clear()
            print("没有需保存的修改；保留最近一次回退记录。")
            return []
        cluster = load_config(self.root / "deploy/lab/cluster.json")
        gateway = copy.deepcopy(self.file(FILES["gateway"]))
        for relative in available_pools(self.root).values():
            pool = self.pool(relative)
            validate({"gateway": gateway, "pool": pool, "engine": resolve_runtime(pool, self.root)}, cluster)
        with operation_lock(self.root):
            self.check_checkpoint()
            for name, data in changes.items():
                if (self.root / name).read_text(encoding="utf-8") != data["before"]:
                    raise RuntimeError("配置已被其他操作修改，重新载入草稿: " + name)
            journal = self.root / "artifacts/kubernetes/control/last-save.json"
            journal.parent.mkdir(parents=True, exist_ok=True)
            previous = publication_record(self.root)
            pending = list(dict.fromkeys([*previous.get("pending", []), *changes]))
            confirmed = dict(previous.get("confirmed", {}))
            for name, data in changes.items():
                confirmed[name] = hashlib.sha256((json.dumps(data["after"], ensure_ascii=False, indent=2) + "\n").encode("utf-8")).hexdigest()
            record = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "status": "writing",
                      "before": {n: v["before"] for n, v in changes.items()},
                      "pending": pending, "confirmed": confirmed}
            atomic_text(journal, json.dumps(record, indent=2))
            written = []
            try:
                for name, data in changes.items():
                    atomic_text(self.root / name, json.dumps(data["after"], ensure_ascii=False, indent=2) + "\n")
                    written.append(name)
            except BaseException:
                for name in written:
                    atomic_text(self.root / name, changes[name]["before"])
                atomic_text(journal, json.dumps(previous, ensure_ascii=False, indent=2))
                raise
            record["status"] = "saved"
            atomic_text(journal, json.dumps(record, indent=2))
            if self.checkpoint_enabled and self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
                self.checkpoint_digest = None
        self.files.clear()
        print("已保存配置；尚未部署。请在变更发布中查看计划并应用。")
        return list(changes)


def edit_instance(root, ident, assignments, write=False):
    draft = Draft(root)
    specs = draft.instances()
    if ident not in specs:
        raise ValueError("Unknown instance ID: " + ident)
    for assignment in assignments:
        key, separator, supplied = assignment.partition("=")
        if not separator:
            raise ValueError("Use key=value")
        aliases = {"tp": "parallelism.tp", "pp": "parallelism.pp", "max_inflight": "traffic.max_inflight"}
        if key in ENGINE_FIELDS_FOR_INSTANCE:
            key = "engine." + key
        key = aliases.get(key, key)
        draft.set("pool", key, supplied, ident)
    draft.preview()
    if write:
        draft.save()
    return draft


ENGINE_FIELDS_FOR_INSTANCE = {k.split(".", 1)[1] for k in FIELDS["engine"]}


def rollback_configuration(execute=False):
    path = ROOT / "artifacts/kubernetes/control/last-save.json"
    record = json.loads(path.read_text())
    allowed = {FILES["gateway"], *available_pools().values()}
    draft = Draft()
    for relative, text in record["before"].items():
        if relative not in allowed:
            raise RuntimeError("Rollback record contains an unsupported target")
        previous = json.loads(text)
        current = draft.file(relative)
        if relative != FILES["gateway"] and "instances" in current and "instances" not in previous:
            from manage_instances import normalized_instances
            defaults = json.loads((ROOT / "deploy/pools/defaults.json").read_text())
            previous["instances"] = normalized_instances({**defaults, **previous})
            previous.pop("replicas", None)
            previous.pop("model_nodes", None)
        draft.files[relative]["after"] = previous
    draft.preview()
    if execute or input("输入 y 回退以上配置（随后仍需应用和验收）: ").strip().lower() == "y":
        draft.save()


def instance_command(args):
    from manage_instances import normalized_instances, validate_instances, inventory, plan, apply, verify, INSTANCE_LABEL
    cluster = load_config(ROOT / "deploy/lab/cluster.json")
    pool = read_configs()["pool"]
    ident = args.ident
    if args.operation in {"set", "resize", "remove", "pause", "resume", "logs"} and not ident:
        raise ValueError("This operation requires an instance ID (resize requires a target count)")
    if args.operation == "list":
        print(json.dumps(inventory(cluster, pool) if args.live else normalized_instances(pool), indent=2, ensure_ascii=False))
        return
    if args.operation == "set":
        edit_instance(ROOT, ident, args.assignments, args.write)
        return
    if args.operation in {"initialize", "add", "resize", "remove", "pause", "resume"}:
        draft = Draft()
        specs = draft.instances()
        if args.operation == "resize":
            draft.resize(int(ident))
        elif args.operation == "add":
            if not ident or ident in specs or not args.node:
                raise ValueError("add requires a new ID and --node")
            specs[ident] = {"node": args.node, "enabled": True, "parallelism": {"tp": args.tp, "pp": args.pp}, "engine": {}}
        elif args.operation == "remove":
            del specs[ident]
        elif args.operation in {"pause", "resume"}:
            specs[ident]["enabled"] = args.operation == "resume"
        draft.preview()
        if args.write:
            draft.save()
        return
    if "instances" not in pool:
        raise ValueError("Initialize stable instances before individual workload operations")
    if ident and ident not in pool["instances"]:
        raise ValueError("Unknown instance ID")
    if args.operation == "plan":
        print(json.dumps(plan(cluster, pool, ident), indent=2, ensure_ascii=False))
    elif args.operation == "logs":
        if not ident:
            raise ValueError("Select an instance ID")
        print(kubectl(cluster, ["logs", "-n", pool["namespace"], "-l", INSTANCE_LABEL + "=" + ident + ",heteroserve.io/pool=" + pool["pool"], "--all-containers=true", "--prefix=true", "--tail=100"]).stdout)
    elif args.operation == "apply":
        with operation_lock(ROOT):
            print(json.dumps(apply(cluster, pool, ident), indent=2))
    elif args.operation == "verify":
        print(json.dumps(verify(cluster, pool, ident), indent=2))


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


def interactive():
    import control_menu
    control_menu.c = sys.modules[__name__]
    control_menu.run()


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
    instance = sub.add_parser("instance", help="Manage a stable vLLM instance")
    instance.add_argument("operation", choices=["list", "initialize", "add", "resize", "remove", "pause", "resume", "set", "plan", "apply", "verify", "logs"])
    instance.add_argument("ident", nargs="?")
    instance.add_argument("assignments", nargs="*")
    instance.add_argument("--write", action="store_true")
    instance.add_argument("--live", action="store_true")
    instance.add_argument("--node")
    instance.add_argument("--tp", type=int, default=1)
    instance.add_argument("--pp", type=int, default=1)
    rollback = sub.add_parser("rollback", help="Preview or restore the previous configuration")
    rollback.add_argument("--write", action="store_true")
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
    if args.action == "instance":
        instance_command(args)
        return
    if args.action == "rollback":
        rollback_configuration(args.write)
        return
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
            if script == "verify_lifecycle.py" and action == "routing" and configs["pool"]["replicas"] == 0:
                print("当前专家池已暂停，没有模型候选；跳过普通路由请求验收。")
                continue
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
