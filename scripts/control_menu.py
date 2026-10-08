"""Six operator workspaces with explicit context and one shared configuration draft."""
import json
import hashlib

import control as c
import manage_instances as instances
from manage_lab import ROOT, kubectl, load_config, names, active_nodes


def pick(title, options, exit_label="返回"):
    print("\n" + title)
    for i, label in enumerate(options, 1):
        print(f"{i}. {label}")
    print("0. " + exit_label)
    value = input("选择: ").strip()
    if value in {"", "0"}:
        return None
    if value.isdigit() and 1 <= int(value) <= len(options):
        return list(options)[int(value) - 1]
    if value in options:
        return value
    raise ValueError("请选择本页列出的项目")


class Menu:
    def __init__(self):
        self.draft = c.Draft()
        self.selected = None
        self.saved = c.publication_record().get("pending", [])

    def context(self, page):
        pool = self.draft.pool()
        self.saved = c.publication_record().get("pending", [])
        print(f"\n位置：{page} | 专家={pool['expert']} | 实例={self.selected or '未选择'} | 未确认草稿={len(self.draft.changes())} | 已确认待更新={len(self.saved)}")

    def confirm_changes(self, update=False):
        if self.draft.changes():
            self.draft.summary()
            print("以上仅修改配置。确认保存后进入待更新状态，正在运行的服务此时不变。")
            if input("输入 y 确认以上修改并保存；其他输入继续编辑: ").strip().lower() != "y":
                print("未确认，草稿保留，配置和运行服务未改变。")
                return
            self.draft.save()
            self.saved = c.publication_record().get("pending", [])
            print("确认完成：已保存、待更新。范围：" + ", ".join(self.saved))
        elif not update:
            print("没有需要确认的草稿。")
        if update:
            self.update_system()

    def update_system(self, scopes=None, selected=None):
        """Plan every confirmed scope before one explicit execution decision."""
        if self.draft.changes():
            raise ValueError("还有未确认草稿；先确认修改，或丢弃草稿后更新")
        record = c.publication_record()
        scopes = list(dict.fromkeys(record.get("pending", []) if scopes is None else scopes))
        if not scopes:
            print("没有已确认待更新的配置。")
            return
        paths = {c.FILES["gateway"], c.FILES["engine"], "deploy/lab/cluster.json",
                 "deploy/pools/defaults.json", *c.available_pools().values()}
        snapshot = {name: (c.ROOT / name).read_bytes() for name in paths}
        for name in scopes:
            if name in record.get("pending", []) and hashlib.sha256(snapshot[name]).hexdigest() != record.get("confirmed", {}).get(name):
                raise RuntimeError("已确认配置发生变化，重新编辑并确认后再更新: " + name)
        previous = c.FILES["pool"]
        try:
            print("\n更新范围：" + ", ".join(scopes) + (" / 仅实例=" + selected if selected else ""))
            for name in scopes:
                print("\n发布影响：" + name)
                if name == c.FILES["gateway"]:
                    c.main(["apply", "gateway", "--plan"])
                elif name in c.available_pools().values():
                    c.FILES["pool"] = name
                    self.publication_plan(selected)
                else:
                    raise ValueError("不支持的更新范围: " + name)
            print("更新会创建/更新/停用对应实例；模型就绪并通过普通/SSE及路由验收后才报告完成。")
            if input("输入 y 确认更新以上系统范围；其他输入保留待更新状态: ").strip().lower() != "y":
                print("更新已取消，配置仍为已确认、待更新。")
                return
            for name in scopes:
                if any((c.ROOT / path).read_bytes() != data for path, data in snapshot.items()):
                    raise RuntimeError("发布计划生成后配置发生变化；未继续更新，请重新查看并确认计划")
                if name != c.FILES["gateway"]:
                    c.FILES["pool"] = name
                target = "gateway" if name == c.FILES["gateway"] else "pool"
                stage = "applying"
                try:
                    c.record_publication(name, stage)
                    c.main(["instance", "apply", selected] if selected else ["apply", target])
                    stage = "verifying"
                    c.record_publication(name, stage)
                    c.main(["instance", "verify", selected] if selected else ["verify", target])
                    if any((c.ROOT / path).read_bytes() != data for path, data in snapshot.items()):
                        raise RuntimeError("更新期间配置发生变化；保留待更新记录，请重新确认运行结果")
                    c.record_publication(name, "instance_verified" if selected else "verified")
                    print("更新完成、验收通过：" + name)
                    if selected:
                        print("选中实例验收通过；专家池整体待更新状态保留，其他已确认修改仍需更新。")
                except BaseException as error:
                    c.record_publication(name, "failed", f"{stage}: {error}")
                    print(f"更新未完成：{name}，失败阶段={stage}；保留待更新记录，运行状态请查看日志。")
                    raise
        finally:
            c.FILES["pool"] = previous
            self.saved = c.publication_record().get("pending", [])

    def overview(self):
        self.context("系统总览 / 实际运行")
        cluster = load_config(ROOT / "deploy/lab/cluster.json")
        rows = instances.inventory(cluster)
        print("实例ID | 专家池 | 节点 | TP/PP | 实际物理卡 | Ready | 重启 | Pod UID")
        for row in rows:
            p = row.get("parallelism", {})
            print(f"{row['instance']} | {row['pool']} | {row['node']} | {p.get('tp','?')}/{p.get('pp','?')} | "
                  f"{row['physical_devices'] if row['physical_devices'] is not None else '未采集'} | {row['ready']} | {row['restarts']} | {row['uid']}")
        print("卡号来自实际 Pod；配置草稿和容器内 npu:0 编号不作为物理分配结果。")

    def cards(self):
        self.context("算力节点与设备 / 实际卡占用")
        cluster = load_config(ROOT / "deploy/lab/cluster.json")
        rows = instances.inventory(cluster)
        from manage_nodes import physical_free
        print("节点 | 物理卡 | 占用实例/服务 | 状态")
        for node in active_nodes(cluster):
            node_name = names(cluster, node)[0]
            runtime = c.read_configs()["engine"]
            if node["host"] not in runtime["device_hosts"]:
                continue
            try:
                free = set(physical_free(node["host"]))
            except Exception:
                free = None
            for device in range(8):
                owners = [r["instance"] + "@" + r["uid"][:8] for r in rows if r["node"] == node_name and device in (r["physical_devices"] or [])]
                state = "项目占用" if owners else "空闲" if free is not None and device in free else "占用/归属未采集"
                print(f"{node['host']} | {device} | {','.join(owners) or '-'} | {state}")

    def field_editor(self, title, fields, section, ident=None):
        while True:
            self.context(title)
            if ident:
                spec = self.draft.instances()[ident]
                p = spec.get("parallelism", {"tp": 1, "pp": 1})
                print(f"请求卡数：{p['tp'] * p['pp']}（TP×PP），卡号由 K8s 分配。")
            keys = list(fields)
            for i, key in enumerate(keys, 1):
                print(f"{i}. {fields[key]} [{key}] = {json.dumps(self.draft.value(section, key, ident), ensure_ascii=False)}")
            print("输入参数编号或 字段名=值；0 返回并保留草稿；c 确认修改（保存不更新）；u 确认修改并更新系统。s/a 分别兼容 c/u。")
            value = input("编辑: ").strip()
            if value in {"", "0"}:
                return
            if value in {"c", "u", "s", "a"}:
                self.confirm_changes(update=value in {"u", "a"})
                continue
            if "=" in value:
                selector, supplied = value.split("=", 1)
            else:
                selector = value
                supplied = None
            selector = selector.strip().lower()
            if selector.isdigit() and 1 <= int(selector) <= len(keys):
                key = keys[int(selector) - 1]
            else:
                matches = [k for k in keys if k == selector or k.rsplit(".", 1)[-1] == selector]
                if len(matches) != 1:
                    raise ValueError("未知或不唯一的字段；请选择本页参数编号")
                key = matches[0]
            if supplied is None:
                supplied = input(f"{fields[key]} 的新值（回车保留）: ").strip()
            if supplied:
                old = self.draft.value(section, key, ident)
                self.draft.set(section, key, supplied.strip(), ident)
                print(f"已加入草稿：{ident or self.draft.pool()['expert']} / {key}: {old} → {self.draft.value(section, key, ident)}；尚未确认或更新。")

    def list_desired(self):
        pool = self.draft.pool()
        specs = self.draft.instances()
        print("实例ID | 节点 | 启用 | TP | PP | 请求卡数 | 引擎覆盖")
        for ident, v in specs.items():
            p = v.get("parallelism", {"tp": 1, "pp": 1})
            print(f"{ident} | {v['node']} | {v.get('enabled',True)} | {p['tp']} | {p['pp']} | {p['tp']*p['pp']} | {json.dumps(v.get('engine',{}))}")
        print("以上是期望配置，实际运行与卡号请查询系统总览。")

    def experts(self):
        while True:
            self.context("专家与 vLLM 实例")
            action = pick("操作", ["选择专家池", "查看实例配置", "选择并编辑实例", "增加实例", "设置启用实例数量", "移除实例配置", "专家默认推理参数"])
            if action is None:
                return
            if action == "选择专家池":
                chosen = pick("专家池", list(c.available_pools()))
                if chosen:
                    c.select_pool(chosen)
                    self.selected = None
            elif action == "查看实例配置":
                self.list_desired()
            elif action == "选择并编辑实例":
                specs = self.draft.instances()
                ident = pick("选择稳定实例 ID", list(specs))
                if ident:
                    self.selected = ident
                    while True:
                        group = pick("实例设置", ["部署与并行", "推理引擎", "健康与退出"])
                        if not group:
                            break
                        self.field_editor("专家与实例 / " + ident + " / " + group, c.INSTANCE_GROUPS[group], "pool", ident)
            elif action == "增加实例":
                specs = self.draft.instances()
                ident = input("新实例 ID（如 awq-05）: ").strip()
                if ident in specs:
                    raise ValueError("实例 ID 已存在")
                cluster = load_config(ROOT / "deploy/lab/cluster.json")
                node = pick("目标节点", [names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in c.read_configs()["engine"]["device_hosts"]])
                if node:
                    specs[ident] = {"node": node, "enabled": True, "parallelism": {"tp": 1, "pp": 1}, "engine": {}}
                    self.selected = ident
                    print(f"新增实例 {ident} 已加入草稿，节点={node}，TP=1，PP=1；可先编辑参数，再进入菜单 5 确认并更新。")
            elif action == "设置启用实例数量":
                before = self.draft.pool()["replicas"]
                self.draft.resize(int(input("目标数量（减少时保留停用实例配置）: ").strip()))
                print(f"启用实例数量 {before} → {self.draft.pool()['replicas']} 已加入草稿；进入菜单 5 确认并更新。")
            elif action == "移除实例配置":
                ident = pick("移除对象（应用时排空并删除对应工作负载，保留模型文件）", list(self.draft.instances()))
                if ident:
                    del self.draft.instances()[ident]
                    print(f"移除 {ident} 已加入草稿；确认并更新后才排空和移除工作负载。")
            elif action == "专家默认推理参数":
                self.field_editor("专家与实例 / 专家默认推理参数", c.FIELDS["engine"], "engine")

    def traffic(self):
        while True:
            self.context("路由与流量")
            choice = pick("配置范围", ["Gateway 全局入口", "专家池发现与默认请求预算", "具体实例请求预算"])
            if not choice:
                return
            if choice == "Gateway 全局入口":
                self.field_editor("路由与流量 / Gateway", c.FIELDS["gateway"], "gateway")
            elif choice == "专家池发现与默认请求预算":
                fields = {k: c.FIELDS["pool"][k] for k in ("requests_per_worker", "discovery_cache_seconds", "router_max_concurrent_requests", "queue_timeout_seconds", "request_timeout_seconds")}
                self.field_editor("路由与流量 / 当前专家池", fields, "pool")
            else:
                ident = pick("选择实例", list(self.draft.instances()))
                if ident:
                    self.field_editor("路由与流量 / " + ident, c.INSTANCE_GROUPS["实例流量"], "pool", ident)

    def nodes(self):
        while True:
            choice = pick("算力节点与设备", ["节点实际状态", "实际卡占用", "接入/退出/恢复节点"])
            if not choice:
                return
            if choice == "节点实际状态":
                c.main(["status", "cluster"])
            elif choice == "实际卡占用":
                self.cards()
            else:
                previous = c.FILES["pool"]
                try:
                    c.select_pool("awq")
                    c.node_menu()
                finally:
                    c.FILES["pool"] = previous

    def publication_plan(self, selected=None):
        if self.draft.changes():
            raise ValueError("还有草稿未保存；先保存才能生成实际发布计划")
        cluster = load_config(ROOT / "deploy/lab/cluster.json")
        pool = c.read_configs()["pool"]
        if "instances" in pool:
            print(json.dumps(instances.plan(cluster, pool, selected), ensure_ascii=False, indent=2))
        else:
            c.main(["apply", "pool", "--plan"])

    def publish(self):
        while True:
            self.context("变更发布与验收")
            action = pick("编辑草稿 → 确认修改 → 更新系统 → 验收结果", ["查看草稿差异", "确认修改（保存全部草稿）", "更新所有已确认范围", "确认修改并更新系统", "查看更新状态", "指定更新范围与验收", "回退或丢弃"])
            if not action:
                return
            if action == "查看草稿差异":
                self.draft.preview()
            elif action == "确认修改（保存全部草稿）":
                self.confirm_changes()
            elif action == "确认修改并更新系统":
                self.confirm_changes(update=True)
            elif action == "查看更新状态":
                record = c.publication_record()
                print(json.dumps({"已确认待更新": record.get("pending", []), "最近更新阶段": record.get("publication", {})}, ensure_ascii=False, indent=2))
            elif action == "更新所有已确认范围":
                self.update_system()
            elif action == "指定更新范围与验收":
                self.publication_tools()
            elif action == "回退或丢弃":
                choice = pick("回退或丢弃", ["回退最近保存的配置", "丢弃未确认草稿"])
                if choice == "回退最近保存的配置":
                    if self.draft.changes():
                        raise ValueError("先确认或丢弃草稿，再回退配置")
                    c.rollback_configuration()
                    self.draft = c.Draft()
                elif choice == "丢弃未确认草稿":
                    self.draft = c.Draft()
                    print("未确认草稿已丢弃；已确认配置和待更新范围保留。")

    def publication_tools(self):
        while True:
            action = pick("指定更新范围与验收", ["查看当前专家发布影响", "更新当前专家", "仅更新选中实例", "更新 Gateway 入口", "验收当前专家", "验收 Gateway", "准备当前专家模型资产"])
            if not action:
                return
            if action == "查看当前专家发布影响":
                self.publication_plan()
            elif action in {"更新当前专家", "仅更新选中实例"}:
                if action == "仅更新选中实例" and not self.selected:
                    raise ValueError("先选择具体实例")
                self.update_system([c.FILES["pool"]], self.selected if action == "仅更新选中实例" else None)
            elif action == "更新 Gateway 入口":
                self.update_system([c.FILES["gateway"]])
            elif action == "验收当前专家":
                c.main(["verify", "pool"])
            elif action == "验收 Gateway":
                c.main(["verify", "gateway"])
            else:
                c.run_interactive("prepare", "pool")

    def observability(self):
        while True:
            action = pick("日志与监控", ["当前实例日志", "专家池模型日志", "专家池 Router 日志", "Gateway 日志", "现有监控接入验收"])
            if not action:
                return
            if action == "当前实例日志":
                if not self.selected:
                    raise ValueError("先选择实例")
                c.main(["instance", "logs", self.selected])
            elif action == "现有监控接入验收":
                c.subprocess.run([c.sys.executable, str(ROOT / "scripts/manage_monitoring.py"), "verify"], check=True)
            else:
                target = {"专家池模型日志": "pool", "专家池 Router 日志": "router", "Gateway 日志": "gateway"}[action]
                c.main(["logs", target])

    def run(self):
        actions = {"系统总览": self.overview, "专家与 vLLM 实例": self.experts,
                   "路由与流量": self.traffic, "算力节点与设备": self.nodes,
                   "变更发布与验收": self.publish, "日志与监控": self.observability}
        print("HeteroServe 总控 | 编辑草稿 → 确认修改 → 更新系统 → 验收结果")
        try:
            while True:
                self.context("主菜单")
                try:
                    action = pick("功能分组", list(actions), "退出")
                    if not action:
                        if self.draft.changes():
                            print("尚有未保存草稿，本次退出丢弃；已保存/已发布操作保留。")
                        return
                    actions[action]()
                except c.ERRORS as error:
                    print("操作未完成: " + str(error))
                    print("草稿仍保留；应用失败时查看日志和阶段记录。")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出，未保存草稿已丢弃。")


def run():
    Menu().run()
