"""Six operator workspaces with explicit context and one shared configuration draft."""
import json

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
        self.saved = []

    def context(self, page):
        pool = self.draft.pool()
        print(f"\n位置：{page} | 专家={pool['expert']} | 实例={self.selected or '未选择'} | 未保存文件={len(self.draft.changes())}")

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
            print("输入参数编号后填写新值，也可输入 字段名=值；0 返回并保留草稿；s 保存全部草稿；a 保存后进入发布。")
            value = input("编辑: ").strip()
            if value in {"", "0"}:
                return
            if value in {"s", "a"}:
                self.draft.preview()
                if input("输入 y 保存全部草稿: ").strip().lower() == "y":
                    self.saved = self.draft.save()
                    if value == "a":
                        self.publish()
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
                self.draft.set(section, key, supplied.strip(), ident)

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
            elif action == "设置启用实例数量":
                self.draft.resize(int(input("目标数量（减少时保留停用实例配置）: ").strip()))
            elif action == "移除实例配置":
                ident = pick("移除对象（应用时排空并删除对应工作负载，保留模型文件）", list(self.draft.instances()))
                if ident:
                    del self.draft.instances()[ident]
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
            action = pick("操作", ["查看草稿差异", "保存全部草稿", "查看当前专家发布影响", "应用当前专家", "仅应用选中实例", "验收当前专家", "准备当前专家模型资产", "回退最近保存的配置", "丢弃未保存草稿", "应用 Gateway 入口", "验收 Gateway", "应用本次保存范围"])
            if not action:
                return
            if action == "查看草稿差异":
                self.draft.preview()
            elif action == "保存全部草稿":
                self.draft.preview()
                if input("输入 y 保存: ").strip().lower() == "y":
                    self.saved = self.draft.save()
            elif action == "查看当前专家发布影响":
                self.publication_plan()
            elif action in {"应用当前专家", "仅应用选中实例"}:
                if action == "仅应用选中实例" and not self.selected:
                    raise ValueError("先选择具体实例")
                self.publication_plan(self.selected if action == "仅应用选中实例" else None)
                if input("输入 y 应用以上工作负载并验收: ").strip().lower() == "y":
                    if action == "仅应用选中实例":
                        c.main(["instance", "apply", self.selected])
                        c.main(["instance", "verify", self.selected])
                    else:
                        c.main(["apply", "pool"])
                        c.main(["verify", "pool"])
            elif action == "验收当前专家":
                c.main(["verify", "pool"])
            elif action == "准备当前专家模型资产":
                c.run_interactive("prepare", "pool")
            elif action == "丢弃未保存草稿":
                self.draft = c.Draft()
            elif action == "应用 Gateway 入口":
                if self.draft.changes():
                    raise ValueError("先保存草稿")
                c.run_interactive("apply", "gateway")
            elif action == "验收 Gateway":
                c.main(["verify", "gateway"])
            elif action == "应用本次保存范围":
                if self.draft.changes():
                    raise ValueError("先保存草稿")
                if not self.saved:
                    raise ValueError("当前会话没有保存记录，按专家/实例选择发布范围")
                print("本次保存文件：" + ", ".join(self.saved))
                if input("输入 y 应用并验收本次保存范围: ").strip().lower() == "y":
                    previous = c.FILES["pool"]
                    try:
                        for relative in self.saved:
                            if relative in c.available_pools().values():
                                c.FILES["pool"] = relative
                                c.main(["apply", "pool"])
                                c.main(["verify", "pool"])
                        if c.FILES["gateway"] in self.saved:
                            c.main(["apply", "gateway"])
                            c.main(["verify", "gateway"])
                    finally:
                        c.FILES["pool"] = previous
            else:
                c.rollback_configuration()
                self.draft = c.Draft()

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
        print("HeteroServe 总控 | 配置草稿集中保存，发布单独执行")
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
