"""Six operator workspaces with explicit context and one shared configuration draft."""
import json
import hashlib
import copy
import shlex
import uuid
import time

import control as c
import manage_instances as instances
from manage_lab import ROOT, OWNER, kubectl, load_config, names, active_nodes


def pick(title, options, exit_label="返回", shortcuts=None):
    print("\n" + title)
    for i, label in enumerate(options, 1):
        print(f"{i}. {label}")
    print("0. " + exit_label)
    if shortcuts:
        print("快捷键：" + "；".join(key + " " + label for key, label in shortcuts.items()))
    value = input("选择: ").strip()
    if value.lower() in (shortcuts or {}):
        return shortcuts[value.lower()]
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
        self.selected_scope = "pool"
        self.saved = c.publication_record().get("pending", [])
        self.resource_rows = []
        self.resource_error = "尚未刷新"
        self.resource_at = None
        self.resource_filter = ""
        self.job = None
        if self.draft.restore():
            context = self.draft.restored_context
            if context.get("pool") in c.available_pools().values():
                c.FILES["pool"] = context["pool"]
            if context.get("selected") in self.draft.instances():
                self.selected = context["selected"]
            if context.get("selected") in self.draft.file(c.FILES["l1"])["instances"]:
                self.selected, self.selected_scope = context["selected"], "l1"
            print("已恢复上次未确认草稿；请查看差异后继续编辑或确认。")

    def checkpoint(self):
        self.draft.checkpoint({"pool": c.FILES["pool"], "selected": self.selected})

    def discard_draft(self):
        self.draft.discard()
        self.draft = c.Draft()
        self.draft.restore()

    def refresh_resources(self):
        print("正在读取项目工作负载状态……")
        try:
            cluster = load_config(c.ROOT / "deploy/lab/cluster.json")
            result = kubectl(cluster, ["get", "pods", "-n", "heteroserve", "-l", OWNER + "=" + cluster["name"], "-o", "json"], timeout=15)
            self.resource_rows = json.loads(result.stdout)["items"]
            self.resource_error = None
        except c.ERRORS as error:
            self.resource_rows = []
            self.resource_error = str(error)
        self.resource_at = c.datetime.datetime.now(c.datetime.timezone(c.datetime.timedelta(hours=8))).isoformat(timespec="seconds")

    def resource_index(self):
        index = {}
        short = {}
        for expert, relative in c.available_pools().items():
            pool = self.draft.pool(relative)
            for ident, spec in instances.normalized_instances(pool).items():
                index[expert + "/" + ident] = {"expert": expert, "relative": relative, "pool": pool["pool"], "ident": ident, "spec": spec}
                short.setdefault(ident, []).append(index[expert + "/" + ident])
        index.update({ident: values[0] for ident, values in short.items() if len(values) == 1})
        for ident, spec in self.draft.file(c.FILES["l1"])["instances"].items():
            row={"expert":"l1","relative":c.FILES["l1"],"pool":"l1-router","ident":ident,"spec":spec}
            index["l1/"+ident]=row
            if ident not in index:index[ident]=row
        return index

    def resources(self):
        print("\n实例工作台 | 采集时间=" + str(self.resource_at) + " | 筛选=" + (self.resource_filter or "全部"))
        print("实例 / 专家 | 工作负载状态 | 实际节点 | 配置 TP/PP | 申请卡数 | 变更")
        pending = set(c.publication_record().get("pending", []))
        for key, row in self.resource_index().items():
            if "/" not in key:
                continue
            spec = row["spec"]
            if self.resource_filter and self.resource_filter.lower() not in (key + " " + spec["node"]).lower():
                continue
            pods = [p for p in self.resource_rows if p["metadata"].get("labels", {}).get("heteroserve.io/pool") == row["pool"] and
                    p["metadata"].get("labels", {}).get(instances.INSTANCE_LABEL) == row["ident"]]
            live = [p for p in pods if not p["metadata"].get("deletionTimestamp")]
            ready = sum(instances.ready(p) for p in live)
            status = "未采集" if self.resource_error else "更新中" if len(pods) > 1 or any(p["metadata"].get("deletionTimestamp") for p in pods) else "就绪" if ready else "加载/调度中" if live else "未部署" if spec.get("enabled", True) else "已停用"
            actual = ",".join(sorted({p["spec"].get("nodeName", "待调度") for p in pods})).replace("heteroserve-lab-", ".") or "--"
            parallel = spec.get("parallelism", {"tp": 1, "pp": 1})
            changed = "草稿" if row["relative"] in self.draft.changes() else "待发布（池）" if row["relative"] in pending else "--"
            print(f"{key} | {status} | {actual} | {parallel['tp']}/{parallel['pp']} | {parallel['tp']*parallel['pp'] if spec.get('enabled',True) else 0} | {changed}")
        if self.resource_error:
            print("采集不可用：" + self.resource_error + "；配置仍可编辑，运行状态未核实。")
        conflicts = [name for name, data in self.draft.changes().items() if (c.ROOT / name).read_text(encoding="utf-8") != data["before"]]
        if conflicts:
            print("草稿与当前配置冲突，禁止覆盖：" + ", ".join(conflicts))
        print("输入实例 ID（例如 awq-01 / gptq-01）直接编辑；/awq 筛选，/ 清除；r 刷新；+ 新增；c 保存；u 发布；jobs 任务。")
        print("申请卡数和 TP/PP 是期望配置；进入实例页查看实际参数、物理卡及 Pod UID。")

    def select_instance(self, row):
        if row["expert"] == "l1":
            self.selected, self.selected_scope = row["ident"], "l1"
            self.field_editor("一级 Router / "+self.selected,{"node":"运行节点","enabled":"启用实例","max_inflight":"路由请求预算"},"l1",self.selected)
            return
        self.selected_scope = "pool"
        c.select_pool(row["expert"])
        self.selected = row["ident"]
        fields = {key: label for group in c.INSTANCE_GROUPS.values() for key, label in group.items()}
        self.field_editor("实例工作台 / " + self.selected, fields, "pool", self.selected)

    def create_instance(self):
        expert = pick("新增实例 / 服务类型", [*c.available_pools(), "一级 Router"])
        if not expert:
            return
        if expert == "一级 Router":
            specs=self.draft.file(c.FILES["l1"])["instances"]
            seq=1
            while "l1-"+str(seq).zfill(2) in specs:seq+=1
            suggested="l1-"+str(seq).zfill(2)
            ident=input(f"一级 Router ID（回车={suggested}）: ").strip() or suggested
            if ident in specs:raise ValueError("实例 ID 已存在")
            cluster=load_config(c.ROOT/'deploy/lab/cluster.json')
            choices={n['host']:names(cluster,n)[0] for n in active_nodes(cluster) if n['host'] in c.read_configs()['engine']['device_hosts']}
            chosen=pick("选择一级 Router 节点（每实例一张 NPU）",list(choices))
            if chosen:
                specs[ident]={"node":choices[chosen],"enabled":True}
                self.checkpoint();self.select_instance({"expert":"l1","ident":ident})
            return
        c.select_pool(expert)
        specs = self.draft.instances()
        seq = 1
        while expert + "-" + str(seq).zfill(2) in specs:
            seq += 1
        suggested = expert + "-" + str(seq).zfill(2)
        ident = input(f"实例 ID（回车使用 {suggested}）: ").strip() or suggested
        if ident in specs:
            raise ValueError("实例 ID 已存在")
        cluster = load_config(c.ROOT / "deploy/lab/cluster.json")
        runtime = c.resolve_runtime(self.draft.pool(), c.ROOT)
        choices = {n["host"]: names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in runtime["device_hosts"]}
        chosen = pick("选择目标节点", list(choices))
        if not chosen:
            return
        tp = int(input("TP（回车=1）: ").strip() or "1")
        pp = int(input("PP（回车=1）: ").strip() or "1")
        if tp < 1 or pp < 1 or tp * pp > 8:
            raise ValueError("单节点 TP×PP 必须在 1～8 之间")
        engine = {}
        params = runtime["model_parameters"]
        if pp > 1 and params["max_num_batched_tokens"] < params["max_model_len"]:
            engine["max_num_batched_tokens"] = params["max_model_len"]
            print("PP 启动约束：批处理 token 预算初始化为 " + str(params["max_model_len"]))
        specs[ident] = {"node": choices[chosen], "enabled": True, "parallelism": {"tp": tp, "pp": pp}, "engine": engine}
        self.selected = ident
        self.checkpoint()
        print("实例已加入持久草稿；正在打开参数页，可确认保存或发布。")
        self.select_instance({"expert": expert, "ident": ident})

    def job_status(self):
        directory = c.ROOT / "artifacts/kubernetes/control/jobs"
        if not directory.exists():
            print("暂无发布任务。")
            return
        for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            job = json.loads(path.read_text(encoding="utf-8"))
            print(f"{job['id']} | {job['status']} | {job.get('stage','--')} | {job.get('scope','--')} | {job['log']}")

    def command(self, args, log_path):
        expert = next(k for k, v in c.available_pools().items() if v == c.FILES["pool"])
        argv = [c.sys.executable, str(c.ROOT / "scripts/control.py"), "--pool", expert, *args]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("ab") as output:
            output.write(("\nCOMMAND " + json.dumps(argv, ensure_ascii=False) + "\n").encode("utf-8")); output.flush()
            process = c.subprocess.Popen(argv, stdin=c.subprocess.DEVNULL, stdout=output, stderr=c.subprocess.STDOUT)
            try:
                while process.poll() is None:
                    print(f"{args[0]} 进行中，已等待 {int(time.monotonic()-started)} 秒；详细输出：{log_path}", flush=True)
                    try:
                        process.wait(timeout=5)
                    except c.subprocess.TimeoutExpired:
                        pass
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except c.subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise
        if process.returncode:
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]))
            raise c.subprocess.CalledProcessError(process.returncode, argv)

    def write_job(self):
        path = c.ROOT / "artifacts/kubernetes/control/jobs" / (self.job["id"] + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        c.atomic_text(path, json.dumps(self.job, ensure_ascii=False, indent=2))

    def context(self, page):
        pool = self.draft.pool()
        self.saved = c.publication_record().get("pending", [])
        print(f"\n位置：{page} | 服务={'一级Router' if self.selected_scope=='l1' else pool['expert']} | 实例={self.selected or '未选择'} | 未确认草稿={len(self.draft.changes())} | 已确认待更新={len(self.saved)}")

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
                 "deploy/pools/defaults.json", c.FILES["l1"], *c.available_pools().values()}
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
                elif name == c.FILES["l1"]:
                    self.publication_plan(selected, section="l1")
                elif name in c.available_pools().values():
                    c.FILES["pool"] = name
                    self.publication_plan(selected, section="pool")
                else:
                    raise ValueError("不支持的更新范围: " + name)
            print("更新会创建/更新/停用对应实例；模型就绪并通过普通/SSE及路由验收后才报告完成。")
            if input("输入 y 确认更新以上系统范围；其他输入保留待更新状态: ").strip().lower() != "y":
                print("更新已取消，配置仍为已确认、待更新。")
                return
            ident = uuid.uuid4().hex
            log_path = c.ROOT / "artifacts/kubernetes/control/jobs" / (ident + ".log")
            self.job = {"id": ident, "status": "running", "scopes": scopes, "selected": selected,
                        "log": str(log_path.relative_to(c.ROOT)), "completed": [],
                        "configuration_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in snapshot.items()},
                        "controller_hashes": {name: hashlib.sha256((c.ROOT / "scripts" / name).read_bytes()).hexdigest()
                                              for name in ("control.py", "control_menu.py") if (c.ROOT / "scripts" / name).exists()},
                        "started_at": c.datetime.datetime.now(c.datetime.timezone.utc).isoformat()}
            self.write_job()
            for name in scopes:
                if any((c.ROOT / path).read_bytes() != data for path, data in snapshot.items()):
                    raise RuntimeError("发布计划生成后配置发生变化；未继续更新，请重新查看并确认计划")
                if name != c.FILES["gateway"]:
                    c.FILES["pool"] = name
                target = "gateway" if name == c.FILES["gateway"] else "l1" if name == c.FILES["l1"] else "pool"
                stage = "applying"
                try:
                    self.job.update(scope=name, stage=stage); self.write_job()
                    c.record_publication(name, stage)
                    self.command(["l1" if target=="l1" else "instance", "apply", selected] if selected else ["apply", target], log_path)
                    stage = "verifying"
                    self.job.update(stage=stage); self.write_job()
                    c.record_publication(name, stage)
                    self.command(["l1" if target=="l1" else "instance", "verify", selected] if selected else ["verify", target], log_path)
                    if any((c.ROOT / path).read_bytes() != data for path, data in snapshot.items()):
                        raise RuntimeError("更新期间配置发生变化；保留待更新记录，请重新确认运行结果")
                    c.record_publication(name, "instance_verified" if selected else "verified")
                    self.job["completed"].append({"scope": name, "selected": selected})
                    self.write_job()
                    print("更新完成、验收通过：" + name)
                    if selected:
                        print("选中实例验收通过；专家池整体待更新状态保留，其他已确认修改仍需更新。")
                except BaseException as error:
                    self.job.update(status="failed", error=str(error), stage=stage); self.write_job()
                    c.record_publication(name, "failed", f"{stage}: {error}")
                    print(f"更新未完成：{name}，失败阶段={stage}；保留待更新记录，运行状态请查看日志。")
                    raise
            self.job.update(status="verified", stage="complete"); self.write_job()
        except BaseException as error:
            if self.job and self.job["status"] == "running":
                self.job.update(status="failed", error=str(error)); self.write_job()
            raise
        finally:
            c.FILES["pool"] = previous
            self.saved = c.publication_record().get("pending", [])
            if self.job and self.job["status"] in {"verified", "failed"}:
                self.refresh_resources()

    def overview(self):
        self.context("系统总览 / 实际运行")
        cluster = load_config(ROOT / "deploy/lab/cluster.json")
        rows = instances.inventory(cluster, runtime=True)
        print("实例ID | 专家池 | 实际节点 | 实际 TP/PP | 物理卡 | 运行序列数 | 上下文 | Ready")
        for row in rows:
            p = row.get("parallelism", {})
            runtime = row.get("runtime_values", {})
            print(f"{row['instance']} | {row['pool']} | {row['node']} | {p.get('tp','?')}/{p.get('pp','?')} | "
                  f"{row['physical_devices'] if row['physical_devices'] is not None else '未采集'} | {runtime.get('engine.max_num_seqs','未采集')} | {runtime.get('engine.max_model_len','未采集')} | {row['ready']}")
            if row.get("runtime_error"):
                print("  采集失败：" + row["runtime_error"])
        print("运行参数来自进程快照/实际启动参数，卡号来自实际 Pod；详细对照及修改进入菜单 2 的实例页面。")

    def live_instance(self, ident):
        try:
            cluster = load_config(ROOT / "deploy/lab/cluster.json")
            return instances.inventory(cluster, self.draft.pool(), runtime=True, selected=ident), None
        except c.ERRORS as error:
            return [], str(error)

    def parameter_panel(self, fields, section, ident, rows, error=None):
        saved = c.Draft()
        print("编号 | 参数 | 运行值 | 已保存值 | 待修改值 | 来源 / 状态")
        for i, (key, label) in enumerate(fields.items(), 1):
            wanted = self.draft.value(section, key, ident)
            try:
                stored = saved.value(section, key, ident)
            except KeyError:
                stored = None
            values = [row.get("runtime_values", {}).get(key) for row in rows]
            def show(v):
                if v is None:
                    return "未采集"
                if type(v) is bool:
                    return "开启" if v else "关闭"
                return str(v).replace("heteroserve-lab-", ".")
            running = show(values[0]) if len(values) == 1 else "; ".join(row["uid"][:6] + "=" + show(v) for row, v in zip(rows, values)) if values else "无运行实例/未采集"
            state = "待确认" if wanted != stored else "未核实" if not values or None in values else "一致" if all(v == stored for v in values) else "待更新/更新中"
            source = "全局配置"
            if ident:
                spec = self.draft.file(c.FILES["l1"])["instances"][ident] if section == "l1" else self.draft.instances()[ident]
                group, _, name = key.partition(".")
                source = "实例配置" if not name or name in spec.get(group, {}) else "专家默认"
                if section == "l1":source = "一级实例覆盖" if key in spec else "一级默认"
                if group == "engine" and name not in spec.get("engine", {}) and name not in self.draft.pool().get("runtime", {}).get("model_parameters", {}):
                    source = "共享默认"
            if key == "lifecycle.startup_seconds" and state == "待更新/更新中" and type(stored) is int and all(v == ((stored + 4) // 5) * 5 for v in values):
                state = "一致（5s取整）"
            if not ident:
                running = "逐实例查看" if section != "gateway" else "未采集"
                source = "专家默认（实例可覆盖）" if section != "gateway" else "Gateway 配置"
            print(f"{i}. {label} | {running} | {show(stored) if stored is not None else '新增实例'} | {show(wanted)} | {source} / {state}")
        for row in rows:
            print(f"实际实例 {row['instance']}：节点={row['node']}，卡={row['physical_devices']}，Ready={row['ready']}，退出中={row.get('draining',False)}，UID={row['uid']}，采集时间={row['observed_at']}")
            if row.get("runtime_error"):
                print("运行参数采集失败：" + row["runtime_error"])
            elif row.get("runtime_source") == "live_process_arguments":
                print("旧进程已核实 vLLM 启动参数；请求/排空预算未核实，实例更新后可通过进程快照显示。")
        if error:
            print("运行状态暂不可读取：" + error)
        print("运行值是本次采集的启动配置；已保存值不代表已生效。运行序列数和入口请求预算分别控制引擎和请求准入。")

    def edit_fields(self, value, fields, section, ident):
        keys = list(fields)
        entries = shlex.split(value) if "=" in value else [value]
        before = copy.deepcopy(self.draft.files)
        try:
            for entry in entries:
                selector, sep, supplied = entry.partition("=")
                selector = selector.strip().lower()
                if selector.isdigit() and 1 <= int(selector) <= len(keys):
                    key = keys[int(selector) - 1]
                else:
                    matches = [k for k in keys if k == selector or k.rsplit(".", 1)[-1] == selector]
                    if len(matches) != 1:
                        raise ValueError("未知或不唯一的字段：" + selector)
                    key = matches[0]
                if not sep:
                    if key == "node":
                        cluster = load_config(ROOT / "deploy/lab/cluster.json")
                        supplied = pick("选择目标节点（回车保留）", [names(cluster, n)[0] for n in active_nodes(cluster) if n["host"] in c.read_configs()["engine"]["device_hosts"]])
                    else:
                        supplied = input(f"{fields[key]} 的新值（回车保留）: ").strip()
                if supplied:
                    supplied = supplied.strip()
                    if supplied == "default" and ident:
                        if section == "l1" and key == "max_inflight":
                            self.draft.file(c.FILES["l1"])["instances"][ident].pop(key,None)
                        else:self.draft.inherit(key, ident)
                        continue
                    if key == "node":
                        cluster = load_config(ROOT / "deploy/lab/cluster.json")
                        candidates = {names(cluster, n)[0]: n["host"] for n in active_nodes(cluster) if n["host"] in c.read_configs()["engine"]["device_hosts"]}
                        selected = [n for n, host in candidates.items() if supplied in {n, host, host.rsplit('.',1)[-1], '.' + host.rsplit('.',1)[-1]}]
                        if len(selected) != 1:
                            raise ValueError("请选择可用节点或输入它的 IP/末段")
                        supplied = selected[0]
                    if type(self.draft.value(section, key, ident)) is bool:
                        supplied = {"on": "true", "off": "false", "启用": "true", "停用": "false"}.get(supplied.lower(), supplied)
                    self.draft.set(section, key, supplied, ident)
        except BaseException:
            self.draft.files = before
            raise
        print("修改已加入草稿，请核对本页待修改列；尚未更新运行服务。")
        self.checkpoint()

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
        self.selected_scope = "l1" if section == "l1" else "pool"
        if ident:self.selected = ident
        def observed():
            if section == "l1":
                try:return instances.inventory(load_config(c.ROOT/'deploy/lab/cluster.json'), {"pool":"l1-router"}, runtime=True, selected=ident),None
                except c.ERRORS as exc:return [],str(exc)
            return self.live_instance(ident) if ident else ([],None)
        rows, error = observed()
        while True:
            self.context(title)
            if ident:
                spec = self.draft.file(c.FILES["l1"])["instances"][ident] if section == "l1" else self.draft.instances()[ident]
                p = spec.get("parallelism", {"tp": 1, "pp": 1})
                print(f"请求卡数：{p['tp'] * p['pp']}（TP×PP），卡号由 K8s 分配。")
            self.parameter_panel(fields, section, ident, rows, error)
            print("输入编号或多项赋值 tp=2 pp=1 max_num_seqs=4；字段=default 恢复默认。? 查看字段名；r 刷新；c 保存；u 发布；0 返回。")
            value = input("编辑: ").strip()
            if value in {"", "0"}:
                return
            try:
                if value == "?":
                    for i, (key, label) in enumerate(fields.items(), 1):
                        print(f"{i}. {label}: {key}")
                elif value == "r":
                    rows, error = observed()
                elif value in {"c", "u", "s", "a"}:
                    self.confirm_changes(update=value in {"u", "a"})
                    if value in {"u", "a"} and ident:
                        rows, error = observed()
                else:
                    self.edit_fields(value, fields, section, ident)
            except c.ERRORS as exc:
                print("本次操作未完成：" + str(exc) + "；仍在当前编辑页，之前草稿保留。")

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
                    fields = {key: label for group in c.INSTANCE_GROUPS.values() for key, label in group.items()}
                    self.field_editor("专家与实例 / " + ident + " / 全部参数", fields, "pool", ident)
            elif action == "增加实例":
                self.create_instance()
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
            self.checkpoint()

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

    def publication_plan(self, selected=None, section=None):
        if self.draft.changes():
            raise ValueError("还有草稿未保存；先保存才能生成实际发布计划")
        cluster = load_config(ROOT / "deploy/lab/cluster.json")
        if (section or self.selected_scope) == "l1":
            import manage_l1
            pool=c.read_configs()["pool"]
            rows=manage_l1.plan(cluster,pool,c.read_configs()["engine"],self.draft.file(c.FILES["l1"]),c.read_configs()["gateway"],selected)
            print("一级实例 | 操作 | 节点 | 一实例一张 NPU")
            for row in rows:print(f"{row['instance']} | {row['action']} | {row['node']} | {int(row['enabled'])}")
            return
        pool = c.read_configs()["pool"]
        if "instances" in pool:
            print("实例 | 操作 | 节点 | TP/PP | 申请卡数")
            for row in instances.plan(cluster, pool, selected):
                p = row.get("parallelism", {})
                action = {"create": "新增", "update": "更新/停用", "unchanged": "保持", "remove": "移除"}[row["action"]]
                print(f"{row['instance']} | {action} | {row.get('node') or '--'} | {p.get('tp','-')}/{p.get('pp','-')} | {row['cards'] if row['enabled'] else 0}")
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
                    self.draft.restore()
                elif choice == "丢弃未确认草稿":
                    self.discard_draft()
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
                self.update_system([c.FILES["l1"] if self.selected_scope == "l1" else c.FILES["pool"]], self.selected if action == "仅更新选中实例" else None)
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
                c.main(["l1" if self.selected_scope=="l1" else "instance", "logs", self.selected])
            elif action == "现有监控接入验收":
                c.subprocess.run([c.sys.executable, str(ROOT / "scripts/manage_monitoring.py"), "verify"], check=True)
            else:
                target = {"专家池模型日志": "pool", "专家池 Router 日志": "router", "Gateway 日志": "gateway"}[action]
                c.main(["logs", target])

    def run(self):
        actions = {"系统总览": self.overview, "专家与 vLLM 实例": self.experts,
                   "路由与流量": self.traffic, "算力节点与设备": self.nodes,
                   "变更发布与验收": self.publish, "日志与监控": self.observability,
                   "确认保存草稿（暂不更新）": self.confirm_changes,
                   "确认并更新系统": lambda: self.confirm_changes(update=True)}
        print("HeteroServe 总控 | 编辑草稿 → 确认修改 → 更新系统 → 验收结果")
        try:
            self.refresh_resources()
            while True:
                self.context("主菜单")
                try:
                    self.resources()
                    if self.draft.changes():
                        print("有尚未确认的修改：按 7/c 确认保存，或按 8/u 确认并更新系统。")
                    shortcuts = {"c": "确认保存草稿（暂不更新）", "u": "确认并更新系统", "r": "刷新资源", "+": "新增实例", "jobs": "发布任务", **{key: key for key in self.resource_index()}}
                    action = self.home_choice(list(actions), shortcuts)
                    if not action:
                        if self.exit_menu():
                            return
                        continue
                    if action in self.resource_index():
                        self.select_instance(self.resource_index()[action])
                    elif action == "刷新资源":
                        self.refresh_resources()
                    elif action == "新增实例":
                        self.create_instance()
                    elif action == "发布任务":
                        self.job_status()
                    elif action.startswith("/"):
                        self.resource_filter = action[1:]
                    else:
                        actions[action]()
                    self.checkpoint()
                except c.ERRORS as error:
                    print("操作未完成: " + str(error))
                    print("草稿仍保留；应用失败时查看日志和阶段记录。")
        except (EOFError, KeyboardInterrupt):
            self.checkpoint()
            print("\n已退出；未确认草稿已保留，下次启动自动恢复。")

    def home_choice(self, actions, shortcuts):
        for i, label in enumerate(actions, 1):
            print(f"{i}. {label}")
        print("0/q 退出 | 实例 ID 直接编辑 | + 新增 | c 保存 | u 发布 | r 刷新 | jobs 任务")
        value = input("操作: ").strip()
        if value in {"", "0", "q"}:
            return None
        if value.startswith("/"):
            return value
        if value.lower() in shortcuts:
            return shortcuts[value.lower()]
        if value.isdigit() and 1 <= int(value) <= len(actions):
            return actions[int(value)-1]
        if value in actions:
            return value
        if value == "?":
            print("输入资源 ID 进入参数页；支持编号编辑、多项 key=value、default 恢复默认。c 确认保存，u 查看计划并发布；0 返回/安全退出。")
            return "刷新资源"
        raise ValueError("输入实例 ID、操作编号或 ? 查看帮助")

    def exit_menu(self):
        if not self.draft.changes():
            if self.saved:
                print("已确认待更新的配置记录保留；下次打开可以继续更新。")
            return True
        self.draft.summary()
        choice = pick("尚有未确认草稿，请选择退出方式", ["确认保存并退出（暂不更新）",
                    "确认并更新成功后退出", "丢弃未确认草稿并退出"], "继续编辑（不退出）")
        if choice == "确认保存并退出（暂不更新）":
            self.confirm_changes()
            return not self.draft.changes()
        if choice == "确认并更新成功后退出":
            self.confirm_changes(update=True)
            return not self.draft.changes() and not c.publication_record().get("pending", [])
        if choice == "丢弃未确认草稿并退出":
            self.discard_draft()
            print("已明确丢弃未确认草稿；已保存配置和运行服务保留。")
            return True
        print("已取消退出，草稿保留，继续编辑。")
        return False


def run():
    try:
        Menu().run()
    except c.ERRORS as error:
        print("控制台未完成操作：" + str(error) + "；草稿/配置文件保留，请检查后重试。")
