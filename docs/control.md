# HeteroServe 总控操作说明

在 `.209` 的正式项目目录执行：

```bash
cd /root/zhangjinhao/LLM-Serving-System
python3 scripts/control.py
```

总控复用现有模型环境、镜像、管理脚本和固定外层 Docker 容器。卡号由 Kubernetes 自动分配，界面展示实际结果，不提供物理卡号锁定。

## 六组功能

| 主菜单 | 功能 |
|---|---|
| 1 系统总览 | 实际实例 ID、专家、节点、TP/PP、物理卡、Ready、重启和 Pod UID |
| 2 专家与 vLLM 实例 | 选择专家、选择具体实例、部署/并行/引擎/退出配置、数量调整 |
| 3 路由与流量 | Gateway 入口、池级发现与排队、实例请求预算与超时 |
| 4 算力节点与设备 | 节点状态、逐卡占用、接入/退出/恢复节点 |
| 5 变更发布与验收 | 草稿差异、集中保存、发布影响、指定实例/专家发布、资产准备、回退 |
| 6 日志与监控 | 实例、模型池、Router、Gateway 日志及已有监控验收 |

每页显示当前位置、专家、实例 ID、未确认草稿数和已确认待更新文件数，编号只作用于当前页。编辑页可以输入参数编号再输入新值，也可以直接输入 `tp=2`、`max_num_seqs=3` 等本页字段名。
`0` 返回上一级并保留本次草稿；退出整个程序丢弃未保存草稿，已确认待更新范围保留到下次启动。保存配置与部署分开，显示配置值不代表运行值。

## 编辑、保存与发布

1. 菜单 2 选择专家，再选择具体实例 ID。
2. 在“部署与并行”“推理引擎”“健康与退出”中修改参数；请求准入及超时统一放在菜单 3。
3. 可跨实例、跨专家连续修改，草稿保留在内存。
4. 在编辑页输入 `c`，或菜单 5 选择“确认修改（保存全部草稿）”。核对实例 ID、参数旧值→新值、启用实例数和申请卡数，输入 `y` 确认保存。服务此时不变，界面显示“已确认待更新”。
5. 编辑页输入 `u`，或菜单 5 选择“确认修改并更新系统”，可以连续完成确认和更新。也可以稍后选“更新所有已确认范围”。先显示所有待更新专家/Gateway 的发布计划，再输入 `y` 更新。
6. 更新逐项执行部署与验收：成功显示“更新完成、验收通过”并移除该范围的待更新记录；失败显示应用/验收阶段，停止后续更新，保留未完成范围。菜单 5“查看更新状态”可查看最近阶段与失败原因。

`s/a` 分别兼容 `c/u`。新增、移除或调整实例数量同样先进入草稿，再通过菜单 5 确认并更新。
取消确认保留草稿；取消更新保留已确认配置及待更新范围。更新失败可能已经改变部分运行实例，应先查询实际状态和日志；总控不将失败描述为自动回退。

保存时校验候选配置、检查并发修改，并保留最近保存前的配置。多文件保存失败会恢复本次已写入文件。
连续保存涉及多个专家时，待更新范围累计保留；程序重启后可以继续更新。确认后的配置以及发布计划涉及的配置发生变化时，阻止执行过期计划。
指定单实例发布不会隐式应用其他已保存变更；即使该实例验收通过，专家池整体待更新记录仍保留，避免漏掉其他实例或 Router 参数。完成专家池整体更新与验收后才清除。
待更新、确认文件摘要及最近更新阶段复用 `artifacts/kubernetes/control/last-save.json`，不增加另一套配置入口。最新保存前的配置仍用于回退；回退后也进入待更新状态。

## 稳定实例与配置层级

每个逻辑 vLLM 实例有稳定 ID，例如 `awq-01`、`awq-02`、`gptq-01`；Pod 重建后 UID 和物理卡号可能改变，稳定 ID 不变。
各实例使用独立 Deployment 和 ConfigMap，复用一套生成模板、镜像、适配器及模型文件。专家池 Service、Router 和 PDB 保持池级。

配置层级为：共享部署/硬件默认值 → 专家默认推理值 → 实例覆盖值。
`deploy/pools/ascend-awq.json` 和 `ascend-gptq.json` 的 `instances` 是实例数量的唯一来源；启用数量从清单计算，不维护另一份手写 replicas。
减少数量会停用多余实例并保留配置；移除实例配置并应用后，清理对应工作负载及 ConfigMap，保留权重、节点卷和外层容器。

```json
"instances": {
  "awq-01": {
    "node": "heteroserve-lab-209",
    "enabled": true,
    "parallelism": {"tp": 1, "pp": 1},
    "engine": {},
    "traffic": {"max_inflight": 8},
    "lifecycle": {"startup_seconds": 1200, "drain_seconds": 240, "termination_seconds": 300}
  }
}
```

实例设置按组整理：

| 设置组 | 参数 |
|---|---|
| 部署与并行 | 节点、启用状态、TP、PP；请求卡数自动计算 |
| 推理引擎 | 上下文长度、运行序列数、批处理 token 预算、每卡内存比例、eager 模式 |
| 健康与退出 | 启动、排空、终止预算 |
| 路由与流量 | 每实例 max_inflight、请求超时；池的默认预算、发现缓存、核心并发与排队超时 |

同池启用实例的 `max_model_len` 必须一致，保证路由可互换；不同实例可以设置不同 TP/PP、max_num_seqs、内存比例和请求预算。
上下文上限需要统一调整专家默认值或相关实例覆盖值；草稿可一次修改多个实例，保存时统一检查。

## 单节点多卡 TP/PP

本轮仅支持同一个节点内的多卡，使用已有 multiprocessing 后端；跨物理节点并行后置。
实例申请 `TP × PP` 张独占 NPU，与服务实例数量分别显示。例如一个 TP2/PP2 实例申请四张卡。
不能在 engine defaults 中再配置 tensor_parallel_size/pipeline_parallel_size，以免与实例并行配置重复。

现有转换后 AWQ/GPTQ 权重无需重新量化。多卡进程使用项目内的 INT4 加载扩展，共用原有适配器，不改写外部共享适配目录。
当前 PP 使用已有 V0 引擎，`max_num_batched_tokens` 必须不小于 `max_model_len`；例如上下文 4096 时将批处理预算设为 4096。
未经当前权重、镜像、加载扩展和引擎配置组合验收的并行配置，会先用不匹配生产发现的临时模型 Pod 做普通/SSE 验收，通过后才发布正式实例。
临时 Pod/ConfigMap 按 UID 和归属清理；失败记录保留，原有实例保持可用。

发布前核对节点 Ready、硬件标签、调度状态、空闲卡、CPU/内存请求以及已发布的模型清单。
实例滚动更新默认先启动一个新副本，Ready 后排空旧副本，因此需要预热/更新用的额外卡；卡不足时停止发布，先暂停或迁移实例释放资源。
首次迁移旧共享 Deployment 时逐个启动和验收独立实例，再减少旧副本，最后清理确认废弃的旧 Deployment/ConfigMap。

## 命令行

```bash
# 查看稳定实例配置与实际卡分配
python3 scripts/control.py --pool awq instance list
python3 scripts/control.py --pool awq instance list --live

# 同时调整一个实例的并行、引擎和请求预算；--write 才保存
python3 scripts/control.py --pool awq instance set awq-02 tp=2 pp=1 engine.max_num_seqs=3 traffic.max_inflight=4 --write
python3 scripts/control.py --pool awq instance plan awq-02
python3 scripts/control.py --pool awq instance apply awq-02
python3 scripts/control.py --pool awq instance verify awq-02

# PP 需要联动调整 V0 的 token 预算
python3 scripts/control.py --pool gptq instance set gptq-01 pp=2 engine.max_num_batched_tokens=4096 --write

# 数量、启停及移除
python3 scripts/control.py --pool awq instance resize 5 --write
python3 scripts/control.py --pool awq instance add awq-06 --node heteroserve-lab-211 --tp 2 --pp 1 --write
python3 scripts/control.py --pool awq instance pause awq-06 --write
python3 scripts/control.py --pool awq instance resume awq-06 --write
python3 scripts/control.py --pool awq instance remove awq-06 --write
# 清单增减后应用整个专家池；单实例修改可只应用该 ID
python3 scripts/control.py --pool awq apply pool
python3 scripts/control.py --pool awq verify pool

# 回退最近保存的配置；不会自行回退进程，随后应用和验收
python3 scripts/control.py rollback --write
python3 scripts/control.py --pool awq instance apply awq-02
python3 scripts/control.py --pool awq instance verify awq-02

# 日志与流量
python3 scripts/control.py --pool awq instance logs awq-02
python3 scripts/control.py set gateway max_inflight=24 --write
python3 scripts/control.py apply gateway
python3 scripts/control.py verify gateway
```

原 `show/set/check/apply/status/prepare/verify/logs` CLI 保留。`set pool replicas=N` 对独立实例池等价于调整清单启用数量；节点属于具体实例，应使用 `instance set ID node=...`。
`set engine ...` 修改当前专家默认值，实例覆盖值优先；所有权重身份、镜像、卡型仍由正式配置管理，镜像环境不做预检或自动传输。

## 实际卡号与资源

系统总览显示实际 Pod 的 `/identity`，核对 Pod UID、节点和稳定实例 ID；Gateway 也显示占用的实际卡。
卡视图按节点反查物理卡 → 实例/服务，并结合驱动独占记录判断空闲。无法核实的占用显示为未知，不把利用率 0 当作空闲。
显示采集时间，Pod 重建后重新采集；容器内 `npu:0` 不当作宿主机物理卡 0。

## 节点、发现与监控

物理节点入群/退出仍是全局操作，外层容器、卷、模型和原暂停服务保留。
启用实例绑定的节点先迁移/停用相关实例并应用，再退出节点；不自动修改其他专家或停止未知工作负载。

```bash
python3 scripts/control.py node remove 10.107.206.216
python3 scripts/control.py node remove 10.107.206.216 --execute
python3 scripts/control.py node add 10.107.206.216 --execute
python3 scripts/control.py node recover 10.107.206.216 --execute
```

新接入节点提供可用设备，实例由清单显式配置，不再为了入群隐式增加模型数量。
新节点或实例迁移目标缺少权重时，使用当前专家 `prepare pool` 发布资产；K8s 不会复制宿主机目录中的权重。
每池一个官方 vLLM Router；实例独立请求预算通过 Pod 注解进入发现，池准入上限按 Ready 实例预算之和计算。
就绪、退出和 Pod UID 过滤仍保留；不重写官方负载均衡算法，不启用跨 Router 计数协调或 HPA。
监控复用已有 Prometheus/Grafana，通过菜单 6 查询和验收，不再部署第二套。

总控配置写入与发布共用进程锁，不要绕过总控并行修改文件或运行底层部署命令。
运行和验收记录集中在 `artifacts/kubernetes/`，并行能力缓存放在 `state/capabilities/`；总控不自动推送 Git。
