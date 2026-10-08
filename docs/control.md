# HeteroServe 总控操作说明

在 `.209` 的正式项目目录执行：

```bash
cd /root/zhangjinhao/LLM-Serving-System
python3 scripts/control.py
```

总控复用现有模型环境、镜像、管理脚本和固定外层 Docker 容器。卡号由 Kubernetes 自动分配，界面展示实际结果，不提供物理卡号锁定。

## 默认实例工作台

启动后直接显示 AWQ/GPTQ 实例列表，包含工作负载状态、实际节点、配置 TP/PP、申请卡数和草稿/待发布标记。首次只读取项目 Pod 状态，不逐个加载模型或采集设备信息。

日常操作围绕对象进行：

```text
awq-01                       # 直接打开 AWQ 实例全部参数
gptq-01                      # 直接打开 GPTQ 实例，无需先换专家池
/awq                         # 筛选 AWQ；单独输入 / 清除筛选
r                            # 重新读取工作负载状态
+                            # 新增向导：专家 → ID → 节点 → TP/PP → 参数页
c                            # 确认保存草稿
u                            # 查看发布计划，确认更新和验收
jobs                         # 最近发布任务：阶段、结果、日志位置
q                            # 安全退出
```

同名 ID 存在于不同池时，使用完整标识 `awq/awq-01`。新实例 ID 可回车自动生成，TP/PP 可回车采用 1；PP 的默认批处理预算不足时，向导明确初始化满足约束的预算。
首页的“就绪”来自 Kubernetes 工作负载状态；申请卡数及 TP/PP 列是配置值。实际进程参数、卡号和 UID 在实例页查询，不把配置列当成运行值。
状态带采集时间，`r` 刷新；API 不可达时显示未核实，仍允许编辑配置。

草稿每次编辑后写入 `artifacts/kubernetes/control/draft.json`。正常返回、Ctrl+C 或输入流结束时保留，重开自动恢复专家与实例上下文。
确认保存后清除工作副本，保留待发布记录；明确丢弃时只清除工作副本。两会话写入同一草稿时拒绝覆盖，并把冲突会话的修改保存在 `artifacts/kubernetes/control/conflicts/`，错误消息会显示具体路径。配置被其他操作修改时也拒绝覆盖；通过“查看差异”检查，再明确丢弃旧草稿重新编辑，不能直接覆盖冲突文件。

发布先展示新增/更新/保持/移除的实例清单。确认后创建独立任务记录，逐阶段显示等待时间和状态；CLI 的完整输出写入任务日志，不在控制台展开大段 JSON。
任务记录与日志在 `artifacts/kubernetes/control/jobs/`，包含发布范围、配置/控制代码摘要、阶段、已完成范围和失败原因。失败时保留待发布记录，不自动回退已经提交的 Kubernetes 更新。
操作仍在当前控制进程中执行，不是跨终端后台任务；中断后先查看任务记录及实际资源，再恢复或重试。

界面参考 [K9s 的资源视图、筛选与对象操作](https://k9scli.io/topics/commands/)和 [OpenShift 的应用状态视图](https://docs.redhat.com/en/documentation/openshift_container_platform/4.7/html/web_console/web-console-overview)；任务历史借鉴 [AWS ECS 的服务更新与部署记录](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/update-service-console-v2.html)，发布流程沿用 [Kubernetes 的声明式配置与差异检查](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/declarative-config/)。这些是公开产品工作流的参考，不代表项目已具备它们的全部能力。项目继续使用自己的参数表单和部署脚本。

## 六组功能

| 主菜单 | 功能 |
|---|---|
| 1 系统总览 | 实际实例 ID、专家、节点、TP/PP、物理卡、Ready、重启和 Pod UID |
| 2 专家与 vLLM 实例 | 选择专家、选择具体实例、部署/并行/引擎/退出配置、数量调整 |
| 3 路由与流量 | Gateway 入口、池级发现与排队、实例请求预算与超时 |
| 4 算力节点与设备 | 节点状态、逐卡占用、接入/退出/恢复节点 |
| 5 变更发布与验收 | 草稿差异、集中保存、发布影响、指定实例/专家发布、资产准备、回退 |
| 6 日志与监控 | 实例、模型池、Router、Gateway 日志及已有监控验收 |
| 7 确认保存草稿 / `c` | 从主菜单直接确认全部草稿，保存后暂不更新服务 |
| 8 确认并更新系统 / `u` | 从主菜单直接确认修改、查看计划、更新并验收 |

每页显示当前位置、专家、实例 ID、未确认草稿数和已确认待更新文件数，编号只作用于当前页。选择具体实例后，部署、TP/PP、引擎、退出和请求预算集中在同一编辑页，不必切换参数子菜单。
子菜单的 `0` 返回上一级并保留本次草稿。主菜单有草稿时按 `0/q`，必须选择确认保存后退出、确认并更新成功后退出、明确丢弃后退出，或继续编辑；回车/`0` 默认继续编辑。
已确认待更新范围保留到下次启动。确认或更新取消/失败时保留相应草稿或待更新范围。编辑成功后已写入工作副本，可在下次启动恢复；工作副本不等于确认配置，也不会自动部署。
保存配置与部署分开，显示配置值不代表运行值。

## 编辑、保存与发布

1. 菜单 2 选择专家，再选择具体实例 ID。
2. 选择“选择并编辑实例”，直接查看该实例全部参数。输入编号修改一项，或一次输入 `tp=2 pp=1 max_num_seqs=4 max_inflight=6`。节点可以选择列表，也可以输入 `node=.210`；布尔值支持 `true/false`、`on/off`。
3. 可跨实例、跨专家连续修改，草稿保留在内存。
4. 在编辑页输入 `c`，或菜单 5 选择“确认修改（保存全部草稿）”。核对实例 ID、参数旧值→新值、启用实例数和申请卡数，输入 `y` 确认保存。服务此时不变，界面显示“已确认待更新”。
5. 编辑页输入 `u`，或菜单 5 选择“确认修改并更新系统”，可以连续完成确认和更新。也可以稍后选“更新所有已确认范围”。先显示所有待更新专家/Gateway 的发布计划，再输入 `y` 更新。
6. 更新逐项执行部署与验收：成功显示“更新完成、验收通过”并移除该范围的待更新记录；失败显示应用/验收阶段，停止后续更新，保留未完成范围。菜单 5“查看更新状态”可查看最近阶段与失败原因。

`s/a` 分别兼容 `c/u`。新增、移除或调整实例数量同样先进入草稿，再通过菜单 5 确认并更新。
取消确认保留草稿；取消更新保留已确认配置及待更新范围。更新失败可能已经改变部分运行实例，应先查询实际状态和日志；总控不将失败描述为自动回退。

## 看清运行值与修改值

实例编辑页并列显示“运行值 / 已保存值 / 待修改值”，以及参数来源（实例覆盖、专家默认、共享默认）和状态（一致、待确认、待更新、未核实）。
例如正在运行的 `max_num_seqs=2`，保存文件也是 2，本次输入 4 后显示 `2 / 2 / 4`；确认保存后显示 `2 / 4 / 4`，完成更新后重新采集显示 `4 / 4 / 4`。
输入 `r` 重新采集运行状态；修改不会自动刷新或替换运行值。每次采集显示时间、实际节点、物理卡号、Ready、退出状态及 Pod UID；滚动更新同时存在旧/新 Pod 时，运行值逐个 UID 显示。

运行引擎和请求参数来自运行进程的只读 `/configuration` 启动快照；启动探针和退出总预算来自当前 Pod 的实际配置。读取 ConfigMap 或期望 Deployment 不能证明旧进程已加载新参数，因此不作为运行值来源。
旧版本进程缺少快照接口时，直接读取实际 vLLM API 进程的启动参数，未核实的请求/排空预算显示未知；更新实例后可显示完整快照。进程身份和采集 Pod UID、节点、实例 ID 不一致时拒绝使用结果。
运行状态读取失败会显示原因并保留编辑功能，不把已保存值填到运行列。Gateway 全局参数和专家默认参数页明确显示未采集/逐实例查看，不能用一个默认值代表所有实例的生效值。

输入 `max_num_seqs=default` 可以删除该实例覆盖值，恢复继承专家/共享默认值；同样适用于请求和退出参数。
同一行多项修改中出现无效字段或无效输入时，撤销整行修改，保留此前草稿，并留在编辑页继续修正。跨参数约束（如 PP 的 token 预算）在确认保存时统一校验。

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
