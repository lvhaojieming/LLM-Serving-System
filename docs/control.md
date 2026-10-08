# 总控操作说明

`scripts/control.py` 是运维命令入口，复用原来的 JSON 和管理脚本。
不增加常驻服务、不占 NPU，也不进入推理请求路径。
本地可查看、修改、校验配置；连接集群的操作在 `.209` 正式项目目录执行：
`/root/zhangjinhao/LLM-Serving-System`。总控本身只依赖标准库，已在管理节点的 Python 3.9.9 验证；模型容器继续使用原有 Python 环境。

## 日常流程

直接运行即可进入中文交互菜单，无需记忆参数名：

```bash
python3 scripts/control.py
```

菜单每次读取现有正式配置，显示当前专家池、副本数、Gateway 节点和各层并发预算。
选择 `1/2/3` 查看对应参数，按序号输入新值，可连续修改多个参数。
`s` 预览并保存，`a` 预览保存后执行部署计划和真实验收，`0` 返回并丢弃未保存的编辑。
保存前会校验整个候选配置；外部修改导致编辑内容过期时，需返回菜单重新读取。
参数列表显示“原值 -> 待保存值”。这些是本地配置值，不代表集群已生效。

主菜单 `4` 可查看完整 JSON（包括模型、镜像、概率路由资产及节点配置）；
只允许修改已支持的常用参数。`6` 查询集群状态，`7` 应用已有配置并验收，
`8` 单独验收，`9` 查看日志，`10` 准备已有资产。
保存配置与执行部署分别确认，退出菜单不会撤销已保存配置或已执行操作。
部署失败停止后续验收，配置不会自动回滚，可查看日志修正后重试。
菜单部署复用同样的 CLI，原命令行模式继续可用于脚本：

```bash
python3 scripts/control.py show
python3 scripts/control.py set pool replicas=3
python3 scripts/control.py set pool replicas=3 --write
python3 scripts/control.py check
git diff -- deploy
```

`set` 默认预览，加 `--write` 才保存。一次可以修改同一配置中的多个字段，全部校验通过才原子写入。
修改不会自动部署。若在本地编辑，提交并同步到 `.209` 的同一个正式项目，再执行：

```bash
python3 scripts/control.py apply pool --plan
python3 scripts/control.py apply pool
python3 scripts/control.py verify pool
python3 scripts/control.py status all
python3 scripts/control.py logs router --tail 100
```

`apply pool` 同时发布模型和二级 Router，保证请求预算配置都更新；更新不是跨组件原子事务，失败会停止，需查日志并修正后重试。
`verify pool` 执行真实普通/SSE 模型验收及 Router 路由检查，会产生测试请求。
`apply` 本身不代表模型验收通过。离线 `check` 不证明节点在线、权重存在或设备空闲。

## 常用参数

```bash
# 一级入口并发和默认输出长度
python3 scripts/control.py set gateway max_inflight=24 default_max_tokens=256 --write
python3 scripts/control.py apply gateway
python3 scripts/control.py verify gateway

# 每实例准入预算与引擎并发是两层不同参数
python3 scripts/control.py set pool requests_per_worker=8 --write
python3 scripts/control.py set engine model_parameters.max_num_seqs=2 model_parameters.max_model_len=4096 --write
python3 scripts/control.py apply pool
python3 scripts/control.py verify pool

# 当前四个推理节点（只能选择已加入且准备好模型资产的节点）
python3 scripts/control.py set pool replicas=4 model_nodes=heteroserve-lab-209,heteroserve-lab-210,heteroserve-lab-211,heteroserve-lab-216 --write

# Gateway 换节点：先将现有资产发布到目标节点，再更新部署
python3 scripts/control.py set gateway deployment.node=heteroserve-lab-210 --write
python3 scripts/control.py prepare gateway
python3 scripts/control.py apply gateway
python3 scripts/control.py verify gateway

# 排空与退出时间要一起调整，给引擎退出保留至少 35 秒
python3 scripts/control.py set pool drain_seconds=300 termination_seconds=360 --write
```

完整可编辑字段和当前值由 `show gateway|pool|engine` 查看。
`gateway` 对应 `deploy/gateway.json`，`pool` 对应 `deploy/pools/ascend-awq.json`，
`engine` 对应 `deploy/lab/npu.json` 的模型参数。没有第二份总控配置。
参数变更可能导致 Pod 更新并暂时降低容量，单 Gateway/Router 更新可能中断入口。
现有外层 Docker 容器与镜像继续复用。

`status all` 查询节点、模型池和 Gateway；`logs pool|router|gateway` 获取有行数上限的日志。
`prepare pool` 复用并发布 Router 依赖，仅在目标节点缺少资产时使用，不需要每次启动执行。
`apply all --plan` 查看更新模型、Router、Gateway 的命令顺序。
服务暂停期间不要执行 `apply`，因为它会恢复配置声明的副本数。

当前保留一个二级 Router、一实例一张 NPU、固定容器身份和取消 CPU/内存硬限制的约定。
总控不自动停止旧模型容器、不修改镜像/模型身份，也不自动回滚。
更换权重或启用 GPTQ 需要真实模型兼容验收。
配置回退使用 Git 恢复经过验证的配置，再显式 apply 和 verify，不能仅回退 Deployment 而保留新 ConfigMap。

此入口不自动提交或推送 Git，不自动准备所有资产，也不自动执行故障注入实验。
验收记录继续写入现有 `artifacts/kubernetes/`，不复制结果目录。

## 接入和退出算力节点

在 `.209` 运行总控，选择 **11**，再选择接入、退出或恢复中断的退出操作，输入完整 IP。
菜单先读取真实状态并显示计划，确认后执行。命令行等价入口：

```bash
# 只预检和显示计划，不修改集群
python3 scripts/control.py node remove 10.107.206.216
# 保持目标模型副本数，迁移完成后退出；保留容器、卷、镜像和权重
python3 scripts/control.py node remove 10.107.206.216 --execute
# 明确重新接入刚才退出的节点，复用相同容器 ID
python3 scripts/control.py node add 10.107.206.216 --execute
# 退出中途失败后，先检查计划，再明确恢复
python3 scripts/control.py node recover 10.107.206.216
python3 scripts/control.py node recover 10.107.206.216 --execute
```

`add/remove` 不隐式修改模型副本数量。新候选节点数量不能超过副本数；需要时先通过菜单 2 调整副本数。
节点接入后仍由调度器放置副本，软拓扑分布不等于严格一机一副本。验收要求新节点上实际运行一个正式模型实例并被 Router 请求覆盖，否则操作不能报告成功。

接入仅适配现有 Ascend 910B4、8 张独占卡的实验节点。检查授权地址、外层容器身份和驱动独占记录；
已有占卡服务不会被自动停止。新节点首次接入确实需要一个长期复用的 K3s 节点容器；重新接入复用原容器。
接入流程直接使用目标节点已准备好的镜像环境，不预检镜像名称或缓存，也不自动拉取、导入、传输镜像。
首次创建节点容器使用本地既有镜像，禁止自动拉取；镜像缺失或运行环境不兼容时，由实际容器/Pod 启动及推理验收报告错误。
镜像启动引用仍由现有部署配置提供。已有模型和适配器先核对 SHA-256，缺少的目录才复制；文件不一致时停止，不覆盖。
权重发布、设备插件、双 Pod 设备隔离复用已有脚本。临时单卡模型 Pod 使用 `expert-candidate` 标签，
不匹配正式 Service 或 Router，完成模型、普通/SSE 推理以及 Router 到目标 Pod 的网络检查后清理。
最后更新正式节点候选配置、应用 Deployment，验证整个专家池和 Gateway。

退出先检查剩余节点的就绪、标签、污点、NPU、CPU、内存及 Pod 数量预算，
同时读取物理设备独占记录，不能仅凭 allocatable 判断有空闲卡。
承载 Gateway、Router 候选配置或其他未纳管工作负载的节点被拒绝，必须先完成对应服务迁移。
仅支持退出工作节点，控制平面不能通过此入口退出。
随后 cordon、drain，遵守 PDB、preStop 和退出预算，不使用 `--force` 或 `--disable-eviction`。
`--delete-emptydir-data` 仅在已核验该节点只含项目专家 Pod 和允许的 DaemonSet 后使用，清除的是这些 Pod 的临时缓存。
服务迁移通过普通/SSE 及全副本路由验收后，记录退休状态，停止固定节点容器、删除 Kubernetes Node 记录。
不运行卸载脚本，不删除 Docker 容器、卷、宿主机模型和数据目录。

`cluster.json` 的 `nodes` 和 `locked_container_ids` 同时保留活跃与已退出节点的身份档案；
`retired_node_hosts` 标记已退出的工作节点，常规 `up` 和集群验证忽略它们。
已授权的 `.208` 存入 `authorized_additional_hosts`，不改变旧容器的创建指纹。
部署候选节点和可用设备仍使用现有 pool、npu 配置，没有第二套业务配置。
编辑 pool 的节点亲和性会触发模型滚动更新，可能需要多轮加载模型。

节点流程和总控配置写入使用同一个进程锁；不要绕过总控并行运行底层部署命令或直接改配置。
失败保留 `artifacts/kubernetes/nodes/<末段IP>/<操作>.json` 的阶段、错误和验收记录。
排空失败不会继续停止节点，保持 cordon 供检查；`recover` 只处理有对应中断退出记录的同一容器。
已经成功退出的节点用 `add` 明确重新接入；普通 `up` 不会擅自恢复暂停状态。
接入失败的节点可能已注册但仍处于禁止生产模型调度状态，应检查最后阶段、修复原因后重试 `add`；不得把注册成功当作推理验收成功。

行为依据：[Kubernetes drain](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_drain/)、
[K3s 节点身份与重新注册](https://docs.k3s.io/architecture#node-password-secrets)。
