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
未提供通过此入口直接增加物理节点、启停旧容器、修改镜像/模型身份或自动回滚功能；
这些仍需对应的正式管理流程。更换权重或启用 GPTQ 需要真实模型兼容验收。
配置回退使用 Git 恢复经过验证的配置，再显式 apply 和 verify，不能仅回退 Deployment 而保留新 ConfigMap。

此入口不自动提交或推送 Git，不自动准备所有资产，也不自动执行故障注入实验。
验收记录继续写入现有 `artifacts/kubernetes/`，不复制结果目录。
