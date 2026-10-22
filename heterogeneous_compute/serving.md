# 异构原生 AWQ 接入

`serving.py` 复用三个厂商的独立原生 AWQ 实例、现有 Gateway 和官方
`vllm-router==0.1.15`。配置集中在 `configs/serving.json`，不创建 K8s 集群或容器。
跨厂商 PP 执行器仍由原 `manage.py` 管理，本入口不将 AWQ 请求拆成跨厂商 PP。

| 入口模型名 | 服务 | 分配 |
| --- | --- | --- |
| `awq` | 官方 Router 按 round_robin 分配 | 目标为五台节点的 32 个 AWQ 副本，实际仅选择健康副本 |
| `ascend-awq` | `.24:18424` | 昇腾芯片 0/1，TP2，项目 INT4 适配器 |
| `musa-awq` | `.97:18497` | 摩尔设备 6/7，复用已有 TP2 实例 |
| `iluvatar-awq` | `.60:18460` | 天数设备 2/3，复用已有 TP2 实例 |
| `embedding` | `.24:18427` | 昇腾芯片 2，Qwen3-Embedding-0.6B BF16 |

统一地址 `http://192.168.162.60:18600`，生成用 `/v1/chat/completions`，
embedding 用 `/v1/embeddings`。昇腾容器映射八张物理卡对应的十六个芯片设备，
`ascend-awq`、`musa-awq`、`iluvatar-awq` 是原主实例的诊断入口；常规调用
使用 `awq`，覆盖全部副本。保持 `resource_limits_enabled=false`。

`replica_groups` 按设备集合展开服务和 Router worker，不复制项目或实现。
启动前验证设备集合无重叠且完整覆盖，`.24` 的 Embedding 芯片 2 单独保留。

| 节点 | AWQ 目标分配 | 其他服务 | 配置覆盖的逻辑设备 |
| --- | --- | --- | --- |
| 昇腾 `.23` | 8 个 TP2 副本 | — | 0–15，共 16 |
| 昇腾 `.24` | 7 个 TP2＋1 个 TP1 副本 | 芯片 2 的 Embedding | 0–15，共 16 |
| 摩尔 `.97` | 4 个 TP2 副本 | — | 0–7，共 8 |
| 摩尔 `.96` | 4 个 TP2 副本 | — | 0–7，共 8 |
| 天数 `.60` | 8 个 TP2 副本 | Gateway、Router 使用 CPU | 0–15，共 16 |
| 天数 `.58` | 未激活 | SSH 不可达 | 待恢复连接后核对 |

五台配置共覆盖 64 个逻辑加速设备；这不表示全部设备已经验收通过。
驱动暴露的逻辑设备数不等于物理板卡数。
`.58` 保留为禁用的待接入项，不能在状态未知时计作已部署。

AWQ 权重来源与算子差异见 `configs/native_awq.json`。摩尔和天数现有实例
显式保留原配置的 eager；昇腾新启动默认厂商图模式，`.24` 开启 cudagraph metrics。
实际图捕获／回放必须依据运行证据，不能仅凭省略 eager 参数判断。
本轮不启用原 AWQ/GPTQ 学习路由头，其标签不代表三家硬件。`auto` 不开放。
Embedding 单独提供向量接口，下载模型不等于恢复旧训练头。
摩尔原生后端不提供 `/get_load`，因此使用官方轮询策略，不把策略名当成已经完成
设备负载感知；路由继续使用官方健康检查排除故障副本。

## 下载与复现

官方小模型是 [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
`.24` 的现有容器内执行以下命令，复用现有可写 `/quantization` 挂载：

```bash
python3 /workspace/shared/download_model.py \
 --model Qwen/Qwen3-Embedding-0.6B \
 --revision dd3e37eb506b3addbd930037796e25d349ae9010 \
 --destination /quantization/models/Qwen3-Embedding-0.6B \
 --report /workspace/artifacts/model-download/embedding/download.json --workers 4
```

逐文件校验大小、SHA-256 和 safetensors 结构。单文件模型与多分片索引均支持。
下载完成标志、源清单和依赖版本保留于模型目录和报告目录。
官方 Router 隔离依赖复用 `/workspace/artifacts/runtime/vllm-router`；wheel 版本、
来源与 SHA-256 见 `artifacts/dependencies/router-manifest.json`。
启动保留厂商原有 `PYTHONPATH` 和共享依赖，不重新建立环境。
AWQ 权重分发复用 `shared/distribute_model.py` 和 `configs/awq_distribution.json`；
已校验的模型复用，新增 `.23`、`.96` 分发逐文件校验后才发布完成标志。
临时传输授权与密钥在分发结束后清理。

工作站在仓库根目录执行：

```powershell
python heterogeneous_compute/serving.py sync --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py start --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py status --ssh-config "$env:USERPROFILE/.ssh/config"
python heterogeneous_compute/serving.py verify --ssh-config "$env:USERPROFILE/.ssh/config"
```

`sync` 仅同步本入口需要的源码、配置和测试。启动不等待长时间加载，
完整覆盖就绪后再验收。控制器在 `.60` 容器内执行健康检查和请求。
`--only router gateway` 可只操作接入层。停止本入口管理的进程：

```powershell
python heterogeneous_compute/serving.py stop --ssh-config "$env:USERPROFILE/.ssh/config"
```

默认先排空并停止 Gateway，再停止 Router 和各模型副本。
Gateway/Router/Embedding/原生 AWQ 校验 boot ID、PID 启动时间与进程组，昇腾 AWQ
复用量化部署器的模型／命令校验。不恢复旧 PP 实验。
摩尔和天数的原有 AWQ 经命令参数、设备环境、启动时间和独立进程组核验后
接管为正式服务，之后也由本入口启停。停止仅作用于登记的进程身份。
可达五台上的旧 BF16／同厂商 PP／跨厂商 PP 已停用，`.58` 的旧任务状态尚未知。
`legacy_baselines_enabled=false` 会在
本入口启动时检查并拒绝混入这些旧进程，不删除其模型和有效结果。

`/ready` 的 `expert_coverage_complete=true` 才表示全部配置入口就绪；
HTTP 200 也可能只表示部分服务可用。Gateway 还核对官方 `/workers` 中的
健康 URL 集合是否恰好等于全部 32 个预期副本，避免少数主实例健康掩盖缺失副本。
验收要求全部副本逐一完成真实生成、
各入口普通与完整 SSE 请求正确，并通过官方 Router 的 worker 计数验证小批
并发请求确实经过每一个健康 worker；请求数为 `max(24, 2 × 预期副本数)`，
并发 4，仅为功能验证。
设备缺失时仍记录全部 worker 的逐一生成结果，最终 `passed=false` 并返回非零状态，
不能把可用部分的功能通过当成全设备验收。Embedding 检查批量数量、1024 维、有限数值、
重复输入一致性及不同输入区分。

日志、PID、状态和验收报告位于 `artifacts/serving/`，不提交原始运行产物。
功能验收不代表完成吞吐上限、持续稳定性、同条件性能比较或 K8s 生命周期验收。

本轮全设备部署结果摘要见 [`results/all_devices_serving.json`](results/all_devices_serving.json)。
`.23` 的逻辑设备 4–9 在 `torch.npu.set_device` 阶段出现 507033、TSD/HDC 错误；
对应三个副本已停止，不能计入实际覆盖。其他同容器昇腾副本已真实生成并回放图。
同机存在其他用户容器；未重置设备、重启主机或重启共享驱动服务。
`.97` 的慢启动副本保持原启动任务，不因等待较久而反复重启。
结果中的未通过项必须在恢复后重新执行 `verify`，不能只依据进程存在或健康接口判断通过。
