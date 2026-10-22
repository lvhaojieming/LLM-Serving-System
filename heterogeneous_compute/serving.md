# 异构原生 AWQ 接入

`serving.py` 复用三个厂商的独立原生 AWQ 实例、现有 Gateway 和官方
`vllm-router==0.1.15`。配置集中在 `configs/serving.json`，不创建 K8s 集群或容器。
跨厂商 PP 执行器仍由原 `manage.py` 管理，本入口不将 AWQ 请求拆成跨厂商 PP。

| 入口模型名 | 服务 | 分配 |
| --- | --- | --- |
| `awq` | 官方 Router 按 round_robin 分配 | 目标为六台节点的 32 个 AWQ TP2 副本，实际仅选择健康副本 |
| `ascend-awq` | `.24:18424` | 昇腾芯片 0/1，TP2，项目 INT4 适配器 |
| `musa-awq` | `.97:18497` | 摩尔设备 6/7，复用已有 TP2 实例 |
| `iluvatar-awq` | `.60:18460` | 天数设备 2/3，复用已有 TP2 实例 |
| `embedding` | `.58:18458` | 天数逻辑设备 12/13，Qwen3-Embedding-0.6B BF16，TP2/PP1 pooling |

统一地址 `http://192.168.162.60:18600`，生成用 `/v1/chat/completions`，
embedding 用 `/v1/embeddings`。昇腾容器映射八张物理卡对应的十六个芯片设备，
`ascend-awq`、`musa-awq`、`iluvatar-awq` 是原主实例的诊断入口；常规调用
使用 `awq`，覆盖全部副本。保持 `resource_limits_enabled=false`。

`replica_groups` 按设备集合展开服务和 Router worker，不复制项目或实现。
启动前验证 AWQ 分区与预留设备无重叠且覆盖完整清单。天数的“AWQ 使用 8 张卡”
在本配置中指 8 个 `ixsmi` 逻辑设备；`.58` 的 Embedding 另用 2 个设备，不挤占 AWQ 预算。

| 节点 | AWQ 目标分配 | 其他服务 | 配置覆盖的逻辑设备 |
| --- | --- | --- | --- |
| 昇腾 `.23` | 8 个 TP2 副本 | — | 0–15，共 16 |
| 昇腾 `.24` | 8 个 TP2 副本 | 原 Embedding 已停止 | 0–15，共 16 |
| 摩尔 `.97` | 4 个 TP2 副本 | — | 0–7，共 8 |
| 摩尔 `.96` | 4 个 TP2 副本 | — | 0–7，共 8 |
| 天数 `.60` | 4 个 TP2 副本，0/1、2/3、4/5、6/7 | Gateway、Router 使用 CPU；8–15 不分配给 AWQ | AWQ 共 8 |
| 天数 `.58` | 4 个 TP2 副本，0/3、4/5、6/7、8/9 | Embedding 12/13；保留 G9 1/2、MinerU 15；10/11/14 不使用 | AWQ 8，Embedding 另 2 |

六台 AWQ 配置共使用 64 个逻辑加速设备；Embedding 另用 2 个。
配置清单不表示全部设备已经验收通过。
驱动暴露的逻辑设备数不等于物理板卡数。

AWQ 权重来源与算子差异见 `configs/native_awq.json`。摩尔和天数现有实例
显式保留原配置的 eager；`.58` 新副本默认厂商图模式并开启 cudagraph metrics，
昇腾新启动默认厂商图模式，`.24` 开启 cudagraph metrics。
两台摩尔节点的全部 AWQ 副本必须显式设置 `--disable-custom-all-reduce`，
统一使用标准 MCCL。`MCCL_P2P_DISABLE=1` 仅约束 MCCL，不能替代这个 vLLM 参数。
启动验收同时核对实际命令和日志中的 `disable_custom_all_reduce=True`。
实际图捕获／回放必须依据运行证据，不能仅凭省略 eager 参数判断。
本轮不启用原 AWQ/GPTQ 学习路由头，其标签不代表三家硬件。`auto` 不开放。
Embedding 单独提供向量接口，下载模型不等于恢复旧训练头。
摩尔原生后端不提供 `/get_load`，因此使用官方轮询策略，不把策略名当成已经完成
设备负载感知；路由继续使用官方健康检查排除故障副本。

## 下载与复现

官方小模型是 [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)。
`.58` 的模型挂载在容器内为只读；在宿主机复用现有工具写入对应模型目录：

```bash
python3 /home/zhangjinhao/LLM-Serving-System/shared/download_model.py \
 --model Qwen/Qwen3-Embedding-0.6B \
 --revision dd3e37eb506b3addbd930037796e25d349ae9010 \
 --destination /home/zhangjinhao/models/Qwen3-Embedding-0.6B \
 --report /home/zhangjinhao/LLM-Serving-System/artifacts/model-download/embedding/download.json --workers 4
```

逐文件校验大小、SHA-256 和 safetensors 结构。单文件模型与多分片索引均支持。
下载完成标志、源清单和依赖版本保留于模型目录和报告目录。
官方 Router 隔离依赖复用 `/workspace/artifacts/runtime/vllm-router`；wheel 版本、
来源与 SHA-256 见 `artifacts/dependencies/router-manifest.json`。
启动保留厂商原有 `PYTHONPATH` 和共享依赖，不重新建立环境。
AWQ 权重分发复用 `shared/distribute_model.py` 和 `configs/awq_distribution.json`；
已校验的模型复用，新增节点分发逐文件校验后才发布完成标志。
`.58` 分发复用 `--peer iluvatar150-01`，不会重新分发已验证的其他节点。
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
六台上的旧项目 BF16／同厂商 PP／跨厂商 PP 已停用；`.58` 按进程组、
启动时间、模型命令和设备环境核对后停止 17 个旧进程，保留外部 G9 和 MinerU。
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
当前天数每节点八设备 AWQ 与 `.58` 独立 Embedding 的验收摘要见
[`results/iluvatar_serving.json`](results/iluvatar_serving.json)。两台天数各四个 TP2 副本均完成
真实生成；`.58` 四个 AWQ 副本均验证 FULL decode 图回放。Embedding 原生与 Gateway
接口均返回有效的 1024 维向量；pooling 图回放尚未证明，不宣称其图模式性能。
64 个轮询请求覆盖全部 29 个健康 worker，另外三个 `.23` worker 仍未恢复，
因此全设备验收保持 `passed=false`。本结果不包含最大稳定请求率或持续稳定性结论。
`.23` 的逻辑设备 4–9 在 `torch.npu.set_device` 阶段出现 507033、TSD/HDC 错误；
对应三个副本已停止，不能计入实际覆盖。其他同容器昇腾副本已真实生成并回放图。
同机存在其他用户容器；未重置设备、重启主机或重启共享驱动服务。
`.97` 设备 2/3 的副本曾在权重加载完成后长期停留在启动试运行。
关闭 vLLM 自定义 AllReduce、走标准 MCCL 后，已通过连续／并发、长输入、SSE
和官方 Router 分发验证；这不证明硬件 P2P 已损坏，也不等于长期稳定性验收。
同一标准 MCCL 配置应用于全部摩尔副本，修复及滚动更新结果按实际验收摘要记录。
全部摩尔副本的标准 MCCL 验收与复现信息见
[`results/musa_standard_mccl.json`](results/musa_standard_mccl.json)。
结果中的未通过项必须在恢复后重新执行 `verify`，不能只依据进程存在或健康接口判断通过。

## 系统压测入口

`benchmark.py --system-config configs/system_benchmark.json` 使用固定到达率独立发送，
关闭重试，保存逐请求时间、SLO、发送滞后、错误、厂商与副本关联。主负载 256/128，
TTFT≤2 秒、TPOT≤0.1 秒/token、E2E≤20 秒；成功率至少 99%，全部已发送请求的 SLO
达标比例至少 95%。容量搜索、两小时验证、三档过载恢复均由配置控制。
实际执行模式混合，不能把本系统结果解释为三厂商在相同图模式下的硬件比较。

在 `.96` 现有容器内，以 `/workspace` 为工作目录启动控制器；结果目录每次使用唯一
实验标识，不覆盖旧结果：

```bash
python3 heterogeneous_compute/benchmark.py \
 --system-config heterogeneous_compute/configs/system_benchmark.json \
 --result-dir artifacts/serving/system-benchmark/<RUN_ID>
```

控制器等待采集器最多 120 秒。在六台宿主机各启动同一采集模块，`NODE` 分别取
`ascend910c-01`、`ascend910c-02`、`s5000-01`、`s5000-02`、
`iluvatar150-01`、`iluvatar150-02`；它复用 Docker、厂商 SMI 和后端原生 metrics：

```bash
python3 /home/zhangjinhao/LLM-Serving-System/heterogeneous_compute/benchmark_telemetry.py \
 collect --node <NODE> --central http://192.168.162.96:18890
```

同时在 `.60` 现有特权容器启动只保存请求 ID 与地址的观测器：

```bash
python3 /workspace/heterogeneous_compute/benchmark_telemetry.py \
 observe --central http://192.168.162.96:18890
```

采集器与观测器在控制器完成或中止后自动退出；在结果目录写入 `STOP` 可停止负载，
不会停止或重启模型服务。健康集合、服务进程、容器身份改变、遥测过期、客户端发不出
目标压力或抓包丢失会中止该实验，不能拼接部署前后的数据。SMI 超时后不反复调用，
缺失设备指标如实保留为空；`.23` 的设备指标覆盖仍有此限制。

该入口已通过单元测试及 64 个短请求的 ID 关联烟测，当前没有完成新的固定长度
预检、容量搜索或两小时稳定性实验。短请求抓包零丢失不证明高压力下一定零丢失；
正式结果必须以该轮 `preflight.json`、逐请求数据、遥测与最终状态为准。
2048 输入／128 输出超出当前昇腾 2048 上下文配置，长输入补充不能直接启动，
也不能静默截断；两个补充场景仍须单独冻结 SLO。
