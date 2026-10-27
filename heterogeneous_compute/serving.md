# 异构原生 AWQ 接入

`serving.py` 复用三个厂商的独立原生 AWQ 实例、现有 Gateway 和官方
`vllm-router==0.1.15`。配置集中在 `configs/serving.json`，不创建 K8s 集群或容器。
当前 `awq` 完整路径为：HAProxy 官方有界排队 → 实际 Embedding → 一级等概率随机选厂商池 →
该池自己的官方 vLLM Router 按轮询选择健康实例。三个厂商池概率相同，
不按池中的副本数量加权；Embedding 向量暂不参与随机决策。
最新两级路径验收见 [`results/two_level_routing.json`](results/two_level_routing.json)，
此前的单 Router 和天数分配结果作为历史验收保留。
跨厂商 PP 执行器仍由原 `manage.py` 管理，本入口不将 AWQ 请求拆成跨厂商 PP。

| 入口模型名 | 服务 | 分配 |
| --- | --- | --- |
| `awq` | Embedding → 随机厂商池 → 池内官方 Router | 目标为六台节点的 32 个 AWQ TP2 副本，实际仅选择健康副本 |
| `ascend-awq` | 昇腾池官方 Router，本机 18601 | `.23`／`.24` 全部已配置昇腾副本 |
| `musa-awq` | 摩尔池官方 Router，本机 18603 | `.96`／`.97` 全部已配置摩尔副本 |
| `iluvatar-awq` | 天数池官方 Router，本机 18605 | `.58`／`.60` 全部已配置天数副本 |
| `embedding` | `.58:18458` | 天数逻辑设备 12/13，Qwen3-Embedding-0.6B BF16，TP2/PP1 pooling |

统一地址 `http://192.168.162.60:18600`，生成用 `/v1/chat/completions`，
embedding 用 `/v1/embeddings`。昇腾容器映射八张物理卡对应的十六个芯片设备，
`ascend-awq`、`musa-awq`、`iluvatar-awq` 用于指定厂商池诊断，直接进入该池二级 Router；
常规整系统测试使用 `awq`，包含实际 Embedding 和一级随机选择。
保持 `resource_limits_enabled=false`。

## 入口排队

排队复用 [HAProxy 2.8 官方 HTTP 队列](https://docs.haproxy.org/2.8/configuration.html)，
不在 Python Gateway 中重写队列。现有 `.60` 容器增加一个 HAProxy 进程；
容器和模型进程不替换。公网入口仍为 18600，Gateway 移至回环 18610，
HAProxy 指标仅监听回环 18612。现有官方 vLLM Router 继续负责池内路由。

`configs/serving.json` 中 `admission` 配置初始等待容量 32、等待上限 2 秒，
并发数复用 `gateway.max_inflight=32`。这些是待压测的初始参数，不宣称最优。
采用 HAProxy `server maxconn`、`timeout queue`、`option abortonclose`；
`queue(gateway)` ACL 拒绝超额请求，单调度线程串行执行准入与入队。
不能仅依靠 `server maxqueue`：溢出还可能进入 backend 队列。
流式请求从下游请求开始至响应体结束占用名额；正常排队先于 Embedding。
队列满返回 429（Retry-After: 1），官方等待超时返回 503；压测均计入失败。
`max_queue=0` 关闭等待，用于对照实验。每次切换必须开始独立实验，不拼接容量数据。

健康、模型列表和指标接口不排队；管理接口不通过代理暴露。
服务管理器先启动 Gateway 再启动 admission；停止顺序相反，HAProxy 用 SIGUSR1
软停止并等存量响应完成，然后排空 Gateway。只改前端时使用 `--only gateway admission`。
HAProxy 的 client/server 60 秒是空闲超时，不是总时限；压测客户端保持
`aiohttp.ClientTimeout(total=60)`，从发出请求开始计时，包含排队、Embedding 和生成。
TPOT SLO 仍为 100ms，不包含排队；单独记录 `x-heteroserve-queue-ms`，不恢复 TTFT/E2E 指标。

遥测保留 `haproxy_backend_current_queue{proxy="gateway"}` 与累计最大队列值；
不与 server 同一队列重复相加，也不和模型 waiting 相加作为唯一积压数。
timeline 额外记录 `gateway_waiting`，每阶段按 Gateway 和实例分别记录采样均值与峰值。
5 秒采样峰值可能漏掉短暂尖峰；官方累计最大值覆盖进程生命周期，不能冒充某阶段峰值。
排空、稳定性、恢复判断同时检查入口与实例队列，采集缺失保留缺失并停止实验。

运行时复用 Ubuntu 官方包 `haproxy=2.8.30-0ubuntu0.24.04.1` 和
`liblua5.4-0=5.4.6-3build2`，仅解包到 `artifacts/runtime/haproxy`，不覆盖系统库，
不安装 systemd 服务、不重建环境或容器。该目录的 `manifest.json` 保留包和二进制校验值。
首次准备（后续复用该目录）在现有 `.60` 容器内执行：

```bash
mkdir -p /workspace/artifacts/runtime/haproxy
cd /workspace/artifacts/runtime/haproxy
apt-get download haproxy=2.8.30-0ubuntu0.24.04.1 liblua5.4-0=5.4.6-3build2
dpkg-deb -x haproxy_2.8.30-0ubuntu0.24.04.1_amd64.deb .
dpkg-deb -x liblua5.4-0_5.4.6-3build2_amd64.deb .
```

安装版本的 vLLM Router 0.1.15 虽带 queue 参数，但其准入使用可补充令牌桶，
归还令牌发生在响应头返回时，队列处理器还会派生额外等待任务，不能作为本次
严格流式并发和总等待容量的实现。依据是该版本的
[`src/middleware.rs`](https://github.com/vllm-project/router/blob/v0.1.15/src/middleware.rs)，
没有修改或复制其源码。HAProxy 真实代理行为测试复用 `tests/test_serving.py`，
设置 `HETEROSERVE_HAPROXY` 和相应 `LD_LIBRARY_PATH` 后运行 pytest；
覆盖 SSE 占用、32 请求突发拒绝、等待超时、取消清理和控制接口。
本轮实际结果见 [`results/gateway_admission.json`](results/gateway_admission.json)：
170 项回归通过；80 请求真实模型突发中 42 个完整成功、22 个等待超时、16 个容量拒绝，
在途和等待采样峰值均为 32。此处超时和拒绝是有意触发的过载验收，不代表正常负载失败率。

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

AWQ 权重来源与算子差异见 `configs/native_awq.json`。厂商图模式为默认设置，
`vendors.<vendor>.enforce_eager` 可记录经过验证的厂商例外；副本级配置优先。
昇腾和天数两台的 AWQ 使用厂商图模式，天数开启 cudagraph metrics。
摩尔暂时显式保留 V0 eager：当前安装栈的 V0 图模式在并发请求下超时，
V1 则因 PH1 KV block-size 与 attention backend 的要求冲突而未能初始化。
这些失败不能视为图模式验收通过；复现和最终落地结果见
[`results/runtime_optimization.json`](results/runtime_optimization.json)。
两台摩尔节点的全部 AWQ 副本必须显式设置 `--disable-custom-all-reduce`，
统一使用标准 MCCL，保留 `MCCL_P2P_DISABLE=1`，并设置 `MCCL_SHM_DISABLE=0`
启用同机共享内存通道。此前同时关闭 P2P 和 SHM 的 Socket 路径已被实测确认
会显著增加 TP2 通信开销。`MCCL_P2P_DISABLE=1` 不能替代这个 vLLM 参数。
启动验收同时核对实际命令和日志中的 `disable_custom_all_reduce=True`。
实际图捕获／回放必须依据运行证据，不能仅凭省略 eager 参数判断。
天数当前验证的是 FULL decode 图回放，256-token prefill 仍为 NONE；
这不是全阶段都使用图执行的声明。
本轮不启用原 AWQ/GPTQ 学习路由头，其标签不代表三家硬件。`auto` 不开放。
`route_embeddings={"awq":"embedding"}` 要求 `awq` 每次先完成真实向量计算，
输入是按消息顺序用换行连接的 `role: content`。Embedding 失败则不发送 AWQ，
向量暂不作为选池依据。独立 `/v1/embeddings` 接口继续保留。
响应头记录选池、请求 ID 和 Embedding 阶段耗时。
摩尔原生后端不提供 `/get_load`，因此使用官方轮询策略，不把策略名当成已经完成
设备负载感知；路由继续使用官方健康检查排除故障副本。
每厂商独立复用一个官方 Router 进程，API／指标端口为 18601/18602、
18603/18604、18605/18606，均仅绑定本机。一级健康快照缓存 5 秒，
使用官方 `/readiness` 判断池是否仍有可用实例；`/health` 的全量健康语义
不能用来排除仍有 13 个健康实例的昇腾池。选池后不自动重试。

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
三个注册表的健康 URL 联集是否恰好等于全部 32 个预期副本，避免少数主实例健康掩盖缺失副本。
验收要求全部副本逐一完成真实生成、
各入口普通与完整 SSE 请求正确，并通过官方 Router 的 worker 计数验证小批
并发请求覆盖三个厂商池；统一入口发出 `max(24, 2 × 预期副本数)` 个随机请求，
再通过显式厂商入口逐池核对轮询覆盖，避免把有限随机样本要求成确定的均匀分布。
并发 4，仅为功能验证。
设备缺失时仍记录全部 worker 的逐一生成结果，最终 `passed=false` 并返回非零状态，
不能把可用部分的功能通过当成全设备验收。Embedding 检查批量数量、1024 维、有限数值、
重复输入一致性及不同输入区分。

日志、PID、状态和验收报告位于 `artifacts/serving/`，不提交原始运行产物。
功能验收不代表完成吞吐上限、持续稳定性、同条件性能比较或 K8s 生命周期验收。

本轮全设备部署结果摘要见 [`results/all_devices_serving.json`](results/all_devices_serving.json)。
2026-10-22 天数每节点八设备 AWQ 与 `.58` 独立 Embedding 的历史验收摘要见
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
仅保留 TPOT≤0.1 秒/token 的延迟 SLO；成功率至少 99%，全部已发送请求的 SLO
达标比例至少 95%。TTFT 与端到端延迟不再统计、展示或用于 SLO，60 秒仍是
请求超时界限。保留发送、完成及生成 token 所需内部时间，用于请求率、TPOT、
积压和恢复。容量搜索、两小时验证、三档过载恢复均由配置控制。
当前实际执行模式为昇腾／天数 AWQ 图模式、摩尔共享内存 V0 eager。
不能把本系统结果解释为三厂商在相同图模式下的硬件比较。

优化或滚动更新后，可先对指定的健康副本执行有界验收，复用已有的精确答案检查，
再验证固定 token 预算、完整 SSE、单请求与小批并发的 TPOT。`--workload` 使用
之前冻结的聊天消息列表，响应中的实际 input/output tokens 必须与参数完全一致：

```bash
python3 heterogeneous_compute/benchmark.py \
 --serving-config heterogeneous_compute/configs/serving.json \
 --services musa-96-2 \
 --workload artifacts/optimization/awq-runtime/workload.json \
 --result-dir artifacts/optimization/<UNIQUE_ACCEPTANCE_ID> \
 --num-prompts 4 --concurrencies 1 4 --input-tokens 256 --output-tokens 128 \
 --seed 20261022 --timeout-seconds 60
```

此检查不替代完整容量、过载恢复和两小时稳定性实验。配置或执行模式变化后，
必须使用新 run ID，重新冻结健康清单；不把旧 eager／Socket 结果与新结果拼接。

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
