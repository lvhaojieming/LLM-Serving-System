# 异构原生 AWQ 接入

`serving.py` 复用三个厂商的独立原生 AWQ 实例、现有 Gateway 和官方
`vllm-router==0.1.15`。配置集中在 `configs/serving.json`，不创建 K8s 集群或容器。
当前 `awq` 完整路径为：HAProxy 官方有界排队 → 实际 Embedding → 一级等概率随机选厂商池 →
该池自己的官方 vLLM Router 按 cache_aware 策略选择健康实例。三个厂商池概率相同，
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

`configs/serving.json` 中 `admission.max_inflight=512`、等待容量 128、等待上限 10 秒。
Gateway 内部保护上限为 640，为客户端取消后仍在清理的流预留空间，避免内外同为 32 时
发生二次 429；对外正常准入仍由 HAProxy 的 512 控制。这些是待压测参数，不宣称最优。
HTTPX 上游连接池显式配置为 672，避免默认 100 连接形成新的隐藏瓶颈。
Gateway 同时需要接收入站连接并持有上游连接。`min_open_files=65536` 显式预留
文件描述符预算，启动时只提升进程软上限，保留原硬上限；硬上限不足则拒绝启动。
预算至少覆盖入口在途、上游连接池和 256 个控制／保活余量。原进程软上限仅 1024，
24 请求/秒阶段接近 1010 个 TCP 连接时指标接口返回 502；现已对原 Gateway PID
使用 `resource.prlimit` 提升软上限，无需重启。后续启动复用正式配置与同一检查。
采集器记录 Gateway 实际软硬上限，预检核对正在运行的进程，防止只改配置而未生效。
指标采集失败的错误对象保留为缺失，停止时给出明确原因，不再当作指标列表读取，
也不将缺失计作零负载。真实 Gateway 的 CPU 后端测试保持 512 个 SSE，验证打开
超过 1024 个 FD 后连续 20 次指标查询正常；该验收不替代模型容量压测。
采用 HAProxy `server maxconn`、`timeout queue`、`option abortonclose`；
`queue(gateway)` ACL 拒绝超额请求，单调度线程串行执行准入与入队。
不能仅依靠 `server maxqueue`：溢出还可能进入 backend 队列。
流式请求从下游请求开始至响应体结束占用名额；正常排队先于 Embedding。
队列满返回 429（Retry-After: 1），官方等待超时返回 503；压测均计入失败。
`max_queue=0` 关闭等待，用于对照实验。每次切换必须开始独立实验，不拼接容量数据。

健康、模型列表和指标接口不排队；管理接口不通过代理暴露。
服务管理器先启动 Gateway 再启动 admission；停止顺序相反，HAProxy 用 SIGUSR1
软停止并等存量响应完成，然后排空 Gateway。只改前端时使用 `--only gateway admission`。
HAProxy 的 client/server 空闲超时为 180 秒，软停止预算为 190 秒。
压测客户端使用 `aiohttp.ClientTimeout(total=150)`，从发出请求开始计时，包含排队、
Embedding 和生成。长输出 1,024 token 在 TPOT 100ms 下仅生成就约需 102 秒，
历史 60 秒总超时会提前截断这类合格请求。新旧超时策略必须分轮保存，不合并容量结果。
客户端连接保活 15 秒，入口保活 75 秒，Gateway 保活 90 秒；Gateway 访问上游的闲置连接
2 秒后退役。恢复判断仅统计完整 150 秒期限已结束的发送批次，避免遗漏仍在途的长请求。
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
初始 32/32/2 秒配置的历史验收见 [`results/gateway_admission.json`](results/gateway_admission.json)：
170 项回归通过；80 请求真实模型突发中 42 个完整成功、22 个等待超时、16 个容量拒绝，
在途和等待采样峰值均为 32。此处超时和拒绝是有意触发的过载验收，不代表正常负载失败率。
当前 mixed 配置调整及历史瓶颈证据见 [`results/admission_tuning.json`](results/admission_tuning.json)：
175 项回归通过，入口重启后仍有 29 个健康 AWQ 实例；入口更新保留六个容器、模型和
Router 进程身份。随后预检发现 `.97:18741` 重复返回 `5!!!!!!!!!!!!!!!`，仅原地重启
`musa-97-1`，未改变容器、设备或运行参数。重启后六个短答案检查、三类并发 mixed
请求及随后三个短答案检查均通过；异常根因未确定，不能把重启恢复视作永久修复。
新容量和稳定性结论仍需独立实验，不能由配置放大直接推断。

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
摩尔原生后端不提供 `/get_load`。池内使用官方 `cache_aware` 的本地在途计数，
`balance_abs_threshold=0`、`balance_rel_threshold=1.0`，有负载差时优先最少在途实例；
`cache_threshold=1.0` 仅在负载相同时保留原生前缀选择，不宣称后端 KV 缓存命中。
这不依赖厂商 `/get_load`，也不把观测到的利用率当路由负载。路由继续用官方健康检查排除故障副本。
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
再通过显式厂商入口逐池核对实际分配。轮询策略要求覆盖全部健康副本；
自适应策略保留实际覆盖，不把有限低并发样本要求成确定的均匀分布。
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
关闭重试，保存逐请求时间、SLO、发送滞后、错误、厂商与副本关联。主负载为 `mixed`：
256 输入／64 输出占 40%，2048 输入／64 输出占 30%，256 输入／1024 输出占 30%。
配置 `workload.profiles` 的 token 数均为实际 chat template 编码后的预算；禁用提前 EOS，
实际输入／输出长度和完整 SSE 必须匹配，不静默截断。
固定 seed 构造 300 个合成请求，包含 120／90／90 个三类请求，每完整 10 请求块包含
4／3／3 个并随机打乱。阶段发送数不是 10 的倍数时，末尾部分保留实际整数比例，
逐请求记录 `workload_name` 和计划长度。整体汇总另含 `by_workload`，
`workload-pressure.csv` 提供三类各自的成功率、达标率和 Goodput。
仅保留 TPOT≤0.1 秒/token 的延迟 SLO；成功率至少 99%，全部已发送请求的 SLO
达标比例至少 95%。TTFT 与端到端延迟不再统计、展示或用于 SLO，60 秒仍是
请求超时界限。保留发送、完成及生成 token 所需内部时间，用于请求率、TPOT、
积压和恢复。容量搜索、两小时验证、三档过载恢复均由配置控制。
按用户要求，当前粗扫从 8 请求/秒开始，之后为 12、16、24、32、48、64，
每档过渡 60 秒、正式测量 300 秒。首档 8 不达标时以它为上界，先查 4 和 2
请求/秒再细搜；不再从 0.25 逐档爬升。此前低速率 mixed 实验单独保留。
队列稳定性使用本实验自定的保守规则：分别比较每个节点及入口在测量前 1/3 与后 1/3
时间内的平均等待请求数，任一项增长超过 2 即未通过。这不是厂商标准，也不是持续无界
积压的统计证明。16 请求/秒阶段虽全部成功且 TPOT 达标，`.96` 前后 100 秒均值从
4.2 增至 10.8，因而未通过。逐请求关联发现摩尔轮询后各副本均为 201 请求，但长输出
分布为 52–70 个；重负载副本经常达到配置的 16 序列上限，KV 缓存仍有余量。
证据与结论边界见 [`results/queue_stability.json`](results/queue_stability.json)。
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

各厂商卡池利用率是实验结果的一部分。复用已有约 5 秒一条的芯片遥测，按
`preflight.json` 冻结的健康 AWQ 实例设备汇总；排除故障实例、Embedding、预留设备和
其他服务。按逻辑芯片等权计算池均值，在入口节点采样时刻对齐各节点最近的有效采样，
过期上限复用实验的 25 秒。输出 `pool-utilization.csv` 时间序列及
`pool-utilization-summary.json`，包含每池设备数、有效采样覆盖率、已完成测量阶段的
利用率均值、池均值峰值和单设备峰值。跨厂商 SMI 的利用率定义可能不同。
缺失、过期或非法数据不记为 0；覆盖不完整时整池均值和峰值留空，仅报告观测部分的值。
当前昇腾健康 AWQ 使用 26 个芯片，`.23` 的 10 个芯片利用率缺失时，仅覆盖 `.24` 的
16/26；摩尔与天数各池各统计 16 个 AWQ 芯片，天数 Embedding 的两个芯片单独排除。
`system_benchmark.py` 的最终导出自动生成该结果；对于已在运行、尚未加载新增导出代码的
本轮实验，在现有容器执行同一模块的仅导出入口，先补齐已采集数据，结束后自动刷新一次：

```bash
python3 heterogeneous_compute/system_benchmark.py \
 --export-pool-utilization --follow \
 --result-dir artifacts/serving/system-benchmark/20261027T073431Z
```

这个处理进程只读取冻结快照、原始遥测和阶段结果并写派生产物，不发模型请求、不新增 SMI
查询、不改变压测配置或服务进程。处理器版本以结果内 SHA-256 与进程启动记录单独保留，
原始压测的源码版本仍以其启动时的 `sources.json` 和 `launch.json` 为准。

固定长度实验 `20261027T054003Z` 已因用户切换 mixed 负载而停止，保留为历史数据，
不与 mixed 容量结果拼接。新混合负载最长需要 2112 上下文；昇腾两个 replica group
通过 `max_model_len=4096` 覆盖旧的 2048 上限，包含原地复用的 `ascend-awq` 服务。
只重启这两个节点原来健康的 13 个实例；`.23` 的三个故障实例保持停止。
其余原生 AWQ 实例已为 4096，不重启。Embedding 的输入按现有消息转文本路径实际验收。
预检核对正在运行的 AWQ 上下文和冻结负载所需长度，防止配置改了而进程未更新。
172 项回归通过，包含三类混合请求的 token 预算、比例、复现性、失败分组和端口复用。
正式容量和稳定性结论以新实验 `preflight.json`、逐请求数据、遥测与最终状态为准。

## cache_aware 路由切换

本轮只变更池内策略，模型并发仍为 16；一级厂商池等概率随机选择、Embedding、
29 个健康实例、设备与容器、mixed、TPOT 和超时预算保持一致。新的实验从 16 请求/秒开始，
`minimum_search_rate=16` 禁止重测更低档位。旧轮询数据独立保留，不充当新策略的容量下界。
低并发自适应路由不保证访问每个空闲副本：每个健康副本先直连生成验收，统一入口再完成
至少 58 个真实 mixed 请求并覆盖三个池，保留实际副本覆盖及请求 ID 对应关系。

官方发布版 0.1.15/0.1.16 的健康检查每 10 轮会清零本地在途计数。采用官方仓库
[PR #216](https://github.com/vllm-project/router/pull/216) 的普通 HTTP 路由修复回移到
0.1.15；该 PR 尚未合并，必须标注为本地带补丁构建，不宣称官方发布版已经修复。
完整 diff 和来源、SHA-256、固定上游提交保留在 `artifacts/optimization/load-aware-routing`；
仅排除与本次普通 HTTP 部署无关且上下文不同的 `src/routers/http/vllm_pd_router.rs`。
源码复用官方 v0.1.15 压缩包，在原 `.60` 容器内构建；没有重建模型环境或容器。

构建复现：在 `artifacts/runtime/vllm-router-source` 解包固定源码，执行
`git apply --exclude=src/routers/http/vllm_pd_router.rs <固定补丁路径>`；
设置 `CARGO_HOME=/workspace/artifacts/runtime/cargo`、
`RUSTUP_HOME=/workspace/artifacts/runtime/rustup`、`PYO3_PYTHON=/usr/local/bin/python3`，
使用 Rust 1.90.0 执行 `cargo build --release --lib --locked -j 4`。
验收后原子替换现有 Router target 中的 `vllm_router_rs.abi3.so`，复用 Python 包及依赖。
`load-accounting.json` 记录二进制校验与验收；启动器核对配置中的 SHA-256，避免误用旧二进制。

真实官方 Router 的 CPU 模拟后端测试覆盖：长 SSE 返回响应头后仍占用计数，
跨越至少 12 轮健康检查保持计数、后续请求选择空闲实例、取消和正常结束后归零。
该测试不请求模型、不查询 SMI；模型与整系统功能验收另行记录。

## 短时边界扫描与队列观测口径

用户要求先观察最大负载和吞吐，因此当前 `run_mode=sweep`，按从高到低的顺序测 64、48、32、24
请求/秒，每档仍为 60 秒过渡、300 秒测量，然后等待请求完成并排空。未达标的档位也
保留并继续扫描后续配置档位；健康、遥测、OOM、客户端容量和排空保护仍会中止实验。
本模式不执行 15 分钟细搜、重复验证、两小时稳定性和过载恢复，也不生成长期容量声明。
输出 `sweep-summary.json`，记录已测最大到达率、通过的最高新测档位、按执行顺序首个未通过档位和最低未通过档位，
以及峰值 Goodput 所在的档位；扫描到 64 不表示已经证明硬件绝对上限。

`require_queue_stability=false` 将平均等待数增长改为观测指标。`queue_stable`、
`queue_changes`、阶段队列统计、各池利用率继续保留，CSV 增加 `queue_stability_required`。
成功率至少 99%、TPOT 100ms 达标率至少 95%、有效发压和资源约束仍为通过门槛，
429、503、超时和异常截断继续计入失败。队列容量 128 和等待上限 10 秒也没有取消。

`20261027T092553Z` 的完整 16 请求/秒阶段只因队列增长未通过，新口径下可解释为
通过，但原阶段和原始请求不改写、不拼接。初次粗扫预检 `20261027T102326Z` 发现
`musa-97-1` 再次返回 `5!!!!!!!!!!!!!!!`。原容器内重启后六次固定答案通过，
但一次 256/1024 长输出后连续三次再次返回异常答案，因此已停止并隔离该实例。
其余七个摩尔实例分别通过 1,024-token 长输出及前后固定答案检查。
当前扫描冻结六节点 28 个健康实例：昇腾 13、摩尔 7、天数 8；不得与原 29 实例
数据拼接。从配置的最高档 64 向下扫描，不导入旧下界，16 仅作为历史数据保留。
异常根因仍未确定，不能凭这个复现认定为硬件故障，也不能把重启视为永久修复。
可复用的可选 `search_reference` 支持严格核对负载、SLO、服务配置、健康实例、
容器／进程身份和精确 workload 后重新解释完整阶段；只允许判定口径、运行模式和
扫描速率改变。使用时在新目录记录 `search-reference.json`，也不能替代长期验证。

`20261027T103740Z` 在预检结束、24 请求/秒初始阶段收到从高到低扫描要求，
通过 STOP 文件停止；其未完成阶段仅保留中止证据，不合入新的降序扫描。
每档先排空，避免高压档残留请求污染后续低压档。
