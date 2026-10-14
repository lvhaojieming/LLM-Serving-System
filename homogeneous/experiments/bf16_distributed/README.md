# Qwen3-14B BF16 / AWQ / GPTQ 跨节点性能实验

本目录独立保存部署、运行和压测代码，不复制主工程、不创建环境或镜像。
复用 `scripts/manage_lab.py` 的私有集群入口、`manage_pool.py` 的工作负载模板、`manage_nodes.py` 的权重校验/传输，以及 `manage_npu.py` 的硬链接发布。

当前对照配置按用户要求比较部署策略：BF16 在 `.210`、`.211` 使用 TP2×PP2（四张 NPU），现有自定义 W4A16 AWQ/GPTQ 在 `.211` 使用 TP2×PP1（各两张 NPU）。三个模型依次启停测量，输入集合、并发 8、固定输出 64 token 一致。卡数和跨节点通信不同，结果不单独代表量化算子加速比。使用现有 vLLM 0.8.4 和旧转换权重，不能作为新 compressed-tensors 模型或 SGLang 的测速结果。此前同节点 TP4×PP1 对照结果保留。
物理卡号通过 `/identity` 记录；不锁定卡号。机内 TP 不跨越 PP 节点。
现有 AWQ/GPTQ 服务按用户要求保持暂停，不能在本实验结束后自动恢复。

重新量化由同目录 `quantize.py` 和 `config.json` 的 `requantization` 配置控制，使用原始 BF16 权重、llmcompressor 0.6.0、compressed-tensors 0.10.2。
AWQ 使用 W4A16_ASYM，GPTQ 使用 W4A16 且关闭 actorder；group size 为 128，lm_head 不量化。
两种算法复用同一 256 条校准数据、seed 20261010、最大长度 512。校准数据排除已固定的 203 条测速输入及重复原始文本。
量化 Job 复用已导入的推理镜像和 `.211` 固定实验容器，由 Kubernetes 分配一张隔离 NPU；一张卡仅用于校准，推理对照仍为四卡 TP4×PP1。
缺少的量化依赖安装在项目的 `moqe-runtime/quantization-deps` 目录，通过 PYTHONPATH 加载，不覆盖原推理环境中的 compressed-tensors 0.9.3、Torch 或 vLLM。
依赖与 `calibration.jsonl` 必须先发布到 `.211` 的 `heteroserve-assets/quantization/`，随后启动：

```bash
python3 experiments/bf16_distributed/manage.py quantize --expert awq
# 两个量化任务可使用不同的隔离 NPU；推理测速仍需逐模型启停。
python3 experiments/bf16_distributed/manage.py quantize --expert gptq
```

任务拒绝覆盖既有 Job 和模型目录。输出先写 `.incomplete` 目录，验证 compressed-tensors W4A16 配置、全部 280 个投影压缩、tokenizer 一致性后才发布正式目录。
输出保留原始权重校验、校准样本、工具版本、脚本 SHA、配置和各文件 SHA。此检查不等于完成推理质量或速度验收。
官方 AscendTurbo 推理尚未部署；不得使用旧 `moqe_ascend_int4` 测量冒充重新量化结果。
参考 [官方 W4A16 量化指导](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_5910012.html)，官方启动要求 `VLLM_PLUGINS=ascend_vllm`、V1、`--quantization compressed-tensors`、`ascend_turbo_graph_config.enabled=true`，关闭 eager 与自定义适配器。
已核对的 6.5.908 官方配套要求为 vLLM 0.9.0、CANN 8.2.RC1、HDK 25.2.1；软件包需从 Support-E 获取。当前软件版本不满足此组合，宿主机驱动未修改，推理与测速仍待官方包及配套环境就绪。

在 `.209` 正式项目目录运行：

```bash
python3 experiments/bf16_distributed/manage.py prepare
python3 experiments/bf16_distributed/manage.py apply
python3 experiments/bf16_distributed/manage.py status
python3 experiments/bf16_distributed/manage.py prepare-data
python3 experiments/bf16_distributed/manage.py benchmark
python3 experiments/bf16_distributed/manage.py collect
```

修改 `config.json` 可以调整节点、TP/PP、引擎参数和压测规模。PP 数须等于不同节点数量。当前 PP1 对照的启停状态及原暂停服务清单保存在 `system/diagnostics/pp1-sequential-state.json`；仅对该清单中原本运行的实验 Deployment 允许恢复，且校验其 UID，不能恢复原先暂停的正式服务。
`verify` 是可选的功能验收入口，本轮只测性能，不需要执行。
`benchmark` 在现有 Gateway Pod 内后台测量；`collect` 显示进度，完成后导出 `artifacts/bf16_distributed/benchmark.json` 和日志，不发起额外推理请求。
AWQ/GPTQ 使用同一套实现，以 `--expert awq` 或 `--expert gptq` 切换，权重路径和 manifest 复用正式专家池配置；不启用原来暂停的专家 Deployment 或 Router。
例如：

```bash
python3 experiments/bf16_distributed/manage.py prepare --expert awq
python3 experiments/bf16_distributed/manage.py apply --expert awq
python3 experiments/bf16_distributed/manage.py status --expert awq
# 两个阶段都 Ready 后启动；GPTQ 将上述 awq 替换为 gptq。
python3 experiments/bf16_distributed/manage.py benchmark --expert awq
python3 experiments/bf16_distributed/manage.py collect --expert awq
```

量化结果分别写入 `artifacts/bf16_distributed/awq/` 和 `gptq/`，BF16 基线保留。
两个量化检查点都是已转换布局的 W4A16（INT4 权重、FP16 激活），复用现有适配器和 TP 扩展，不重新量化。
Ray 计算进程通过显式 worker 类导入适配器；Ray 控制进程保持原环境。
三种模型共用 BF16 准备好的 203 个 input token 序列，并验证词表文件相同，因此 GPTQ 文件中的不同聊天模板不会改变输入。
顺序测量每个模型，避免同时发起多个模型的压测。BF16 的 BF16 激活与量化模型的 FP16 激活不同，结果是当前实际部署方案的比较。

## 全系统入口与路由耗时

本节及后续初始基线、INT4 调用开销数据是缓存修复和 Q/K norm 优化前的历史测量。当前部署与最新对照见最后一节，历史结果不作为修复后的效果验收。

BF16 已有结果保留为模型服务基线，不重复测量。完整系统使用 `model=auto`，保留原 checkpoint 的概率与阈值选择策略：Gateway → 一级 NPU 路由 → 所选池官方 vLLM Router → AWQ/GPTQ 模型。
当前两个一级实例的 `embedding_graph=false`；可选图代码存在，但本次不启用尚未在当前环境重新验收的图路径。

```bash
# 在 AWQ/GPTQ 组件测量已导出、没有运行中请求时执行；更新实验专家 Pod 的计时接口。
python3 experiments/bf16_distributed/manage.py prepare-chat
python3 experiments/bf16_distributed/manage.py apply-routing
# 等待两个专家池、两级路由和 Gateway 均就绪，再测量。
python3 experiments/bf16_distributed/manage.py benchmark --entry gateway
python3 experiments/bf16_distributed/manage.py collect --entry gateway
```

`apply-routing` 复用现有 Gateway 和两份一级路由，在两个分布式实验池各配置一个官方 vLLM Router。
Gateway 的本次生效配置与完整部署清单存于 `artifacts/bf16_distributed/system/`；正式池原来暂停的 Deployment/Router 继续保持零副本。
选中的原始 ShareGPT 首轮消息会重新分词，逐条确认与 BF16 请求的 input token 序列完全一致；不更换提示或重新抽样。

逐请求保存 trace ID、实际选中专家、一级及模型 Pod UID、一级排队/计算/RPC 时间、二级转发区间和实际图命中状态。
可观测前置路由及网络开销 = Gateway 从 handler 入口到下游响应头的时长 − 模型 worker 从入口到引擎响应头的时长。
二级路由及网络开销 = 二级守卫前置处理时间 + 官方 Router 往返响应头时间 − worker 的上述时长；避免把后端等待当成 Router 调度时间。
这些是嵌套区间的本地单调时钟时长，不比较跨机器时间戳；包含节点间转发，不宣称是官方 Router 内部的纯选择计算时间。
占比先逐请求除以客户端 TTFT/总耗时，再统计均值/p95/p99；不能拿独立的两个 p99 相除。
上述路由开销不包含响应头之后的 SSE 转发成本，端到端 TTFT/TPOT/总耗时则包含完整链路。

当前完整系统有两个四卡专家池和两个单卡一级路由，共预留十张 NPU，并记录实际选中专家的请求数；BF16 基线使用四张模型 NPU。
结果必须同时说明卡数和入口不同，不视为同资源预算的速度提升。

全系统实测：同一组 200 个请求全部成功，AWQ 141 次、GPTQ 59 次；两个一级路由各服务 100 次，图启用/命中/回退均为 0。

| 实际入口 | TTFT p95/p99（ms） | TPOT p95/p99（ms/token） | 输出吞吐量（token/s） |
| --- | --- | --- | ---: |
| BF16 模型 Service（已有基线） | 365.86 / 1217.46 | 100.94 / 101.40 | 80.27 |
| AWQ 模型 Service（组件对照） | 357.73 / 484.49 | 107.57 / 109.15 | 74.98 |
| GPTQ 模型 Service（组件对照） | 354.94 / 5260.38 | 109.80 / 110.15 | 72.76 |
| 完整自动路由系统 | 1446.54 / 3237.68 | 163.65 / 293.25 | 58.94 |

完整系统测量耗时 434.351 秒，总 token 吞吐量 118.842 token/s，请求吞吐量 0.46046 request/s；请求总耗时 p95/p99 为 21.052/37.476 秒。
当前部署在这个并发档没有吞吐收益，尾延迟也高于 BF16 模型服务基线，不能据此宣称量化或两级路由已经加速。

前置路由及转发平均 88.248 ms（p95/p99：156.709/254.339 ms）。一级 RPC 平均 80.556 ms，其中实际路由计算 62.608 ms、排队 10.755 ms；二级路由及转发平均 4.194 ms。
逐请求占比的均值：TTFT 25.093%、总耗时 0.613%；将各请求耗时合计后计算的加权占比：18.321% 和 0.581%，两种口径分别保存在结果中。
完整系统数据和逐请求分段见 `artifacts/bf16_distributed/system/benchmark.json`，对照及专家分布见 `system/comparison.json`；实际配置、UID/卡号和源码校验值见 `system/reproducibility.json`。
初版不支持对运行中的 PP 集群无中断更新，部署使用 Recreate；Ray 运行在专属模型 Pod 内，原外层 Docker 容器保持不变。
前端 Service 为 `bf16-distributed.heteroserve.svc.cluster.local:8000`。Ray 控制入口仅用于内部成员发现，启动阶段允许发布未就绪地址；模型 Service 仍只发布就绪前端。
NetworkPolicy 仅允许本实验组成员互通其内部通信端口，不公开 Ray 端口。

性能测量直接在现有 Gateway Pod 内运行 `benchmark.py`，不额外执行普通/SSE 功能验收。
客户端复用镜像内的 httpx/anyio，不加载模型、不占用 NPU。
使用 ShareGPT 原始文件，固定 revision 和 SHA-256；从真实对话第一轮提问按固定 seed 抽样、去重，保留自然输入长度并加入 Qwen 原始聊天模板。
默认过滤原始输入少于 4 token、模板化输入超过 1024 token 或输入加输出超过 2048 token 的样本，不截断输入。
复用现有模型 Pod 的 tokenizer 在 CPU 上准备 200 个测量样本和 3 个独立预热样本；固定输出 128 token、温度 0、并发 8。
同一提示集合可在不同并发档复用，结果、请求 ID/分词数据和数据版本保存在主工程 `artifacts/bf16_distributed/`。

- TTFT：发起请求到第一个非空文本输出，忽略空 SSE 事件。
- TPOT：最后一个非空文本输出时间减首个输出时间，除以实际输出 token 数减一；这是每请求平均值，不把 SSE 事件数量当 token 数。
- 吞吐量：成功请求的输出 token、总 token 或请求数除以完整测试档的墙钟时间。
- 同时报告失败数和 p50/p95/p99；失败请求不计入成功吞吐量。

这些结果描述指定 ShareGPT 子集、固定输出预算和闭环并发档，不等于自然输出长度分布、最大吞吐量、开放到达率 SLO 或真实任务质量验收。
200 请求的 p99 是本次样本估计；更稳定的尾延迟结论需要扩大样本量并重复测量。

## 已验证的性能基线

2026-10-10，在 `.210/.211` 的四张 Ascend 910B4 上运行 Qwen3-14B 原始 BF16，机内 TP=2、跨机 PP=2。
ShareGPT 子集使用 seed=20261010，3 个独立预热请求后测量 200 个请求，并发 8、每请求固定输出 128 token。
输入保持真实长度，加入原始聊天模板后平均 130.095 token，范围 16～1022 token。

| 指标 | 均值 | p95 | p99 |
| --- | ---: | ---: | ---: |
| TTFT（ms） | 253.15 | 365.86 | 1217.46 |
| TPOT（ms/token，每请求平均） | 98.42 | 100.94 | 101.40 |
| 请求总耗时（s） | 12.75 | 13.06 | 13.54 |

200/200 成功，0 失败；测量耗时 318.914 秒。
整个四卡服务合计输出吞吐量 80.272 token/s，总 token 吞吐量 161.858 token/s，请求吞吐量 0.627 request/s。
这是一档闭环并发下直接访问模型 Service 的暖机基线，未测两级路由开销或最大容量。

完整逐请求记录见 `artifacts/bf16_distributed/benchmark.json`；`dataset.json` 和 `requests.jsonl` 保存数据版本、过滤信息、模板校验值和抽样请求。
`reproducibility.json` 保存代码基准提交、修改后源码 SHA-256、Pod UID/节点、外层容器 ID 和暂停状态。
原始 ShareGPT 数据在 `.209` 的 `/root/zhangjinhao/datasets/sharegpt/`，不提交数据和运行产物到 Git；本节保留必要结果摘要。

## INT4 调用开销诊断

2026-10-10 的直接引擎对照已确认：真实模型中的 INT4 矩阵核平均约 68 微秒，BF16 矩阵核约 125 微秒。
量化线性层原先在每次调用中执行输入 reshape、contiguous 和输出 reshape，主机调用约 103～107 微秒；BF16 约 55～57 微秒。
这与 NPU 计算时间是不同的区间，不能相加解释整体延迟。

`native_parallel.install(fast_path=True)` 对连续二维输入直接调用现有 `torch_npu.npu_weight_quant_batchmatmul`，高维及非连续输入保留原路径。
它复用原权重、scale/offset、TP 分片和同一个计算算子。四种线性层形状、三种 batch、大于二维及非连续输入共 20 组真实 NPU 验证输出逐元素相等；独立调用的主机耗时约从 85 降到 55 微秒。
实验的 `config.json` 用 `native_int4_fast_path` 控制此路径，量化 Ray worker 显式安装它；BF16 不启用该量化方法。

```bash
python3 experiments/bf16_distributed/manage.py apply --expert awq
python3 experiments/bf16_distributed/manage.py apply --expert gptq
```

关闭张量处理后，INT4 原生 API 的主机调用仍约 48～56 微秒，BF16 约 41～51 微秒。
当前 V0/eager、TP2/PP2 下主机执行和跨卡等待主导吞吐：低开销计时的关键阶段每步约 52.27 ms（INT4）和 49.92 ms（BF16）。
两个虚拟引擎轮流执行，与每请求 TPOT 约 104.8/100.1 ms 相符。Qwen3 还显式调用 Q/K norm 的 `forward_native`，其主机区间每个阶段合计约 8 ms。
这些区间包含等待且存在嵌套，不能把各标签直接求和。

最终关闭采样后，用同一 32-token 输入、并发 8、输出 64 token，逐模型预热再按正反顺序测量，48/48 请求成功。
BF16 两轮 TPOT 为 105.56/99.33 ms，AWQ 为 103.35/102.37 ms，GPTQ 为 103.70/104.60 ms。
Pod 更换时物理卡被重新分配，BF16 两轮也有明显波动；这些小样本不能证明量化已稳定跑赢 BF16，也不能把整模型的全部变化归因于快速路径。
已证实的收益是减少 INT4 的主机调用开销；完整部署仍受主机执行、通信和同步限制。

逐请求结果、四个 rank 的算子轨迹、低开销计时、修复验证及源码校验值保存在 `artifacts/bf16_distributed/system/diagnostics/`。
通过现有管理入口可再次启用采样；这会重建实验模型 Pod，但不会替换外层固定 Docker 容器，也拒绝修改暂停或不属于项目的 Deployment：

```bash
python3 experiments/bf16_distributed/manage.py configure-profiler --expert awq --profile-dir /scratch/heteroserve-profile-timing --profile-mode timing
# 模型 Ready 后，使用引擎 localhost 的 /start_profile、受控请求和 /stop_profile。
# 关闭前导出 /scratch 中的每个 rank 文件；该目录随 Pod 更换而消失。
python3 experiments/bf16_distributed/manage.py configure-profiler --expert awq
```

`operators` 模式保留 CPU/NPU 算子轨迹并关闭内存、调用栈采样；`timing` 模式只记录主机墙钟区间。
采样器已关闭，实验模型全部 Ready，原暂停服务仍为零副本，资源上限开关保持 false。

## Q/K norm 与标准量化入口复核

`qwen3_fused_qknorm` 控制 Qwen3 128 维 Q/K norm 的昇腾算子路径，当前启用 `preserve_rounding`。
`qknorm_mode=native` 使用单个加权 RMSNorm；`preserve_rounding` 使用 RMSNorm 加乘法，保留原实现先转换 dtype 再乘权重的顺序。
实际量化模型的 80 个 norm 权重、FP16/BF16、不同形状、幅度和步长共 5760 组 NPU 测试中，后者逐元素相等；前者存在约一个 ULP 的差异。
复测发现，即使关闭 norm 优化，原部署也会在并发请求间串题。安装的 Ascend V0 attention 后端只缓存第一次收到的 K/V 视图，而 PP 调度器有两个独立 virtual engine 的 KV 张量；切换 virtual engine 时仍使用第一次的缓存。
`ascend_pp_cache_views=true` 按实际传入的 KV 张量选择缓存视图，每个张量只切片一次，保留张量引用避免对象 ID 重用。此修复同时用于 BF16、AWQ 和 GPTQ，不修改权重或计算算子。

修复缓存后重新验收 norm：三种模型共 12 个 rank、每 rank 40 个 norm、每个前 4 次真实服务调用，共 1920 次逐元素比较全部相等。优化后第一次自然输入测试与修复缓存但关闭 norm 的同组首次结果均为 16/16 相同。后续重复仍有少量措辞变化，原路径也有此现象；这不等于已完成量化模型的任务准确率验收。
`qknorm_verify_calls=4` 只在每个 norm 的前四次实际服务调用中核对参考输出，完成后写入 `/scratch/qknorm-audit/` 并停止核对，不在稳态计时中引入同步核对。

当前直接引擎短测使用相同 16 个真实 ShareGPT 输入、并发 8、固定输出 64 token；两批预热后测两批，并反向按 GPTQ、AWQ、BF16 顺序复测，复测时 Pod UID 和配置不变。

| 模型 | 首次优化后 TPOT（ms/token，两批） | 反向复测 TPOT（ms/token，两批） | 反向复测输出吞吐量（token/s） |
| --- | --- | --- | --- |
| BF16 | 96.85 / 95.50 | 96.11 / 96.78 | 82.00 / 81.25 |
| AWQ | 93.08 / 93.46 | 94.61 / 93.43 | 83.12 / 84.10 |
| GPTQ | 95.94 / 95.16 | 94.62 / 94.56 | 82.91 / 83.15 |

修复缓存但关闭 norm 时，两批 TPOT 均值为 BF16 103.40、AWQ 101.19、GPTQ 98.95 ms。启用 norm 前后 Pod 更新导致物理卡重新分配，因此不能把全部差值归因于 norm。固定 Pod 的正反向短测曾显示 INT4 有小幅优势，但随后扩大到 200 个不同输入时结论反转，不能宣称已稳定跑赢 BF16。

同一批优化后 Pod，使用原数据集的 200 个正式测量输入、并发 8、每请求固定输出 64 token，按 AWQ、GPTQ、BF16 顺序单独测量，600/600 成功，输入 SHA 相同：

| 模型 | TPOT 均值（ms/token） | TPOT p95 / p99（ms/token） | 输出吞吐量（token/s） |
| --- | ---: | --- | ---: |
| BF16 | 93.80 | 96.12 / 97.93 | 83.55 |
| AWQ | 95.73 | 100.00 / 101.72 | 81.84 |
| GPTQ | 95.36 | 98.93 / 101.15 | 82.22 |

这档 INT4 平均 TPOT 仍比 BF16 高约 1.7%～2.1%。已验证局部调用优化和缓存正确性修复；尚未取得稳定的整模型速度优势。测量包含不同自然输入和连续批处理，与此前固定提示的小样本不同；每种模型仅一轮，物理卡也各自不同。输出预算与历史 128-token 表格不同，不能直接作为同条件对照。完整 Gateway/L1/L2 链路尚未复测。

NPUGraph 实际捕获因当前 CANN 缺少 `aclmdlRICaptureBegin` 失败。复用 TorchNPU 内置 TorchAir 的孤立 INT4 MLP 编译也在 SwiGlu 的 GE 预编译阶段失败；未把这两条图路径部署到服务，未更新驱动、环境或镜像。当前仍是 V0/eager，CPU 下发、TP/PP 通信仍限制收益。
当前代码、配置、逐请求结果、真实输入核对与部署状态见 `artifacts/bf16_distributed/system/diagnostics/execution-reproducibility.json`。旧缓存下的历史测试保留为排查证据，不能作为当前质量验收。

标准入口复核关闭了 `MOQE_ASCEND_INT4_ADAPTER` 和 `MOQE_ASCEND_INT4_PARALLEL`，复用正在运行的实验容器，未加载另一份模型或更换环境。
vLLM 0.8.4 的 `NPUPlatform.supported_quantization` 仅为 `ascend`；调用 `verify_quantization('awq')` / `verify_quantization('gptq')` 均抛出 NPU 不支持的 ValueError。
`torch.ops._C` 中也未注册标准路径使用的 `awq_gemm`、`awq_dequantize`、`gptq_gemm`。
目前部署仍是由 AWQ/GPTQ 转换打包格式后的 `moqe_ascend_int4`，不能把启动参数改名后宣称完成标准部署。
标准权重文件与官方未修改的推理后端是两个不同要求；官方标准后端在此环境下无法直接运行。
复核源码和结果保存为 `artifacts/bf16_distributed/system/diagnostics/standard-quantization-support.json`。

## BF16 四卡 PP2 与旧量化两卡 PP1 的部署策略对照

按用户要求，BF16 在 `.210 + .211` 使用 TP2×PP2，AWQ/GPTQ 在 `.211` 使用 TP2×PP1。现有 vLLM 0.8.4/V0 eager、旧自定义 W4A16 权重；不是新 compressed-tensors 或 SGLang 的验证结果。相同 200 个输入 token 序列、并发 8、每请求实际输出 64 token、seed 0，两个八请求预热批次后每模型测一轮。三组依次启停。

| 模型 | NPU 数 | 输出吞吐（token/s） | TPOT 均值（ms/token） | TTFT 均值（ms） |
| --- | ---: | ---: | ---: | ---: |
| BF16 TP2×PP2 | 4 | 85.18 | 91.98 | 208.91 |
| AWQ TP2×PP1 | 2 | 99.08 | 76.58 | 335.99 |
| GPTQ TP2×PP1 | 2 | 106.29 | 71.25 | 321.64 |

600/600 成功，逐请求输入哈希和 64-token 输出长度一致，局部 Q/K norm CPU 位检查无差异。AWQ/GPTQ 吞吐分别提高 16.3%/24.8%，平均 TPOT 降低 16.7%/22.5%，首 token 延迟更高。这是不同卡数、TP/PP、模型精度和前端节点的部署策略比较，不能独立归因于量化算子。模型质量和完整 Gateway 链路未在本轮评估。

结果、逐请求数据、实际卡号、配置、源码快照及编排代码见 `artifacts/bf16_distributed/system/diagnostics/topology-comparison.json`、`topology-speed-*.json`、`topology-comparison-state.json`、`topology-runtime-sources.json`。已停止测试模型，原正式服务保持暂停，固定外层容器 ID 与无限制资源策略已复核。当前使用现有 vLLM 部署入口。

## 已撤销的 SGLang 实验

按用户要求撤销 SGLang 部署入口、兼容补丁及镜像，继续使用现有 vLLM。实验模型保持暂停。重新量化和标准 AWQ 导出模块、模型及 manifest 保留；它们尚未通过当前 vLLM 的部署验收，当前量化推理仍使用已有转换权重。

SGLang 两卡 TP2×PP1 eager 的 200 请求结果为 85.28 output tokens/s、平均 TPOT 89.85 ms；整模型图回放未通过。结果和复现快照保存在 `artifacts/bf16_distributed/system/diagnostics/sglang-awq-results.json` 等历史诊断文件中，不再作为当前运行配置。资源清理记录见同目录 `sglang-removal.json`。

## AWQ CPU 下发优化验收

当前 AWQ 保持 `.211` 两卡 TP2×PP1，使用现有 vLLM 0.8.4 和原转换 W4A16 权重。`model_overrides.awq.host_execution` 固定为 `task_queue_enable=2`、`cpu_affinity_conf=0`、`executor=mp`。BF16 保持 `.210 + .211` 四卡 TP2×PP2；GPTQ 配置保持原样。

优化已接入正式 AWQ 专家池。实验与正式池共用 `src/heteroserve/execution_worker.py`、`ascend_execution.py` 和主机执行参数解析；原本实验目录中的 worker 已迁入正式模块。
正式池的 TP/PP 由实例配置控制，不固定成实验的两卡拓扑；`executor=auto` 根据 PP 选择 mp/ray。
正式部署仍暂停，原直接引擎压测数据不能作为完整 Gateway 链路或其他拓扑的成绩。发布方式见 `docs/deployment.md`。

现有快速调用、TorchScript MLP 和保留舍入顺序的 Q/K norm 继续启用。二级下发队列减少 CPU 准备开销，本地多进程替代单节点 Ray 调度。`runtime.py` 在创建 Ray/推理进程之前设置环境，避免仅设置前端环境而没有进入 rank；采样结果确认两个 rank 的 `TASK_QUEUE_ENABLE=2`。

真实模型的固定提示、并发 8、输出 64 token CPU 采样：下表为各 rank 调用中位数的均值，单位 μs。区间包含等待、标签存在嵌套，不能相加；BF16 与 AWQ 部署拓扑不同，比较的是同形状线性层调用。

| 调用 | BF16 当前基线 | AWQ 原配置 | AWQ 二级队列 + mp |
| --- | ---: | ---: | ---: |
| QKV apply | 52.06 | 65.47 | 44.99 |
| 输出投影 apply | 57.95 | 70.07 | 49.90 |
| MLP forward | 328.16 | 223.13 | 161.06 |

AWQ 三组在同一个 Pod、同一对物理卡 `[4,5]` 上依次执行，CPU 采样结束后使用同一 200 个输入、并发 8、每请求 64 输出 token、seed 0 测速，全部成功。采样不在测速期间开启，计时包装仍存在：

| AWQ 设置 | 输出 tokens/s | 平均 TPOT ms |
| --- | ---: | ---: |
| 原配置 / Ray / 队列 1 | 101.30 | 74.97 |
| Ray / 队列 2 | 123.66 | 60.88 |
| mp / 队列 2 | 128.88 | 58.37 |

同 Pod 对比吞吐提升 27.23%、TPOT 降低 22.14%。撤掉采样 worker 后重新启动，仍分配 `[4,5]`，200/200 请求成功，复验 124.37 tokens/s、平均 TPOT 60.49 ms；Pod UID 已改变，不能把两轮波动归因于去掉采样包装。四个基础答案检查全部通过；实际 Q/K norm 的两 rank 有限逐位核对没有差异。这不是完整准确率评估，也不是同拓扑 BF16/AWQ 的量化加速比。

在同一批隔离的两张卡上做的算子诊断显示，batch 8 的 AWQ 当前接口 CPU 调用从 48.71 μs 降至 37.51 μs，原设置 BF16 为 36.59 μs。自动绑核没有额外收益，因此不启用。直接 dispatcher / 单线性层 TorchScript 候选未采用；随机矩阵测试也发现当前接口自身重复有微小浮点差异，逐位不同不能单独证明候选计算错误。部署保留原来的算子入口。

复现入口：`python3 experiments/bf16_distributed/manage.py benchmark-dispatch --expert awq`；整模型采样使用 `python3 experiments/bf16_distributed/host_benchmark.py`。入口拒绝覆盖已有结果；再次运行前应先归档相应结果和状态，并核对实验部署 UID。`host_benchmark.py` 保持实验 Pod 的 NPU 分配，只更新 ConfigMap 并重启 Pod 内进程，在结束或失败时恢复默认暂停模板。

保留 `system/diagnostics/dispatch-benchmark.json`、`host-execution-state.json`、`host-*.json` 中的配置、源码、逐请求数据和各 rank 的 CPU 区间。最终无采样验收见 `host-awq-acceptance.json`，整理和资源验证见 `host-optimization.json`。测试后模型保持暂停，固定外层容器和主机驱动不更换，CPU/内存上限仍关闭。

## 正式系统 TP2 全链路压测

`python3 experiments/bf16_distributed/host_benchmark.py --mode system` 复用正式实例、Gateway、两级路由和现有镜像。
`config.json` 的 `system_benchmark.instances` 指定一个 AWQ 和一个 GPTQ 稳定实例，`parallelism` 指定各模型的 TP/PP。
本轮为 AWQ `awq-03`（`.211`）与 GPTQ `gptq-01`（`.216`），各 TP2×PP1。
两者保留各自 W4A16 权重，复用正式 AWQ 已验证的执行优化和相同引擎预算；实际部署与真实普通/SSE 验收均记录。

客户端在现有 Gateway Pod 内，经 Service 访问 Gateway，再经过学习型 L1、官方 vLLM Router 和模型 worker。
复用原 203 条聊天输入，其中 3 条预热，200 条正式测量；并发 8，实际输出长度固定 64 token，seed 0。
模型占 4 张 NPU，既有两路 L1 另占 2 张。该结果包含完整服务链路，不能直接归因于量化或与两卡单引擎作同卡数比较。
结束或失败后恢复原 Deployment/ConfigMap 和暂停状态，保留固定外层容器；不创建镜像、环境或另一套模型。
结果、逐请求路由身份、配置、源码和状态保存在 `system/diagnostics/system-speed*.json`；入口拒绝覆盖已有结果。

本轮 200/200 成功，输出吞吐 125.20 token/s（1.956 请求/s），完整用时 102.24 秒。
TTFT 平均/P95/P99 为 323.37/490.94/681.17 ms；TPOT 为 59.44/65.12/66.52 ms；请求总耗时为 4.07/4.54/4.58 秒。
学习型路由选择 AWQ 136 次、GPTQ 64 次，两路 L1 各 100 次。路由与转发的启动区间平均 94.87 ms，其中 L1 计算平均 54.73 ms、排队 24.83 ms；该区间排除 worker/backend 等待和响应头之后的流转发。
实际 AWQ 为 `.211` 的 `[6,7]`、GPTQ 为 `.216` 的 `[2,3]`，两者均加载队列 2、mp、INT4 快路径、MLP 和 Q/K Norm 优化。
这是集群内客户端的单轮闭环测试，不是最大容量或外部网络 SLO 测量。测量后恢复原暂停模板；原两卡 AWQ 127.18 token/s 的直接引擎结果不能作为同资源、同专家分布的链路开销对照。

## L1 编码器下发优化

复用现有 L1 镜像、checkpoint、Embedding 模型和 `.209/.210` 副本；在原 `.209` Pod 和同一 NPU 上对照 200 个相同输入，生成预算 64，未启动暂停的生成模型。
阶段诊断的 20 个输入中，编码器同步墙钟均值为 56.33 ms，聊天分词/编码器分词合计 1.77 ms，池化和路由头约 2.55 ms。同步诊断有额外等待，不能作为纯 NPU kernel 时间或直接替代无采样测量。

只启用 `TASK_QUEUE_ENABLE=2` 的无采样对照：

| L1 计算延迟 | 原进程 | 队列 2 |
| --- | ---: | ---: |
| 平均（ms） | 55.31 | 46.13 |
| P95（ms） | 63.02 | 51.84 |
| P99（ms） | 66.60 | 55.21 |

平均延迟降低 16.6%。200 个路由概率逐项一致，专家选择无变化。TorchScript norm/MLP 分组的额外收益仅约 1%，没有足够证据确认价值，已从正式实现删除；候选的有效测量及其源码快照保留为复现证据。
连续容器进程重启曾触发 Kubernetes 退避，收尾等待超时；随后通过既有正式入口正常发布两副本并恢复健康。这一编排问题保留在报告中，不能把中断写成成功的并发验收。

最终 `.209` 单请求计算平均/P95/P99 为 48.19/54.45/57.48 ms，`.210` 为 46.18/51.34/54.40 ms；两副本共 600 条正式验证均与原 200 条概率/选择一致（含 `.209` 并发 4）。正常发布后的 Pod 和卡号与原对照不同，应单独看待。
正式配置 `deploy/gateway.json`、运行时默认值和 L1 发布模板都保留队列 2，诊断采样关闭；既有 AWQ/GPTQ 的执行优化不回退。
本轮未重新测试完整生成链路，旧系统 125.20 token/s 不是本轮 L1 优化后的吞吐。
复现使用 `host_benchmark.py --mode l1` 与 `benchmark.diagnose_l1`；有效源码、配置、状态和全部概率见 `system/diagnostics/l1-optimization*.json`、`l1-final-benchmark.json` 和 `l1-final-acceptance.json`。
重跑前归档已有 `l1-optimization.json` 和 `l1-optimization-before.json`；入口自动采集当前 Pod/ConfigMap 快照，拒绝使用不同 Pod 的旧快照。临时队列 1 只用于对照，正式配置和最终运行保持队列 2。

## TTFT 分解与两卡诊断对照

复用当前无 profiler 的 vLLM 服务，重新测量同一 200 个 ShareGPT 输入、并发 8、输出 64 token、seed 0。每组通过 vLLM 原有 Prometheus histogram 的完成请求增量读取排队与 prefill 时间，计数必须与客户端成功请求数一致；同时记录响应头、首个 SSE choice 和首段非空文本的时间。Prefill 指首次调度到首 token 的引擎区间，包含模型执行、采样和输出处理，不是纯 NPU kernel 时间。引擎外残差包含输入处理、HTTP 与流式发送，不能单独解释成网络或 CPU 时间。

| 部署 | TTFT ms | 排队 ms | Prefill ms | 引擎外残差 ms |
| --- | ---: | ---: | ---: | ---: |
| BF16 四卡 TP2×PP2 / Ray / 队列 1 | 217.28 | 81.42 | 119.32 | 16.53 |
| AWQ 两卡 TP2×PP1 / mp / 队列 2 | 299.85 | 58.59 | 209.31 | 31.95 |
| BF16 两卡 TP2×PP1 / mp / 队列 2（诊断对照） | 293.43 | 59.79 | 204.28 | 29.37 |

三组各 200/200 成功，无 preemption。两卡 BF16 在原 50% NPU 内存预算下无法建立 KV cache，诊断对照单独使用 80%；AWQ 与正式四卡 BF16 保持 50%。两卡 BF16 的物理卡是 `.211 [0,7]`，AWQ 是 `.211 [1,2]`，不是同卡配对统计实验，不能将剩余约 6.4 ms 差距精确归因于量化。

主要差距出现在四卡 PP2 与两卡 PP1 的执行策略之间：相同两卡拓扑和 host 设置下，AWQ 与 BF16 的 TTFT/prefill 接近，量化没有明显额外拉长整模型首 token，也尚未取得明显的 prefill 加速。两卡 BF16 与 AWQ 的输出吞吐分别为 125.93/127.18 tokens/s，接近；此前相对四卡 PP2 的 decode 速度优势也包含部署策略变化，不能归因于量化本身。固定 512-token 输入、并发 8 的 prefill，四卡 BF16 为 219.02 ms、两卡 BF16 为 525.91 ms、两卡 AWQ 为 523.67 ms。不能把四卡对两卡的 TTFT 差距全部解释为量化算子问题。

两卡隔离算子探针在同一批 `[2,3]` NPU 上比较相同 TP2 投影形状，固定队列 2，覆盖输入矩阵行数 8/32/128/512/1024/4096。W4A16 存在形状相关的性能劣势：device 0、M=32 的 `down_proj` 同步完成中位数 333.97 μs，BF16 为 94.60 μs，CPU 下发分别为 48.01/31.75 μs。四个投影的同步完成中位数之和在两 rank 上平均，M=32 为 1041.67/548.60 μs，M=4096 为 6946.17/6528.88 μs。这里是随机权重、孤立同步算子实验，不能直接相加预测整模型 TTFT，也不能据此判断模型准确率。

固定输入的四卡 BF16 单请求测试出现反复的大幅波动，排队时间很低、异常落在首次调度后的区间；没有证据证明只是首次编译，不作为稳定低并发性能结论。该次逐请求证据保留，PP2 低并发波动的内部原因尚未单独定位。

复现命令：`python3 experiments/bf16_distributed/host_benchmark.py --mode ttft`、`python3 experiments/bf16_distributed/host_benchmark.py --mode ttft-local-bf16`、`python3 experiments/bf16_distributed/manage.py benchmark-dispatch --expert awq --dispatch-mode prefill`。仍拒绝覆盖有效结果，重新运行前先归档对应状态与结果。两卡 BF16 是临时诊断参数；结束后恢复默认四卡 BF16 和两卡 AWQ 的暂停模板。有效源码快照、部署身份、完成计数与逐请求结果见 `system/diagnostics/ttft-diagnosis.json`、`ttft-*.json`、`dispatch-prefill*.json`；已删除完成的探针 Job/ConfigMap 和归档后的重复临时输出。
