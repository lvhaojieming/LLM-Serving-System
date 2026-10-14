# 异构算力适配

此目录独立管理跨厂商算力适配。已完成的同构集群、扩容和路由管理位于
`homogeneous/`，不在这里重新实现。`configs/baselines/` 保存多厂商原生基线；
`platforms/iluvatar/` 保存天数部署与官方 benchmark 入口；`probes/` 保存通信预检。
`quantization/` 保存共用校准集与标准 AWQ/GPTQ W4A16 权重生成入口，
部署继续复用现有后端。详见 [量化说明](quantization/README.md)。

## Qwen3-14B Ascend / MUSA TP2 PP2 experimental executor

Functional acceptance on 2026-10-14: actual Qwen3-14B BF16 weights and all
40 decoder layers execute across .23/.97, using two accelerator processes
per stage. Retained containers `heteroserve-ascend` / `heteroserve-musa` and
their devices 0/1 are reused; baseline vLLM services remain running.

Stage 0 owns embedding and layers [0,20); stage 1 owns layers [20,40), final
norm and LM head. Each rank loads approximately 7.384 GB of parameters.
vLLM's own PP-aware Qwen3 construction, safetensor weight loaders, fused
projection layout and sharding are reused. Stage weights are strictly checked
for missing parameters. Local TP all-reduces/gathers use HCCL or MCCL through
a process-local group adapter. The PP groups exchange BF16 hidden states AND
residuals over two TCP lanes with CPU staging. No whole-model HTTP chaining.

## Scope

This is a vLLM-component experimental executor, not an unmodified vLLM
Engine/AsyncLLMEngine or a vendor-supported cross-vendor deployment.
The common reference attention keeps KV locally, applies causal masking and
GQA, accumulates attention scores/softmax/weighted values in FP32 and returns
BF16 activations. The output projection uses BF16 weights/activations before
converting logits to FP32, retaining the serial control's precision policy. RMSNorm, RoPE and
activations use vLLM reference methods; TP linear modules remain reused.

Rank zero owns bounded admission and broadcasts each scheduling operation to its
local TP peer. `execution.scheduler=continuous` admits up to 8 requests, packs
up to 4 distinct requests / 1024 tokens into one model forward and allows two
independent microbatches in flight. Each request has at most one chunk/token in
flight. Stage zero executes the next microbatch while stage one processes the
previous one; feedback has its own CPU receive thread, so SSE token emission does
not wait for the next stage-zero forward to finish. Shorter requests release their
KV and admission slots without waiting for longer requests; queued work refills
available slots. When the pipeline is empty, ready work is split between two
microbatches (four clients become 2+2) instead of packing every request into one
batch and leaving the first stage waiting. The serial control remains available
through `scheduler=serial`.

KV is owned per request and per layer. Packed linears and RoPE use individual token
positions, attention never crosses request boundaries, and decode attention uses
batched FP32 scores with padding masks for different sequence lengths. Prefill is
chunked at 512 tokens so long prompts can share execution with decode requests.
The reference BF16 LM head is applied per sampled row to retain the original
output-rounding path. A local KV owner is removed only after the peer acknowledges
release. Cancellations are broadcast to both TP lanes; peer failures wake HTTP
waiters with errors. Automatic rank recovery is not implemented.

The JSON tensor protocol is version 2 for continuous scheduling, version 1 for the
serial control; mixed versions fail immediately. Clock calibration and per-lane
execution windows are retained to verify overlap between distinct microbatches.
RTT bounds the initial offset; the short acceptance assumes stable monotonic
clock rates and does not measure clock drift. The overlap includes dispatch and native TP operations; it is not a GPU kernel
profiler or a hardware utilization measurement. No graph execution, paged attention
or quantization is added by this change. This remains a custom vLLM-component
reference executor, not the native vLLM Engine scheduler.

## Manage

Run from the formal project on the client. No environment installation needed:

```powershell
python heterogeneous_compute/manage.py start --ssh-config '<cluster_ssh_config>'
python heterogeneous_compute/manage.py status --ssh-config '<cluster_ssh_config>'
python heterogeneous_compute/manage.py stop --ssh-config '<cluster_ssh_config>'
```

The default config is `heterogeneous_compute/configs/ascend_musa.json`; change parameters
there. Start transfers the executor, scheduler module and config, then starts four Python
processes inside existing containers. Existing pipeline processes block a
duplicate start. Stop sends SIGTERM only to the scheduler head, drains pending
work and terminates both stages through the protocol. It does not stop native
baseline services or remove containers. Inspect status after drain completes.
There are no CPU/RAM/PID limits imposed by this manager.

API: `http://192.168.162.23:18230/v1/chat/completions`, `/v1/models`, `/health`.
It binds the cluster LAN. The 172.22 external port mapping has not been tested.
Supported path: text chat, `model=Qwen3-14B`, `temperature=0`, `stream=false`,
`max_tokens` 1..128; total context bounded by 2048 tokens. Thinking is disabled.
The queue holds up to 16 pending requests; full queue returns 429 and draining
returns 503. A canceled queued request is skipped; active requests
finish their in-flight microbatch and release their KV cache. No native throughput claim is made.

```json
{"model":"Qwen3-14B","messages":[{"role":"user","content":"2+3等于多少？只输出数字。"}],"temperature":0,"max_tokens":8,"stream":false}
```

## Earlier serial control validation and retained evidence

- Native HCCL/MCCL TP2 and device->CPU->TCP->CPU->device transport passed.
- All 40 real layers loaded; missing-weight check passed on every rank.
- Four initial offline cases plus a repeated case across intervening requests
  produced 5 / 巴黎 / 3 / 5, matching both native services.
- Final HTTP acceptance simultaneously submitted seven pipeline requests:
  arithmetic, capital, English arithmetic, repeat, 32-token explanation,
  1..10 enumeration and context interference. All seven output texts exactly
  matched both native Ascend and MUSA greedy references.
- Both stage-zero lanes generated identical token IDs; every request release
  left zero cache entries. Native baseline math acceptance remained valid.
- Six invalid/unsupported request cases returned 400: streaming, stochastic
  temperature, zero/excessive output bound, overlong context and unknown model.
- KV ownership/stale-position and protocol-version unit checks passed without
  adding dependencies. Drain/start and duplicate-start protection were exercised.

These are limited functional checks, not a general accuracy benchmark,
bitwise equivalence proof, failure-recovery acceptance or performance study.
Earlier FP32-head/BF16-probability prototype produced a long-text divergence;
the final attention accumulation/head rounding changes passed the same case.

Evidence: `artifacts/baselines/cross-vendor-pp/` with node snapshots,
preflight, initial execution and final `adapted/acceptance.json`, per-lane logs,
per-request token IDs/step times and cache-release records. Cold-start and
queued client timings are diagnostics and must not be reported as TTFT/TPOT
benchmark results. Source/config SHA256 and dependency snapshots are retained
with the final verification report. Existing uncommitted project work is preserved.

## 连续调度验收及当前限制

连续调度复用已有容器、实际模型权重、TP 分片和 PP 传输。验收分别报告
`functional_passed` 和 `batch_invariance_passed`；只有两者同时满足才将
总 `passed` 标为 true。严格验收失败不会伪装成全部通过。

已观察到批大小相关的长回答差异：原来的七条对照请求中六条文本一致，
长解释在批处理时出现 token 分歧，单请求执行可匹配原基线。旧逐行 LM head
未消除这个差异；当前改为整批 LM head。不同批大小的
逐 token 一致性与更广泛的质量评估仍未通过。vLLM 官方也为独立于批大小
的确定性提供专门的 [batch invariance 支持](https://docs.vllm.ai/en/stable/features/batch_invariance/)，
这里没有安装或宣称适配这些内核，也不能仅凭文档认定本次差异的确切来源。

摩尔原生 TP 曾在恢复下一轮请求时发生 MCCL 超时，保留了诊断。当前此执行器
使用配置驱动的 Socket / 禁用 P2P、SHM。当前本地 TP 等待对应 collective 的
`Work.wait()`，不再在每层后执行整设备同步；
这些设置仅作用于本次执行进程，不修改驱动、原生服务或容器。它们影响性能，
正式 benchmark 必须随结果保留这些配置。质量一致性和性能均独立于功能验收。

实际验收结果保存在 [调度结果摘要](results/cross_vendor_scheduling.json)：
10 项单元验证通过；12 条不同输出长度的流式请求、补位、双 TP lane token、
827-token 长提示词分块、客户端断开后释放 KV、空闲后再次并发请求均通过。
记录到两个阶段不同微批次的执行窗口重叠；短验收使用 RTT 校准的时钟偏移。
七条原生文本对照为 6/7，批大小的逐 token 一致性未通过；因此严格总验收
`passed=false`，功能项 `functional_passed=true`。旧连续调度官方测速已完成
600/600，但各并发吞吐均低于旧串行控制组，不能认定性能优化成功。

## 批量计算和异步 PP 协议

现有执行器扩展协议版本 3，不复制工程或改厂商 site-packages。前向包包含：

| 内容 | 字段 |
| --- | --- |
| 批次 | `batch_id`、`step_id`、`phase`（prefill/decode/mixed） |
| 请求 | `request_order`、每个 segment 的 request_id、step_id、phase、positions、length、past |
| 采样 | `sample_token`、max_tokens、ignore_eos、sampling.temperature=0 |
| 张量 | name、shape、dtype=bfloat16、bytes；hidden_states 与 residual 正文 |
| 返回 | batch_id、step_id、request_order、每请求 next_token、finished、sampling |

批次 step_id 是单调派发序号；请求 step_id 是已生成 token 数，分块 prefill
期间可以不变，依靠 batch_id 和连续 positions 区分。接收端检查 FIFO 顺序、
布局、长度和采样配置；返回端检查 token、结束状态和请求所有权。

Prefill、Decode、混合 attention 使用独立 KV 的 padding 与因果 mask，一次
批量 QK 和 AV 计算；LM head 一次批量矩阵计算。不同长度和已有历史 KV 的
混合请求通过独立请求数学对照与两次 matmul 调用数检查。padding 会增加计算
与内存，当前仍是可移植参考 attention，尚未等同于厂商原生 paged/fused 内核。

stage0 使用有界 FIFO 后台发送和专用复制 stream，主线程仅记录本批设备事件。
stage1 后台预取网络包并在独立 stream 复制，计算 stream 等待输入事件。
本地 TP 使用对应通信 Work.wait 保持依赖，不能删除必要的数据就绪关系。
CPU staging 仍保留，未宣称 RDMA 或设备直连。设备事件记录本地执行耗时；
跨节点时钟校准的 host 窗口仍包含复制与等待，不能当作 GPU kernel timeline。

新版功能与测速产物分别在 `artifacts/baselines/cross-vendor-pp/batched/` 和
`artifacts/baselines/cross-vendor-pp/performance-batched/`；旧结果保留用于对照。
当前目标是已请求的 greedy Qwen3-14B TP2/PP2；随机采样、beam search、
其他模型/拓扑和通用跨厂商原生 vLLM Engine 尚未实现或验证。

## 独立验收入口

从仓库根目录执行 `python heterogeneous_compute/validate.py --ssh-config <cluster_ssh_config>`。
它同时提交配置中的七条请求，并与两个原生服务对照，校验六类非法请求；
连续调度还验证 12 条不同输出长度的流式请求、请求补位、双 TP lane 输出、
KV 释放及两个阶段不同微批次的执行窗口重叠。原生输出一致性单独报告；
完整原始结果保存在被 Git 忽略的 `artifacts/`，必要摘要位于 `results/`。
单元验证：`python -m unittest discover -s heterogeneous_compute/tests -v`。
管理器识别整理前的旧执行进程，避免误启动第二套；排空后再启动会使用新目录。

## Official vLLM streaming benchmark

This executor is the cross-vendor reference baseline. The measurement client
is the unmodified installed `vllm bench serve` v0.23.0 on .58, the same version
used for native-vendor results. `/v1/completions` accepts a raw text prompt,
`stream=true`, greedy `temperature=0`, `ignore_eos=true`, and usage summaries.
It emits exactly one choices event per generated token, then a separate usage
summary with empty choices and `[DONE]`. It does not wait for full generation
before emitting the first token. `/v1/chat/completions` remains non-streaming.

Run inside the existing .58 `heteroserve-iluvatar` client container:

```bash
bash /workspace/heterogeneous_compute/platforms/iluvatar/benchmark.sh \
 /workspace/heterogeneous_compute/configs/cross_vendor_benchmark.env
```

The config sets 200 ShareGPT requests per case, 3 warmups, 64 output tokens,
EOS ignored, seed 20261010, concurrency 1/4/8, unlimited offered request rate,
and percentiles 50/95/99 for TTFT/TPOT/ITL/E2EL. Cases execute sequentially.
The dataset and tokenizer are reused from the earlier native baseline; no
additional copies or benchmark environments are created. Each result includes
per-request success, TTFT, ITL and generated output from the official client.

TTFT is measured by the client from request start to the first token event,
including queue, prefill and network. TPOT is the official per-request mean
post-first-token latency divided by output length minus one. ITL percentiles
pool the observed token gaps. Output throughput is generated tokens divided
by measured case duration for the whole four-device PP instance, not per GPU.
The retained `results/cross_vendor_performance.json` measures the earlier serial
control. The current configuration writes new results to
`artifacts/baselines/cross-vendor-pp/performance-continuous/` and preserves the old
measurements. Reference attention and CPU staging remain part of both paths.
Only completed official benchmark results establish throughput or latency changes.
