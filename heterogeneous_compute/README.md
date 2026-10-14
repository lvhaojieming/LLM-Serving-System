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
converting logits to FP32, matching native output rounding. RMSNorm, RoPE and
activations use vLLM reference methods; TP linear modules remain reused.

Rank zero admits requests into one bounded FIFO and broadcasts scheduling to
its local TP peer. Only one request executes at a time; pending HTTP requests
queue. Stage ownership and contiguous token positions are checked, tokens
return from the final stage on each lane, and cache release must be acknowledged
before switching requests. Protocol version, payload size and tensor shapes
are validated. Peer disconnects fail the operation; automatic rank recovery
is not implemented. A serving idle wait does not expire the PP connection.

No continuous batching, overlapped pipeline microbatches, graph execution,
paged attention or quantization is implemented here. The raw completions API
now streams one SSE event per generated token for official client timing. This
correctness-first path is not a throughput baseline and must not be mixed with
the earlier native vLLM performance measurements.

## Manage

Run from the formal project on the client. No environment installation needed:

```powershell
python heterogeneous_compute/manage.py start --ssh-config '<cluster_ssh_config>'
python heterogeneous_compute/manage.py status --ssh-config '<cluster_ssh_config>'
python heterogeneous_compute/manage.py stop --ssh-config '<cluster_ssh_config>'
```

The default config is `heterogeneous_compute/configs/ascend_musa.json`; change parameters
there. Start transfers only this module and config, then starts four Python
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
returns 503. A canceled queued request is skipped; already-executing work
finishes and releases its KV cache. No native throughput claim is made.

```json
{"model":"Qwen3-14B","messages":[{"role":"user","content":"2+3等于多少？只输出数字。"}],"temperature":0,"max_tokens":8,"stream":false}
```

## Validation and retained evidence

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

## 独立验收入口

从仓库根目录执行 `python heterogeneous_compute/validate.py --ssh-config <cluster_ssh_config>`。
它同时提交配置中的七条请求，并与两个原生服务对照，校验六类非法请求；
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
Reference attention, CPU staging and serialized scheduling remain part of the
baseline. Continuous batching/pipeline overlap is still absent; a higher
client concurrency does not imply simultaneous model execution.
