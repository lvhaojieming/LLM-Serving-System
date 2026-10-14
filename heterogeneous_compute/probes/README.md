# Ascend / MUSA TP2 PP2 transport preflight

Subsequent real-model adaptation and functional acceptance are documented in
`heterogeneous_compute/README.md`. The following records the earlier
transport-only preflight; it is not the current deployment status.

2026-10-14: reused .23 `heteroserve-ascend` and .97 `heteroserve-musa`.
No weights were loaded, no existing model process stopped, and no container recreated.
This is a communication probe, not a Qwen3 pipeline inference implementation.
Configuration: `heterogeneous_compute/configs/ascend_musa.json`.

All four final ranks passed native local TP2 all-reduce (HCCL / MCCL),
four-rank CPU Gloo all-reduce, and the two PP pairs [0,2] / [1,3].
The PP test sends BF16 [262,5120] activations (2682880 bytes) through
device->CPU->Gloo->CPU->device, modifies the received tensor on the target
accelerator, returns it, and checks every element. These are synthetic
activations, not hidden states produced by Qwen3 layers.

Both accelerator types fail direct device Gloo send/recv with
`No backend type associated with device type npu/musa`.
MUSA Torch also reports that error while destroying the Gloo process group;
it is captured separately, and all short-lived probe processes exit.
CPU staging therefore establishes a possible transport path, not compatible
vLLM worker/scheduler semantics. Existing stacks are vLLM 0.11.0 / Torch 2.7.1
on Ascend and vendor vLLM 0.9.3.dev / Torch 2.5.0 on MUSA. Neither environment
registers the other vendor's native communication backend. Installed
GroupCoordinator code creates device groups separately from CPU Gloo groups.

## Reproduce

Run four commands concurrently, using the retained containers. For each
stage, set the environment from the config with `docker exec -e KEY=VALUE`;
also set `OMP_NUM_THREADS=1`. Each rank uses local device `rank % 2`.

```bash
docker exec <environment flags> <container> timeout 65 python3 \
 /workspace/heterogeneous_compute/probes/communication_probe.py \
 --rank <0..3> --device <npu|musa> \
 --master 192.168.162.23 --local-master <stage host> --port 29650 \
 --output /workspace/artifacts/baselines/cross-vendor-pp/isolated-ports/rank<rank>.json
```

The native TP rendezvous uses port 29651 separately on each host. CPU world
rendezvous uses 29650 on .23. Bind Gloo to the configured LAN interface;
automatic interface selection failed on the first attempt. Isolate HCCL host
and device ports when sharing devices with an existing model; default ports
collided on the second attempt. Neither failure proves a cross-vendor hardware
limitation. Final results are in `isolated-ports/`; earlier small JSON records
are retained as useful networking/port diagnostics. Single cold-start tensor
timings include context initialization and rendezvous skew and must not be
reported as bandwidth or model latency.

## Remaining model acceptance

Proposed split: Ascend layers 0..19 + embedding, MUSA layers 20..39 + final
norm/head. This has not been deployed. Integrate a compatible PP communicator,
shared scheduling/metadata, intermediate hidden states AND residuals,
per-stage KV cache ownership, sampling/token feedback, and error handling.
Then verify greedy outputs against single-instance BF16 on multiple prompts,
multi-request isolation and streamed completion before measuring TTFT/TPOT.
Do not concatenate two whole-model HTTP responses and call that PP.

Official multi-node vLLM setup requires identical execution environments:
https://docs.vllm.ai/en/v0.10.0/serving/distributed_serving.html
HCCL port configuration:
https://www.hiascend.com/doc_center/source/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0143.html
