# Multi-vendor BF16 serving baseline

This deploys Qwen3-14B directly through each vendor's existing vLLM runtime.
It does not change the Ascend Kubernetes lab, Gateway, Router, or paused services.
`vllm.yaml` is shared; `cluster.json` records node, image, device, port, and
vendor-specific settings. The current scope is one validated node per vendor,
not simultaneous deployment on every node in the inventory.

Validated backends on 2026-10-13:

| Backend | Node | vLLM | Container | Port |
|---|---|---|---|---|
| Ascend 910C | 172.22.162.23 | 0.11.0 + Ascend 0.11.0 | heteroserve-ascend | 18023 |
| MTT S5000 | 172.22.162.97 | 0.9.3 vendor development build, V0 | heteroserve-musa | 18097 |
| Hygon BW1100 | 172.22.162.252 | 0.18.1 DAS / DTK26.04 | heteroserve-hygon | 18252 |

Each new backend uses devices 0/1, BF16, TP2/PP1, 2048-token context,
maximum 8 sequences, eager execution, and seed 20261010. The existing
Iluvatar backend on .58 stays in its independently validated graph profile.
Do not compare throughput until execution mode, cache and sampling settings,
input data, output budget, and client protocol are aligned.

The persistent containers mount the host repository
`/home/zhangjinhao/LLM-Serving-System` at `/workspace` and the existing
`/home/zhangjinhao/models` read-only at `/models`. Containers were needed because
shared containers lacked these mounts and belong to other tasks. Reuse the
validated container IDs recorded in `artifacts/baseline/deployment.json`;
do not recreate them for each experiment.

After checking devices, ports, and existing processes, the model-only restart
commands on the corresponding hosts are:

```bash
# Ascend
docker exec -d heteroserve-ascend bash -lc \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh; if test -f /usr/local/Ascend/nnal/atb/set_env.sh; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi; exec vllm serve --config /workspace/heterogeneous_compute/configs/baselines/vllm.yaml --port 18023 --block-size 128 > /workspace/artifacts/baseline/vllm.log 2>&1'

# Moore Threads: this image's V1 path failed its block-size check even with 32.
docker exec -d -e VLLM_USE_V1=0 heteroserve-musa bash -lc \
  'exec vllm serve --config /workspace/heterogeneous_compute/configs/baselines/vllm.yaml --port 18097 --block-size 32 > /workspace/artifacts/baseline/vllm.log 2>&1'

# Hygon
docker exec -d heteroserve-hygon bash -lc \
  'exec vllm serve --config /workspace/heterogeneous_compute/configs/baselines/vllm.yaml --port 18252 > /workspace/artifacts/baseline/vllm.log 2>&1'
```

Do not issue a restart if the user has paused the service. The commands manage
only this task's model processes and do not resume existing shared services.
No CPU, memory, or process quotas were added; device assignment remains scoped.

Compatibility findings:

- The requested Ascend 0.8.4rc2 image failed a real tensor operation because
  its stack rejected `Ascend910_9362`. The A3 0.11.0 image passed tensor and
  model-generation acceptance; the host driver was unchanged.
- Moore Threads' new MUSA5.2 image failed import against host libmusa4.3.8
  (`muThreadExchangeStreamCaptureMode` missing). Reused the .97 vendor image
  with Torch/torch-musa2.5.0, which passed tensor operations and real generation.
  The V0 setting is retained; no vendor source was patched.
- .251 Hygon cards were occupied. .252 cards were free and had an existing
  official 0.18.1 image, which was reused. Shared processes stayed untouched.

Acceptance checks returned `2`, `巴黎`, and streamed `5` with `[DONE]` on all
three backends. Results, runtime package versions, image/container IDs, config
hash, repository commit, and resource settings are in
`artifacts/baseline/{acceptance,deployment}.json` on each host. Startup logs are
`artifacts/baseline/vllm.log`. These are serving smoke checks, not labeled
accuracy, throughput, graph-execution, or AWQ/GPTQ W4A16 validation.

## Comparable performance profile

`benchmark.env` reuses `heterogeneous_compute/platforms/iluvatar/benchmark.sh` and the installed official
vLLM 0.23 benchmark client on .58. Pass the shared config path as its first
argument; set `BASE_URL` and `RESULT_DIR` per backend through the environment.
The wrapper snapshots the effective client config and refuses to overwrite
existing result JSON. For example, inside the existing .58 container:

```bash
BASE_URL=http://192.168.162.23:18023 \
RESULT_DIR=/workspace/artifacts/baselines/performance/ascend \
bash /workspace/heterogeneous_compute/platforms/iluvatar/benchmark.sh /workspace/heterogeneous_compute/configs/baselines/benchmark.env
```

The aligned profile uses BF16, TP2, eager execution, GPU memory fraction 0.60,
2048-token context, 8 maximum sequences, prefix caching disabled, and greedy
sampling (`--temperature=0`). Disable prefix caching using the explicit
`--no-enable-prefix-caching` flag or the true-valued negative YAML key retained
in `vllm.yaml`; a false-valued positive YAML key was ignored in a previous CLI
version. Verify `enable_prefix_caching=False` in the actual engine startup log
before accepting measurements.

Each backend has concurrency 1/4/8, 200 measured requests per case, 3 warmups,
and fixed 64-token output with EOS ignored. Seed 20261010 selects the same
ShareGPT inputs through the same client implementation and local tokenizer.
The public data remains at `artifacts/datasets/sharegpt/` on .58; downloading
additional copies for these HTTP-only tests is unnecessary.

Results live on .58 in `artifacts/baselines/performance/<backend>/`, with
TTFT/TPOT/ITL/E2E distributions and per-request details. The actual server
profile, image/container identity, and cache/mode assertions are recorded on
each serving host in `artifacts/baseline/performance-profile.json`;
startup logs are `vllm-nocache.log`. Network topology is part of this test:
the .58 client uses loopback/LAN-local access for Iluvatar and a LAN hop for
the other three servers. Different vendor engine versions remain necessary.
These are comparisons of deployed stacks, not isolated hardware rankings.

For this profile only, the Iluvatar service temporarily uses the shared eager
config on its existing devices 4/5. Restore its retained
`heterogeneous_compute/platforms/iluvatar/vllm.yaml` graph profile after measurement; never recreate
the container. Preserve earlier cache-enabled and graph results separately.

The repository baseline is commit `977d541b63303d2d395806d26ddfa54daa41d755`.
Network restrictions required a Git bundle transport; each checkout's origin
remains https://github.com/lvhaojieming/LLM-Serving-System.git. Deployment configs
are additional uncommitted changes. Existing local work was preserved.
