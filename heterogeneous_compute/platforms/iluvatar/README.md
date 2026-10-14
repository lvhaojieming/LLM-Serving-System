# Iluvatar Qwen3-14B backend

Validated on 172.22.162.58 (BI-V150, driver 4.5.0), using the existing
official CoreX 4.5 image with vLLM/vllm-iluvatar 0.23.0, Python 3.12,
and Torch 2.10.0. This is a BF16 backend deployment, not a completed
AWQ/GPTQ or Kubernetes/Gateway integration.

Image:

```text
registry.iluvatar.com.cn:10443/vllm_plugin/corex_vllm_plugin:0.23.0-corex-4.5.0.20260619-ubuntu24.04-py3.12-x86_64-v1
```

The persistent container is `heteroserve-iluvatar`. Its ID is
`3b1a689e8564e59c1065dc08d63c5b4ce2cf1ac6465bf306d910ae0e5e1ab710`.
Reuse it for subsequent experiments. It was created because existing shared
containers did not mount this user's model and repository directories.
The host repository `/home/zhangjinhao/LLM-Serving-System` is mounted at
`/workspace`; `/home/zhangjinhao/models` is mounted read-only at `/models`.

The server uses devices 4 and 5, TP2/PP1, seed 20261010, and port 18058.
Prefix caching is enabled, as verified in the actual engine configuration.
The original YAML `enable-prefix-caching: false` did not disable it in this
vLLM CLI version. Baseline result metadata and reports contain this correction.
Concurrency 4/8 reuse prompts from the concurrency-1 case and can hit the cache;
do not describe these TTFT results as cold-cache measurements.
Change `vllm.yaml` and `env.sh` for subsequent experiments. Before starting,
check device availability, the port, and existing processes to avoid duplicate
servers. After confirming no existing server, start inside the same container:

```bash
docker exec -d heteroserve-iluvatar bash -lc \
  'source /workspace/heterogeneous_compute/platforms/iluvatar/env.sh; exec vllm serve --config /workspace/heterogeneous_compute/platforms/iluvatar/vllm.yaml > /workspace/artifacts/iluvatar/vllm.log 2>&1'
```

TP initialization initially failed with the default NCCL path. Selecting
CoreX `libixsysnccl.so`, disabling cuMem and disabling IB resolved startup.
These choices are retained in `env.sh`; no engine source or shared environment
was modified.

The API endpoint is `http://192.168.162.58:18058/v1`. `/v1/models`, two local
chat responses, and one request from node .23 passed. Chat requests used
`chat_template_kwargs: {"enable_thinking": false}` and temperature zero.
Model loading occupied about 13.88 GiB per device. These are deployment
smoke checks, not throughput or accuracy benchmarks.

Logs and reproducibility records are in `artifacts/iluvatar/` on the host:
`vllm.log`, `deployment.json`, `acceptance.json`, and `startup-diagnostics.txt`.
The repository baseline commit is `977d541b63303d2d395806d26ddfa54daa41d755`,
verified against GitHub main on 2026-10-13. Network restrictions required a Git
bundle transfer; the Git remote remains the original GitHub URL.

## Performance baseline

Reuse the installed `vllm bench serve` implementation:

```bash
docker exec heteroserve-iluvatar bash /workspace/heterogeneous_compute/platforms/iluvatar/benchmark.sh
```

`benchmark.env` controls the dataset, seed, concurrency, output length, warmups,
request count, and output directory. The initial profile uses ShareGPT, 200
requests per case, three warmups, 64 generated tokens with EOS ignored, and
closed-loop concurrency 1/4/8. The endpoint is `/v1/completions`; prompts use the
installed official ShareGPT sampler. This is a controlled output-budget test,
not natural answer-length measurement, accuracy evaluation, or an open-loop
SLO capacity test. ITL is the benchmark client's observed streaming interval.

The public dataset is retained outside tracked source files at
`artifacts/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`, with
`manifest.json` recording its pinned revision, size, SHA256, and download URL.
It uses the same dataset revision as the existing distributed experiment.
Download reuses `shared/download_model.py`'s `fetch` function; no second
downloader was introduced.

Results, per-request timing, engine metrics, and device memory snapshots are
retained in `artifacts/iluvatar/benchmark/`. Package versions and benchmark
source SHA256 are recorded in `benchmark-versions.json`; deployment identity
and configuration hashes are in `deployment.json`. The shell wrapper refuses
to overwrite retained JSON results. For a new useful experiment, select its
own descriptive result directory in the config and retain the exact config
with the results. Run `bash -n heterogeneous_compute/platforms/iluvatar/benchmark.sh` before launch.

The current profile enables `FULL_DECODE_ONLY` graphs with capture sizes
1/2/4/8 and disables Torch compilation (mode 0). Prefill retains eager execution.
`cudagraph-metrics` records runtime graph dispatch. The graph profile uses
`artifacts/iluvatar/benchmark-graph/`; the earlier eager profile stays in
`artifacts/iluvatar/benchmark/`. Captured graphs and generated answers must be
validated before accepting performance results. The wrapper stores configuration
snapshots under each result directory; do not overwrite retained snapshots when
switching profiles.

The graph profile completed 600/600 requests. Output throughput at concurrency
1/4/8 was 30.12/83.67/161.77 tokens/s, versus 25.23/88.21/160.64 in eager mode.
The observed changes were +19.37%/-5.15%/+0.71%; graph execution is not uniformly
faster, and the concurrency-8 difference is negligible in a single run.
The model's default sampling parameters (temperature 0.6, top_k 20, top_p 0.95)
were used in both benchmark profiles. The benchmark seed fixes dataset sampling;
it does not guarantee identical generated text. Future greedy experiments can
explicitly add `--temperature=0`, but must not mix that change into this comparison.

An independent 32-token greedy profile exported 31 `cudaGraphLaunch` events on
each TP rank. Traces and verification are retained in `artifacts/iluvatar/graph-profile/`.
This diagnostic was performed after the performance runs and does not contribute
to reported throughput. Profiling was disabled again after verification. Complete
metrics and configuration corrections are in `graph-comparison.json`.
