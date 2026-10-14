#!/usr/bin/env bash
# Reuse the installed vLLM benchmark; all experiment settings live in one config.
set -euo pipefail
CONFIG_FILE="${1:-/workspace/heterogeneous_compute/platforms/iluvatar/benchmark.env}"
source "$CONFIG_FILE"
mkdir -p "$RESULT_DIR"
for concurrency in $CONCURRENCIES; do
  result="sharegpt-c${concurrency}.json"
  if test -e "$RESULT_DIR/$result"; then
    echo "Refusing to overwrite retained results: $RESULT_DIR/$result" >&2
    exit 1
  fi
done
mkdir -p "$RESULT_DIR/config"
cp "${SERVER_CONFIG:-/workspace/heterogeneous_compute/platforms/iluvatar/vllm.yaml}" "$RESULT_DIR/config/"
if test "${CAPTURE_DEVICE_STATS:-true}" = true; then
  cp /workspace/heterogeneous_compute/platforms/iluvatar/env.sh "$RESULT_DIR/config/"
elif test -f /workspace/heterogeneous_compute/configs/baselines/cluster.json; then
  cp /workspace/heterogeneous_compute/configs/baselines/cluster.json "$RESULT_DIR/config/"
fi
cp "$CONFIG_FILE" "$RESULT_DIR/config/benchmark.env"
printf 'BASE_URL=%q\nRESULT_DIR=%q\n' "$BASE_URL" "$RESULT_DIR" \
  >> "$RESULT_DIR/config/benchmark.env"
SAMPLING_ARGS=()
if test -n "${TEMPERATURE:-}"; then
  SAMPLING_ARGS+=(--temperature "$TEMPERATURE")
fi
for concurrency in $CONCURRENCIES; do
  label="sharegpt-c${concurrency}"
  echo "CASE_START=$label"
  curl --fail --silent "$BASE_URL/metrics" > "$RESULT_DIR/$label-metrics-before.txt"
  if test "${CAPTURE_DEVICE_STATS:-true}" = true; then
    ixsmi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader -i 4,5 > "$RESULT_DIR/$label-devices-before.csv"
  fi
  vllm bench serve \
    --backend vllm --endpoint /v1/completions --base-url "$BASE_URL" \
    --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" --dataset-name sharegpt \
    --dataset-path "$DATASET_PATH" --sharegpt-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --num-warmups "$NUM_WARMUPS" \
    --max-concurrency "$concurrency" --request-rate inf --ignore-eos \
    --seed "$BENCHMARK_SEED" "${SAMPLING_ARGS[@]}" --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 --save-result --save-detailed \
    --result-dir "$RESULT_DIR" --result-filename "$label.json" \
    > "$RESULT_DIR/$label.log" 2>&1
  curl --fail --silent "$BASE_URL/metrics" > "$RESULT_DIR/$label-metrics-after.txt"
  if test "${CAPTURE_DEVICE_STATS:-true}" = true; then
    ixsmi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader -i 4,5 > "$RESULT_DIR/$label-devices-after.csv"
  fi
  echo "CASE_COMPLETE=$label"
done
echo BENCHMARK_COMPLETE
