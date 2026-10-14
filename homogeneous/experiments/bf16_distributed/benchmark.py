"""Closed-loop SSE benchmark with exact token counts and per-request timing."""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import platform
import socket

import anyio
import httpx


def percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows, elapsed):
    passed = [row for row in rows if row["success"]]
    result = {"requests": len(rows), "successful": len(passed), "failed": len(rows) - len(passed),
              "elapsed_seconds": elapsed, "requests_per_second": len(passed) / elapsed,
              "output_tokens_per_second": sum(r["output_tokens"] for r in passed) / elapsed,
              "total_tokens_per_second": sum(r["input_tokens"] + r["output_tokens"] for r in passed) / elapsed}
    timing_keys = sorted({key for row in passed for key in row.get("routing", {}) if key.endswith(("_ms", "_percent"))})
    for key in ("ttft_ms", "tpot_ms", "e2e_ms", *timing_keys):
        values = [row.get(key, row.get("routing", {}).get(key)) for row in passed if row.get(key, row.get("routing", {}).get(key)) is not None]
        result[key] = {"mean": statistics.mean(values) if values else None,
                       "p50": percentile(values, .5), "p95": percentile(values, .95), "p99": percentile(values, .99)}
    return result


async def request(client, url, model, prompt, output_length, index, timeout, sample_id=None, messages=None):
    prompt_length = len(prompt)
    payload = {"model": model, "prompt": prompt, "temperature": 0, "seed": 0,
               "max_tokens": output_length, "ignore_eos": True, "skip_special_tokens": False, "stream": True,
               "stream_options": {"include_usage": True}}
    endpoint = "/v1/completions"
    if messages is not None:
        payload.pop("prompt")
        payload.update(messages=messages, chat_template_kwargs={"enable_thinking": False})
        endpoint = "/v1/chat/completions"
    row = {"index": index, "sample_id": sample_id, "success": False, "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}
    started = time.perf_counter()
    first, last, usage, done, content = None, None, None, False, []
    first_choice = None
    try:
        with anyio.fail_after(timeout):
            async with client.stream("POST", url + endpoint, json=payload) as response:
                row["headers_ms"] = (time.perf_counter() - started) * 1000
                row["status"] = response.status_code
                response.raise_for_status()
                row["pod_uid"] = response.headers.get("x-heteroserve-pod-uid")
                if messages is not None:
                    row.update(request_id=response.headers.get("x-request-id"), expert=response.headers.get("x-moqe-expert"),
                        l1_instance=response.headers.get("x-moqe-l1-instance"), l1_node=response.headers.get("x-moqe-l1-node"),
                        l1_pod_uid=response.headers.get("x-moqe-l1-pod-uid"))
                    keys = {"gateway_headers_ms": "x-heteroserve-gateway-headers-ms",
                        "gateway_pre_dispatch_ms": "x-heteroserve-gateway-pre-dispatch-ms", "l1_rpc_ms": "x-heteroserve-l1-rpc-ms",
                        "l1_compute_ms": "x-moqe-router-ms", "l1_queue_ms": "x-heteroserve-l1-queue-ms",
                        "l1_service_ms": "x-heteroserve-l1-service-ms", "l2_pre_dispatch_ms": "x-heteroserve-l2-pre-dispatch-ms",
                        "l2_core_headers_ms": "x-heteroserve-l2-core-headers-ms", "worker_headers_ms": "x-heteroserve-worker-headers-ms",
                        "worker_engine_headers_ms": "x-heteroserve-worker-engine-headers-ms"}
                    row["routing"] = {key: float(response.headers[header]) for key, header in keys.items()}
                    row["routing"].update({key: response.headers["x-heteroserve-l1-" + key.replace("_", "-")] == "true"
                        for key in ("graph_enabled", "graph_used", "graph_fallback")})
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    event = json.loads(data)
                    if event.get("choices") and first_choice is None:
                        first_choice = time.perf_counter()
                    if event.get("error"):
                        raise RuntimeError("SSE reported a model error")
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        text = choice.get("text") or choice.get("delta", {}).get("content")
                        if text:
                            now = time.perf_counter()
                            first = now if first is None else first
                            last = now
                            content.append(text)
        if not done or first is None or usage is None:
            raise RuntimeError("Incomplete SSE stream, missing content or missing usage")
        if usage["prompt_tokens"] != prompt_length or usage["completion_tokens"] != output_length:
            raise RuntimeError("Server token counts differ from the benchmark profile")
        row.update(success=True, input_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
                   ttft_ms=(first - started) * 1000,
                   tpot_ms=(last - first) * 1000 / (usage["completion_tokens"] - 1) if usage["completion_tokens"] > 1 else None,
                   e2e_ms=(time.perf_counter() - started) * 1000,
                   output_sha256=hashlib.sha256("".join(content).encode()).hexdigest())
        row["first_choice_ms"] = (first_choice - started) * 1000 if first_choice is not None else None
        row["text_after_first_choice_ms"] = (first - first_choice) * 1000 if first_choice is not None else None
        if messages is not None:
            routing = row["routing"]
            routing["routing_transport_ms"] = routing["gateway_headers_ms"] - routing["worker_headers_ms"]
            routing["l2_routing_transport_ms"] = routing["l2_pre_dispatch_ms"] + routing["l2_core_headers_ms"] - routing["worker_headers_ms"]
            if not row["request_id"] or row["expert"] not in {"awq", "gptq"} or not row["l1_pod_uid"] or any(
                    not isinstance(v, bool) and (not math.isfinite(v) or v < 0) for v in routing.values()):
                raise RuntimeError("Missing identity or invalid nested routing durations")
            routing["routing_ttft_percent"] = 100 * routing["routing_transport_ms"] / row["ttft_ms"]
            routing["routing_e2e_percent"] = 100 * routing["routing_transport_ms"] / row["e2e_ms"]
            if routing["routing_ttft_percent"] > 100:
                raise RuntimeError("Routing interval exceeds observed TTFT")
    except Exception as exc:
        row.update(success=False, error=type(exc).__name__ + ": " + str(exc), e2e_ms=(time.perf_counter() - started) * 1000)
    return row


def engine_metrics(text):
    """Read cumulative engine timings; histogram deltas isolate a completed case."""
    names = ("time_to_first_token_seconds", "request_queue_time_seconds",
             "request_prefill_time_seconds", "request_prompt_tokens", "num_preemptions_total")
    result = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        metric, value = line.rsplit(" ", 1)
        key = metric.split("{", 1)[0].removeprefix("vllm:")
        if any(key == name or key in {name + "_sum", name + "_count"} for name in names):
            result[key] = result.get(key, 0) + float(value)
    return result


async def diagnose_ttft(config, rows, sample, url=None):
    """Compare client TTFT with existing engine queue/prefill metrics, without profiling."""
    url = url or config["engine_url"]
    model = config["model"]
    cases = []
    async with httpx.AsyncClient(timeout=300, trust_env=False,
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16)) as client:
        async def snapshot():
            response = await client.get(url + "/metrics")
            response.raise_for_status()
            return engine_metrics(response.text)

        async def batch(inputs, concurrency, output):
            semaphore = asyncio.Semaphore(concurrency)
            async def one(index, item):
                async with semaphore:
                    return await request(client, url, model, item["input_ids"], output, index, 240, item["id"])
            started = time.perf_counter()
            results = await asyncio.gather(*(one(i, item) for i, item in enumerate(inputs)))
            return results, time.perf_counter() - started

        warmups, _ = await batch([sample] * 8, 8, 16)
        assert all(x["success"] for x in warmups), warmups
        await asyncio.sleep(.1)
        plans = [("sharegpt_c8", rows, 8, 64)]
        longest = max(rows, key=lambda row: len(row["input_ids"]))
        for length in (32, 512, min(1024, len(longest["input_ids"]))):
            prompt = {"id": "fixed-" + str(length), "input_ids": longest["input_ids"][:length]}
            for concurrency in (1, 8):
                plans.append((f"fixed{length}_c{concurrency}", [prompt] * (8 if concurrency == 1 else 32), concurrency, 8))
        for name, inputs, concurrency, output in plans:
            before = await snapshot()
            results, elapsed = await batch(inputs, concurrency, output)
            assert all(x["success"] for x in results), results
            expected = len(results)
            for _ in range(100):
                after = await snapshot()
                count = after.get("request_prefill_time_seconds_count", 0) - before.get("request_prefill_time_seconds_count", 0)
                if count == expected:
                    break
                await asyncio.sleep(.1)
            assert count == expected, (name, count, expected)
            delta = {key: value - before.get(key, 0) for key, value in after.items()}
            components = {}
            for label, metric in (("engine_ttft_ms", "time_to_first_token_seconds"),
                                  ("queue_ms", "request_queue_time_seconds"),
                                  ("prefill_ms", "request_prefill_time_seconds")):
                assert delta[metric + "_count"] == expected
                components[label] = 1000 * delta[metric + "_sum"] / expected
            summary = summarize(results, elapsed)
            components["outside_engine_ttft_ms"] = summary["ttft_ms"]["mean"] - components["engine_ttft_ms"]
            components["engine_ttft_minus_queue_prefill_ms"] = components["engine_ttft_ms"] - components["queue_ms"] - components["prefill_ms"]
            components["headers_ms"] = statistics.mean(x["headers_ms"] for x in results)
            components["text_after_first_choice_ms"] = statistics.mean(x["text_after_first_choice_ms"] for x in results)
            result = {"name": name, "concurrency": concurrency, "output_tokens": output,
                      "input_rows_sha256": hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
                      "summary": summary, "components": components, "metrics_before": before,
                      "metrics_after": after, "metrics_delta": delta, "samples": results}
            cases.append(result)
            print(json.dumps({"event": "ttft_case", "name": name, "components": components}), flush=True)
    return {"configuration": config, "cases": cases,
            "scope": "Uninstrumented engine; completed-case histogram deltas; prefill includes execution, sampling and output processing; outside-engine residual includes input processing and stream delivery"}


async def diagnose_l1(rows, url, concurrency=1, warmup=5):
    """Measure the existing L1 service and retain probabilities for parity checks."""
    async with httpx.AsyncClient(timeout=120, trust_env=False,
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)) as client:
        identity = (await client.get(url + "/identity")).json()
        async def one(index, row):
            started = time.perf_counter()
            response = await client.post(url + "/route", json={"messages": row["messages"],
                "max_new_tokens": 64, "template_kwargs": {"enable_thinking": False}})
            response.raise_for_status()
            result = response.json()
            assert result["identity"]["pod_uid"] == identity["pod_uid"]
            decision = result["decision"]
            assert decision["expert"] in {"awq", "gptq"} and all(math.isfinite(x) for x in decision["probabilities"].values())
            return {"index": index, "id": row["id"], "decision": decision, "timing": result["timing"],
                    "rpc_ms": 1000 * (time.perf_counter() - started)}
        for i in range(warmup): await one(-i-1, rows[i % len(rows)])
        semaphore = asyncio.Semaphore(concurrency)
        async def task(index, row):
            async with semaphore: return await one(index, row)
        started = time.perf_counter()
        samples = await asyncio.gather(*(task(i, row) for i, row in enumerate(rows)))
        elapsed = time.perf_counter() - started
        summary = {"elapsed_seconds": elapsed, "requests_per_second": len(rows) / elapsed}
        for label, getter in (("compute_ms", lambda x: x["timing"]["compute_ms"]),
                              ("queue_ms", lambda x: x["timing"]["queue_ms"]),
                              ("rpc_ms", lambda x: x["rpc_ms"])):
            values = [getter(x) for x in samples]
            summary[label] = {"mean": statistics.mean(values), "p50": percentile(values, .5),
                              "p95": percentile(values, .95), "p99": percentile(values, .99)}
        return {"identity": identity, "concurrency": concurrency, "summary": summary, "samples": samples}


async def run(config, url, output, requests_path):
    settings = config["benchmark"]
    dataset = [json.loads(line) for line in requests_path.read_text().splitlines() if line.strip()]
    needed = settings["requests_per_case"] + settings["warmup_requests"]
    if len(dataset) < needed or len({row["id"] for row in dataset}) != len(dataset):
        raise ValueError("Prepared real dataset is insufficient or contains duplicate IDs")
    warmups, samples = dataset[:settings["warmup_requests"]], dataset[settings["warmup_requests"]:needed]
    report = {"success": False, "config": config, "url": url, "cases": [],
              "dataset_file_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
              "input_tokens": {"mean": statistics.mean(len(row["input_ids"]) for row in samples),
                  "min": min(len(row["input_ids"]) for row in samples), "max": max(len(row["input_ids"]) for row in samples),
                  "p95": percentile([len(row["input_ids"]) for row in samples], .95), "p99": percentile([len(row["input_ids"]) for row in samples], .99)},
              "client": {"hostname": socket.gethostname(), "python": platform.python_version(), "httpx": httpx.__version__},
              "scope": settings.get("scope", "ShareGPT real first-turn chat prompts with canonical Qwen template; fixed output budget; warm closed-loop concurrency; direct model Service, not open-loop SLO or maximum-capacity"),
              "definitions": {"ttft": "request start to first nonempty completion text",
                  "tpot": "last nonempty text time minus first, divided by actual output_tokens - 1; per-request average, not per-SSE-event ITL",
                  "throughput": "successful output or total tokens divided by full case wall time",
                  "routing_transport_ms": "Gateway handler entry to downstream headers minus worker entry to engine headers; includes L1, L2 and inter-hop transport; excludes worker/backend wait and stream relay after headers",
                  "routing_percent": "Per-request observed routing/transport startup duration divided by client TTFT or E2E; summarize individual percentages, never divide separate percentiles"},
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
    save()
    timeout = settings["timeout_seconds"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout), trust_env=False,
            limits=httpx.Limits(max_connections=max(settings["concurrency"]), max_keepalive_connections=max(settings["concurrency"]))) as client:
        try:
            ready = await client.get(url + "/ready")
            ready.raise_for_status()
            if settings.get("entry") == "gateway" and not ready.json().get("auto_expert_coverage_complete"):
                raise RuntimeError("Automatic routing requires complete healthy expert coverage")
        except Exception as exc:
            report["error"] = {"phase": "readiness", "error": type(exc).__name__ + ": " + str(exc)}
            save()
            raise
        report["server_identity"] = ready.json()
        for concurrency in settings["concurrency"]:
            for index, sample in enumerate(warmups):
                warmup = await request(client, url, config["runtime"]["served_model"], sample["input_ids"],
                    settings["output_tokens"], -index - 1, timeout, sample["id"], sample.get("messages"))
                if not warmup["success"]:
                    report["error"] = {"phase": "warmup", "sample": warmup}
                    save()
                    raise RuntimeError("Warmup failed; measurement stopped")
            semaphore = asyncio.Semaphore(concurrency)
            completed = []
            start = time.perf_counter()
            async def sample_request(index):
                async with semaphore:
                    row = samples[index]
                    result = await request(client, url, config["runtime"]["served_model"], row["input_ids"],
                        settings["output_tokens"], index, timeout, row["id"], row.get("messages"))
                    completed.append(result)
                    if len(completed) % 10 == 0:
                        report["progress"] = {"completed": len(completed), "total": len(samples),
                            "failed": sum(not r["success"] for r in completed), "elapsed_seconds": time.perf_counter() - start}
                        save()
                        print(json.dumps({"event": "progress", **report["progress"]}), flush=True)
                    return result
            rows = await asyncio.gather(*(sample_request(index) for index in range(settings["requests_per_case"])))
            result = {"dataset": settings["dataset"]["name"], "output_tokens": settings["output_tokens"], "concurrency": concurrency,
                      "samples": rows, "summary": summarize(rows, time.perf_counter() - start)}
            report["cases"].append(result)
            save()
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
            if result["summary"]["failed"]:
                report["error"] = "Some requests failed; do not treat throughput as a passing measurement"
                save()
                raise RuntimeError(report["error"])
        report["success"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(json.loads(args.config.read_text()), args.url.rstrip("/"), args.output, args.requests))


if __name__ == "__main__":
    main()
