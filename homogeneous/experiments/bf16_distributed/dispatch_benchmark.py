"""Compare host dispatch on the same isolated NPUs, without loading a model."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def measure(function, torch, rounds=5, samples=20):
    for _ in range(8):
        function()
    torch.npu.synchronize()
    blocks, totals = [], []
    for _ in range(rounds):
        launches = []
        for _ in range(samples):
            started = time.perf_counter_ns()
            result = function()
            dispatched = time.perf_counter_ns()
            torch.npu.synchronize()
            finished = time.perf_counter_ns()
            launches.append((dispatched - started) / 1000)
            totals.append((finished - started) / 1000)
        blocks.append(statistics.median(launches))
    return {"launch_median_us": statistics.median(blocks),
            "block_medians_us": blocks, "synchronized_median_us": statistics.median(totals)}


def child(seed, mode="dispatch"):
    import importlib.metadata
    from types import SimpleNamespace
    import torch
    import torch_npu
    import adapter_patch
    adapter_patch.install()
    from native_config import NativeInt4LinearMethod
    from native_parallel import install
    install(fast_path=True)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    assert torch.npu.device_count() == 2, "Requires two Kubernetes-isolated NPUs"
    method = object.__new__(NativeInt4LinearMethod)
    method.quant_config = SimpleNamespace(group_size=128)
    op = torch.ops.npu.npu_weight_quant_batchmatmul.default

    def linear(x, weight, scale, offset):
        return torch.ops.npu.npu_weight_quant_batchmatmul.default(
            x, weight, scale, offset, antiquant_group_size=128)

    results = []
    for device in range(2):
        torch.npu.set_device(device)
        traced = None
        for projection, k, n in [("qkv", 5120, 3584), ("o", 2560, 5120),
                                 ("gate_up", 5120, 17408), ("down", 8704, 5120)]:
            weight = torch.randint(-2147483648, 2147483647, (k, n // 8),
                                   dtype=torch.int32, device="npu")
            scale = torch.full((k // 128, n), .01, dtype=torch.float16, device="npu")
            offset = torch.zeros_like(scale)
            layer = SimpleNamespace(qweight=weight, scales=scale, qzeros=offset)
            dense = torch.randn((n, k), dtype=torch.bfloat16, device="npu") * .01
            dense_fp16 = dense.to(torch.float16) if mode == "prefill" else None
            for batch in ((8, 32, 128, 512, 1024, 4096) if mode == "prefill" else (1, 8, 16)):
                x = torch.randn((batch, k), dtype=torch.float16, device="npu")
                bf = x.to(torch.bfloat16)
                if traced is None:
                    traced = torch.jit.trace(linear, (x, weight, scale, offset), check_trace=False)
                variants = {"current_fast_path": lambda: method.apply(layer, x),
                            "direct_op": lambda: op(x, weight, scale, offset, antiquant_group_size=128),
                            "scripted_linear": lambda: traced(x, weight, scale, offset),
                            "bf16": lambda: torch.nn.functional.linear(bf, dense)}
                if mode == "prefill":
                    variants = {"current_fast_path": variants["current_fast_path"], "bf16": variants["bf16"],
                                "fp16": lambda: torch.nn.functional.linear(x, dense_fp16)}
                reference_tensor = variants["current_fast_path"]().cpu()
                reference = reference_tensor.view(torch.int16)
                validation = {}
                for name in (() if mode == "prefill" else ("direct_op", "scripted_linear")):
                    candidate = variants[name]().cpu()
                    equal = torch.equal(candidate.view(torch.int16), reference)
                    validation[name] = {"exact_equal": equal,
                        "max_abs_difference": (candidate.float() - reference_tensor.float()).abs().max().item()}
                    if not equal:
                        del variants[name]
                repeat = variants["current_fast_path"]().cpu()
                validation["current_repeat"] = {"exact_equal": torch.equal(repeat.view(torch.int16), reference)}
                # Alternating order reduces systematic order/thermal bias.
                keys = list(variants)
                if len(results) % 2:
                    keys.reverse()
                row = {"device": device, "projection": projection, "batch": batch,
                       "k": k, "n": n, "validation": validation,
                       "measurements": {name: measure(variants[name], torch,
                            rounds=3 if mode == "prefill" else 5, samples=10 if mode == "prefill" else 20) for name in keys}}
                results.append(row)
            del weight, scale, offset, layer, dense, dense_fp16, x, bf, reference, reference_tensor, repeat, variants
            torch.npu.empty_cache()
    return {"seed": seed, "mode": mode, "env": {k: os.environ.get(k) for k in
            ("TASK_QUEUE_ENABLE", "CPU_AFFINITY_CONF", "ASCEND_LAUNCH_BLOCKING", "OMP_NUM_THREADS")},
            "versions": {k: importlib.metadata.version(k) for k in ("torch", "torch-npu", "vllm")},
            "physical_devices": sorted(p.name for p in Path("/dev").glob("davinci[0-9]*")),
            "operator_schema": str(op._schema), "cases": results}


def run(output, seed, mode="dispatch"):
    if output.exists():
        raise FileExistsError("Refusing to overwrite dispatch results")
    report = {"scope": "Same devices and projection shapes; CPU enqueue intervals exclude synchronization; "
                       "random packed INT4 and dense BF16 weights; not model throughput or kernel-only time",
              "plans": []}
    for queue, affinity in ([(2, 0)] if mode == "prefill" else [(1, 0), (2, 0), (2, 1)]):
        env = {**os.environ, "TASK_QUEUE_ENABLE": str(queue), "CPU_AFFINITY_CONF": str(affinity),
               "ASCEND_LAUNCH_BLOCKING": "0", "OMP_NUM_THREADS": "4"}
        process = subprocess.run([sys.executable, __file__, "--child", "--seed", str(seed), "--mode", mode],
                                 env=env, text=True, capture_output=True, timeout=600 if mode == "prefill" else 240)
        if process.returncode:
            raise RuntimeError(process.stdout[-2000:] + process.stderr[-2000:])
        line = next(line for line in reversed(process.stdout.splitlines()) if line.startswith("DISPATCH_RESULT="))
        report["plans"].append(json.loads(line.split("=", 1)[1]))
        print("DISPATCH_PLAN_COMPLETED", queue, affinity, flush=True)
    output.write_text(json.dumps(report, indent=2))
    print("DISPATCH_BENCHMARK_COMPLETED", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", choices=("dispatch", "prefill"), default="dispatch")
    args = parser.parse_args()
    if args.child:
        print("DISPATCH_RESULT=" + json.dumps(child(args.seed, args.mode)), flush=True)
    else:
        if args.output is None:
            parser.error("--output is required")
        run(args.output, args.seed, args.mode)
