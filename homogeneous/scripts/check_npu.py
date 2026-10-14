"""Small deterministic Ascend execution check; does not load a model or start a service."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time


def check(device):
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is unavailable in this process")
    count = torch.npu.device_count()
    if count != 1:
        raise RuntimeError(f"Expected one process-visible NPU, observed {count}; check device assignment")
    target = torch.device(device)
    torch.npu.set_device(target)
    with torch.inference_mode():
        x = torch.ones((32, 32), dtype=torch.float32, device=target)
        y = x @ x
        torch.npu.synchronize()
        if y.device.type != "npu" or not torch.equal(y.cpu(), torch.full((32, 32), 32.0)):
            raise RuntimeError("Exact NPU matrix multiplication check failed")
        a = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
        expected = a @ a.T
        a_npu = a.to(target)
        timings = []
        maximum_error = 0.0
        for _ in range(3):
            started = time.perf_counter()
            actual = a_npu @ a_npu.T
            torch.npu.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            cpu_result = actual.cpu()
            maximum_error = max(maximum_error, float((cpu_result - expected).abs().max()))
            if actual.device.type != "npu" or not torch.allclose(cpu_result, expected, rtol=1e-3, atol=1e-3):
                raise RuntimeError("NPU result differs from the CPU reference")
    return {"passed": True, "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "torch_version": torch.__version__, "torch_npu_version": getattr(torch_npu, "__version__", "unknown"),
            "visible_npus": count, "device": str(target), "device_name": torch.npu.get_device_name(0),
            "checks": ["npu_available", "one_visible_device", "exact_matmul", "cpu_reference_matmul"],
            "max_absolute_error": maximum_error, "iteration_ms": timings,
            "scope": "existing_container_hardware_check", "kubernetes_device_allocation": "not_tested",
            "model_inference": "not_tested"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assigned-device-files", action="store_true")
    parser.add_argument("--runtime-device-map", choices=["physical", "automatic"], default="physical")
    args = parser.parse_args()
    try:
        physical = []
        denied = []
        if args.assigned_device_files:
            physical = sorted(int(m.group(1)) for p in Path("/dev").glob("davinci*")
                              if (m := re.fullmatch(r"davinci([0-9]+)", p.name)))
            if len(physical) != 1:
                raise RuntimeError(f"Kubernetes must expose exactly one compute device node: {physical}")
            if args.runtime_device_map == "physical":
                os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(physical[0])
            else:
                os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
            for index in range(8):
                if index in physical:
                    continue
                try:
                    fd = os.open(f"/dev/davinci{index}", os.O_RDWR | os.O_NONBLOCK)
                except OSError as exc:
                    if exc.errno not in {1, 2, 13}:
                        raise
                    denied.append(index)
                else:
                    os.close(fd)
                    raise RuntimeError(f"Unallocated compute device {index} can be opened")
        report = check(args.device)
        if args.assigned_device_files:
            report.update(scope="kubernetes_pod", allocated_physical_npus=physical,
                          denied_unallocated_npus=denied, kubernetes_device_allocation="passed",
                          pod_uid=os.environ["POD_UID"], node=os.environ["NODE_NAME"])
    except Exception as exc:
        report = {"passed": False, "error_type": type(exc).__name__, "error": str(exc),
                  "scope": "kubernetes_pod" if args.assigned_device_files else "existing_container_hardware_check",
                  "allocated_physical_npus": physical, "denied_unallocated_npus": denied}
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
