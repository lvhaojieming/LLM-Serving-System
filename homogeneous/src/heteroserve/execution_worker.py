"""Shared Ascend worker for production and benchmark execution optimizations."""
import json
import os
from pathlib import Path
import socket

from vllm_ascend.worker.worker import NPUWorker as AscendWorker


class NPUWorker(AscendWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = json.loads(Path(__file__).with_name("worker.json").read_text())
        self.execution_optimizations = config.get("execution_optimizations", {})
        self.qknorm_audit = None
        self.qknorm_audit_written = False

    def load_model(self):
        super().load_model()
        if self.execution_optimizations.get("native_int4_script_mlp", False):
            if self.model_config.hf_config.model_type != "qwen3":
                raise ValueError("Scripted INT4 MLP is validated for Qwen3 only")
            from ascend_execution import install_qwen3_int4_script_mlp
            count = install_qwen3_int4_script_mlp(self.model_runner.model)
            if not count:
                raise RuntimeError("No supported Qwen3 INT4 MLP callsites found")
            print(json.dumps({"event": "ascend_int4_script_mlp_installed", "rank": self.rank,
                              "mlp_count": count}), flush=True)
        if self.execution_optimizations.get("ascend_pp_cache_views", False):
            from ascend_execution import install_ascend_pp_cache_views
            count = install_ascend_pp_cache_views(self.model_runner.model)
            if not count:
                raise RuntimeError("No supported Ascend V0 attention cache callsites found")
            print(json.dumps({"event": "ascend_pp_cache_views_installed", "rank": self.rank,
                              "attention_count": count}), flush=True)
        if self.execution_optimizations.get("qwen3_fused_qknorm", False):
            if self.model_config.hf_config.model_type != "qwen3":
                raise ValueError("Q/K norm optimization is validated for Qwen3 only")
            from ascend_execution import install_qwen3_qknorm
            mode = self.execution_optimizations.get("qknorm_mode", "native")
            verify = self.execution_optimizations.get("qknorm_verify_calls", 0)
            if verify:
                self.qknorm_audit = {"enabled": False, "calls_per_norm": verify, "norms": {}}
            count = install_qwen3_qknorm(self.model_runner.model, mode=mode, audit=self.qknorm_audit)
            if not count:
                raise RuntimeError("No supported Qwen3 Q/K norm callsites were found")
            print(json.dumps({"event": "ascend_qknorm_optimized", "rank": self.rank,
                              "norm_count": count, "mode": mode}), flush=True)

    def execute_model(self, *args, **kwargs):
        audit = self.qknorm_audit
        if audit is None or self.qknorm_audit_written:
            return super().execute_model(*args, **kwargs)
        # Startup memory profiling calls the runner directly. Audit only serving
        # calls, so real prefill/decode tensors are compared with the original math.
        audit["enabled"] = True
        try:
            return super().execute_model(*args, **kwargs)
        finally:
            audit["enabled"] = False
            if all(row["checked"] >= audit["calls_per_norm"] for row in audit["norms"].values()):
                report = {"rank": self.rank, "pid": os.getpid(), "mode":
                          self.execution_optimizations.get("qknorm_mode", "native"),
                          "scope": "Bitwise comparison on actual serving Q/K norm inputs",
                          **audit}
                directory = Path("/scratch/qknorm-audit")
                directory.mkdir(exist_ok=True)
                (directory / (socket.gethostname() + "-" + str(os.getpid()) + ".json")).write_text(json.dumps(report))
                self.qknorm_audit_written = True
                print(json.dumps({"event": "qknorm_audit_complete", "rank": self.rank,
                                  "checks": sum(row["checked"] for row in audit["norms"].values()),
                                  "mismatches": sum(row["mismatches"] for row in audit["norms"].values())}), flush=True)
