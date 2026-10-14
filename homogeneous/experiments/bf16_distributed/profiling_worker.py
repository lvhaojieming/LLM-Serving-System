"""Use the existing Ascend worker with a small operator trace, without allocation stacks."""
from execution_worker import NPUWorker as AscendWorker
import json
import os
from pathlib import Path
import socket
import time


class NPUWorker(AscendWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm import envs
        self.profile_mode = json.loads(Path(__file__).with_name("profiling.json").read_text())["mode"]
        self.timing_active = False
        self.timings = {}
        if self.profile_mode == "timing":
            self.profiler = None
            return
        if envs.VLLM_TORCH_PROFILER_DIR:
            import torch_npu
            self.profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, with_stack=False, profile_memory=False,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    export_type=torch_npu.profiler.ExportType.Text,
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    envs.VLLM_TORCH_PROFILER_DIR))

    def timed(self, name, function):
        def call(*args, **kwargs):
            if not self.timing_active:
                return function(*args, **kwargs)
            started = time.perf_counter_ns()
            try:
                return function(*args, **kwargs)
            finally:
                self.timings.setdefault(name, []).append((time.perf_counter_ns()-started)/1000)
        return call

    def load_model(self):
        super().load_model()
        if self.profile_mode != "timing":
            return
        runner = self.model_runner
        for name in ("execute_model", "prepare_model_input"):
            if hasattr(runner, name):
                setattr(runner, name, self.timed("runner."+name, getattr(runner, name)))
        for name in ("forward", "compute_logits", "sample"):
            if hasattr(runner.model, name):
                setattr(runner.model, name, self.timed("model."+name, getattr(runner.model, name)))
        for name, module in runner.model.named_modules():
            label = name.rsplit(".", 1)[-1]
            if label in {"mlp", "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "rotary_emb",
                         "q_norm", "k_norm", "input_layernorm", "post_attention_layernorm"}:
                module.forward = self.timed("layer."+label, module.forward)
            if label in {"q_norm", "k_norm"}:
                module.forward_native = self.timed("layer."+label+".native", module.forward_native)
            if label in {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"}:
                method = module.quant_method
                method.apply = self.timed("linear.apply."+label, method.apply)
        import torch_npu
        torch_npu.npu_weight_quant_batchmatmul = self.timed(
            "backend.weight_quant_batchmatmul", torch_npu.npu_weight_quant_batchmatmul)
        from vllm.distributed.parallel_state import GroupCoordinator
        for name in ("all_reduce", "broadcast_tensor_dict", "send_tensor_dict", "recv_tensor_dict"):
            setattr(GroupCoordinator, name, self.timed("communication."+name, getattr(GroupCoordinator, name)))
        import torch
        torch.nn.functional.linear = self.timed("backend.linear", torch.nn.functional.linear)
        torch.npu.synchronize = self.timed("device.synchronize", torch.npu.synchronize)

    def execute_model(self, *args, **kwargs):
        return self.timed("worker.execute_model", super().execute_model)(*args, **kwargs)

    def start_profile(self):
        if self.profile_mode != "timing":
            return super().start_profile()
        self.timings = {}
        self.timing_active = True

    def stop_profile(self):
        if self.profile_mode != "timing":
            return super().stop_profile()
        self.timing_active = False
        from vllm import envs
        directory = Path(envs.VLLM_TORCH_PROFILER_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        report = {"rank": self.rank, "pid": os.getpid(), "device": str(self.device),
                  "dispatch_env": {name: os.environ.get(name) for name in
                      ("TASK_QUEUE_ENABLE", "CPU_AFFINITY_CONF", "ASCEND_LAUNCH_BLOCKING")},
                  "scope": "CPU wall intervals in microseconds; nested intervals overlap; includes waits; no NPU instrumentation",
                  "measurements": self.timings}
        (directory / (socket.gethostname()+"-"+str(os.getpid())+"-timing.json")).write_text(json.dumps(report))
