"""Static-buffer adapters for official MLP and local TP graph execution.

These are compute-island graphs, not an unmodified vLLM whole-engine runner.
Attention/KV metadata and cross-node transport stay outside graph capture.
"""
from contextlib import nullcontext


class MLPGraphRunner:
    def __init__(self, forward, device, vllm_config, max_tokens=4):
        if device.type == 'npu':
            from vllm_ascend.utils import enable_custom_op
            if not enable_custom_op():
                raise RuntimeError('Official Ascend graph extension could not be initialized')
        self.forward = forward
        self.device = device
        self.vllm_config = vllm_config
        self.max_tokens = max_tokens
        self.entries = {}
        self.captures = self.replays = self.uncaptured_calls = 0

    def __call__(self, x):
        import torch
        if x.ndim != 2 or not 1 <= x.shape[0] <= self.max_tokens:
            self.uncaptured_calls += 1
            return self.forward(x)
        key = (tuple(x.shape), x.dtype, x.device)
        if key not in self.entries:
            entry = dict(input=x.clone())
            # Pure MLP warmup does not advance KV or sample tokens. Both local
            # TP ranks execute the same deterministic shape/capture sequence.
            for _ in range(2):
                self.forward(entry['input'])
            accelerator = getattr(torch, self.device.type)
            accelerator.synchronize()
            if self.device.type == 'npu':
                from vllm.config import CUDAGraphMode
                from vllm.compilation.cuda_graph import CUDAGraphOptions
                from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
                entry['runner'] = ACLGraphWrapper(self.forward, self.vllm_config,
                    CUDAGraphMode.PIECEWISE,
                    cudagraph_options=CUDAGraphOptions(weak_ref_output=False))
                entry['output'] = self._npu_call(entry)
            elif self.device.type == 'musa':
                from vllm.distributed import get_tp_group
                communicator = get_tp_group().device_communicator
                custom = getattr(communicator, 'ca_comm', None)
                capture = custom.capture() if custom is not None and not custom.disabled else nullcontext()
                entry['graph'] = accelerator.MUSAGraph()
                with capture, accelerator.graph(entry['graph']):
                    entry['output'] = self.forward(entry['input'])
            else:
                raise ValueError('This graph adapter supports installed NPU/MUSA APIs only')
            self.entries[key] = entry
            self.captures += 1
            # Capture records operations; its output is not a completed live
            # forward. Replay once before using the first request's result.
            if self.device.type == 'npu':
                self._npu_call(entry)
            else:
                entry['graph'].replay()
            self.replays += 1
        else:
            entry = self.entries[key]
            entry['input'].copy_(x)
            if self.device.type == 'npu':
                self._npu_call(entry)
            else:
                entry['graph'].replay()
            self.replays += 1
        # The next in-flight microbatch can replay this graph while the PP
        # sender copies the previous result. Give each caller owned storage.
        return entry['output'].clone()

    def _npu_call(self, entry):
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, get_forward_context
        context = get_forward_context()
        previous = (context.cudagraph_runtime_mode, context.batch_descriptor,
                    getattr(context, 'capturing', False))
        try:
            context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE
            context.batch_descriptor = BatchDescriptor(num_tokens=entry['input'].shape[0])
            return entry['runner'](entry['input'])
        finally:
            context.cudagraph_runtime_mode, context.batch_descriptor, context.capturing = previous

    def stats(self):
        return dict(captures=self.captures, replays=self.replays,
                    uncaptured_calls=self.uncaptured_calls)
