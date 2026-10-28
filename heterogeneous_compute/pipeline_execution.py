"""Experimental Qwen3 heterogeneous PP executor using vLLM TP model layers.

Both serialized and continuously batched schedules reuse eager attention.
No vendor engine files are modified. Tensor transport is
framed JSON plus raw BF16, never Python pickle or whole-model HTTP chaining.
"""
import argparse
import datetime
import json
import socket
import struct
import time
import queue
import concurrent.futures
import threading
import uuid
import signal
import select
from functools import lru_cache
from pathlib import Path
if __package__:
    from .scheduling import RequestState, BatchState, ContinuousScheduler
else:
    from scheduling import RequestState, BatchState, ContinuousScheduler


class TensorChannel:
    VERSION = 1
    MAX_HEADER = 65536
    MAX_PAYLOAD = 64 * 1024 * 1024

    def __init__(self, connection, version=1):
        self.connection = connection
        self.VERSION = version
        connection.settimeout(180)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def read(self, length):
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self.connection.recv(length - len(chunks))
            if not chunk:
                raise ConnectionError('PP peer disconnected')
            chunks.extend(chunk)
        return chunks

    def send(self, metadata, tensors=None):
        payloads, descriptors = [], []
        for name, tensor in (tensors or {}).items():
            import torch
            if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                raise ValueError('PP activations must be two-dimensional BF16')
            cpu = tensor.detach().cpu().contiguous()
            payload = cpu.view(torch.uint16).numpy().tobytes()
            descriptors.append(dict(name=name, shape=list(cpu.shape), dtype='bfloat16', bytes=len(payload)))
            payloads.append(payload)
        header = json.dumps(dict(version=self.VERSION, metadata=metadata,
                                 tensors=descriptors)).encode()
        if len(header) > self.MAX_HEADER or sum(map(len, payloads)) > self.MAX_PAYLOAD:
            raise ValueError('PP packet exceeds configured bounds')
        self.connection.sendall(struct.pack('!I', len(header)) + header)
        for payload in payloads:
            self.connection.sendall(payload)

    def receive(self, device, allow_idle=False):
        if allow_idle:
            # Keep the send/partial-packet timeout bounded on this duplex socket.
            select.select([self.connection], [], [])
        length, = struct.unpack('!I', self.read(4))
        if not 0 < length <= self.MAX_HEADER:
            raise ValueError('Invalid PP header length')
        header = json.loads(self.read(length))
        if header['version'] != self.VERSION:
            raise ValueError('Incompatible PP protocol')
        tensors, total = {}, 0
        for desc in header['tensors']:
            shape = desc['shape']
            size = desc['bytes']
            total += size
            if (desc.get('dtype') != 'bfloat16' or len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape)
                    or size != shape[0] * shape[1] * 2 or total > self.MAX_PAYLOAD
                    or desc['name'] in tensors):
                raise ValueError('Invalid PP tensor descriptor')
            import torch
            tensors[desc['name']] = torch.frombuffer(self.read(size), dtype=torch.bfloat16).clone().reshape(shape).to(device)
        return header['metadata'], tensors


class AsyncTensorSender:
    """Bounded FIFO transport worker; only batch events gate activation copies."""
    def __init__(self, channel, device, trace, capacity=2):
        self.channel, self.device, self.trace = channel, device, trace
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.thread = threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def send(self, metadata, tensors=None, events=None, started=None):
        self.check()
        while True:
            try:
                self.queue.put(None if metadata is None else (metadata,tensors,events,started),timeout=.1)
                break
            except queue.Full:
                self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Asynchronous PP transport failed') from self.error

    def run(self):
        try:
            import torch
            accelerator=getattr(torch,self.device.type)
            accelerator.set_device(self.device)
            stream=accelerator.Stream()
            with torch.inference_mode():
                while True:
                    item=self.queue.get()
                    if item is None:
                        break
                    metadata,tensors,events,started=item
                    copied={}
                    if tensors:
                        if getattr(self.channel, 'device_transport', False):
                            # MUSA stream waits alone did not make elapsed_time
                            # ready reliably. Device-buffer transport also needs
                            # the producer complete before the native stream reads.
                            events[0].synchronize()
                            events[1].synchronize()
                        with accelerator.stream(stream):
                            stream.wait_event(events[1])
                            copied={name:(t.detach() if getattr(self.channel, 'device_transport', False)
                                          else t.detach().to('cpu')) for name,t in tensors.items()}
                        # CPU reads only these copies; producer may enqueue next batch.
                        stream.synchronize()
                        self.trace.write(event='execution',batch_id=metadata['batch_id'],
                            start=started,end=time.perf_counter(),
                            device_ms=events[0].elapsed_time(events[1]),
                            batch_size=len(metadata['segments']),
                            tokens=sum(len(s['positions']) for s in metadata['segments']),
                            scope='host execution window includes copy; device_ms is local device event time')
                    sent=time.perf_counter()
                    self.channel.send(metadata,copied)
                    self.trace.write(event='transport_send',batch_id=metadata.get('batch_id'),
                                     start=sent,end=time.perf_counter())
        except Exception as error:
            self.error=error
            print(json.dumps(dict(event='transport_error', error=repr(error))), flush=True)
            self.trace.write(event='transport_error', error=repr(error))
            try:
                self.channel.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self):
        if self.thread.is_alive() and self.error is None:
            self.send(None)
            # None metadata is the worker sentinel, not a protocol message.
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                self.channel.connection.shutdown(socket.SHUT_RDWR)
                self.thread.join(timeout=5)
                raise TimeoutError('PP transport did not drain')
        self.check()


class AsyncTensorReceiver:
    """Prefetch ordered PP packets on a separate copy stream."""
    def __init__(self,channel,device,capacity=2):
        self.channel,self.device=channel,device
        self.queue=queue.Queue(maxsize=capacity)
        self.closing=False
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def publish(self,value):
        while not self.closing:
            try:
                self.queue.put(value,timeout=.1)
                return
            except queue.Full:
                pass

    def run(self):
        try:
            import torch
            accelerator=getattr(torch,self.device.type)
            accelerator.set_device(self.device)
            stream=accelerator.Stream()
            with torch.inference_mode():
                while not self.closing:
                    packet,cpu=self.channel.receive('cpu',allow_idle=True)
                    with accelerator.stream(stream):
                        tensors={name:value.to(self.device) for name,value in cpu.items()}
                        event=accelerator.Event()
                        event.record()
                    self.publish((packet,tensors,event))
                    if packet['op']=='stop':
                        break
        except Exception as error:
            self.publish(error)

    def receive(self):
        value=self.queue.get()
        if isinstance(value,Exception):
            raise value
        packet,tensors,event=value
        import torch
        compute_stream=getattr(torch,self.device.type).current_stream()
        compute_stream.wait_event(event)
        for tensor in tensors.values():
            tensor.record_stream(compute_stream)
        return packet,tensors

    def close(self):
        self.closing=True
        try:
            self.channel.connection.shutdown(socket.SHUT_RD)
        except OSError:
            pass
        self.thread.join(timeout=5)


class ParallelGroup:
    """vLLM group interface: native TP stays local, PP is transport-managed."""
    def __init__(self, rank, size, device, native=True):
        self.rank_in_group = rank
        self.world_size = size
        self.device = device
        self.native = native

    @property
    def is_first_rank(self):
        return self.rank_in_group == 0

    @property
    def is_last_rank(self):
        return self.rank_in_group == self.world_size - 1

    def all_reduce(self, tensor):
        import torch.distributed as dist
        work=dist.all_reduce(tensor,async_op=True)
        work.wait()
        return tensor

    def all_gather(self, tensor, dim=-1):
        import torch
        import torch.distributed as dist
        pieces = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work=dist.all_gather(pieces,tensor,async_op=True)
        work.wait()
        return torch.cat(pieces, dim=dim)


@lru_cache(maxsize=16)
def npu_causal_mask(device):
    import torch
    return torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1)


def vendor_attention(pieces, scale, device_type):
    """Use installed official vendor kernels; adapt layouts, never replace kernels."""
    import torch
    if device_type == 'npu':
        import torch_npu
        query_lengths, key_lengths = [], []
        for q, k, _, _ in pieces:
            query_lengths.append(q.shape[1]+(query_lengths[-1] if query_lengths else 0))
            key_lengths.append(k.shape[1]+(key_lengths[-1] if key_lengths else 0))
        # The vendor TND kernel implements independent variable-length requests
        # and bottom-right causal alignment for chunks with existing KV.
        mask = npu_causal_mask(pieces[0][0].device)
        output = torch_npu.npu_fusion_attention(
            torch.cat([p[0].transpose(0, 1) for p in pieces]).contiguous(),
            torch.cat([p[1].transpose(0, 1) for p in pieces]).contiguous(),
            torch.cat([p[2].transpose(0, 1) for p in pieces]).contiguous(),
            head_num=pieces[0][0].shape[0], input_layout='TND', scale=scale,
            keep_prob=1.0, sparse_mode=3, atten_mask=mask,
            actual_seq_qlen=query_lengths, actual_seq_kvlen=key_lengths,
            pre_tockens=2147483647, next_tockens=0)[0]
        return output.reshape(output.shape[0], -1)
    if device_type == 'cuda':
        from vllm_iluvatar.attention.flash_attn_interface import flash_attn_varlen_func
        q_lengths = [p[0].shape[1] for p in pieces]
        k_lengths = [p[1].shape[1] for p in pieces]
        def offsets(lengths):
            values = [0]
            for length in lengths:
                values.append(values[-1]+length)
            return torch.tensor(values, dtype=torch.int32, device=pieces[0][0].device)
        output = flash_attn_varlen_func(
            torch.cat([p[0].transpose(0, 1) for p in pieces]).contiguous(),
            torch.cat([p[1].transpose(0, 1) for p in pieces]).contiguous(),
            torch.cat([p[2].transpose(0, 1) for p in pieces]).contiguous(),
            max_seqlen_q=max(q_lengths), cu_seqlens_q=offsets(q_lengths),
            max_seqlen_k=max(k_lengths), cu_seqlens_k=offsets(k_lengths),
            softmax_scale=scale, causal=True)
        return output.reshape(output.shape[0], -1)
    if device_type != 'musa':
        raise ValueError('Official attention adapter supports npu, musa and iluvatar cuda only')
    # This installed MUSA flash op uses top-left causal alignment. Decode sees
    # all existing KV; cached multi-token chunks right-align Q by left-padding.
    groups = {}
    for row, (q, k, v, past) in enumerate(pieces):
        groups.setdefault((q.shape[1], k.shape[1]), []).append((row, q, k, v))
    outputs = [None]*len(pieces)
    for (nq, nk), rows in groups.items():
        q = torch.stack([r[1] for r in rows])
        k = torch.stack([r[2] for r in rows])
        v = torch.stack([r[3] for r in rows])
        if nq > 1 and nk > nq:
            q = torch.nn.functional.pad(q, (0, 0, nk-nq, 0))
        result = torch.ops.aten._scaled_dot_product_attention_flash_musa(
            q, k, v, dropout_p=0.0, is_causal=nq > 1, scale=scale)[0]
        result = result[:, :, -nq:, :].transpose(1, 2)
        for offset, (row, _, _, _) in enumerate(rows):
            outputs[row] = result[offset].reshape(nq, -1)
    return torch.cat(outputs, dim=0)


def packed_attention(q, k, v, attention, state, index):
    """Packed projections, separate KV owners; fuse independent decode rows."""
    import torch
    n, d = q.shape[0], attention.head_dim
    if n != sum(s['length'] for s in state.segments):
        raise ValueError('Packed attention rows disagree with scheduler layout')
    queries = q.reshape(n, attention.num_heads, d).transpose(0, 1)
    keys = k.reshape(n, attention.num_kv_heads, d).transpose(0, 1)
    values = v.reshape(n, attention.num_kv_heads, d).transpose(0, 1)
    ratio = attention.num_heads // attention.num_kv_heads
    pieces = []
    for segment in state.segments:
        start, count = segment['start'], segment['length']
        owner = segment['owner']
        previous = owner.caches.get(index)
        past = 0 if previous is None else previous[0].shape[1]
        if past != segment['past']:
            raise ValueError('KV length disagrees with request token positions')
        kk, vv = keys[:, start:start+count], values[:, start:start+count]
        if previous is not None:
            kk, vv = torch.cat((previous[0], kk), 1), torch.cat((previous[1], vv), 1)
        owner.caches[index] = (kk, vv)
        pieces.append((queries[:, start:start+count], kk, vv, past))
    if getattr(state, 'attention_backend', 'reference') == 'vendor':
        return vendor_attention(pieces, attention.scaling, q.device.type)
    # One batched attention calculation for prefill, decode and mixed phases.
    # Padding preserves independent KV owners; causal positions include past KV.
    maximum_k = max(kk.shape[1] for _,kk,_,_ in pieces)
    maximum_q = max(qq.shape[1] for qq,_,_,_ in pieces)
    qq = torch.stack([torch.nn.functional.pad(p[0], (0,0,0,maximum_q-p[0].shape[1])) for p in pieces])
    kk = torch.stack([torch.nn.functional.pad(p[1].repeat_interleave(ratio, 0), (0,0,0,maximum_k-p[1].shape[1])) for p in pieces])
    vv = torch.stack([torch.nn.functional.pad(p[2].repeat_interleave(ratio, 0), (0,0,0,maximum_k-p[2].shape[1])) for p in pieces])
    lengths = torch.tensor([p[1].shape[1] for p in pieces],device=q.device)
    past = torch.tensor([p[3] for p in pieces],device=q.device)
    query_positions = past[:,None]+torch.arange(maximum_q,device=q.device)[None,:]
    key_positions = torch.arange(maximum_k,device=q.device)[None,None,:]
    mask = (key_positions>=lengths[:,None,None]) | (key_positions>query_positions[:,:,None])
    scores = torch.matmul(qq.float(),kk.float().transpose(-1,-2))*attention.scaling
    scores.masked_fill_(mask[:,None,:,:],float('-inf'))
    output = torch.matmul(scores.softmax(-1),vv.float()).to(q.dtype).transpose(1,2)
    return torch.cat([output[row,:p[0].shape[1]].reshape(p[0].shape[1],-1) for row,p in enumerate(pieces)],dim=0)


def load_stage(config, stage, lane):
    import torch
    import torch.distributed as dist
    spec = config['stages'][stage]
    if spec['device'] == 'npu':
        import torch_npu  # noqa: F401
    elif spec['device'] == 'musa':
        import torch_musa  # noqa: F401
    device = torch.device(f"{spec['device']}:{lane}")
    getattr(torch, spec['device']).set_device(lane)
    dist.init_process_group(spec['tp_backend'],
                            init_method=f"tcp://{spec['host']}:{config['execution']['tp_port']}",
                            rank=lane, world_size=2,
                            timeout=datetime.timedelta(seconds=180))
    # The command broadcast may wait while HTTP admission is idle. Native
    # accelerator operations still retain their bounded execution timeout.
    cpu_group = dist.new_group([0, 1], backend='gloo',
                               timeout=datetime.timedelta(days=365))
    from vllm.distributed import parallel_state
    parallel_state._TP = ParallelGroup(lane, 2, device)
    parallel_state._PP = ParallelGroup(stage, 2, device, native=False)
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import set_current_vllm_config
    tp_communicator = config['execution'].get('tp_communicator', 'torch')
    if tp_communicator not in {'torch', 'vllm'}:
        raise ValueError('Unknown configured TP communicator')
    official_tp = tp_communicator == 'vllm'
    execution_mode = config['execution'].get('execution_mode', 'eager')
    if execution_mode not in {'eager', 'piecewise_mlp_graph'}:
        raise ValueError('Unknown configured execution mode')
    args = EngineArgs(model=config['execution']['model_path'], dtype='bfloat16',
                      tensor_parallel_size=2, pipeline_parallel_size=2,
                      max_model_len=config['execution']['max_model_len'],
                      enforce_eager=execution_mode == 'eager', disable_custom_all_reduce=not official_tp,
                      enable_prefix_caching=False)
    vconfig = args.create_engine_config()
    if execution_mode == 'piecewise_mlp_graph':
        # The installed engines have incompatible whole-engine graph runners.
        # Capture the official MLP+local TP path via the vendor graph APIs;
        # dynamic attention/KV and cross-node transport remain outside graphs.
        if hasattr(vconfig.compilation_config, 'mode'):
            from vllm.config import CompilationMode
            vconfig.compilation_config.mode = CompilationMode.NONE
        else:
            from vllm.config import CompilationLevel
            vconfig.compilation_config.level = CompilationLevel.NO_COMPILATION
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(vconfig), torch.device(device):
        if official_tp:
            parallel_state.set_custom_all_reduce(True)
            parallel_state._TP = parallel_state.init_model_parallel_group(
                [list(range(config['tp']))], local_rank=lane,
                backend=spec['tp_backend'], group_name='tp')
            if spec['device'] == 'npu':
                parallel_state._DP = parallel_state.init_model_parallel_group(
                    [[0], [1]], local_rank=lane, backend=spec['tp_backend'], group_name='dp')
        model = Qwen3ForCausalLM(vllm_config=vconfig)
    from safetensors import safe_open
    start, end = model.model.start_layer, model.model.end_layer

    def weights():
        for path in sorted(Path(config['execution']['model_path']).glob('*.safetensors')):
            with safe_open(path, framework='pt', device='cpu') as source:
                for name in source.keys():
                    if name.startswith('model.layers.'):
                        if not start <= int(name.split('.')[2]) < end:
                            continue
                    elif (name.startswith('model.embed_tokens.') and stage != 0
                          or name.startswith(('model.norm.', 'lm_head.')) and stage != 1):
                        continue
                    yield name, source.get_tensor(name)
    loaded = model.load_weights(weights())
    missing = set(dict(model.named_parameters())) - loaded
    if missing:
        raise RuntimeError(f'Uninitialized stage weights: {sorted(missing)}')
    # Retain the portable reference for comparison. The vendor path preserves
    # official custom-op dispatch and adapts only packed attention/KV layouts.
    from vllm.model_executor.custom_op import CustomOp
    backend = config['execution'].get('attention_backend', 'reference')
    if backend not in {'reference', 'vendor'}:
        raise ValueError('Unknown configured attention backend')
    for module in model.modules():
        if (backend == 'reference' and hasattr(module, 'forward_native')
                and type(module).forward_native is not CustomOp.forward_native):
            module.forward = module.forward_native
    state = BatchState(config['execution'].get('max_active_requests', 8))
    state.attention_backend = backend
    state.vllm_config = vconfig
    state.execution_mode = execution_mode
    state.official_tp = official_tp
    state.graph_runners = []
    if execution_mode == 'piecewise_mlp_graph':
        if __package__:
            from .graph_execution import MLPGraphRunner
        else:
            from graph_execution import MLPGraphRunner
        for index in range(start, end):
            module = model.model.layers[index].mlp
            runner = MLPGraphRunner(module.forward, device, vconfig,
                                    max_tokens=config['execution'].get('graph_max_tokens', 4))
            module.forward = runner
            state.graph_runners.append(runner)
    for index in range(start, end):
        attention = model.model.layers[index].self_attn

        def bind(attention, index):
            def forward(q, k, v, **kwargs):
                return packed_attention(q, k, v, attention, state, index)
            return forward
        attention.attn.forward = bind(attention, index)
    model.eval()
    if execution_mode == 'piecewise_mlp_graph' and config['execution'].get('transport') == 'flagcx':
        # Lazy capture synchronizes the whole device. Once FlagCX receive/send
        # streams exist, that synchronization can wait on peer progress while
        # the peer waits for this rank's next TP collective. Capture every
        # supported shape before creating any PP channel or transport thread.
        with torch.inference_mode(), set_current_vllm_config(vconfig):
            for tokens in range(1, config['execution'].get('graph_max_tokens', 4) + 1):
                inputs = torch.zeros((tokens, vconfig.model_config.hf_config.hidden_size), device=device,
                                     dtype=torch.bfloat16)
                synchronize_tp_inputs(state, cpu_group)
                for runner in state.graph_runners:
                    runner(inputs)
            getattr(torch, device.type).synchronize()
        print(json.dumps(dict(event='graphs_prewarmed_before_pp', stage=stage, lane=lane,
                              graphs=graph_stats(state))), flush=True)
    print(json.dumps(dict(event='stage_loaded', stage=stage, lane=lane,
                          start_layer=start, end_layer=end,
                          parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                          tp_implementation=type(parallel_state._TP).__module__+'.'+type(parallel_state._TP).__name__,
                          tp_device_communicator=type(getattr(parallel_state._TP, 'device_communicator', None)).__module__+'.'+type(getattr(parallel_state._TP, 'device_communicator', None)).__name__,
                          attention_backend=backend, execution_mode=execution_mode,
                          torch=torch.__version__)), flush=True)
    return model, state, device, cpu_group


def model_forward(model, state, device, *args):
    """Provide the installed Ascend plugin's official per-forward context."""
    if device.type == 'npu' and state.attention_backend == 'vendor':
        from vllm_ascend.ascend_forward_context import set_ascend_forward_context
        import inspect
        if not hasattr(state, 'ascend_context_has_prefill'):
            state.ascend_context_has_prefill = 'with_prefill' in inspect.signature(set_ascend_forward_context).parameters
        options = {}
        if state.ascend_context_has_prefill:
            options['with_prefill'] = any(s['past'] == 0 or s['length'] > 1 for s in state.segments)
        with set_ascend_forward_context(None, state.vllm_config,
                num_tokens=sum(s['length'] for s in state.segments),
                model_instance=model, **options):
            return model(*args)
    return model(*args)


def graph_stats(state):
    return dict(mode=state.execution_mode, modules=len(state.graph_runners),
                captures=sum(r.captures for r in state.graph_runners),
                replays=sum(r.replays for r in state.graph_runners),
                uncaptured_calls=sum(r.uncaptured_calls for r in state.graph_runners))


def synchronize_tp_inputs(state, cpu_group):
    if state.official_tp:
        from vllm.distributed import get_tp_group
        get_tp_group().barrier()  # Official CPU-group barrier; never a GPU barrier.
    else:
        import torch.distributed as dist
        dist.monitored_barrier(group=cpu_group, timeout=datetime.timedelta(seconds=180))


def open_channel(config, stage, lane):
    address = (config['master'], config['execution']['pp_port'] + lane)
    if stage == 0:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
        listener.listen(1)
        listener.settimeout(300)
        connection, _ = listener.accept()
        listener.close()
    else:
        deadline = time.monotonic() + 300
        while True:
            try:
                connection = socket.create_connection(address, timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    version = 3 if config['execution'].get('scheduler') == 'continuous' else 1
    channel = TensorChannel(connection, version)
    if config['execution'].get('transport') == 'flagcx':
        if __package__:
            from .flagcx_transport import FlagCXChannel
        else:
            from flagcx_transport import FlagCXChannel
        import torch
        spec = config['stages'][stage]
        return FlagCXChannel(channel, config['flagcx']['library'],
                             torch.device(f"{spec['device']}:{lane}"), stage)
    return channel


def admission_metrics(lifecycle, queued):
    values = {'queued_requests': queued}
    for key, value in lifecycle.items():
        if isinstance(value, (int, float)):
            values[key] = value
        elif isinstance(value, dict):
            values.update({f'{key}_{name}': number for name, number in value.items()
                           if isinstance(number, (int, float))})
    return ''.join(f'heteroserve_pp_{key} {int(value) if isinstance(value, bool) else value}\n'
                   for key, value in values.items())


def create_admission(config, lane, tokenizer):
    """Shared HTTP admission for both scheduling policies."""
    admission = queue.Queue(maxsize=16)
    server = None
    lifecycle = {'draining': False, 'completed': 0, 'active_requests': 0,
                 'inflight_batches': 0, 'kv_owners': 0, 'max_batch_size': 0,
                 'max_inflight_batches': 0, 'admitted_requests': 0}
    if lane == 0 and config['execution'].get('mode') == 'http':
        import asyncio
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse, Response
        app = FastAPI()

        @app.get('/health')
        async def health():
            return dict(ready=not lifecycle['draining'], tp=2, pp=2, queue=admission.qsize(),
                        execution=config['execution'].get('scheduler', 'serial'), **lifecycle)

        @app.get('/v1/models')
        async def models():
            return dict(object='list', data=[dict(id='Qwen3-14B', object='model')])

        @app.get('/metrics')
        async def metrics():
            return Response(admission_metrics(lifecycle, admission.qsize()),
                            media_type='text/plain')

        @app.post('/v1/completions')
        async def completions(body: dict):
            if lifecycle['draining']:
                raise HTTPException(503, 'PP service is draining')
            if body.get('model', 'Qwen3-14B') != 'Qwen3-14B':
                raise HTTPException(400, 'Unknown model')
            if body.get('temperature', 0) != 0 or body.get('n', 1) != 1:
                raise HTTPException(400, 'Only one greedy completion is supported')
            if body.get('logprobs') is not None or body.get('stop') is not None:
                raise HTTPException(400, 'logprobs and custom stop strings are unsupported')
            prompt = body.get('prompt')
            if not isinstance(prompt, str):
                raise HTTPException(400, 'A single text prompt is required')
            maximum = body.get('max_tokens', 32)
            output_limit = config['execution'].get('max_output_tokens', 128)
            if type(maximum) is not int or not 1 <= maximum <= output_limit:
                raise HTTPException(400, f'max_tokens must be between 1 and {output_limit}')
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids or len(ids) + maximum > config['execution']['max_model_len']:
                raise HTTPException(400, 'Empty or excessive context')
            future = concurrent.futures.Future()
            future.token_events = queue.Queue()
            item = dict(id=str(uuid.uuid4()), max_tokens=maximum, prompt_token_ids=ids,
                        prompt=prompt, ignore_eos=bool(body.get('ignore_eos', False)))
            try:
                admission.put_nowait((item, future))
            except queue.Full:
                raise HTTPException(429, 'PP admission queue is full')
            if not body.get('stream', False):
                result = await asyncio.wrap_future(future)
                lifecycle['completed'] += 1
                return dict(id='cmpl-'+item['id'], object='text_completion', model='Qwen3-14B',
                    choices=[dict(index=0, text=result['text'],
                                  finish_reason='stop' if result['stopped_on_eos'] else 'length')],
                    usage=dict(prompt_tokens=len(ids), completion_tokens=len(result['generated_token_ids']),
                               total_tokens=len(ids)+len(result['generated_token_ids'])))

            async def events():
                try:
                    while True:
                        event = await asyncio.to_thread(future.token_events.get)
                        if event is None:
                            break
                        yield 'data: '+json.dumps(dict(id='cmpl-'+item['id'], object='text_completion',
                            created=int(time.time()), model='Qwen3-14B', choices=[event]), ensure_ascii=False)+'\n\n'
                    result = await asyncio.wrap_future(future)
                    lifecycle['completed'] += 1
                    if body.get('stream_options', {}).get('include_usage', False):
                        yield 'data: '+json.dumps(dict(id='cmpl-'+item['id'], choices=[],
                            usage=dict(prompt_tokens=len(ids), completion_tokens=len(result['generated_token_ids']),
                                       total_tokens=len(ids)+len(result['generated_token_ids']))))+'\n\n'
                    yield 'data: [DONE]\n\n'
                finally:
                    if not future.done():
                        future.cancel()
            return StreamingResponse(events(), media_type='text/event-stream')

        @app.post('/v1/chat/completions')
        async def chat(body: dict):
            if lifecycle['draining']:
                raise HTTPException(503, 'PP service is draining')
            if body.get('stream', False) or body.get('temperature', 0) != 0:
                raise HTTPException(400, 'Experimental PP accepts non-streaming greedy requests')
            if body.get('model', 'Qwen3-14B') != 'Qwen3-14B':
                raise HTTPException(400, 'Unknown model')
            messages = body.get('messages')
            if (not isinstance(messages, list) or not messages
                    or any(not isinstance(m, dict) or m.get('role') not in {'user', 'system', 'assistant'}
                           or not isinstance(m.get('content'), str) for m in messages)):
                raise HTTPException(400, 'Text messages required')
            maximum = body.get('max_tokens', 32)
            output_limit = config['execution'].get('max_output_tokens', 128)
            if type(maximum) is not int or not 1 <= maximum <= output_limit:
                raise HTTPException(400, f'max_tokens must be between 1 and {output_limit}')
            ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                      add_generation_prompt=True, enable_thinking=False)
            if len(ids) + maximum > config['execution']['max_model_len']:
                raise HTTPException(400, 'Request exceeds configured context bound')
            future = concurrent.futures.Future()
            item = dict(id=str(uuid.uuid4()), messages=messages, max_tokens=maximum,
                        prompt_token_ids=ids, prompt=json.dumps(messages, ensure_ascii=False))
            try:
                admission.put_nowait((item, future))
            except queue.Full:
                raise HTTPException(429, 'PP admission queue is full')
            result = await asyncio.wrap_future(future)
            lifecycle['completed'] += 1
            return dict(id='chatcmpl-'+item['id'], object='chat.completion', created=int(time.time()),
                        model='Qwen3-14B', choices=[dict(index=0, message=dict(role='assistant', content=result['text']),
                        finish_reason='stop' if result['stopped_on_eos'] else 'length')],
                        usage=dict(prompt_tokens=len(ids), completion_tokens=len(result['generated_token_ids']),
                                   total_tokens=len(ids)+len(result['generated_token_ids'])))
        server = uvicorn.Server(uvicorn.Config(app, host=config['master'],
                                  port=config['execution']['api_port'], log_level='warning'))
        threading.Thread(target=server.run, daemon=True).start()
        def drain(signum, frame):
            lifecycle['draining'] = True
        signal.signal(signal.SIGTERM, drain)
        print(json.dumps(dict(event='http_ready', host=config['master'], port=config['execution']['api_port'])), flush=True)
    return admission, lifecycle, server


def schedule_requests(config, lane, cpu_group, tokenizer):
    """Serialized control: rank zero admits one complete request at a time."""
    import torch.distributed as dist
    admission, lifecycle, server = create_admission(config, lane, tokenizer)
    index = 0
    try:
        while True:
            future = None
            scheduled = [None]
            if lane == 0:
                if server is not None:
                    if lifecycle['draining'] and admission.empty():
                        item = None
                    else:
                        try:
                            item, future = admission.get(timeout=1)
                        except queue.Empty:
                            continue
                    if future is not None and future.cancelled():
                        continue
                else:
                    items = config['execution']['requests']
                    item = items[index] if index < len(items) else None
                    index += 1
                scheduled[0] = item
            dist.broadcast_object_list(scheduled, src=0, group=cpu_group)
            if scheduled[0] is None:
                break
            yield scheduled[0], future
    finally:
        if server is not None:
            server.should_exit = True


def finish_future(future, row):
    if future is None:
        return
    try:
        future.set_result(row)
    except concurrent.futures.InvalidStateError:
        if not future.cancelled():
            raise
    finally:
        # A canceled stream may leave a to_thread(queue.get) waiter behind.
        if hasattr(future, 'token_events'):
            future.token_events.put(None)


def execute_serial(config, stage, lane):
    import torch
    model, state, device, cpu_group = load_stage(config, stage, lane)
    channel = open_channel(config, stage, lane)
    from vllm.sequence import IntermediateTensors
    rows = []
    with torch.inference_mode():
        if stage == 1:
            while True:
                command, tensors = channel.receive(device, allow_idle=True)
                if command['op'] == 'stop':
                    break
                if command['op'] == 'release':
                    state.release(command['request_id'])
                    channel.send(dict(op='released', request_id=command['request_id']))
                    continue
                if command['op'] != 'forward' or set(tensors) != {'hidden_states', 'residual'}:
                    raise ValueError('Invalid PP execution message')
                expected = (len(command['positions']), model.config.hidden_size)
                if any(tuple(t.shape) != expected for t in tensors.values()):
                    raise ValueError('PP activations disagree with token positions')
                state.advance(command['request_id'], command['positions'])
                positions = torch.tensor(command['positions'], dtype=torch.int64, device=device)
                hidden = model(None, positions, IntermediateTensors(tensors))
                logits = torch.nn.functional.linear(hidden[-1:], model.lm_head.weight).float()
                from vllm.distributed import get_tp_group
                logits = get_tp_group().all_gather(logits)
                logits = logits[:, :model.config.vocab_size]
                token = int(logits.argmax(-1).item())
                channel.send(dict(op='token', request_id=command['request_id'],
                                  step=command['step'], token=token))
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config['execution']['model_path'], local_files_only=True)
            for item, future in schedule_requests(config, lane, cpu_group, tokenizer):
                request_id = item['id']
                ids = item.get('prompt_token_ids')
                if ids is None:
                    ids = tokenizer.apply_chat_template([dict(role='user', content=item['prompt'])], return_dict=False,
                              tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if len(ids) + item['max_tokens'] > config['execution']['max_model_len']:
                    raise ValueError('Request exceeds configured context bound')
                output, durations = [], []
                emitted_text = ''
                for step in range(item['max_tokens']):
                    positions = list(range(len(ids))) if step == 0 else [len(ids) + step - 1]
                    state.advance(request_id, positions)
                    tokens = ids if step == 0 else [output[-1]]
                    start = time.perf_counter()
                    intermediate = model(torch.tensor(tokens, dtype=torch.int64, device=device),
                                         torch.tensor(positions, dtype=torch.int64, device=device))
                    channel.send(dict(op='forward', request_id=request_id, step=step, positions=positions),
                                 dict(hidden_states=intermediate['hidden_states'], residual=intermediate['residual']))
                    response, _ = channel.receive(device)
                    if response.get('request_id') != request_id or response.get('step') != step:
                        raise RuntimeError('Stale PP token feedback')
                    output.append(response['token'])
                    durations.append(time.perf_counter() - start)
                    stop = (not item.get('ignore_eos', False)
                            and output[-1] in {tokenizer.eos_token_id, 151645})
                    if future is not None and hasattr(future, 'token_events'):
                        decoded = tokenizer.decode(output, skip_special_tokens=True)
                        if decoded.endswith('\ufffd') and not stop and step + 1 < item['max_tokens']:
                            delta = ''
                        elif decoded.startswith(emitted_text):
                            delta = decoded[len(emitted_text):]
                            emitted_text = decoded
                        else:
                            raise RuntimeError('Incremental decoder changed an emitted text prefix')
                        future.token_events.put(dict(index=0, text=delta,
                            finish_reason='stop' if stop else ('length' if step+1==item['max_tokens'] else None)))
                    if stop:
                        break
                channel.send(dict(op='release', request_id=request_id))
                response, _ = channel.receive(device)
                if response != dict(op='released', request_id=request_id):
                    raise RuntimeError('PP cache release not acknowledged')
                state.release(request_id)
                row = dict(id=request_id, prompt=item['prompt'], prompt_token_ids=ids,
                           generated_token_ids=output, text=tokenizer.decode(output, skip_special_tokens=True),
                           step_seconds=durations, caches_after_release=len(state.caches),
                           stopped_on_eos=not item.get('ignore_eos', False) and output[-1] in {tokenizer.eos_token_id, 151645})
                if config['execution'].get('mode') != 'http':
                    rows.append(row)
                else:
                    directory = Path(config['execution']['result_dir'])
                    directory.mkdir(parents=True, exist_ok=True)
                    with (directory/f'stage0-lane{lane}.jsonl').open('a') as sink:
                        sink.write(json.dumps(row, ensure_ascii=False)+'\n')
                finish_future(future, row)
                print(json.dumps(dict(event='request_complete', **row), ensure_ascii=False), flush=True)
            channel.send(dict(op='stop'))
    channel.connection.close()
    out = Path(config['execution']['result_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out/f'stage{stage}-lane{lane}.json').write_text(json.dumps(dict(stage=stage, lane=lane,
        scope='vLLM model layers, native TP, reference attention, CPU-staged PP, serialized requests',
        requests=rows, caches_after_release=len(state.caches)), ensure_ascii=False, indent=2))


class ExecutionTrace:
    def __init__(self, config, stage, lane):
        directory = Path(config['execution']['result_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory/f'timeline-stage{stage}-lane{lane}.jsonl'
        self.lock = threading.Lock()

    def write(self, **event):
        with self.lock, self.path.open('a') as sink:
            sink.write(json.dumps(event)+'\n')


def calibrate_channel(channel, stage, trace):
    """Bound clock offset using minimum RTT; timelines are execution windows."""
    if stage == 0:
        estimates = []
        for sequence in range(8):
            start = time.perf_counter()
            channel.send(dict(op='clock_ping', sequence=sequence))
            response, _ = channel.receive('cpu')
            end = time.perf_counter()
            if response.get('op') != 'clock_pong' or response.get('sequence') != sequence:
                raise ValueError('Invalid PP clock handshake')
            estimates.append(dict(offset=response['remote_time']-(start+end)/2,
                                  uncertainty=(end-start)/2))
        calibration = min(estimates, key=lambda x:x['uncertainty'])
        channel.send(dict(op='clock_calibrated', **calibration))
    else:
        for sequence in range(8):
            command, _ = channel.receive('cpu')
            if command != dict(op='clock_ping', sequence=sequence):
                raise ValueError('Invalid PP clock handshake')
            channel.send(dict(op='clock_pong', sequence=sequence, remote_time=time.perf_counter()))
        calibration, _ = channel.receive('cpu')
        if calibration.pop('op') != 'clock_calibrated':
            raise ValueError('Invalid PP clock calibration')
    trace.write(event='clock_calibration', **calibration)
    return calibration


class FeedbackReceiver:
    """CPU feedback and SSE emission can run while the main thread executes TP."""
    def __init__(self, channel, tokenizer, eos_ids, trace):
        self.channel, self.tokenizer, self.eos_ids, self.trace = channel, tokenizer, eos_ids, trace
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.pending, self.requests = {}, {}
        self.closing = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def register(self, item, future):
        with self.lock:
            self.requests[item['id']] = dict(item=item, future=future, output=[], text='')

    def expect(self, packet):
        with self.lock:
            if packet['batch_id'] in self.pending:
                raise ValueError('Duplicate dispatched microbatch')
            self.pending[packet['batch_id']] = packet['segments']

    def emit(self, response):
        with self.lock:
            segments = self.pending.pop(response['batch_id'])
            samples = response['samples']
            if set(samples) != {s['request_id'] for s in segments}:
                raise ValueError('Feedback does not match dispatched requests')
            for segment in segments:
                identifier, token = segment['request_id'], samples[segment['request_id']]
                entry = self.requests[identifier]
                if (segment['step'] != len(entry['output']) or
                    (segment['sample_token'] and type(token) is not int) or
                    (not segment['sample_token'] and token is not None)):
                    raise ValueError('Stale or invalid token feedback')
                if token is None:
                    continue
                entry['output'].append(token)
                future, item = entry['future'], entry['item']
                if future is None or future.cancelled() or not hasattr(future, 'token_events'):
                    continue
                stop = not item.get('ignore_eos', False) and token in self.eos_ids
                finished = stop or len(entry['output']) == item['max_tokens']
                decoded = self.tokenizer.decode(entry['output'], skip_special_tokens=True)
                if decoded.endswith('\ufffd') and not finished:
                    delta = ''
                elif decoded.startswith(entry['text']):
                    delta, entry['text'] = decoded[len(entry['text']):], decoded
                else:
                    raise RuntimeError('Incremental decoder changed an emitted text prefix')
                future.token_events.put(dict(index=0, text=delta,
                    finish_reason=('stop' if stop else 'length') if finished else None))

    def run(self):
        try:
            while not self.closing:
                response, tensors = self.channel.receive('cpu', allow_idle=True)
                if tensors or response.get('op') not in {'completed_batch', 'released'}:
                    raise ValueError('Invalid PP feedback')
                if response['op'] == 'completed_batch':
                    with self.lock:
                        segments=self.pending[response['batch_id']]
                        if (response.get('step_id')!=response['batch_id'] or
                                response.get('request_order')!=[s['request_id'] for s in segments] or
                                set(response.get('results',{}))!={s['request_id'] for s in segments}):
                            raise ValueError('Stale or reordered PP completion')
                        for segment in segments:
                            identifier=segment['request_id']
                            result=response['results'][identifier]
                            item=self.requests[identifier]['item']
                            token=response['samples'][identifier]
                            finished=token is not None and (segment['step']+1>=item['max_tokens'] or
                                (not item.get('ignore_eos',False) and token in self.eos_ids))
                            if (result['step_id']!=segment['step_id'] or result['next_token']!=token or
                                    result['finished']!=finished or result['sampling']!={'temperature':0}):
                                raise ValueError('Invalid PP sampling/termination feedback')
                    self.emit(response)
                    self.trace.write(event='peer_execution', **response)
                self.queue.put(response)
        except Exception as error:
            if not self.closing:
                self.queue.put(error)

    def forget(self, identifier):
        with self.lock:
            return self.requests.pop(identifier)

    def close(self):
        self.closing = True
        try:
            self.channel.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.thread.join(timeout=5)


def execute_continuous(config, stage, lane):
    import torch
    import torch.distributed as dist
    from vllm.sequence import IntermediateTensors
    model, state, device, cpu_group = load_stage(config, stage, lane)
    channel = open_channel(config, stage, lane)
    trace = ExecutionTrace(config, stage, lane)
    calibrate_channel(channel, stage, trace)
    accelerator = getattr(torch, config['stages'][stage]['device'])
    directory = Path(config['execution']['result_dir'])
    receiver, server, sender, incoming = None, None, None, None
    try:
        with torch.inference_mode():
            if stage == 1:
                incoming=AsyncTensorReceiver(channel,device,capacity=config['execution'].get('max_inflight_batches',2))
                expected_batch = 0
                while True:
                    packet, tensors = incoming.receive()
                    if packet['op'] == 'stop':
                        if state.requests:
                            raise ValueError('Cannot stop with live KV owners')
                        break
                    if packet['op'] == 'release':
                        state.release(packet['request_id'])
                        trace.write(event='release', request_id=packet['request_id'], kv_owners=len(state.requests))
                        channel.send(dict(op='released', request_id=packet['request_id']))
                        continue
                    if (packet['op'] != 'forward_batch' or packet['batch_id'] != expected_batch
                            or packet.get('step_id') != expected_batch):
                        raise ValueError('Invalid or out-of-order PP microbatch')
                    expected_batch += 1
                    segments = packet['segments']
                    if packet.get('request_order') != [s['request_id'] for s in segments]:
                        raise ValueError('PP request order mismatch')
                    for segment in segments:
                        if (segment['length']!=len(segment['positions']) or segment['past']!=segment['positions'][0]
                                or segment['step_id']!=segment['step'] or segment['phase'] not in {'prefill','decode'}
                                or segment['sampling']!={'temperature':0}):
                            raise ValueError('Invalid PP execution metadata')
                    positions = [p for s in segments for p in s['positions']]
                    expected = (len(positions), model.config.hidden_size)
                    if set(tensors) != {'hidden_states', 'residual'} or any(tuple(t.shape) != expected for t in tensors.values()):
                        raise ValueError('PP activations disagree with packed positions')
                    state.install(segments)
                    # Prepare GPU metadata before either local rank can enqueue
                    # collectives. Synchronous list-to-device conversions after
                    # a peer launches a waiting collective can deadlock MUSA.
                    position_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
                    offset, sample_rows, owners = 0, [], []
                    samples = {s['request_id']:None for s in segments}
                    for segment in segments:
                        offset += len(segment['positions'])
                        if segment['sample_token']:
                            sample_rows.append(offset-1)
                            owners.append(segment['request_id'])
                    sample_indices = torch.tensor(sample_rows, dtype=torch.int64, device=device)
                    synchronize_tp_inputs(state, cpu_group)
                    started = time.perf_counter()
                    begin,end=accelerator.Event(enable_timing=True),accelerator.Event(enable_timing=True)
                    begin.record()
                    hidden = model_forward(model, state, device, None, position_tensor, IntermediateTensors(tensors))
                    if sample_rows:
                        logits = torch.nn.functional.linear(hidden.index_select(0, sample_indices), model.lm_head.weight).float()
                        from vllm.distributed import get_tp_group
                        logits = get_tp_group().all_gather(logits)[:, :model.config.vocab_size]
                        samples.update(zip(owners, logits.argmax(-1).tolist()))
                    end.record()
                    end.synchronize()
                    ended = time.perf_counter()
                    trace.write(event='execution', batch_id=packet['batch_id'], start=started, end=ended,
                                device_ms=begin.elapsed_time(end),batch_size=len(segments), tokens=len(positions),
                                graphs=graph_stats(state))
                    outcomes={s['request_id']:dict(next_token=samples[s['request_id']],step_id=s['step_id'],
                        finished=samples[s['request_id']] is not None and (s['step_id']+1>=s['max_tokens']
                            or (not s['ignore_eos'] and samples[s['request_id']] in {151643,151645})),
                        sampling=s['sampling']) for s in segments}
                    channel.send(dict(op='completed_batch', batch_id=packet['batch_id'],step_id=packet['step_id'],
                                      request_order=packet['request_order'],samples=samples,results=outcomes,
                                      start=started, end=ended, graphs=graph_stats(state)))
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config['execution']['model_path'], local_files_only=True)
                eos_ids = {tokenizer.eos_token_id, 151645}
                settings = config['execution']
                scheduler = ContinuousScheduler(max_active=settings.get('max_active_requests', 8),
                    microbatch_size=settings.get('microbatch_size', 4), max_inflight=settings.get('max_inflight_batches', 2),
                    max_batch_tokens=settings.get('max_batch_tokens', 1024), prefill_chunk_tokens=settings.get('prefill_chunk_tokens', 512))
                admission, lifecycle, server = create_admission(config, lane, tokenizer)
                if lane == 0 and settings.get('mode') != 'http':
                    for original in settings['requests']:
                        item = dict(original)
                        item.setdefault('prompt_token_ids', tokenizer.apply_chat_template([dict(role='user', content=item['prompt'])], return_dict=False,
                            tokenize=True, add_generation_prompt=True, enable_thinking=False))
                        admission.put_nowait((item, None))
                    lifecycle['draining'] = True
                receiver = FeedbackReceiver(channel, tokenizer, eos_ids, trace)
                receiver.thread.start()
                sender = AsyncTensorSender(channel,device,trace,capacity=settings.get('max_inflight_batches',2))
                starts, rows = {}, []
                while True:
                    sender.check()
                    command, future = None, None
                    if lane == 0:
                        while command is None:
                            try:
                                feedback = receiver.queue.get_nowait()
                            except queue.Empty:
                                feedback = None
                            if feedback is not None:
                                command = dict(op='feedback', response=feedback) if not isinstance(feedback, Exception) else dict(op='failure', error=str(feedback))
                            elif len(scheduler.active) < scheduler.max_active and not admission.empty():
                                item, future = admission.get_nowait()
                                if future is not None and future.cancelled():
                                    continue
                                command = dict(op='admit', item=item)
                            else:
                                command = scheduler.plan()
                                if command is None and lifecycle['draining'] and admission.empty() and not scheduler.active:
                                    command = dict(op='stop')
                                elif command is None:
                                    try:
                                        feedback = receiver.queue.get(timeout=0.05)
                                        command = dict(op='feedback', response=feedback) if not isinstance(feedback, Exception) else dict(op='failure', error=str(feedback))
                                    except queue.Empty:
                                        pass
                    if lane == 0 and command['op'] == 'feedback':
                        command['canceled'] = [identifier for identifier,request in scheduler.active.items()
                                               if request.future is not None and request.future.cancelled()]
                    broadcast = [command]
                    dist.broadcast_object_list(broadcast, src=0, group=cpu_group)
                    command = broadcast[0]
                    operation = command['op']
                    if operation == 'failure':
                        raise RuntimeError(command['error'])
                    if operation == 'admit':
                        item = command['item']
                        if len(item['prompt_token_ids'])+item['max_tokens'] > settings['max_model_len']:
                            raise ValueError('Request exceeds configured context bound')
                        if not scheduler.admit(item, future):
                            raise RuntimeError('Scheduler admission overflow')
                        receiver.register(item, future)
                    elif operation == 'forward_batch':
                        if lane != 0 and scheduler.plan() != command:
                            raise RuntimeError('TP lanes disagree on scheduling')
                        state.install(command['segments'])
                        receiver.expect(command)
                        positions = [p for s in command['segments'] for p in s['positions']]
                        tokens = [t for s in command['segments'] for t in s['input_token_ids']]
                        token_tensor = torch.tensor(tokens, dtype=torch.int64, device=device)
                        position_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
                        synchronize_tp_inputs(state, cpu_group)
                        started = time.perf_counter()
                        starts[command['batch_id']] = started
                        begin,end = accelerator.Event(enable_timing=True),accelerator.Event(enable_timing=True)
                        begin.record()
                        intermediate = model_forward(model, state, device, token_tensor, position_tensor)
                        if state.graph_runners:
                            trace.write(event='graph_execution', batch_id=command['batch_id'], graphs=graph_stats(state))
                        end.record()
                        sender.send(command, dict(hidden_states=intermediate['hidden_states'], residual=intermediate['residual']),
                                    events=(begin,end),started=started)
                    elif operation == 'feedback':
                        response = command['response']
                        if lane != 0:
                            own = receiver.queue.get(timeout=180)
                            if isinstance(own, Exception):
                                raise own
                            # Compare ownership and tokens; clocks and duration differ by lane.
                            if {k:v for k,v in own.items() if k not in {'start','end'}} != {k:v for k,v in response.items() if k not in {'start','end'}}:
                                raise RuntimeError('TP lanes disagree on feedback')
                            response = own
                        if response['op'] == 'completed_batch':
                            lifecycle['peer_graphs'] = response.get('graphs')
                            elapsed = time.perf_counter()-starts.pop(response['batch_id'])
                            finished = scheduler.complete(response['batch_id'], response['samples'], elapsed, eos_ids,
                                                          canceled_ids=command['canceled'])
                            for identifier in finished:
                                sender.send(dict(op='release', request_id=identifier))
                        else:
                            identifier = response['request_id']
                            state.release(identifier)
                            request = scheduler.released(identifier)
                            emitted = receiver.forget(identifier)
                            if emitted['output'] != request.output:
                                raise RuntimeError('Emitted tokens disagree with scheduled tokens')
                            row = dict(id=identifier, prompt=request.item['prompt'], prompt_token_ids=request.item['prompt_token_ids'],
                                generated_token_ids=request.output, text=tokenizer.decode(request.output, skip_special_tokens=True),
                                step_seconds=request.step_seconds, stopped_on_eos=request.stopped_on_eos,
                                released_request_kv_entries=0, remaining_kv_owners=len(state.requests))
                            with (directory/f'stage0-lane{lane}.jsonl').open('a') as sink:
                                sink.write(json.dumps(row, ensure_ascii=False)+'\n')
                            if settings.get('mode') != 'http':
                                rows.append(row)
                            finish_future(request.future, row)
                            trace.write(event='release', request_id=identifier, kv_owners=len(state.requests))
                            print(json.dumps(dict(event='request_complete', id=identifier, tokens=len(request.output),
                                                  remaining_kv_owners=len(state.requests))), flush=True)
                    elif operation == 'stop':
                        sender.send(command)
                        break
                    else:
                        raise ValueError('Unknown scheduler operation')
                    lifecycle.update(active_requests=len(scheduler.active), inflight_batches=len(scheduler.inflight),
                        kv_owners=len(state.requests), graphs=graph_stats(state), **scheduler.stats)
                (directory/f'stage0-lane{lane}.json').write_text(json.dumps(dict(stage=0, lane=lane, scheduler='continuous',
                    stats=scheduler.stats, requests=rows, caches_after_release=len(state.requests)), ensure_ascii=False, indent=2))
    finally:
        transport_error=None
        if incoming is not None:
            incoming.close()
        if sender is not None:
            try:
                sender.close()
            except Exception as error:
                transport_error=error
        if stage == 0 and receiver is not None:
            lifecycle['draining'] = True
            # Wake HTTP waiters on peer failure; never leave token-event readers hanging.
            for entry in list(receiver.requests.values()):
                pending = entry['future']
                if pending is not None and not pending.done():
                    pending.set_exception(RuntimeError('PP executor stopped before cache release'))
                if pending is not None and hasattr(pending, 'token_events'):
                    pending.token_events.put(None)
            while not admission.empty():
                _, pending = admission.get_nowait()
                if pending is not None and not pending.done():
                    pending.set_exception(RuntimeError('PP executor stopped before admission'))
                    if hasattr(pending, 'token_events'):
                        pending.token_events.put(None)
            if server is not None:
                server.should_exit = True
            receiver.close()
        channel.connection.close()
        if transport_error is not None:
            raise transport_error


def execute(config, stage, lane):
    scheduler = config['execution'].get('scheduler', 'serial')
    if scheduler == 'continuous':
        execute_continuous(config, stage, lane)
    elif scheduler == 'serial':
        execute_serial(config, stage, lane)
    else:
        raise ValueError('Unknown scheduler policy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', type=int, choices=[0, 1], required=True)
    parser.add_argument('--lane', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if (config['tp'], config['pp']) != (2, 2):
        raise ValueError('This acceptance executor requires TP2 PP2')
    execute(config, args.stage, args.lane)


if __name__ == '__main__':
    main()
