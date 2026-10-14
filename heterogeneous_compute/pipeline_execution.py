"""Experimental Qwen3 heterogeneous PP executor using vLLM TP model layers.

Reference eager attention and a serialized scheduler deliberately precede any
performance claims. No vendor engine files are modified. Tensor transport is
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
from pathlib import Path


class TensorChannel:
    VERSION = 1
    MAX_HEADER = 65536
    MAX_PAYLOAD = 64 * 1024 * 1024

    def __init__(self, connection):
        self.connection = connection
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
            descriptors.append(dict(name=name, shape=list(cpu.shape), bytes=len(payload)))
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
            self.connection.settimeout(None)
        try:
            length, = struct.unpack('!I', self.read(4))
        finally:
            self.connection.settimeout(180)
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
            if (len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape)
                    or size != shape[0] * shape[1] * 2 or total > self.MAX_PAYLOAD
                    or desc['name'] in tensors):
                raise ValueError('Invalid PP tensor descriptor')
            import torch
            tensors[desc['name']] = torch.frombuffer(self.read(size), dtype=torch.bfloat16).clone().reshape(shape).to(device)
        return header['metadata'], tensors


class RequestState:
    """One active request per rank; sequence guards reject stale KV reuse."""
    def __init__(self):
        self.request_id = None
        self.position = 0
        self.caches = {}

    def advance(self, request_id, positions):
        if not positions:
            raise ValueError('Empty token positions')
        if self.request_id is None:
            if not positions or positions[0] != 0:
                raise ValueError('New request must start at position zero')
            self.request_id = request_id
        if request_id != self.request_id:
            raise ValueError('Release active request before switching KV ownership')
        if positions != list(range(self.position, self.position + len(positions))):
            raise ValueError('Non-contiguous or stale PP token positions')
        self.position += len(positions)

    def release(self, request_id):
        if self.request_id != request_id:
            raise ValueError('Release does not own active KV cache')
        self.caches.clear()
        self.request_id = None
        self.position = 0


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
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor, dim=-1):
        import torch
        import torch.distributed as dist
        pieces = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(pieces, tensor)
        return torch.cat(pieces, dim=dim)


def load_stage(config, stage, lane):
    import torch
    import torch.distributed as dist
    spec = config['stages'][stage]
    if spec['device'] == 'npu':
        import torch_npu  # noqa: F401
    else:
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
    args = EngineArgs(model=config['execution']['model_path'], dtype='bfloat16',
                      tensor_parallel_size=2, pipeline_parallel_size=2,
                      max_model_len=config['execution']['max_model_len'],
                      enforce_eager=True, disable_custom_all_reduce=True,
                      enable_prefix_caching=False)
    vconfig = args.create_engine_config()
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(vconfig), torch.device(device):
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
    # Portable reference kernels avoid dependence on incompatible vendor
    # attention metadata and custom-op dispatch. TP projections stay vLLM.
    from vllm.model_executor.custom_op import CustomOp
    for module in model.modules():
        if (hasattr(module, 'forward_native')
                and type(module).forward_native is not CustomOp.forward_native):
            module.forward = module.forward_native
    state = RequestState()
    for index in range(start, end):
        attention = model.model.layers[index].self_attn

        def bind(attention, index):
            def forward(q, k, v, **kwargs):
                n, d = q.shape[0], attention.head_dim
                queries = q.reshape(n, attention.num_heads, d).transpose(0, 1)
                keys = k.reshape(n, attention.num_kv_heads, d).transpose(0, 1)
                values = v.reshape(n, attention.num_kv_heads, d).transpose(0, 1)
                previous = state.caches.get(index)
                past = 0 if previous is None else previous[0].shape[1]
                if previous is not None:
                    keys = torch.cat((previous[0], keys), dim=1)
                    values = torch.cat((previous[1], values), dim=1)
                state.caches[index] = (keys, values)
                ratio = attention.num_heads // attention.num_kv_heads
                keys = keys.repeat_interleave(ratio, dim=0)
                values = values.repeat_interleave(ratio, dim=0)
                scores = torch.matmul(queries.float(), keys.float().transpose(1, 2)) * attention.scaling
                mask = torch.arange(keys.shape[1], device=device)[None, :] > (past + torch.arange(n, device=device)[:, None])
                scores = scores.masked_fill(mask[None, :, :], float('-inf'))
                output = torch.matmul(scores.softmax(-1), values.float()).to(q.dtype)
                return output.transpose(0, 1).contiguous().reshape(n, -1)
            return forward
        attention.attn.forward = bind(attention, index)
    model.eval()
    print(json.dumps(dict(event='stage_loaded', stage=stage, lane=lane,
                          start_layer=start, end_layer=end,
                          parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                          torch=torch.__version__)), flush=True)
    return model, state, device, cpu_group


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
    return TensorChannel(connection)


def schedule_requests(config, lane, cpu_group, tokenizer):
    """Rank zero owns FIFO admission; every TP lane executes one command."""
    import torch.distributed as dist
    admission = queue.Queue(maxsize=16)
    server = None
    lifecycle = {'draining': False, 'completed': 0}
    if lane == 0 and config['execution'].get('mode') == 'http':
        import asyncio
        import uvicorn
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse, Response
        app = FastAPI()

        @app.get('/health')
        async def health():
            return dict(ready=not lifecycle['draining'], tp=2, pp=2, queue=admission.qsize(),
                        execution='serialized_reference_attention')

        @app.get('/v1/models')
        async def models():
            return dict(object='list', data=[dict(id='Qwen3-14B', object='model')])

        @app.get('/metrics')
        async def metrics():
            return Response(f"heteroserve_pp_queued_requests {admission.qsize()}\n"
                            f"heteroserve_pp_completed_requests_total {lifecycle['completed']}\n",
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
            if type(maximum) is not int or not 1 <= maximum <= 128:
                raise HTTPException(400, 'max_tokens must be between 1 and 128')
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
            if type(maximum) is not int or not 1 <= maximum <= 128:
                raise HTTPException(400, 'max_tokens must be between 1 and 128')
            ids = tokenizer.apply_chat_template(messages, tokenize=True,
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


def execute(config, stage, lane):
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
                    ids = tokenizer.apply_chat_template([dict(role='user', content=item['prompt'])],
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
                if future is not None and not future.cancelled():
                    future.set_result(row)
                    if hasattr(future, 'token_events'):
                        future.token_events.put(None)
                print(json.dumps(dict(event='request_complete', **row), ensure_ascii=False), flush=True)
            channel.send(dict(op='stop'))
    channel.connection.close()
    out = Path(config['execution']['result_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out/f'stage{stage}-lane{lane}.json').write_text(json.dumps(dict(stage=stage, lane=lane,
        scope='vLLM model layers, native TP, reference attention, CPU-staged PP, serialized requests',
        requests=rows, caches_after_release=len(state.caches)), ensure_ascii=False, indent=2))


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
