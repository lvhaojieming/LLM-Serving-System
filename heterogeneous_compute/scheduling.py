"""Bounded continuous admission, packed microbatches and per-request KV ownership."""
from collections import deque
from dataclasses import dataclass, field


class RequestState:
    def __init__(self):
        self.request_id = None
        self.position = 0
        self.caches = {}

    def validate(self, request_id, positions):
        if not positions:
            raise ValueError('Empty token positions')
        if self.request_id is None and positions[0] != 0:
            raise ValueError('New request must start at position zero')
        if self.request_id is not None and request_id != self.request_id:
            raise ValueError('Release active request before switching KV ownership')
        if positions != list(range(self.position, self.position+len(positions))):
            raise ValueError('Non-contiguous or stale PP token positions')

    def advance(self, request_id, positions):
        self.validate(request_id, positions)
        self.request_id = request_id
        self.position += len(positions)

    def release(self, request_id):
        if self.request_id != request_id:
            raise ValueError('Release does not own active KV cache')
        self.caches.clear()
        self.request_id = None
        self.position = 0


class BatchState:
    def __init__(self, maximum=8):
        self.maximum = maximum
        self.requests = {}
        self.segments = []

    def install(self, segments):
        identifiers = [s['request_id'] for s in segments]
        if not segments or len(set(identifiers)) != len(identifiers):
            raise ValueError('A microbatch needs distinct request IDs')
        if len(set(self.requests)|set(identifiers)) > self.maximum:
            raise ValueError('Too many active KV owners')
        candidates = {identifier:self.requests.get(identifier, RequestState()) for identifier in identifiers}
        # Validate the complete packet before advancing any owner's position.
        for segment in segments:
            candidates[segment['request_id']].validate(segment['request_id'], segment['positions'])
        layout, offset = [], 0
        for segment in segments:
            owner = candidates[segment['request_id']]
            past = owner.position
            owner.advance(segment['request_id'], segment['positions'])
            self.requests[segment['request_id']] = owner
            length = len(segment['positions'])
            layout.append(dict(request_id=segment['request_id'], start=offset, length=length, past=past, owner=owner))
            offset += length
        self.segments = layout

    def advance(self, request_id, positions):
        self.install([dict(request_id=request_id, positions=positions)])

    def release(self, request_id):
        if request_id not in self.requests:
            raise ValueError('Release does not own active KV cache')
        self.requests.pop(request_id).release(request_id)
        self.segments = [s for s in self.segments if s['request_id'] != request_id]

    @property
    def caches(self):
        return {identifier:owner.caches for identifier,owner in self.requests.items()}


@dataclass
class ScheduledRequest:
    item: dict
    future: object = None
    cursor: int = 0
    output: list = field(default_factory=list)
    step_seconds: list = field(default_factory=list)
    inflight: object = None
    finishing: bool = False
    stopped_on_eos: bool = False


class ContinuousScheduler:
    """A request has at most one token/chunk in flight; independent batches overlap."""
    def __init__(self, max_active=8, microbatch_size=4, max_inflight=2,
                 max_batch_tokens=1024, prefill_chunk_tokens=512):
        if (not 1 <= microbatch_size <= max_active <= 16 or not 1 <= max_inflight <= 2
                or not 1 <= prefill_chunk_tokens <= max_batch_tokens <= 2048):
            raise ValueError('Invalid bounded scheduling configuration')
        self.max_active = max_active
        self.microbatch_size = microbatch_size
        self.max_inflight = max_inflight
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.active = {}
        self.ready = deque()
        self.inflight = {}
        self.sequence = 0
        self.stats = dict(microbatches=0, max_batch_size=0, max_inflight_batches=0,
                          generated_tokens=0, admitted_requests=0)

    def admit(self, item, future=None):
        if len(self.active) >= self.max_active:
            return False
        if item['id'] in self.active or not item['prompt_token_ids']:
            raise ValueError('Duplicate request ID or empty prompt')
        self.active[item['id']] = ScheduledRequest(item=item, future=future)
        self.ready.append(item['id'])
        self.stats['admitted_requests'] += 1
        return True

    def plan(self):
        if len(self.inflight) >= self.max_inflight or not self.ready:
            return None
        segments, budget = [], self.max_batch_tokens
        # Leave ready requests for the second stage's next independent batch.
        # With four clients and a cap of four, packing all four would serialize PP.
        limit = self.microbatch_size
        if self.max_inflight > 1 and not self.inflight and len(self.ready) > 1:
            limit = min(limit, (len(self.ready)+1)//2)
        while self.ready and len(segments) < limit and budget:
            identifier = self.ready.popleft()
            request = self.active[identifier]
            if request.inflight is not None or request.finishing:
                raise RuntimeError('Ready queue contains a busy request')
            ids = request.item['prompt_token_ids']
            phase = 'prefill' if request.cursor < len(ids) else 'decode'
            if request.cursor < len(ids):
                count = min(len(ids)-request.cursor, self.prefill_chunk_tokens, budget)
                positions = list(range(request.cursor, request.cursor+count))
                tokens = ids[request.cursor:request.cursor+count]
                request.cursor += count
                sample = request.cursor == len(ids)
            else:
                if not request.output:
                    raise RuntimeError('Decode cannot precede a prefill sample')
                positions = [len(ids)+len(request.output)-1]
                tokens = [request.output[-1]]
                count, sample = 1, True
            segments.append(dict(request_id=identifier, positions=positions,
                                 input_token_ids=tokens, sample_token=sample, step=len(request.output),
                                 step_id=len(request.output), phase=phase,
                                 length=count, past=positions[0], max_tokens=request.item['max_tokens'],
                                 ignore_eos=request.item.get('ignore_eos',False), sampling=dict(temperature=0)))
            budget -= count
        batch_id = self.sequence
        self.sequence += 1
        for segment in segments:
            self.active[segment['request_id']].inflight = batch_id
        self.inflight[batch_id] = segments
        self.stats['microbatches'] += 1
        self.stats['max_batch_size'] = max(self.stats['max_batch_size'], len(segments))
        self.stats['max_inflight_batches'] = max(self.stats['max_inflight_batches'], len(self.inflight))
        phases={s['phase'] for s in segments}
        return dict(op='forward_batch', batch_id=batch_id, step_id=batch_id,
                    phase=next(iter(phases)) if len(phases)==1 else 'mixed',
                    request_order=[s['request_id'] for s in segments], segments=segments)

    def complete(self, batch_id, samples, elapsed, eos_ids, canceled_ids=None):
        if batch_id not in self.inflight:
            raise ValueError('Unknown or duplicate completed microbatch')
        segments = self.inflight[batch_id]
        if set(samples) != {s['request_id'] for s in segments}:
            raise ValueError('Token feedback does not own this microbatch')
        for segment in segments:
            token = samples[segment['request_id']]
            if (segment['sample_token'] and type(token) is not int
                    or not segment['sample_token'] and token is not None):
                raise ValueError('Prefill/decode feedback disagrees with dispatched segment')
        self.inflight.pop(batch_id)
        finished = []
        for segment in segments:
            request = self.active[segment['request_id']]
            if request.inflight != batch_id:
                raise ValueError('Stale request feedback')
            request.inflight = None
            token = samples[segment['request_id']]
            if token is not None:
                request.output.append(token)
                request.step_seconds.append(elapsed)
                self.stats['generated_tokens'] += 1
                request.stopped_on_eos = not request.item.get('ignore_eos', False) and token in eos_ids
            canceled = (request.future is not None and request.future.cancelled()) if canceled_ids is None else segment['request_id'] in canceled_ids
            request.finishing = canceled or request.stopped_on_eos or len(request.output) >= request.item['max_tokens']
            if request.finishing:
                finished.append(segment['request_id'])
            else:
                self.ready.append(segment['request_id'])
        return finished

    def released(self, identifier):
        request = self.active[identifier]
        if not request.finishing or request.inflight is not None:
            raise ValueError('Cannot reclaim a busy request slot')
        return self.active.pop(identifier)
