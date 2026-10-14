import json
import socket
import struct
import unittest
import concurrent.futures
import queue
import importlib.util
from types import SimpleNamespace
from heterogeneous_compute.pipeline_execution import RequestState, TensorChannel
from heterogeneous_compute.pipeline_execution import FeedbackReceiver
from heterogeneous_compute.pipeline_execution import packed_attention
from heterogeneous_compute.pipeline_execution import finish_future
from heterogeneous_compute.scheduling import BatchState, ContinuousScheduler

class PipelineExecutionTests(unittest.TestCase):
    def test_kv_ownership_and_contiguous_decode_then_release(self):
        state = RequestState()
        state.advance('a', [0, 1, 2])
        state.caches[4] = object()
        with self.assertRaisesRegex(ValueError, 'ownership'):
            state.advance('b', [0])
        with self.assertRaisesRegex(ValueError, 'stale'):
            state.advance('a', [2])
        with self.assertRaisesRegex(ValueError, 'own'):
            state.release('b')
        state.advance('a', [3])
        state.release('a')
        self.assertFalse(state.caches)
        self.assertEqual(state.position, 0)
        state.advance('b', [0, 1])

    def test_invalid_protocol_is_rejected_before_tensor_loading(self):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        sender = socket.create_connection(listener.getsockname())
        receiver, _ = listener.accept()
        try:
            channel = TensorChannel(receiver)
            header = json.dumps(dict(version=999, tensors=[])).encode()
            sender.sendall(struct.pack('!I', len(header)) + header)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                channel.receive('cpu')
        finally:
            sender.close()
            receiver.close()
            listener.close()

    def test_batch_kv_owners_are_isolated_and_invalid_install_is_atomic(self):
        state = BatchState(2)
        state.install([dict(request_id='a', positions=[0,1]), dict(request_id='b', positions=[0])])
        state.requests['a'].caches[0] = 'a-keys'
        state.requests['b'].caches[0] = 'b-keys'
        with self.assertRaises(ValueError):
            state.install([dict(request_id='a', positions=[2]), dict(request_id='b', positions=[2])])
        self.assertEqual(state.requests['a'].position, 2)
        state.release('a')
        self.assertEqual(state.requests['b'].caches[0], 'b-keys')
        state.install([dict(request_id='c', positions=[0]), dict(request_id='b', positions=[1])])
        self.assertFalse(state.requests['c'].caches)

    def test_two_inflight_batches_never_share_requests_and_refill_after_release(self):
        scheduler = ContinuousScheduler(max_active=4, microbatch_size=2)
        for i in range(4):
            scheduler.admit(dict(id=str(i), prompt_token_ids=[1,2], max_tokens=1))
        first, second = scheduler.plan(), scheduler.plan()
        self.assertIsNone(scheduler.plan())
        self.assertFalse({s['request_id'] for s in first['segments']} & {s['request_id'] for s in second['segments']})
        self.assertFalse(scheduler.admit(dict(id='extra', prompt_token_ids=[1], max_tokens=1)))
        for packet in [first, second]:
            finished = scheduler.complete(packet['batch_id'], {s['request_id']:7 for s in packet['segments']}, .1, set())
            for identifier in finished:
                scheduler.released(identifier)
        self.assertTrue(scheduler.admit(dict(id='extra', prompt_token_ids=[1], max_tokens=1)))
        self.assertEqual(scheduler.stats['max_inflight_batches'], 2)

    def test_chunked_prefill_decode_positions_and_duplicate_feedback(self):
        scheduler = ContinuousScheduler(prefill_chunk_tokens=2, max_batch_tokens=2)
        scheduler.admit(dict(id='a', prompt_token_ids=[10,11,12,13,14], max_tokens=2))
        expected = [([0,1], None), ([2,3], None), ([4], 7), ([5], 8)]
        for positions, token in expected:
            packet = scheduler.plan()
            self.assertEqual(packet['segments'][0]['positions'], positions)
            self.assertEqual(packet['segments'][0]['sample_token'], token is not None)
            if token is None:
                with self.assertRaises(ValueError):
                    scheduler.complete(packet['batch_id'], {'a':7}, .1, set())
            scheduler.complete(packet['batch_id'], {'a':token}, .1, set())
            with self.assertRaises(ValueError):
                scheduler.complete(packet['batch_id'], {'a':token}, .1, set())
        self.assertEqual(scheduler.released('a').output, [7,8])

    def test_four_clients_split_into_two_batches_to_overlap_pp(self):
        scheduler = ContinuousScheduler(microbatch_size=4)
        for i in range(4):
            scheduler.admit(dict(id=str(i),prompt_token_ids=[1,2],max_tokens=8))
        first,second=scheduler.plan(),scheduler.plan()
        self.assertEqual([len(first['segments']),len(second['segments'])],[2,2])
        self.assertEqual(scheduler.stats['max_inflight_batches'],2)

    def test_cancellation_is_explicit_and_identical_across_tp_lanes(self):
        schedulers = [ContinuousScheduler(), ContinuousScheduler()]
        future = concurrent.futures.Future()
        future.cancel()
        for lane, scheduler in enumerate(schedulers):
            scheduler.admit(dict(id='a', prompt_token_ids=[1], max_tokens=8), future if lane==0 else None)
            packet = scheduler.plan()
            finished = scheduler.complete(packet['batch_id'], {'a':7}, .1, set(), canceled_ids=['a'])
            self.assertEqual(finished, ['a'])
            scheduler.released('a')

    def test_feedback_streams_token_before_scheduler_consumes_completion(self):
        class Tokenizer:
            def decode(self, output, **kwargs):
                return ''.join(chr(x) for x in output)
        future = concurrent.futures.Future()
        future.token_events = queue.Queue()
        receiver = FeedbackReceiver(None, Tokenizer(), {0}, None)
        receiver.register(dict(id='a', max_tokens=2), future)
        receiver.expect(dict(batch_id=0, segments=[dict(request_id='a', step=0, sample_token=True)]))
        receiver.emit(dict(batch_id=0, samples={'a':65}))
        self.assertEqual(future.token_events.get_nowait(), dict(index=0, text='A', finish_reason=None))
        self.assertFalse(future.done())

    def test_canceled_stream_wakes_blocked_event_waiter(self):
        future = concurrent.futures.Future()
        future.token_events = queue.Queue()
        future.cancel()
        finish_future(future, {})
        self.assertIsNone(future.token_events.get_nowait())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Uses the retained vendor Torch runtime')
    def test_packed_prefill_decode_matches_independent_requests_and_chunking(self):
        import torch
        torch.manual_seed(7)
        attention = SimpleNamespace(head_dim=8, num_heads=4, num_kv_heads=2, scaling=8**-.5)
        q, k, v = [torch.randn(8, width, dtype=torch.bfloat16) for width in [32,16,16]]
        packed = BatchState(2)
        packed.install([dict(request_id='a', positions=list(range(3))), dict(request_id='b', positions=list(range(5)))])
        result = packed_attention(q,k,v,attention,packed,0)
        independent = [BatchState(1), BatchState(1)]
        for identifier, owner, start, end in [('a',independent[0],0,3), ('b',independent[1],3,8)]:
            owner.install([dict(request_id=identifier, positions=list(range(end-start)))])
            expected = packed_attention(q[start:end],k[start:end],v[start:end],attention,owner,0)
            torch.testing.assert_close(result[start:end], expected, rtol=0, atol=0)
        q,k,v = [torch.randn(2,width,dtype=torch.bfloat16) for width in [32,16,16]]
        packed.install([dict(request_id='a', positions=[3]), dict(request_id='b', positions=[5])])
        result = packed_attention(q,k,v,attention,packed,0)
        for row,identifier,position in [(0,'a',3),(1,'b',5)]:
            independent[row].install([dict(request_id=identifier,positions=[position])])
            expected = packed_attention(q[row:row+1],k[row:row+1],v[row:row+1],attention,independent[row],0)
            torch.testing.assert_close(result[row:row+1], expected, rtol=0, atol=0)
        q,k,v = [torch.randn(5,width,dtype=torch.bfloat16) for width in [32,16,16]]
        full, chunked = BatchState(1), BatchState(1)
        full.install([dict(request_id='c',positions=list(range(5)))])
        expected = packed_attention(q,k,v,attention,full,0)
        parts = []
        for start,end in [(0,2),(2,4),(4,5)]:
            chunked.install([dict(request_id='c',positions=list(range(start,end)))])
            parts.append(packed_attention(q[start:end],k[start:end],v[start:end],attention,chunked,0))
        torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
