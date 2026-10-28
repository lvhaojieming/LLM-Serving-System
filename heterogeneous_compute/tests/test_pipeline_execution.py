import json
import socket
import struct
import unittest
import concurrent.futures
import queue
import importlib.util
import os
from unittest.mock import patch
from types import SimpleNamespace
from heterogeneous_compute.pipeline_execution import RequestState, TensorChannel
from heterogeneous_compute.pipeline_execution import FeedbackReceiver
from heterogeneous_compute.pipeline_execution import packed_attention
from heterogeneous_compute.pipeline_execution import finish_future
from heterogeneous_compute.scheduling import BatchState, ContinuousScheduler
from heterogeneous_compute.manage import processes
from heterogeneous_compute.benchmark import summarize

class PipelineExecutionTests(unittest.TestCase):
    def test_cross_vendor_replicas_do_not_share_devices_or_ports(self):
        from pathlib import Path
        from heterogeneous_compute.manage import resolve_instance
        config = json.loads((Path(__file__).parents[1]/'configs/ascend_iluvatar.json').read_text(encoding='utf-8'))
        resolved = resolve_instance(config, 'pp-03')
        self.assertEqual(resolved['stages'][0]['env']['ASCEND_RT_VISIBLE_DEVICES'], '4,5')
        self.assertEqual(resolved['stages'][1]['env']['CUDA_VISIBLE_DEVICES'], '10,11')
        self.assertEqual(config['stages'][0]['devices'], [0, 1])
        config['instances']['pp-03']['devices'][0] = [0, 1]
        with self.assertRaises(ValueError):
            resolve_instance(config, 'pp-03')

    def test_metrics_remain_numeric_after_graph_execution(self):
        from heterogeneous_compute.pipeline_execution import admission_metrics
        text = admission_metrics(dict(active_requests=3, draining=False,
            graphs=dict(mode='piecewise_mlp_graph', captures=40, replays=100)), 2)
        self.assertIn('heteroserve_pp_queued_requests 2\n', text)
        self.assertIn('heteroserve_pp_graphs_replays 100\n', text)
        for line in text.splitlines():
            float(line.split()[1])

    def test_overlap_evidence_excludes_unrelated_requests_and_old_windows(self):
        import tempfile
        import textwrap
        from pathlib import Path
        from heterogeneous_compute.validate import PROGRAM
        block = textwrap.dedent(PROGRAM[PROGRAM.index('    overlaps = []'):PROGRAM.index('    stream_ok = ')])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = [dict(event='clock_calibration', offset=0, uncertainty=.001)]
            for batch, start, end, owners in [(1, 0, 1, ['active']), (2, .5, 1.5, ['active']),
                                               (3, 0, 1000, ['unrelated'])]:
                rows.extend([dict(event='execution', batch_id=batch, start=start, end=end, batch_size=2),
                             dict(event='peer_execution', batch_id=batch, start=start, end=end,
                                  request_order=owners)])
            for lane in range(2):
                (directory / f'timeline-stage0-lane{lane}.jsonl').write_text(
                    '\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            namespace = dict(directory=directory, stream_ids={'active'}, json=json)
            exec(block, namespace)
            for row in namespace['overlaps']:
                self.assertEqual(row['overlapping_windows'], 2)
                self.assertAlmostEqual(row['conservative_max_overlap_ms'], 499)
            namespace['stream_ids'] = {'absent'}
            exec(block, namespace)
            self.assertTrue(all(r['overlapping_windows'] == 0 and r['max_packed_batch_size'] == 0
                                for r in namespace['overlaps']))
    @unittest.skipUnless(os.environ.get('HETEROSERVE_TEST_DEVICE', '').startswith(('npu', 'musa')),
                         'Explicit retained NPU/MUSA device required')
    def test_graph_replay_updates_inputs_and_preserves_previous_microbatch_output(self):
        import torch
        from heterogeneous_compute.graph_execution import MLPGraphRunner
        device = os.environ['HETEROSERVE_TEST_DEVICE']
        if device.startswith('npu'):
            import torch_npu
            from vllm.engine.arg_utils import EngineArgs
            from vllm.forward_context import set_forward_context
            config = EngineArgs(model='/models/Qwen3-14B', dtype='bfloat16').create_engine_config()
            context = set_forward_context(None, config)
        else:
            import torch_musa
            from contextlib import nullcontext
            config, context = None, nullcontext()
        torch.manual_seed(13)
        weight = torch.randn(128,128,dtype=torch.bfloat16,device=device)
        forward = lambda x: torch.nn.functional.silu(x @ weight)
        x = torch.randn(2,128,dtype=torch.bfloat16,device=device)
        group = SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=None))
        with patch('vllm.distributed.get_tp_group', return_value=group), context:
            runner = MLPGraphRunner(forward, torch.device(device), config)
            first = runner(x)
            torch.testing.assert_close(first,forward(x),rtol=.025,atol=.025)
            retained = first.clone()
            second = runner(x*2)
            torch.testing.assert_close(second,forward(x*2),rtol=.025,atol=.025)
            torch.testing.assert_close(first,retained,rtol=0,atol=0)
            self.assertEqual(runner.stats()['captures'],1)
            self.assertEqual(runner.stats()['replays'],2)

    @unittest.skipUnless(os.environ.get('HETEROSERVE_TEST_DEVICE'), 'Explicit retained vendor device required')
    def test_official_attention_matches_reference_with_independent_cached_requests(self):
        import torch
        device = os.environ['HETEROSERVE_TEST_DEVICE']
        if device.startswith('musa'):
            import torch_musa
        elif device.startswith('npu'):
            import torch_npu
        torch.manual_seed(7)
        attention = SimpleNamespace(head_dim=128, num_heads=20, num_kv_heads=4, scaling=128**-.5)
        reference, vendor = BatchState(2), BatchState(2)
        vendor.attention_backend = 'vendor'
        # Unequal initial lengths, single-token decode, then mixed cached chunks.
        for positions in [([0,1,2], [0,1,2,3,4]), ([3], [5]), ([4], [6,7,8]), ([5,6], [9,10])]:
            segments = [dict(request_id=key, positions=list(p)) for key,p in zip(['a','b'],positions)]
            n = sum(len(p) for p in positions)
            q,k,v = [torch.randn(n,w,dtype=torch.bfloat16,device=device) for w in [2560,512,512]]
            reference.install(segments)
            vendor.install(segments)
            expected = packed_attention(q,k,v,attention,reference,0)
            actual = packed_attention(q,k,v,attention,vendor,0)
            torch.testing.assert_close(actual,expected,rtol=.025,atol=.025)
        for state in [reference,vendor]:
            state.release('a')
            self.assertEqual(set(state.requests), {'b'})

    def test_manager_selects_only_the_requested_config(self):
        entry = '/workspace/heterogeneous_compute/pipeline_execution.py'
        wanted = '/workspace/heterogeneous_compute/configs/musa_iluvatar_flagcx.json'
        listing = ('PID COMMAND\n'
                   f'11 python3 {entry} --config {wanted} --lane 0\n'
                   f'12 python3 {entry} --config /workspace/heterogeneous_compute/configs/ascend_musa.json --lane 0\n'
                   f'13 bash -lc "python3 {entry} --config {wanted}"\n')
        with patch('heterogeneous_compute.manage.remote', return_value=listing):
            rows = processes('unused', dict(container='retained', ssh_alias='node'), wanted)
        self.assertEqual([row['pid'] for row in rows], [11])

    def test_speed_summary_rejects_incomplete_output_and_inconsistent_workloads(self):
        data = dict(completed=2, failed=0, duration=1, output_throughput=128,
                    output_lens=[64,64], input_lens=[256,256])
        self.assertEqual(summarize(data, 2, 64, [256,256])['output_throughput'], 128)
        for change in [dict(failed=1), dict(output_lens=[64,63]),
                       dict(input_lens=[256,255]), dict(output_throughput=129)]:
            with self.assertRaises(RuntimeError):
                summarize(dict(data, **change), 2, 64, [256,256])

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

    def test_tensor_dtype_and_byte_count_rejected_before_body_read(self):
        for dtype,size in [('float16',8),('bfloat16',7)]:
            listener=socket.socket()
            listener.bind(('127.0.0.1',0));listener.listen(1)
            a=socket.create_connection(listener.getsockname())
            b,_=listener.accept()
            try:
                header=json.dumps(dict(version=3,metadata={},tensors=[dict(
                    name='hidden_states',shape=[2,2],dtype=dtype,bytes=size)])).encode()
                a.sendall(struct.pack('!I',len(header))+header)
                with self.assertRaisesRegex(ValueError,'descriptor'):
                    TensorChannel(b,version=3).receive('cpu')
            finally:
                a.close();b.close();listener.close()

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

    def test_protocol_tracks_request_order_phase_positions_and_sampling(self):
        scheduler=ContinuousScheduler(max_inflight=1,prefill_chunk_tokens=2)
        scheduler.admit(dict(id='a',prompt_token_ids=[1,2,3],max_tokens=2,ignore_eos=True))
        first=scheduler.plan()
        self.assertEqual(first['request_order'],['a'])
        self.assertEqual(first['step_id'],first['batch_id'])
        self.assertEqual(first['phase'],'prefill')
        scheduler.complete(first['batch_id'],{'a':None},.1,set())
        second=scheduler.plan();scheduler.complete(second['batch_id'],{'a':7},.1,set())
        third=scheduler.plan();segment=third['segments'][0]
        self.assertEqual(third['phase'],'decode')
        self.assertEqual((segment['step_id'],segment['past'],segment['length']),(1,3,1))
        self.assertEqual(segment['sampling'],{'temperature':0})
        self.assertTrue(segment['ignore_eos'])

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
        # Different query lengths and nonzero histories must share two matmuls,
        # with independent causal masks and no cross-request KV contamination.
        q,k,v=[torch.randn(4,width,dtype=torch.bfloat16) for width in [32,16,16]]
        packed.install([dict(request_id='a',positions=[4]),dict(request_id='b',positions=[6,7,8])])
        with patch.object(torch,'matmul',wraps=torch.matmul) as operation:
            result=packed_attention(q,k,v,attention,packed,0)
            self.assertEqual(operation.call_count,2)
        for row,identifier,start,end,positions in [(0,'a',0,1,[4]),(1,'b',1,4,[6,7,8])]:
            independent[row].install([dict(request_id=identifier,positions=positions)])
            expected=packed_attention(q[start:end],k[start:end],v[start:end],attention,independent[row],0)
            torch.testing.assert_close(result[start:end],expected,rtol=0,atol=0)
        q,k,v = [torch.randn(5,width,dtype=torch.bfloat16) for width in [32,16,16]]
        full, chunked = BatchState(1), BatchState(1)
        full.install([dict(request_id='c',positions=list(range(5)))])
        expected = packed_attention(q,k,v,attention,full,0)
        parts = []
        for start,end in [(0,2),(2,4),(4,5)]:
            chunked.install([dict(request_id='c',positions=list(range(start,end)))])
            parts.append(packed_attention(q[start:end],k[start:end],v[start:end],attention,chunked,0))
        torch.testing.assert_close(torch.cat(parts), expected, rtol=0, atol=0)
