import json
import socket
import struct
import unittest
from heterogeneous_compute.pipeline_execution import RequestState, TensorChannel

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
