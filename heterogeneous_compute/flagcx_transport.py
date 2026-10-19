"""FlagCX device-buffer PP transport, with framed JSON control over TCP.

The C bridge avoids requiring identical vendor Torch ABIs. Activations always
use flagcxSend/Recv; no Python CPU/TCP tensor fallback is permitted.
"""
import ctypes
import hashlib
import json
import time


class FlagCXChannel:
    device_transport = True

    def __init__(self, control, library, device, rank):
        self.control, self.connection = control, control.connection
        self.device, self.rank = device, rank
        self.lib = ctypes.CDLL(library)
        pointer = ctypes.c_void_p
        self.lib.hs_flagcx_open.argtypes = [ctypes.POINTER(pointer), ctypes.c_int]
        self.lib.hs_flagcx_unique_id.argtypes = [pointer]
        self.lib.hs_flagcx_init.argtypes = [pointer, pointer, ctypes.c_int]
        self.lib.hs_flagcx_is_homo.argtypes = [pointer, ctypes.POINTER(ctypes.c_int)]
        self.lib.hs_flagcx_transfer.argtypes = [pointer, pointer, ctypes.c_size_t,
                                               ctypes.c_int, ctypes.c_int]
        self.lib.hs_flagcx_close.argtypes = [pointer]
        self.lib.hs_flagcx_error.argtypes = [ctypes.c_int]
        self.lib.hs_flagcx_error.restype = ctypes.c_char_p
        self.handle = pointer()
        self.check(self.lib.hs_flagcx_open(ctypes.byref(self.handle), device.index))
        unique = ctypes.create_string_buffer(256)
        if rank == 0:
            self.check(self.lib.hs_flagcx_unique_id(unique))
            control.send(dict(op='flagcx_unique_id', value=unique.raw.hex()))
        else:
            packet, tensors = control.receive('cpu')
            if packet.get('op') != 'flagcx_unique_id' or tensors:
                raise ValueError('Invalid FlagCX bootstrap message')
            data = bytes.fromhex(packet['value'])
            if len(data) != 256:
                raise ValueError('Invalid FlagCX unique ID size')
            ctypes.memmove(unique, data, 256)
        self.check(self.lib.hs_flagcx_init(self.handle, unique, rank))
        homogeneous = ctypes.c_int()
        self.check(self.lib.hs_flagcx_is_homo(self.handle, ctypes.byref(homogeneous)))
        if homogeneous.value:
            raise RuntimeError('Expected an actual heterogeneous FlagCX communicator')
        self.sent_bytes = self.received_bytes = 0
        self.sent_tensors = self.received_tensors = 0
        print(json.dumps(dict(event='flagcx_initialized', rank=rank,
            device=str(device), homogeneous=False, library=library,
            tensor_transport='FlagCX device buffers; FLAGCX_IB_DISABLE selects Socket')), flush=True)

    def check(self, result):
        if result:
            message = self.lib.hs_flagcx_error(result).decode(errors='replace')
            raise RuntimeError(f'FlagCX error {result}: {message}')

    def send(self, metadata, tensors=None):
        import torch
        values, descriptors = [], []
        for name, value in (tensors or {}).items():
            if value.dtype != torch.bfloat16 or value.ndim != 2 or value.device != self.device:
                raise ValueError('FlagCX PP requires two-dimensional BF16 device tensors')
            value = value.detach().contiguous()
            size = value.numel() * value.element_size()
            descriptors.append(dict(name=name, shape=list(value.shape), dtype='bfloat16', bytes=size))
            values.append(value)
        if sum(d['bytes'] for d in descriptors) > self.control.MAX_PAYLOAD:
            raise ValueError('PP packet exceeds configured bounds')
        self.control.send(dict(op='flagcx_packet', metadata=metadata, tensors=descriptors))
        if values:
            # FlagCX owns a separate native stream. Retain inputs until transfer
            # completion and fence producers before handing over their buffers.
            getattr(torch, self.device.type).synchronize(self.device)
        for value in values:
            self.check(self.lib.hs_flagcx_transfer(self.handle, value.data_ptr(),
                                                   value.numel(), 1-self.rank, 0))
            self.sent_tensors += 1
            self.sent_bytes += value.numel() * value.element_size()

    def receive(self, device, allow_idle=False):
        import torch
        packet, unexpected = self.control.receive('cpu', allow_idle=allow_idle)
        if packet.get('op') != 'flagcx_packet' or unexpected:
            raise ValueError('Invalid FlagCX control packet')
        descriptors = packet['tensors']
        total, names = 0, set()
        for descriptor in descriptors:
            shape, size, name = descriptor['shape'], descriptor['bytes'], descriptor['name']
            total += size
            if (descriptor.get('dtype') != 'bfloat16' or len(shape) != 2
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or size != shape[0]*shape[1]*2 or total > self.control.MAX_PAYLOAD
                    or name in names):
                raise ValueError('Invalid FlagCX tensor descriptor')
            names.add(name)
        tensors = {}
        for descriptor in descriptors:
            value = torch.empty(descriptor['shape'], dtype=torch.bfloat16, device=self.device)
            getattr(torch, self.device.type).synchronize(self.device)
            self.check(self.lib.hs_flagcx_transfer(self.handle, value.data_ptr(),
                                                   value.numel(), 1-self.rank, 1))
            tensors[descriptor['name']] = value
            self.received_tensors += 1
            self.received_bytes += descriptor['bytes']
        return packet['metadata'], tensors

    def stats(self):
        return dict(transport='flagcx', sent_bytes=self.sent_bytes,
                    received_bytes=self.received_bytes, sent_tensors=self.sent_tensors,
                    received_tensors=self.received_tensors)

    def close(self):
        if self.handle:
            self.check(self.lib.hs_flagcx_close(self.handle))
            self.handle = None
        self.connection.close()


def probe(config, stage, lane):
    """Validate exact cross-vendor BF16 values before loading model weights."""
    import torch
    from pathlib import Path
    if __package__:
        from .pipeline_execution import open_channel
    else:
        from pipeline_execution import open_channel
    spec = config['stages'][stage]
    if spec['device'] == 'musa':
        import torch_musa  # noqa: F401
    device = torch.device(f"{spec['device']}:{lane}")
    getattr(torch, device.type).set_device(device)
    channel = open_channel(config, stage, lane)
    rows = []
    try:
        for iteration, tokens in enumerate((1, 128, 512, 1024)*3):
            tensor = ((torch.arange(tokens*5120, device=device).reshape(tokens, 5120)+iteration) % 97).to(torch.bfloat16)
            started = time.perf_counter()
            if stage == 0:
                channel.send(dict(op='probe'), dict(activation=tensor))
                _, reply = channel.receive(device)
                assert torch.equal(reply['activation'], tensor+1), 'FlagCX roundtrip mismatch'
            else:
                _, received = channel.receive(device)
                assert torch.equal(received['activation'], tensor), 'FlagCX activation mismatch'
                channel.send(dict(op='probe_reply'), dict(activation=received['activation']+1))
            rows.append(dict(shape=list(tensor.shape), seconds=time.perf_counter()-started, exact=True))
            print(json.dumps(dict(event='probe_transfer_passed', stage=stage, lane=lane,
                                  iteration=iteration, **rows[-1])), flush=True)
        report = dict(stage=stage, lane=lane, passed=True, tests=rows,
                      config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                      **channel.stats())
        directory = Path(config['execution']['result_dir'])
        directory.mkdir(parents=True, exist_ok=True)
        (directory/f'communication-stage{stage}-lane{lane}.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        channel.close()


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', type=int, choices=[0, 1], required=True)
    parser.add_argument('--lane', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    probe(json.loads(Path(args.config).read_text()), args.stage, args.lane)
