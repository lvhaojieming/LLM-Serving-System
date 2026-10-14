"""Promoted AWQ/GPTQ integer-layout adapter from ascend_int4_test.

Preserves checkpoint scales and integer zero points; no requantization.
"""
import numpy as np
import torch

AWQ_REVERSE_ORDER = np.array([0, 4, 1, 5, 2, 6, 3, 7])


def unpack_last(tensor, awq=False):
    if tensor.dtype != torch.int32 or tensor.ndim != 2:
        raise ValueError('Expected a packed INT32 matrix')
    packed = tensor.detach().cpu().numpy().astype(np.uint32)
    values = ((packed[..., None] >> (np.arange(8, dtype=np.uint32)*4)) & 15).astype(np.uint8)
    if awq:
        values = values[..., AWQ_REVERSE_ORDER]
    return values.reshape(packed.shape[0], packed.shape[1]*8)


def decode(qweight, qzeros, kind):
    if kind == 'awq':
        return unpack_last(qweight, True), unpack_last(qzeros, True).astype(np.float32)
    if kind != 'gptq' or qweight.dtype != torch.int32 or qweight.ndim != 2:
        raise ValueError('Expected legacy GPTQ or AWQ GEMM INT4')
    packed = qweight.detach().cpu().numpy().astype(np.uint32)
    values = ((packed[:, None, :] >> (np.arange(8, dtype=np.uint32)[None, :, None]*4)) & 15)
    values = values.astype(np.uint8).reshape(packed.shape[0]*8, packed.shape[1])
    return values, unpack_last(qzeros).astype(np.float32)+1


def repack(q, z, scales, device):
    import torch_npu
    if scales.ndim != 2 or tuple(scales.shape) != tuple(z.shape) or q.shape[1] != z.shape[1]:
        raise ValueError('Quantized weights/scales/zero points disagree')
    signed = torch.from_numpy(q.astype(np.int32)-8).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(signed)
    offset = torch.from_numpy(8-z).to(device=device, dtype=scales.dtype)
    return packed, scales.to(device).contiguous(), offset.contiguous()


def matmul(x, packed, scales, offset, group_size, bias=None):
    import torch_npu
    if x.dtype != scales.dtype or x.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError('A16 activations must match checkpoint scale dtype')
    return torch_npu.npu_weight_quant_batchmatmul(x, packed, scales, offset,
                                                 bias=bias, antiquant_group_size=group_size)
