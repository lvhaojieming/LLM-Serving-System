"""Process-local vLLM extension reusing the tested integer-layout/NPU adapter.

Dense AWQ GEMM / GPTQ legacy, TP1/TP2, group128. No CUDA or FP16-weight fallback.
"""
import json
import torch
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
try:
    from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
except ImportError:
    # vLLM 0.11 exposes the same loader registry as a class-name list.
    from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
    def register_weight_loader_v2_supported_method(cls):
        if cls.__name__ not in WEIGHT_LOADER_V2_SUPPORTED:
            WEIGHT_LOADER_V2_SUPPORTED.append(cls.__name__)
        return cls
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import (PackedvLLMParameter,
                                         GroupQuantScaleParameter, RowvLLMParameter)
from shared.ascend_int4.layout import decode, repack, matmul
from vllm.platforms import current_platform

@register_quantization_config('moqe_ascend_int4')
class AscendInt4Config(QuantizationConfig):
    def __init__(self, config):
        super().__init__()
        # Ascend RMSNorm queries this vendor metadata even for custom methods.
        self.quant_description = {}
        self.kind = config.get('quant_method')
        self.group_size = config.get('group_size')
        self.pack_factor = 8
        if self.kind not in {'awq','gptq'} or config.get('bits') != 4 or self.group_size != 128:
            raise ValueError('Only AWQ/GPTQ INT4 group128 is supported')
        if config.get('desc_act', False) or config.get('lm_head', False):
            raise ValueError('Act ordering and a quantized LM head are unsupported')
        if self.kind == 'gptq' and config.get('checkpoint_format', 'gptq') != 'gptq':
            raise ValueError('Only legacy GPTQ zero-point convention is supported')
        if self.kind == 'awq' and str(config.get('version','gemm')).lower() != 'gemm':
            raise ValueError('Only AWQ GEMM layout is supported')

    @classmethod
    def get_name(cls):
        return 'moqe_ascend_int4'

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return ['quantize_config.json','quant_config.json']

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if user_quant == cls.get_name() and hf_quant_cfg.get('quant_method') in {'awq','gptq'}:
            return cls.get_name()
        return None

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and not prefix.endswith('lm_head'):
            return AscendInt4LinearMethod(self)
        return None


# Older NPU plugins do not extend their allowlist when a custom config registers.
# This opt-in extension implements the native NPU operator for this exact method.
if current_platform.device_name == 'npu' and 'moqe_ascend_int4' not in current_platform.supported_quantization:
    current_platform.supported_quantization.append('moqe_ascend_int4')


@register_weight_loader_v2_supported_method
class AscendInt4LinearMethod(LinearMethodBase):
    def __init__(self, config):
        self.quant_config = config

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        if get_tensor_model_parallel_world_size() not in {1,2}:
            raise ValueError('This standard-checkpoint adapter supports TP1/TP2 only')
        k, n, config = input_size_per_partition, sum(output_partition_sizes), self.quant_config
        if k % 128 or any(size % 8 for size in output_partition_sizes) or params_dtype != torch.float16:
            raise ValueError('Invalid INT4 group/packing/A16 alignment')
        loader = attrs['weight_loader']
        dim = 1 if config.kind == 'awq' else 0
        shape = (k,n//8) if dim == 1 else (k//8,n)
        layer.register_parameter('qweight', PackedvLLMParameter(data=torch.empty(shape,dtype=torch.int32),
            input_dim=0,output_dim=1,packed_dim=dim,packed_factor=8,weight_loader=loader))
        # Row-parallel projections must shard group rows as well as packed columns.
        layer.register_parameter('qzeros', PackedvLLMParameter(data=torch.empty(k//128,n//8,dtype=torch.int32),
            input_dim=0,output_dim=1,packed_dim=1,packed_factor=8,weight_loader=loader))
        layer.register_parameter('scales', GroupQuantScaleParameter(data=torch.empty(k//128,n,dtype=params_dtype),
            input_dim=0,output_dim=1,weight_loader=loader))
        if config.kind == 'gptq':
            layer.g_idx_group_base = (get_tensor_model_parallel_rank()*(k//128)
                                      if input_size != k else 0)
            layer.register_parameter('g_idx', RowvLLMParameter(data=torch.empty(k,dtype=torch.int32),
                input_dim=0,weight_loader=loader))

    def process_weights_after_loading(self, layer):
        config = self.quant_config
        q, z = decode(layer.qweight, layer.qzeros, config.kind)
        if config.kind == 'gptq':
            expected = torch.arange(q.shape[0],dtype=torch.int64)//config.group_size+layer.g_idx_group_base
            if not torch.equal(layer.g_idx.detach().cpu().to(torch.int64),expected):
                raise ValueError('Nonsequential GPTQ g_idx is unsupported')
        packed, scales, offset = repack(q,z,layer.scales.data,layer.qweight.device)
        layer.qweight = torch.nn.Parameter(packed,requires_grad=False)
        layer.scales = torch.nn.Parameter(scales,requires_grad=False)
        layer.qzeros = torch.nn.Parameter(offset,requires_grad=False)
        if config.kind == 'gptq':
            layer.g_idx = torch.nn.Parameter(torch.empty(0,dtype=torch.int32,device=packed.device),requires_grad=False)
        self.calls = 0
        print(json.dumps(dict(event='w4a16_layer_prepared',algorithm=config.kind,k=q.shape[0],n=q.shape[1],
            tp_rank=get_tensor_model_parallel_rank(),tp=get_tensor_model_parallel_world_size(),
            packed_dtype=str(packed.dtype),packed_bytes=packed.numel()*packed.element_size(),
            scale_dtype=str(scales.dtype),operator='torch_npu.npu_weight_quant_batchmatmul')),flush=True)

    def apply(self, layer, x, bias=None):
        result = matmul(x.reshape(-1,x.shape[-1]).contiguous(),layer.qweight,layer.scales,
                        layer.qzeros,self.quant_config.group_size,bias)
        # Python JSON logging and mutable counters cannot enter a full Dynamo graph.
        # Eager diagnostics remain available; graph tracing is not execution proof.
        if not torch.compiler.is_compiling():
            if self.calls == 0:
                print(json.dumps(dict(event='w4a16_operator_executed',algorithm=self.quant_config.kind,
                                      tp_rank=get_tensor_model_parallel_rank(),tp=get_tensor_model_parallel_world_size(),
                                      activation_dtype=str(x.dtype),packed_dtype=str(layer.qweight.dtype))),flush=True)
            self.calls += 1
        return result.reshape(x.shape[:-1]+(layer.scales.shape[-1],))
