"""Process-local TP extension for existing signed INT4 checkpoints; weights stay unchanged."""


def validate_alignment(input_size, output_sizes, group_size, pack_factor):
    if input_size % group_size:
        raise ValueError("TP shard must preserve whole INT4 quantization groups")
    if any(size % pack_factor for size in output_sizes):
        raise ValueError("TP output shard must preserve packed INT4 columns")


def install(fast_path=None):
    import torch
    from native_config import NativeInt4LinearMethod
    from vllm.model_executor.layers.quantization.awq import AWQLinearMethod
    from vllm.model_executor.parameter import GroupQuantScaleParameter

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        validate_alignment(input_size_per_partition, output_partition_sizes,
                           self.quant_config.group_size, self.quant_config.pack_factor)
        # vLLM's packed parameter loaders already shard K/N and fused QKV projections.
        # The native checkpoint stores floating offsets, rather than packed AWQ zeros.
        AWQLinearMethod.create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                                       input_size, output_size, params_dtype, **extra_weight_attrs)
        layer.qweight.pack_factor = self.quant_config.pack_factor
        layer.qzeros = GroupQuantScaleParameter(
            data=torch.empty(input_size_per_partition // self.quant_config.group_size,
                             sum(output_partition_sizes), dtype=params_dtype),
            input_dim=0, output_dim=1, weight_loader=extra_weight_attrs.get("weight_loader"))

    NativeInt4LinearMethod.create_weights = create_weights
    if fast_path is not None:
        if type(fast_path) is not bool:
            raise ValueError("INT4 fast_path must be a boolean")
        import torch_npu
        original = getattr(NativeInt4LinearMethod, "_original_native_apply", NativeInt4LinearMethod.apply)
        NativeInt4LinearMethod._original_native_apply = original

        def apply(self, layer, x, bias=None):
            # vLLM linear layers normally receive contiguous [tokens, hidden] tensors.
            # Preserve the native fallback for higher dimensions and strided views.
            if x.ndim == 2 and x.is_contiguous():
                return torch_npu.npu_weight_quant_batchmatmul(
                    x, layer.qweight, layer.scales, layer.qzeros, bias=bias,
                    antiquant_group_size=self.quant_config.group_size)
            return original(self, layer, x, bias)

        NativeInt4LinearMethod.apply = apply if fast_path else original
