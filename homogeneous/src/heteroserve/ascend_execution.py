"""Ascend execution fixes and configured Qwen3 normalization optimizations."""


def install_ascend_pp_cache_views(model):
    """Select cached K/V views by the cache tensor passed for each virtual engine.

    The Ascend V0 backend caches its first pair of views on the implementation
    instance. PP schedulers have independent cache tensors, so that pair must
    change when the active scheduler changes. Retain each tensor with its views
    to avoid repeated slicing and Python object-id reuse.
    """
    changed = 0
    for _, layer in model.named_modules():
        impl = getattr(layer, "impl", None)
        if impl is None or type(impl).__module__ != "vllm_ascend.attention.attention":
            continue
        if not hasattr(impl, "key_cache") or not hasattr(impl, "value_cache"):
            continue
        reference = getattr(impl, "_ascend_pp_cache_reference", impl.forward)
        impl._ascend_pp_cache_reference = reference

        def bind(backend, original):
            views = {}

            def forward(layer, query, key, value, kv_cache, *args, **kwargs):
                if kv_cache.numel() > 0:
                    identity = id(kv_cache)
                    cached = views.get(identity)
                    if cached is None:
                        cached = (kv_cache, kv_cache[0], kv_cache[1])
                        views[identity] = cached
                    backend.key_cache, backend.value_cache = cached[1:]
                return original(layer, query, key, value, kv_cache, *args, **kwargs)
            return forward

        impl.forward = bind(impl, reference)
        changed += 1
    return changed


def install_qwen3_qknorm(model, enabled=True, mode="native", audit=None):
    """Replace only Q/K head norm callsites; retain their native fallback."""
    import torch
    import torch_npu
    if mode not in {"native", "preserve_rounding"}:
        raise ValueError("Unknown Q/K normalization mode")

    changed = 0
    for name, norm in model.named_modules():
        if name.rsplit(".", 1)[-1] not in {"q_norm", "k_norm"}:
            continue
        if getattr(norm, "variance_size_override", None) is not None:
            continue
        if not getattr(norm, "has_weight", False) or norm.hidden_size != 128:
            continue
        original = getattr(norm, "_ascend_qknorm_reference", norm.forward_native)
        norm._ascend_qknorm_reference = original
        if enabled:
            weight = norm.weight
            unit_weight = torch.ones_like(weight) if mode == "preserve_rounding" else weight
            epsilon = norm.variance_epsilon

            if audit is not None:
                audit["norms"][name] = {"checked": 0, "mismatches": 0, "shapes": []}

            def bind(reference, gamma, unit, eps, preserve_rounding, label):
                def fused(x, residual=None):
                    if (residual is not None or x.ndim != 3 or x.shape[-1] != 128 or
                            x.dtype != unit.dtype or x.device != unit.device):
                        return reference(x, residual)
                    # Native mode is a single weighted RMSNorm kernel. The optional
                    # compatibility mode preserves cast-before-scale rounding.
                    normalized = torch_npu.npu_rms_norm(x, unit, eps)[0]
                    result = normalized * gamma if preserve_rounding else normalized
                    if audit is not None and audit["enabled"]:
                        record = audit["norms"][label]
                        if record["checked"] < audit["calls_per_norm"]:
                            expected = reference(x)
                            record["checked"] += 1
                            record["shapes"].append(list(x.shape))
                            # NPU floating-point equal can accept tiny differences.
                            # Compare stored FP16/BF16 bits on CPU during the finite audit.
                            actual_bits = result.detach().cpu().view(torch.int16)
                            expected_bits = expected.detach().cpu().view(torch.int16)
                            if not torch.equal(actual_bits, expected_bits):
                                record["mismatches"] += 1
                                raise RuntimeError("Q/K norm differs from reference at " + label)
                    return result
                return fused

            norm.forward_native = bind(original, weight, unit_weight, epsilon,
                                       mode == "preserve_rounding", name)
        else:
            norm.forward_native = original
        changed += 1
    return changed


def install_qwen3_int4_script_mlp(model):
    """Run the existing INT4/SwiGLU/INT4 sequence through one C++ interpreter call.

    Parameters stay runtime inputs. The original row-parallel all-reduce remains
    outside the script, in exactly the same position as RowParallelLinear.forward.
    This reduces Python dispatch; it is not an NPU graph or kernel fusion.
    """
    import torch
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import tensor_model_parallel_all_reduce
    from vllm.model_executor.models.qwen2 import Qwen2MLP

    def local_mlp(x, w1, scale1, offset1, w2, scale2, offset2):
        gate = torch.ops.npu.npu_weight_quant_batchmatmul.default(
            x, w1, scale1, offset1, antiquant_group_size=128)
        hidden = torch.ops.npu.npu_swiglu.default(gate)
        return torch.ops.npu.npu_weight_quant_batchmatmul.default(
            hidden, w2, scale2, offset2, antiquant_group_size=128)

    changed = 0
    compiled = None
    for _, mlp in model.named_modules():
        if not isinstance(mlp, Qwen2MLP) or not isinstance(mlp.act_fn, SiluAndMul):
            continue
        gate, down = mlp.gate_up_proj, mlp.down_proj
        if gate.bias is not None or down.bias is not None or not down.input_is_parallel:
            continue
        if any(not all(hasattr(layer, attr) for attr in ("qweight", "scales", "qzeros"))
               for layer in (gate, down)):
            continue
        configs = [layer.quant_method.quant_config for layer in (gate, down)]
        if any(c.get_name() != "moqe_ascend_int4" or c.group_size != 128 for c in configs):
            continue
        params = (gate.qweight, gate.scales, gate.qzeros, down.qweight, down.scales, down.qzeros)
        if gate.scales.dtype != torch.float16 or gate.scales.device.type != "npu":
            continue
        if compiled is None:
            example = torch.zeros((1, gate.qweight.shape[0]), device=gate.scales.device, dtype=gate.scales.dtype)
            # One script per rank, shared by all layers; no weight constants or
            # batch size arithmetic is captured, and native ops accept varying M.
            compiled = torch.jit.trace(local_mlp, (example, *params), check_trace=False)
        original = getattr(mlp, "_ascend_int4_mlp_reference", mlp.forward)
        mlp._ascend_int4_mlp_reference = original

        def bind(reference, graph, tensors, row):
            def forward(x):
                if (x.ndim != 2 or not x.is_contiguous() or x.dtype != tensors[1].dtype or
                        x.device != tensors[1].device or x.shape[-1] != tensors[0].shape[0]):
                    return reference(x)
                local = graph(x, *tensors)
                if row.reduce_results and row.tp_size > 1:
                    return tensor_model_parallel_all_reduce(local)
                return local
            return forward

        mlp.forward = bind(original, compiled, params, down)
        changed += 1
    return changed
