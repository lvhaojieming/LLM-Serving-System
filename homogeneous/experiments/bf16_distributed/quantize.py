"""Requantize original Qwen weights; publish only verified compressed-tensors output."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_quantization(config):
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "compressed-tensors":
        raise ValueError("Output is not compressed-tensors")
    groups = quant.get("config_groups", {})
    if not groups:
        raise ValueError("Missing quantization groups")
    for group in groups.values():
        weight = group.get("weights", {})
        if (weight.get("num_bits") != 4 or weight.get("type") != "int" or
                weight.get("strategy") != "group" or weight.get("group_size") != 128 or
                weight.get("actorder") not in {None, "weight"} or
                group.get("input_activations") is not None or group.get("output_activations") is not None):
            raise ValueError("Expected W4A16 INT, group size 128, without group activation ordering")
    return quant


def restore_tokenizer(source, staging):
    """Discard calibration-only tokenizer settings after checking the vocabulary."""
    for name in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]:
        a, b = source / name, staging / name
        if not a.exists():
            continue
        original, saved = a.read_bytes(), b.read_bytes()
        if name.endswith(".json"):
            x, y = json.loads(original), json.loads(saved)
            if name == "tokenizer.json":
                for key in ["padding", "truncation"]:
                    x.pop(key, None)
                    y.pop(key, None)
            if name == "tokenizer_config.json":
                if x.get("extra_special_tokens") in (None, {}) and y.get("extra_special_tokens") in (None, {}):
                    x.pop("extra_special_tokens", None)
                    y.pop("extra_special_tokens", None)
            if x != y:
                raise ValueError("Tokenizer vocabulary or configuration changed: " + name)
        elif original != saved:
            raise ValueError("Tokenizer merges changed")
    names = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "added_tokens.json", "special_tokens_map.json"]
    for name in names:
        a, b = source / name, staging / name
        if a.exists():
            shutil.copyfile(a, b)
        elif b.exists():
            b.unlink()  # Generated tokenizer metadata; the original files define the contract.


def quantize(settings, algorithm):
    if algorithm not in {"awq", "gptq"}:
        raise ValueError("Unknown quantization algorithm")
    source = Path(settings["source"])
    output = Path(settings["outputs"][algorithm])
    if output.exists():
        raise FileExistsError("Refusing to overwrite model: " + str(output))
    original = json.loads((source / "config.json").read_text())
    if original.get("quantization_config"):
        raise ValueError("Quantization must start from the original floating-point checkpoint")
    original_manifest = json.loads((source / "weights_manifest.json").read_text())
    files = original_manifest["files"]
    for name, digest in files.items():
        if sha256(source / name) != digest:
            raise ValueError("Original checkpoint checksum differs: " + name)
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization import GPTQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; refusing a silent CPU quantization fallback")
    if torch.npu.device_count() != 1:
        raise RuntimeError("The quantization Job must own exactly one isolated NPU")
    torch.manual_seed(settings["seed"])
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    rows = [json.loads(line) for line in Path(settings["calibration"]).read_text().splitlines()]
    if len(rows) != settings["samples"]:
        raise ValueError("Calibration sample count differs from the pinned configuration")
    dataset = Dataset.from_list([{"text": tokenizer.apply_chat_template(
        [{"role": "user", "content": row["text"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)} for row in rows])
    recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])] if algorithm == "awq" else [
        GPTQModifier(ignore=["lm_head"], scheme="W4A16", targets=["Linear"], actorder=None)]
    started = time.monotonic()
    print(json.dumps({"event": "quantization_loading_original", "algorithm": algorithm,
                      "source": str(source), "samples": len(rows), "seed": settings["seed"]}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype="auto",
        local_files_only=True, attn_implementation="eager")
    oneshot(model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe,
        max_seq_length=settings["max_sequence_length"], num_calibration_samples=len(rows))
    # A task-owned incomplete directory remains distinguishable after an interruption.
    staging = output.with_name(output.name + ".incomplete")
    staging.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(staging, save_compressed=True)
    tokenizer.save_pretrained(staging)
    publish(settings, algorithm, started=started)


def pack_awq_int4(values):
    """Pack unsigned logical columns into the standard AWQ GEMM nibble order."""
    import torch
    if values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("AWQ requires a matrix with output channels divisible by eight")
    if values.min().item() < 0 or values.max().item() > 15:
        raise ValueError("AWQ INT4 values must be unsigned integers from zero to fifteen")
    values = values.to(torch.int32)
    packed = torch.zeros((values.shape[0], values.shape[1] // 8), dtype=torch.int32)
    for lane, shift in enumerate((0, 4, 1, 5, 2, 6, 3, 7)):
        packed.bitwise_or_(values[:, lane::8] << (4 * shift))
    return packed


def unpack_awq_int4(packed):
    import torch
    return torch.stack([(packed >> (4 * shift)) & 15
        for shift in (0, 4, 1, 5, 2, 6, 3, 7)], dim=-1).reshape(packed.shape[0], -1)


def unpack_compressed_int4(packed, shape, packed_dim=1):
    """Decode compressed-tensors' offset-binary INT4 storage, including grouped zero points."""
    import torch
    if packed.dtype != torch.int32 or packed.ndim != 2 or packed_dim not in (0, 1):
        raise ValueError("Invalid compressed INT4 tensor")
    if packed_dim == 1:
        values = torch.stack([(packed >> (4 * lane)) & 15 for lane in range(8)], dim=-1)
        values = values.reshape(packed.shape[0], -1)[:, :shape[1]]
    else:
        values = torch.stack([(packed >> (4 * lane)) & 15 for lane in range(8)], dim=1)
        values = values.reshape(-1, packed.shape[1])[:shape[0], :]
    if tuple(values.shape) != tuple(shape):
        raise ValueError("Packed INT4 shape differs from the checkpoint metadata")
    return (values - 8).to(torch.int8)


def export_awq(settings):
    """Repack freshly calibrated INT4 values without recalibration or dequantizing weights."""
    source = Path(settings["outputs"]["awq"])
    output = Path(settings["standard_outputs"]["awq"])
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists():
        raise FileExistsError("Refusing to overwrite an AWQ export: " + str(output))
    source_manifest = json.loads((source / "quantization_manifest.json").read_text())
    if source_manifest.get("algorithm") != "awq" or source_manifest.get("requantized") is not True:
        raise ValueError("Export must use the freshly calibrated AWQ checkpoint")
    for name, digest in source_manifest["files"].items():
        if sha256(source / name) != digest:
            raise ValueError("AWQ source differs from its quantization manifest: " + name)
    config = json.loads((source / "config.json").read_text())
    quant = validate_quantization(config)
    if any(group["weights"].get("actorder") is not None for group in quant["config_groups"].values()):
        raise ValueError("The AWQ export requires unpermuted input groups")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    global_keys = set(source_index)
    # Exercise every nibble value, the sign bit of int32, and adjacent packed words.
    fixture = torch.arange(64, dtype=torch.int32).reshape(2, 32) % 16
    if not torch.equal(unpack_awq_int4(pack_awq_int4(fixture)), fixture):
        raise ValueError("AWQ packing round trip failed")
    staging.mkdir(parents=True)
    packed_count, total_bytes, weight_map = 0, 0, {}
    audits = []
    for shard in sorted(source.glob("*.safetensors")):
        tensors = {}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            def tensor(key):
                if key in keys:
                    return handle.get_tensor(key)
                with safe_open(source / source_index[key], framework="pt", device="cpu") as other:
                    return other.get_tensor(key)
            prefixes = sorted(key[:-len(".weight_packed")] for key in keys if key.endswith(".weight_packed"))
            consumed = set()
            for prefix in prefixes:
                names = {kind: prefix + "." + kind for kind in
                    ("weight_packed", "weight_shape", "weight_scale", "weight_zero_point")}
                if not set(names.values()) <= global_keys or prefix + ".weight_g_idx" in global_keys:
                    raise ValueError("AWQ projection metadata is missing or permuted: " + prefix)
                shape = tuple(tensor(names["weight_shape"]).tolist())
                n, k = shape
                scale = tensor(names["weight_scale"])
                if k % 128 or scale.shape != (n, k // 128):
                    raise ValueError("Unexpected AWQ group shape: " + prefix)
                # compressed-tensors stores q+8 and packs grouped zero points on axis zero.
                q = unpack_compressed_int4(tensor(names["weight_packed"]), shape).to(torch.int32) + 8
                z = unpack_compressed_int4(tensor(names["weight_zero_point"]),
                    (n, k // 128), packed_dim=0).to(torch.int32) + 8
                qw, qz = pack_awq_int4(q.t()), pack_awq_int4(z.t())
                if not torch.equal(unpack_awq_int4(qw), q.t()) or not torch.equal(unpack_awq_int4(qz), z.t()):
                    raise ValueError("Quantized integer or zero-point round trip differs: " + prefix)
                tensors.update({prefix + ".qweight": qw.contiguous(), prefix + ".qzeros": qz.contiguous(),
                                prefix + ".scales": scale.t().contiguous()})
                audits.append({"projection": prefix, "shape": shape, "integers_equal": True,
                               "zero_points_equal": True, "scale_dtype": str(scale.dtype)})
                consumed.update(names.values())
                packed_count += 1
                del q, z, qw, qz, scale
            for key in keys - consumed:
                if any(key.endswith("." + suffix) for suffix in
                       ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx")):
                    if key.rsplit(".", 1)[0] + ".weight_packed" not in global_keys:
                        raise ValueError("Unmapped compressed tensor: " + key)
                    continue  # Metadata may be in a different shard from its packed weight.
                tensors[key] = handle.get_tensor(key).contiguous()
        save_file(tensors, str(staging / shard.name), metadata={"format": "pt"})
        with safe_open(staging / shard.name, framework="pt", device="cpu") as saved:
            if set(saved.keys()) != set(tensors) or any(not torch.equal(saved.get_tensor(key), value)
                    for key, value in tensors.items()):
                raise ValueError("Serialized AWQ tensor differs: " + shard.name)
        for key, value in tensors.items():
            weight_map[key] = shard.name
            total_bytes += value.numel() * value.element_size()
        del tensors
        print(json.dumps({"event": "awq_shard_exported", "shard": shard.name,
                          "packed_projections": packed_count}), flush=True)
    if packed_count != config["num_hidden_layers"] * 7:
        raise ValueError("All attention/MLP projections must be exported")
    config["quantization_config"] = {"quant_method": "awq", "bits": 4, "group_size": 128,
        "zero_point": True, "version": "gemm", "modules_to_not_convert": ["lm_head"]}
    (staging / "config.json").write_text(json.dumps(config, indent=2))
    (staging / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if (source / name).exists():
            shutil.copyfile(source / name, staging / name)
    manifest = {"algorithm": "awq", "format": "awq", "requantized": True,
        "recalibrated_during_export": False, "source_quantization_manifest_sha256": sha256(source / "quantization_manifest.json"),
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "calibration_sha256": source_manifest["calibration_sha256"], "seed": source_manifest["seed"],
        "source": str(source), "packed_projections": packed_count, "projection_audits": audits,
        "script_sha256": sha256(__file__), "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()},
        "inference_validation": "pending: standard AWQ deployment requires a compatible runtime; current vLLM experiment uses the existing converted checkpoint"}
    (staging / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    if output.exists():
        raise FileExistsError("AWQ output appeared during export")
    staging.rename(output)
    print(json.dumps({"event": "standard_awq_export_completed", "output": str(output),
                      "packed_projections": packed_count}), flush=True)


def publish(settings, algorithm, started=None, quantization_script_sha256=None):
    """Validate saved weights and recover publication without recalibrating the model."""
    from safetensors import safe_open
    source, output = Path(settings["source"]), Path(settings["outputs"][algorithm])
    if output.exists():
        raise FileExistsError("Refusing to overwrite model: " + str(output))
    staging = output.with_name(output.name + ".incomplete")
    original = json.loads((source / "config.json").read_text())
    if original.get("quantization_config"):
        raise ValueError("Original source is already quantized")
    recipe = (staging / "recipe.yaml").read_text()
    if ("AWQModifier" if algorithm == "awq" else "GPTQModifier") not in recipe:
        raise ValueError("Saved quantization recipe does not match the selected algorithm")
    rows = [json.loads(line) for line in Path(settings["calibration"]).read_text().splitlines()]
    if len(rows) != settings["samples"]:
        raise ValueError("Calibration count differs")
    quant = validate_quantization(json.loads((staging / "config.json").read_text()))
    packed = 0
    for shard in staging.glob("*.safetensors"):
        with safe_open(shard, framework="np", device="cpu") as handle:
            packed += sum(key.endswith("weight_packed") for key in handle.keys())
    if packed != original["num_hidden_layers"] * 7:
        raise ValueError("Not all seven attention/MLP projections in every layer were packed")
    restore_tokenizer(source, staging)
    tokenizers = ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]
    if any(sha256(source / name) != sha256(staging / name) for name in tokenizers if (source / name).exists()):
        raise ValueError("Quantized tokenizer differs from the original token contract")
    manifest = {"requantized": True, "algorithm": algorithm, "format": "compressed-tensors",
        "source": str(source), "source_manifest_sha256": sha256(source / "weights_manifest.json"),
        "calibration_sha256": sha256(settings["calibration"]), "calibration_ids": [row["id"] for row in rows],
        "seed": settings["seed"], "samples": len(rows), "max_sequence_length": settings["max_sequence_length"],
        "quantization_config": quant, "packed_projections": packed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started if started is not None else None,
        "recovered_from_saved_weights": started is None,
        "quantization_script_sha256": quantization_script_sha256 or sha256(__file__),
        "script_sha256": sha256(__file__), "versions": {name: importlib.metadata.version(name)
            for name in ["torch", "torch-npu", "transformers", "llmcompressor", "compressed-tensors", "datasets"]},
        "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()},
        "inference_validation": "pending: official AscendTurbo deployment and TP4/PP1 benchmark"}
    (staging / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        raise FileExistsError("Output appeared during quantization; refusing replacement")
    staging.rename(output)
    print(json.dumps({"event": "quantization_completed", "output": str(output),
                      "algorithm": algorithm, "elapsed_seconds": manifest["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", required=True, choices=["awq", "gptq"])
    parser.add_argument("--finalize", action="store_true", help="Verify and publish existing compressed weights")
    parser.add_argument("--export-awq", action="store_true", help="Repack fresh AWQ into standard AWQ GEMM storage on CPU")
    parser.add_argument("--quantization-script-sha256", help="Original producer script digest when recovering publication")
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())["requantization"]
    if args.export_awq:
        if args.algorithm != "awq" or args.finalize:
            raise ValueError("AWQ export is a separate CPU operation")
        export_awq(settings)
    elif args.finalize:
        publish(settings, args.algorithm, quantization_script_sha256=args.quantization_script_sha256)
    else:
        quantize(settings, args.algorithm)
