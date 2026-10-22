"""Single-file embedding checkpoints receive the same structural validation as shards."""
import json
import struct
import pytest
from shared.download_model import verify_weights


def checkpoint(tmp_path, offsets=(0, 4)):
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16", "model_type": "qwen3"}))
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [1, 2], "data_offsets": list(offsets)}}).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


def test_unsharded_embedding_has_verified_tensor_count_and_payload(tmp_path):
    checkpoint(tmp_path)
    report = verify_weights(tmp_path)
    assert report["parameters"] == 2 and report["tensor_bytes"] == 4 and report["tensor_count"] == 1


def test_unsharded_embedding_rejects_invalid_tensor_extent(tmp_path):
    checkpoint(tmp_path, (0, 8))
    with pytest.raises(ValueError, match="offsets"):
        verify_weights(tmp_path)
