"""Generate standard AWQ/GPTQ checkpoints with a pinned GPTQModel runtime."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Reuse the existing checkpoint verification/tokenizer contract.
from homogeneous.experiments.bf16_distributed.quantize import sha256, restore_tokenizer


def restore_original_tokenizer(source, staging, rows):
    """Discard Transformers serialization changes, after checking vocabulary and IDs."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    original = json.loads((source/'tokenizer.json').read_text())
    saved = json.loads((staging/'tokenizer.json').read_text())
    if original['model'] != saved['model'] or original['added_tokens'] != saved['added_tokens']:
        raise ValueError('Export changed tokenizer vocabulary, merges or special token IDs')
    raw = Tokenizer.from_file(str(source/'tokenizer.json'))
    used = AutoTokenizer.from_pretrained(source, local_files_only=True)
    for row in rows:
        if raw.encode(row['text'], add_special_tokens=False).ids != used.encode(row['text'], add_special_tokens=False):
            raise ValueError('Calibration tokenizer changes the original token IDs')
    # Restore the source contract rather than keeping a regenerated v5 tokenizer.
    for name in ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                 'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja']:
        a, b = source/name, staging/name
        if a.exists():
            shutil.copyfile(a, b)
        elif b.exists():
            b.unlink()
    restore_tokenizer(source, staging)


def validate_checkpoint(source, output, algorithm):
    """Check every packed projection against the original tensor geometry."""
    import torch
    from safetensors import safe_open
    hf = json.loads((output/'config.json').read_text())
    quant = hf.get('quantization_config', {})
    if (str(quant.get('quant_method', quant.get('method'))).lower() != algorithm
            or quant.get('bits') != 4 or quant.get('group_size') != 128):
        raise ValueError('Export is not the requested standard W4/group128 format')
    if algorithm == 'gptq' and quant.get('desc_act', False):
        raise ValueError('Existing backend requires sequential GPTQ groups')
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    exported = {}
    for path in output.glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in exported:
                    raise ValueError('Duplicate exported tensor: '+name)
                exported[name] = path
    qweights = [name for name in exported if name.endswith('.qweight')]
    if len(qweights) != 280 or 'lm_head.weight' not in exported:
        raise ValueError('Expected 40 layers x 7 packed linears and a floating LM head')
    for name in qweights:
        prefix = name.removesuffix('.qweight')
        with safe_open(source/index[prefix+'.weight'], framework='pt', device='cpu') as handle:
            n, k = handle.get_slice(prefix+'.weight').get_shape()
        for suffix, shape in {
            'qweight': (k, n//8) if algorithm == 'awq' else (k//8, n),
            'qzeros': (k//128, n//8),
            'scales': (k//128, n),
        }.items():
            key = prefix+'.'+suffix
            with safe_open(exported[key], framework='pt', device='cpu') as handle:
                tensor = handle.get_tensor(key)
            if tuple(tensor.shape) != shape:
                raise ValueError('Packed tensor geometry differs: '+key)
            if suffix != 'scales' and tensor.dtype != torch.int32:
                raise ValueError('Packed tensor must be INT32: '+key)
            if suffix == 'scales' and (not torch.isfinite(tensor).all() or not (tensor > 0).all()):
                raise ValueError('Quantization scales must be positive and finite: '+key)
        if algorithm == 'gptq':
            key = prefix+'.g_idx'
            with safe_open(exported[key], framework='pt', device='cpu') as handle:
                tensor = handle.get_tensor(key)
            if not torch.equal(tensor.long(), torch.arange(k)//128):
                raise ValueError('Nonsequential GPTQ groups are unsupported: '+key)
    with safe_open(exported['lm_head.weight'], framework='pt', device='cpu') as handle:
        if handle.get_tensor('lm_head.weight').dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError('LM head must remain floating point')
    return len(qweights)


def run(config, algorithm):
    source = Path(config['source'])
    output = Path(config['outputs'][algorithm])
    staging = output.with_name(output.name+'.incomplete')
    if output.exists() or staging.exists():
        raise FileExistsError('Refusing to overwrite existing model/staging output')
    print(json.dumps(dict(event='source_verification_start', algorithm=algorithm)), flush=True)
    for line in (source/'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(None, 1)
        path = (source/name.strip()).resolve()
        if source.resolve() not in path.parents or sha256(path) != expected:
            raise ValueError('Original model checksum failed: '+name)
        print(json.dumps(dict(event='source_file_verified', file=name.strip())), flush=True)
    calibration = Path(config['calibration'])
    if sha256(calibration) != config['calibration_sha256']:
        raise ValueError('Calibration set checksum differs')
    # JSON strings may contain Unicode line separators; only LF delimits JSONL.
    rows = [json.loads(line) for line in calibration.read_text(encoding='utf-8').split('\n') if line]
    if len(rows) != config['samples'] or any(r['tokens'] > config['max_sequence_length'] for r in rows):
        raise ValueError('Invalid calibration size/length')
    import torch
    import torch_npu  # noqa: F401
    from gptqmodel import GPTQModel, GPTQConfig, AWQConfig
    from gptqmodel.utils import BACKEND
    from transformers import AutoTokenizer
    if torch.npu.device_count() != 1:
        raise RuntimeError('Quantization requires one isolated NPU')
    versions = {n:importlib.metadata.version(n) for n in config['versions']}
    if versions != config['versions']:
        raise RuntimeError('Installed dependencies differ from pinned versions: '+str(versions))
    torch.manual_seed(config['seed'])
    torch.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if any(len(tokenizer.encode(row['text'], add_special_tokens=False)) != row['tokens'] for row in rows):
        raise ValueError('Calibration tokenizer differs from the prepared token lengths')
    work = Path(config['work'])/algorithm
    work.mkdir(parents=True, exist_ok=True)
    cls = AWQConfig if algorithm == 'awq' else GPTQConfig
    qconfig = cls(bits=4, group_size=128, desc_act=False, sym=(algorithm=='gptq'),
                  device='npu:0', offload_to_disk=True, offload_to_disk_path=str(work))
    started = time.monotonic()
    print(json.dumps(dict(event='quantization_start', algorithm=algorithm, versions=versions,
                         samples=len(rows), device='npu:0', config=str(qconfig)), default=str), flush=True)
    model = GPTQModel.load(str(source), quantize_config=qconfig,
                          backend=BACKEND.TORCH, dtype=torch.float16,
                          attn_implementation='eager', trust_remote_code=False)
    print(json.dumps(dict(event='model_loaded', algorithm=algorithm)), flush=True)
    model.quantize([r['text'] for r in rows], batch_size=1, tokenizer=tokenizer,
                   backend=BACKEND.TORCH, calibration_data_min_length=10)
    staging.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(staging), max_shard_size='4GB')
    publish(config, algorithm, started=started)


def publish(config, algorithm, started=None):
    """Validate a saved checkpoint; permit recovery without repeating calibration."""
    source = Path(config['source'])
    output = Path(config['outputs'][algorithm])
    staging = output.with_name(output.name+'.incomplete')
    if output.exists() or not staging.exists():
        raise FileExistsError('Publishing requires an existing incomplete checkpoint and no output')
    calibration = Path(config['calibration'])
    if sha256(calibration) != config['calibration_sha256']:
        raise ValueError('Calibration set checksum differs')
    rows = [json.loads(line) for line in calibration.read_text(encoding='utf-8').split('\n') if line]
    versions = {n:importlib.metadata.version(n) for n in config['versions']}
    if versions != config['versions']:
        raise ValueError('Publish runtime differs from pinned versions')
    restore_original_tokenizer(source, staging, rows)
    packed_count = validate_checkpoint(source, staging, algorithm)
    files = {p.name:sha256(p) for p in staging.iterdir() if p.is_file()}
    manifest = dict(algorithm=algorithm, bits=4, activations=16, group_size=128,
        desc_act=False, symmetric=(algorithm=='gptq'), source=str(source),
        source_checksums_sha256=sha256(source/'SHA256SUMS'),
        calibration_sha256=sha256(calibration), samples=len(rows), seed=config['seed'],
        versions=versions, packed_linear_layers=packed_count, files=files,
        elapsed_seconds=time.monotonic()-started if started is not None else None,
        recovered_from_saved_checkpoint=(started is None),
        code_sha256=sha256(Path(__file__)),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inference_backend_validation='pending; weight export is not runtime acceptance')
    (staging/'quantization_manifest.json').write_text(json.dumps(manifest,indent=2))
    (staging/'QUANTIZATION_COMPLETE').write_text(algorithm+'\n')
    staging.rename(output)
    print(json.dumps(dict(event='quantization_complete', algorithm=algorithm,
                         output=str(output), elapsed_seconds=manifest['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--algorithm', choices=['awq','gptq'], required=True)
    parser.add_argument('--publish-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.publish_only:
        publish(config, args.algorithm)
    else:
        run(config, args.algorithm)


if __name__ == '__main__':
    main()
