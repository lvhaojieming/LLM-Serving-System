"""Download pinned ModelScope assets with existing curl; resume and verify every file."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import time
import urllib.parse
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inside(path, root):
    """Support the existing Python 3.7 runtime as well as newer nodes."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def model_files(model, revision):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model):
        raise ValueError("Expected an organization/model identifier")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Pin an immutable 40-character repository revision")
    query = urllib.parse.urlencode({"Revision": revision, "Recursive": "true"})
    url = "https://modelscope.cn/api/v1/models/" + model + "/repo/files?" + query
    request = urllib.request.Request(url, headers={"User-Agent": "heteroserve-asset-download/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    if not data.get("Success"):
        raise RuntimeError("ModelScope did not return a valid repository listing")
    files = []
    for item in data["Data"]["Files"]:
        if item.get("Type") != "blob":
            continue
        relative = Path(item["Path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["Path"] or not relative.parts:
            raise ValueError("Unsafe model asset path")
        checksum, size = item.get("Sha256", ""), item.get("Size")
        if not re.fullmatch(r"[a-f0-9]{64}", checksum) or type(size) is not int or size <= 0:
            raise ValueError("Every asset requires a positive size and SHA-256")
        query = urllib.parse.urlencode({"Revision": revision, "FilePath": item["Path"]})
        files.append({"path": item["Path"], "size": size, "sha256": checksum,
                      "url": "https://modelscope.cn/api/v1/models/" + model + "/repo?" + query})
    names = [item["path"] for item in files]
    if (len(names) != len(set(names)) or not {"config.json", "tokenizer.json"} <= set(names)
            or not {"model.safetensors.index.json", "model.safetensors"}.intersection(names)):
        raise ValueError("Repository assets are duplicated or incomplete")
    return sorted(files, key=lambda item: (-item["size"], item["path"]))


def fetch(item, destination):
    path = destination / item["path"]
    if path.is_symlink() or not inside(path.resolve(), destination):
        raise ValueError("Model asset escaped the destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise RuntimeError("Existing asset differs from pinned source; refusing to overwrite: " + str(path))
        return {**item, "reused": True}
    partial = path.with_name(path.name + ".part")
    if partial.is_symlink() or (partial.exists() and partial.stat().st_size > item["size"]):
        raise ValueError("Unexpected partial model asset")
    for attempt in range(6):
        if partial.exists() and partial.stat().st_size == item["size"]:
            break
        for url in item.get('download_urls', [item['url']]):
            response = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "15", "--speed-time", "60", "--speed-limit", "1024",
                "--continue-at", "-", "--output", str(partial), url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if response.returncode == 0:
                break
        if response.returncode == 0:
            break
        print(json.dumps({"event": "retry", "file": item["path"], "attempt": attempt + 1,
                          "curl_exit": response.returncode}), flush=True)
        time.sleep(min(2 ** attempt, 20))
    if not partial.exists() or partial.stat().st_size != item["size"] or digest(partial) != item["sha256"]:
        raise RuntimeError("Size or SHA-256 verification failed: " + item["path"])
    partial.replace(path)
    print(json.dumps({"event": "verified", "file": item["path"], "bytes": item["size"]}), flush=True)
    return {**item, "reused": False}


def verify_weights(destination, quantization='none'):
    config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    quant = config.get('quantization_config', {})
    if quantization == 'none':
        if quant or config.get("torch_dtype") != "bfloat16":
            raise ValueError("This acceptance requires original, unquantized BF16 weights")
    elif quant.get('quant_method') != quantization or quant.get('bits') != 4 or quant.get('group_size') != 128:
        raise ValueError('Expected the requested W4A16 group128 checkpoint')
    elif quantization == 'gptq' and (quant.get('desc_act') or quant.get('checkpoint_format', 'gptq') != 'gptq'):
        raise ValueError('Existing adapters require legacy GPTQ without activation ordering')
    index_path = destination / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        # Small embedding models use a single safetensors file without an index.
        path = destination / "model.safetensors"
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= length <= min(128 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(length))
        index = {"weight_map": {k: path.name for k in header if k != "__metadata__"},
                 "metadata": {"total_size": path.stat().st_size - 8 - length}}
    mapping = index["weight_map"]
    seen, dtypes, parameters, tensor_bytes, packed, schema = set(), {}, 0, 0, 0, {}
    for name in sorted(set(mapping.values())):
        path = destination / name
        if not inside(path.resolve(), destination):
            raise ValueError("Weight index escaped the model directory")
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if not 2 <= length <= min(128 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError("Invalid safetensors header")
            header = json.loads(stream.read(length))
        payload_size, intervals = path.stat().st_size - 8 - length, []
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if key in seen or mapping.get(key) != name:
                raise ValueError("Safetensors names disagree with the weight index")
            dtype = value["dtype"]
            widths = {'BF16': 2, 'F32': 4} if quantization == 'none' else {'BF16': 2, 'F16': 2, 'F32': 4, 'I32': 4, 'I64': 8}
            if dtype not in widths:
                raise ValueError("Unexpected checkpoint tensor dtype")
            if key.endswith('.qweight'):
                if dtype != 'I32':
                    raise ValueError('Quantized weights must be packed INT32')
                packed += 1
            count = 1
            for dimension in value["shape"]:
                if type(dimension) is not int or dimension < 0:
                    raise ValueError("Invalid tensor shape")
                count *= dimension
            start, end = value["data_offsets"]
            if not 0 <= start <= end <= payload_size or end - start != count * widths[dtype]:
                raise ValueError("Invalid tensor offsets or payload size")
            seen.add(key)
            schema[key] = value
            intervals.append((start, end))
            parameters += count
            tensor_bytes += end - start
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError("Overlapping or missing tensor data")
            cursor = end
        if cursor != payload_size:
            raise ValueError("Unreferenced bytes in a safetensors shard")
    if seen != set(mapping) or tensor_bytes != index["metadata"]["total_size"]:
        raise ValueError("Weight index is incomplete")
    if quantization != 'none' and config['model_type']=='qwen3' and packed != config['num_hidden_layers']*7:
        raise ValueError('Expected seven packed projections per Qwen3 decoder layer')
    if quantization != 'none':
        if schema.get('lm_head.weight', {}).get('dtype') not in {'F16', 'BF16'}:
            raise ValueError('W4A16 checkpoint must preserve the floating LM head')
        for name, value in schema.items():
            if not name.endswith('.qweight'):
                continue
            prefix = name[:-len('.qweight')]
            a, b = value['shape']
            k, n = (a*8, b) if quantization=='gptq' else (a, b*8)
            for suffix, shape, allowed in [('qzeros', [k//128, n//8], {'I32'}),
                                           ('scales', [k//128, n], {'F16', 'BF16', 'F32'})]:
                item = schema.get(prefix+'.'+suffix, {})
                if item.get('shape') != shape or item.get('dtype') not in allowed:
                    raise ValueError('Packed tensor metadata differs: '+prefix+'.'+suffix)
            if quantization=='gptq' and schema.get(prefix+'.g_idx', {}).get('shape') != [k]:
                raise ValueError('GPTQ grouping metadata is missing or incomplete')
    return {"dtype": config["torch_dtype"], "quantized": quantization!='none', "quantization": quantization,
            "model_type": config["model_type"], "tensor_count": len(seen), "dtype_counts": dtypes,
            ('parameters' if quantization=='none' else 'stored_elements'): parameters,
            "packed_linear_layers": packed, "tensor_bytes": tensor_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--mirror', choices=['modelscope', 'hf-mirror'], default='modelscope')
    parser.add_argument('--quantization', choices=['none', 'awq', 'gptq'], default='none')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("Use 1 to 16 concurrent file downloads")
    if not shutil.which("curl"):
        parser.error("Reuse an installed curl executable")
    destination, report = args.destination.resolve(), args.report.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with report.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = model_files(args.model, args.revision)
        if args.mirror == 'hf-mirror':
            for item in files:
                if not item['path'].endswith('.safetensors'):
                    continue
                mirror = 'https://hf-mirror.com/'+args.model+'/resolve/main/'+urllib.parse.quote(item['path'])
                item['download_urls'] = [mirror, item['url']]
        source = {"model": args.model, "revision": args.revision, "source": "modelscope.cn", "files": files}
        manifest = report.with_name(report.stem + "-source.json")
        write_json(manifest, source)
        needed = sum(item["size"] for item in files if not (destination / item["path"]).exists())
        if shutil.disk_usage(destination).free < needed + 1024 ** 3:
            raise RuntimeError("Insufficient disk space for the remaining assets")
        start = time.monotonic()
        status = {"success": False, "started_at": datetime.now(timezone.utc).isoformat(),
                  "model": args.model, "revision": args.revision, "destination": str(destination),
                  "workers": args.workers, "total_bytes": sum(item["size"] for item in files),
                  "source_manifest_sha256": digest(manifest), "python": platform.python_version(),
                  "curl": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}
        write_json(report, status)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                results = list(executor.map(lambda item: fetch(item, destination), files))
            status.update(success=True, elapsed_seconds=round(time.monotonic() - start, 3),
                          completed_at=datetime.now(timezone.utc).isoformat(), weights=verify_weights(destination, args.quantization), files=results)
        except Exception as exc:
            status.update(error=type(exc).__name__ + ": " + str(exc), elapsed_seconds=round(time.monotonic() - start, 3))
            write_json(report, status)
            raise
        write_json(report, status)
        write_json(destination/'download_manifest.json', status)
        (destination/'SHA256SUMS').write_text(''.join(item['sha256']+'  '+item['path']+'\n' for item in files))
        (destination/'DOWNLOAD_COMPLETE').write_text(status['completed_at']+'\n')
        print(json.dumps({"event": "complete", "success": True, "elapsed_seconds": status["elapsed_seconds"],
                          "weights": status["weights"]}), flush=True)


if __name__ == "__main__":
    main()
