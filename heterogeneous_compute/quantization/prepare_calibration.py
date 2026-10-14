"""Reuse pinned ShareGPT while excluding the official benchmark request set."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import ShareGPTDataset
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = ShareGPTDataset(dataset_path=args.dataset, random_seed=20261010)
    heldout = dataset.sample(tokenizer=tokenizer, num_requests=200, output_len=64)
    reserved = {digest(item.prompt) for item in heldout}
    candidates = list(dataset.data)
    random.Random(20261014).shuffle(candidates)
    rows, selected = [], set()
    for entry in candidates:
        prompt = entry['conversations'][0]['value']
        key = digest(prompt)
        if key in reserved or key in selected:
            continue
        text = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        length = len(tokenizer.encode(text, add_special_tokens=False))
        if not 32 <= length <= 512:
            continue
        rows.append(dict(id=entry.get('id'), prompt_sha256=key, text=text, tokens=length))
        selected.add(key)
        if len(rows) == 256:
            break
    if len(rows) != 256 or selected & reserved:
        raise ValueError('Calibration size or holdout isolation failed')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve the existing pinned calibration set')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    manifest = dict(dataset='anon8231489123/ShareGPT_Vicuna_unfiltered',
        revision='192ab2185289094fc556ec8ce5ce1e8e587154ca',
        dataset_sha256='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4',
        calibration_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        seed=20261014, samples=256, max_tokens=512,
        actual_min_tokens=min(r['tokens'] for r in rows),
        actual_max_tokens=max(r['tokens'] for r in rows),
        actual_mean_tokens=sum(r['tokens'] for r in rows)/len(rows),
        heldout_requests=200, heldout_seed=20261010, overlap=0,
        heldout_mean_input_tokens=sum(r.prompt_len for r in heldout)/len(heldout),
        chat_template_sha256=digest(tokenizer.chat_template), thinking=False)
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
