"""Official vLLM benchmark client, configured same-vendor native PP workloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

ROOT = Path('/workspace')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--vendor', required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    spec = config['vendors'][args.vendor]
    profile = config['benchmark']
    result = ROOT/'artifacts/baselines/same-vendor-pp'/args.vendor
    if spec.get('instance_id'):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,23}', spec['instance_id']):
            raise ValueError('Invalid instance ID')
        result /= spec['instance_id']
    result /= 'performance'
    result.mkdir(parents=True, exist_ok=True)
    marker = result/'status.json'
    if marker.exists():
        raise RuntimeError('Refusing duplicate benchmark or overwrite of retained results')
    state = dict(status='waiting_for_model', started_at=datetime.now(timezone.utc).isoformat(),
                 vendor=args.vendor, instance_id=spec.get('instance_id'), tp=2, pp=2, total_devices=4,
                 precision={'float16':'FP16', 'bfloat16':'BF16'}[config['dtype']],
                 execution='native vLLM '+spec.get('executor','ray')+' executor; eager, prefix cache off',
                 client_vllm=importlib.metadata.version('vllm'),
                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
    def save(**values):
        state.update(values)
        marker.write_text(json.dumps(state, indent=2))
        print(json.dumps(values), flush=True)
    save()
    try:
        endpoint = 'http://'+spec['nodes'][0]['host']+':'+str(spec['api_port'])
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(endpoint+'/health', timeout=5) as response:
                    if response.status == 200:
                        body = dict(model=config['model'], messages=[dict(role='user', content='2+3等于多少？只输出数字。')],
                                    temperature=0, max_tokens=16, chat_template_kwargs={'enable_thinking':False})
                        request = urllib.request.Request(endpoint+'/v1/chat/completions', data=json.dumps(body).encode(),
                                                         headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(request, timeout=120) as answer:
                            acceptance = json.load(answer)
                        if acceptance['choices'][0]['message']['content'].strip() != '5':
                            raise RuntimeError('Pre-benchmark real decode acceptance failed')
                        (result/'acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2))
                        break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError('Model readiness timeout; inspect server stage/vllm logs')
        env_path = result/'benchmark.env'
        values = dict(MODEL_PATH=config['model_path'], SERVED_MODEL=config['model'],
            DATASET_PATH='/workspace/artifacts/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json',
            SERVER_CONFIG=str(config_path), BASE_URL=endpoint, RESULT_DIR=str(result),
            CONCURRENCIES=' '.join(map(str, profile['concurrencies'])), NUM_PROMPTS=profile['num_prompts'],
            NUM_WARMUPS=profile['num_warmups'], OUTPUT_LEN=profile['output_len'],
            BENCHMARK_SEED=profile['seed'], TEMPERATURE=profile['temperature'], CAPTURE_DEVICE_STATS='false')
        env_path.write_text(''.join(k+'='+shlex.quote(str(v))+'\n' for k,v in values.items()))
        dataset = Path(values['DATASET_PATH'])
        actual = hashlib.sha256(dataset.read_bytes()).hexdigest()
        if actual != '35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4':
            raise RuntimeError('ShareGPT dataset differs from retained native/cross-vendor baseline')
        wrapper = ROOT/'heterogeneous_compute/platforms/iluvatar/benchmark.sh'
        save(status='running', endpoint=endpoint, dataset_sha256=actual,
             benchmark_wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
        with (result/'run.log').open('w') as log:
            completed = subprocess.run(['bash', str(wrapper), str(env_path)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log)
        (result/'run.exit').write_text(str(completed.returncode)+'\n')
        if completed.returncode:
            raise RuntimeError('Official benchmark failed; inspect run.log and per-case logs')
        cases = {}
        lengths = None
        reference_path = ROOT/'artifacts/baselines/performance/ascend/sharegpt-c1.json'
        reference_lengths = json.loads(reference_path.read_text())['input_lens'] if reference_path.exists() else None
        for concurrency in profile['concurrencies']:
            data = json.loads((result/('sharegpt-c'+str(concurrency)+'.json')).read_text())
            if data['completed'] != profile['num_prompts'] or data.get('failed', 0) != 0:
                raise RuntimeError('A benchmark case has failed requests')
            if any(n != profile['output_len'] for n in data['output_lens']):
                raise RuntimeError('Output length differs from fixed token budget')
            if lengths is not None and lengths != data['input_lens']:
                raise RuntimeError('Case input lengths differ')
            lengths = data['input_lens']
            if reference_lengths is not None and lengths != reference_lengths:
                raise RuntimeError('Input token lengths differ from original single-node baseline')
            recalculated = sum(data['output_lens'])/data['duration']
            if abs(recalculated-data['output_throughput']) > 1e-6:
                raise RuntimeError('Output throughput recomputation failed')
            keys = ['completed', 'failed', 'duration', 'request_throughput', 'output_throughput',
                    'total_token_throughput', 'mean_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms',
                    'mean_tpot_ms', 'p95_tpot_ms', 'p99_tpot_ms']
            cases[str(concurrency)] = {k:data[k] for k in keys if k in data}
        (result/'summary.json').write_text(json.dumps(dict(**state, validation='all requests passed; fixed outputs; identical case input lengths; throughput recomputed',
                    cases=cases, completed_at=datetime.now(timezone.utc).isoformat()), indent=2))
        save(status='complete', requests=profile['num_prompts']*len(profile['concurrencies']))
    except Exception as exc:
        save(status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
