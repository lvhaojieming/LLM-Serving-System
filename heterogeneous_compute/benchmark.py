"""Acceptance and official vLLM speed tests for a configured PP executor.

Runs inside an existing container with the installed official benchmark client.
Only synthetic random-token workload is used; it is not ShareGPT throughput.
With --system-config, run the unified Gateway's open-loop SLO experiment instead.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request


def request(endpoint, body, timeout=180):
    query = urllib.request.Request(endpoint+'/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(query, timeout=timeout) as response:
        return json.load(response)


def acceptance(endpoint, directory, served_model='Qwen3-14B', require_kv_state=True, timeout=180):
    cases = [('2+3等于多少？只输出数字。', '5'),
             ('法国的首都是哪里？只回答城市名。', '巴黎'),
             ('What is 7 minus 4? Reply with only the digit.', '3')]

    def check(case):
        prompt, expected = case
        started = time.perf_counter()
        response = request(endpoint, dict(model=served_model,
            messages=[dict(role='user', content=prompt)], temperature=0, max_tokens=16,
            chat_template_kwargs=dict(enable_thinking=False)), timeout=timeout)
        actual = response['choices'][0]['message']['content'].strip()
        row = dict(prompt=prompt, expected=expected, actual=actual,
                    passed=actual == expected, seconds=time.perf_counter()-started,
                    response=response)
        print(json.dumps(dict(event='model_acceptance', **row), ensure_ascii=False), flush=True)
        return row

    rows = [check(case) for case in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        concurrent_rows = list(pool.map(check, cases))
    with urllib.request.urlopen(endpoint+'/health', timeout=10) as response:
        payload = response.read()
        health = json.loads(payload) if payload else dict(http_status=response.status)
    report = dict(passed=all(row['passed'] for row in rows+concurrent_rows)
                  and (not require_kv_state or
                       (health['kv_owners'] == 0 and health['active_requests'] == 0)),
                  sequential=rows, concurrent=concurrent_rows, health_after=health,
                  scope='Six exact-answer smoke checks; not general model quality or batch invariance')
    (directory/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise RuntimeError('Model acceptance failed; benchmark was not started')
    return report


def summarize(data, count, output_tokens, previous_inputs):
    if data['completed'] != count or data.get('failed', 0):
        raise RuntimeError('Failed or missing benchmark requests')
    if len(data['output_lens']) != count or any(n != output_tokens for n in data['output_lens']):
        raise RuntimeError('Benchmark output budget mismatch')
    if previous_inputs is not None and data['input_lens'] != previous_inputs:
        raise RuntimeError('Benchmark input lengths differ between concurrency cases')
    throughput = sum(data['output_lens']) / data['duration']
    if abs(throughput-data['output_throughput']) > 1e-5*max(throughput, 1):
        raise RuntimeError('Benchmark throughput recomputation mismatch')
    metrics = ['completed', 'failed', 'duration', 'request_throughput', 'output_throughput',
               'total_token_throughput']
    for name in ('ttft', 'tpot', 'itl', 'e2el'):
        metrics += [f'{percent}_{name}_ms' for percent in ('mean', 'p50', 'p95', 'p99')]
        percentiles = [data.get(f'{percent}_{name}_ms') for percent in ('p50', 'p95', 'p99')]
        if all(n is not None for n in percentiles) and percentiles != sorted(percentiles):
            raise RuntimeError('Latency percentile ordering mismatch')
    return {key: data[key] for key in metrics if key in data}


def benchmark_command(endpoint, model_path, served_model, input_tokens, output_tokens,
                      count, warmups, concurrency, seed, output):
    return ['vllm', 'bench', 'serve', '--backend', 'vllm',
        '--endpoint', '/v1/completions', '--base-url', endpoint,
        '--model', model_path, '--tokenizer', model_path,
        '--served-model-name', served_model, '--dataset-name', 'random',
        '--random-input-len', str(input_tokens), '--random-output-len', str(output_tokens),
        '--random-range-ratio', '0.0', '--num-prompts', str(count), '--num-warmups', str(warmups),
        '--max-concurrency', str(concurrency), '--request-rate', 'inf',
        '--ignore-eos', '--temperature', '0', '--seed', str(seed),
        '--percentile-metrics', 'ttft,tpot,itl,e2el', '--metric-percentiles', '50,95,99',
        '--save-result', '--save-detailed', '--result-dir', str(output.parent),
        '--result-filename', output.name]


def native_matrix(config, vendor, config_path):
    """Measure native vendor servers using the same installed official client."""
    target = config['vendors'][vendor]
    settings = config['benchmark']
    endpoint = f"http://{target['host']}:{target['port']}"
    directory = Path(target['result_dir'])/'performance'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'status.json'
    with marker.open('x') as stream:
        stream.write('{}')
    state = dict(status='accepting_model', started_at=datetime.now(timezone.utc).isoformat(),
                 vendor=vendor, model=config['model'], tp=config['tp'], pp=config['pp'],
                 total_devices=len(target['devices']), dtype=config['dtype'],
                 quantization=config['quantization'], kernel_path=target['kernel_path'],
                 weight_provenance=config['weight_provenance'], benchmark=settings,
                 client_vllm=importlib.metadata.version('vllm'),
                 config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 workload='synthetic fixed-length random tokens; not ShareGPT', cases={})

    def save(**updates):
        state.update(updates)
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(marker)
        print(json.dumps(updates, ensure_ascii=False), flush=True)

    save()
    try:
        acceptance(endpoint, directory, config['model'], require_kv_state=False)
        (directory/'config.json').write_text(json.dumps(config, indent=2))
        for input_tokens, output_tokens in settings['lengths']:
            inputs = None
            for concurrency in settings['concurrencies']:
                label = f'random-i{input_tokens}-o{output_tokens}-c{concurrency}'
                output = directory/(label+'.json')
                if output.exists():
                    raise RuntimeError('Refusing to overwrite retained benchmark results')
                command = benchmark_command(endpoint, config['model_path'], config['model'],
                    input_tokens, output_tokens, settings['num_prompts'], settings['warmups'],
                    concurrency, settings['seed'], output)
                (directory/(label+'-command.json')).write_text(json.dumps(command, indent=2))
                save(status='running', current_case=label, command=command)
                with (directory/(label+'.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=log, check=True)
                data = json.loads(output.read_text())
                if len(data['input_lens']) != settings['num_prompts'] or any(
                        n != input_tokens for n in data['input_lens']):
                    raise RuntimeError('Benchmark fixed input length mismatch')
                state['cases'][label] = summarize(data, settings['num_prompts'], output_tokens, inputs)
                inputs = data['input_lens']
                save(completed_cases=len(state['cases']))
        save(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
             validation='All requests passed; fixed inputs/outputs; throughput recomputed')
        (directory/'summary.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as error:
        save(status='failed', error=str(error))
        raise


def fixed_stream(endpoint, served_model, messages, input_tokens, output_tokens, seed, timeout=60):
    """Validate a complete fixed-token SSE response and measure decode TPOT."""
    body = dict(model=served_model, messages=messages, temperature=0, seed=seed,
                max_tokens=output_tokens, ignore_eos=True, stream=True,
                stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False})
    query = urllib.request.Request(endpoint + '/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    arrivals, content, usage, finish, done = [], [], None, None, False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(query, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            payload = line[5:].strip()
            if payload == b'[DONE]':
                done = True
                continue
            event = json.loads(payload)
            if event.get('error'):
                raise RuntimeError('Backend stream error: ' + str(event['error']))
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
                text = choice.get('delta', {}).get('content')
                if text:
                    arrivals.append(time.perf_counter())
                    content.append(text)
    if not (done and usage and finish == 'length' and len(arrivals) > 1
            and usage.get('prompt_tokens') == input_tokens
            and usage.get('completion_tokens') == output_tokens):
        raise RuntimeError('Incomplete SSE or fixed-token budget mismatch: ' + str(usage))
    return dict(passed=True, usage=usage, done=done, finish_reason=finish,
                tpot_ms=1000 * (arrivals[-1] - arrivals[0]) / (output_tokens - 1),
                content_sha256=hashlib.sha256(''.join(content).encode()).hexdigest())


def runtime_acceptance(args):
    """Reuse the exact-answer checks for selected live workers, then probe SSE.

    This bounded rollout check is separate from the open-loop capacity experiment.
    A frozen workload file makes the token budget identical across vendor clients.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from heterogeneous_compute.serving import expand_config, generation_workers
    config = expand_config(json.loads(args.serving_config.read_text()))
    workers = generation_workers(config)
    names = args.services or list(workers)
    if not set(names).issubset(workers):
        raise ValueError('Unknown generation worker')
    if args.num_prompts < 1 or any(n < 1 for n in args.concurrencies):
        raise ValueError('Request counts and concurrencies must be positive')
    if args.output_tokens < 2 or args.input_tokens < 1:
        raise ValueError('Runtime probes need positive input and at least two output tokens')
    prompts = json.loads(args.workload.read_text())
    if not prompts:
        raise ValueError('Empty frozen workload')
    directory = args.result_dir
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'status.json'
    with marker.open('x') as stream:
        stream.write('{}')
    report = dict(status='running', seed=args.seed, input_tokens=args.input_tokens,
                  output_tokens=args.output_tokens, requests_per_concurrency=args.num_prompts,
                  workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Exact-answer and fixed-token SSE rollout acceptance; not capacity or long-term stability',
                  workers={})
    timeout = getattr(args, 'timeout_seconds', 60)
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        for name in names:
            report['current_worker'] = name
            marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            worker = workers[name]
            endpoint = worker['base_url'].removesuffix('/v1')
            output = directory / name
            output.mkdir()
            checks = acceptance(endpoint, output, worker['model'], require_kv_state=False, timeout=timeout)
            row = dict(worker=worker, exact_answer_passed=checks['passed'], cases={})
            # The warmup is retained separately and excluded from the comparison.
            row['warmup'] = fixed_stream(endpoint, worker['model'], prompts[0],
                args.input_tokens, args.output_tokens, args.seed, timeout)
            for concurrency in args.concurrencies:
                def one(index):
                    return fixed_stream(endpoint, worker['model'], prompts[index % len(prompts)],
                        args.input_tokens, args.output_tokens, args.seed, timeout)
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows = list(pool.map(one, range(args.num_prompts)))
                times = [r['tpot_ms'] for r in rows]
                row['cases'][str(concurrency)] = dict(requests=rows,
                    median_tpot_ms=statistics.median(times), max_tpot_ms=max(times),
                    tpot_slo_ms=args.tpot_slo_ms,
                    slo_pass=sum(t <= args.tpot_slo_ms for t in times))
            report['workers'][name] = row
            marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps(dict(event='runtime_worker_accepted', name=name,
                cases={c:{k:v for k,v in r.items() if k != 'requests'} for c,r in row['cases'].items()})), flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        marker.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--system-config', type=Path, help='Open-loop unified-system SLO capacity and stability configuration')
    parser.add_argument('--serving-config', type=Path, help='Bounded acceptance of selected live generation workers')
    parser.add_argument('--services', nargs='+', help='Worker names for --serving-config')
    parser.add_argument('--workload', type=Path, help='Frozen chat-message workloads for --serving-config')
    parser.add_argument('--tpot-slo-ms', type=float, default=100)
    parser.add_argument('--timeout-seconds', type=float, default=60)
    parser.add_argument('--vendor', help='Native vendor key from the configuration')
    parser.add_argument('--result-dir', type=Path, help='Cross-vendor validation output; never overwrites an existing run')
    parser.add_argument('--concurrencies', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--num-prompts', type=int, default=200)
    parser.add_argument('--input-tokens', type=int, default=256)
    parser.add_argument('--output-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261010)
    args = parser.parse_args()
    if args.serving_config:
        if args.config or args.system_config or args.vendor or not args.result_dir or not args.workload:
            parser.error('--serving-config requires --workload/--result-dir and is a separate mode')
        runtime_acceptance(args)
        return 0
    if args.system_config:
        if args.config or args.vendor or not args.result_dir:
            parser.error('--system-config requires --result-dir and cannot combine with --config/--vendor')
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from heterogeneous_compute.system_benchmark import main as system_main
        return system_main(args.system_config, args.result_dir)
    if args.config is None:
        parser.error('--config or --system-config is required')
    config = json.loads(args.config.read_text())
    if args.vendor:
        if args.vendor not in config.get('vendors', {}):
            parser.error('Requested vendor is absent from the configuration')
        native_matrix(config, args.vendor, args.config)
        return
    endpoint = f"http://{config['master']}:{config['execution']['api_port']}"
    directory = args.result_dir or Path(config['execution']['result_dir'])/'performance'
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'status.json'
    # An exclusive marker also prevents simultaneous duplicate benchmark clients.
    with marker.open('x') as stream:
        stream.write('{}')
    state = dict(status='accepting_model', started_at=datetime.now(timezone.utc).isoformat(),
                 model=config['model'], tp=config['tp'], pp=config['pp'], total_devices=4,
                 dtype=config['dtype'], transport=config['execution']['transport'],
                 flagcx=config.get('flagcx'), stages=config['stages'],
                 execution_mode=config['execution'].get('execution_mode', 'eager'),
                 tp_communicator=config['execution'].get('tp_communicator', 'torch'),
                 attention_backend=config['execution'].get('attention_backend', 'reference'),
                 client_vllm=importlib.metadata.version('vllm'),
                 workload='synthetic random tokens; not ShareGPT',
                 input_tokens=args.input_tokens, output_tokens=args.output_tokens,
                 requests_per_case=args.num_prompts, concurrencies=args.concurrencies,
                 seed=args.seed, cases={},
                 scope=config['execution']['scope'],
                 config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest())

    def save(**updates):
        state.update(updates)
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(marker)
        print(json.dumps(updates, ensure_ascii=False), flush=True)

    save()
    try:
        acceptance(endpoint, directory)
        (directory/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
        inputs = None
        for concurrency in args.concurrencies:
            label = f'random-c{concurrency}'
            output = directory/(label+'.json')
            if output.exists():
                raise RuntimeError('Refusing to overwrite retained benchmark results')
            command = benchmark_command(endpoint, config['execution']['model_path'], 'Qwen3-14B',
                args.input_tokens, args.output_tokens, args.num_prompts, 3,
                concurrency, args.seed, output)
            (directory/(label+'-command.json')).write_text(json.dumps(command, indent=2))
            save(status='running', current_concurrency=concurrency, command=command)
            with (directory/(label+'.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=log, check=True)
            data = json.loads(output.read_text())
            state['cases'][str(concurrency)] = summarize(data, args.num_prompts, args.output_tokens, inputs)
            inputs = data['input_lens']
            save(completed_cases=len(state['cases']))
        save(status='complete', completed_at=datetime.now(timezone.utc).isoformat(),
             validation='All requests passed; fixed outputs; identical case input lengths; throughput recomputed')
        (directory/'summary.json').write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as error:
        save(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
