"""Own only this experiment's Ray node and native vLLM TP2/PP2 frontend."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def run_native(config, spec, node, stage, directory, processes, save):
    executor = spec.get('executor', 'ray')
    command = ['vllm', 'serve', config['model_path'], '--served-model-name', config['model'],
               '--host', node['host'], '--port', str(spec['api_port']), '--dtype', config['dtype'],
               '--tensor-parallel-size', '2', '--pipeline-parallel-size', '2',
               '--distributed-executor-backend', executor, '--enforce-eager', '--no-enable-prefix-caching',
               '--max-model-len', '2048', '--max-num-seqs', '8', '--max-num-batched-tokens', '2048',
               '--gpu-memory-utilization', '0.60', '--seed', str(config['benchmark']['seed']), *spec['engine_args']]
    if executor == 'mp':
        command += ['--nnodes', '2', '--node-rank', str(stage), '--master-addr', spec['nodes'][0]['host'],
                    '--master-port', str(spec['ray_port'])]
        if stage == 1:
            command += ['--headless']
    with (directory/('vllm.log' if stage == 0 else 'vllm-worker.log')).open('w') as engine_log:
        engine = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=engine_log, stderr=engine_log)
        processes.append(engine)
        save(status='loading_model', executor=executor, engine_pid=engine.pid, engine_command=command)
        if stage == 1:
            raise RuntimeError('Native MP worker exited with '+str(engine.wait()))
        deadline = time.monotonic()+900
        endpoint = 'http://'+node['host']+':'+str(spec['api_port'])
        while time.monotonic() < deadline:
            if engine.poll() is not None:
                raise RuntimeError('vLLM exited before readiness: '+str(engine.returncode))
            try:
                with urllib.request.urlopen(endpoint+'/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError('Native PP engine startup timed out')
        body = dict(model=config['model'], messages=[dict(role='user', content='2+3等于多少？只输出数字。')],
                    temperature=0, max_tokens=16, chat_template_kwargs={'enable_thinking':False})
        request = urllib.request.Request(endpoint+'/v1/chat/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            acceptance = json.load(response)
        if acceptance['choices'][0]['message']['content'].strip() != '5':
            raise RuntimeError('Greedy arithmetic acceptance failed')
        (directory/'acceptance.json').write_text(json.dumps(acceptance, ensure_ascii=False, indent=2))
        save(status='ready', endpoint=endpoint, model_inference_verified=True, ready_at=datetime.now(timezone.utc).isoformat())
        raise RuntimeError('Native PP engine exited with '+str(engine.wait()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--vendor', required=True)
    parser.add_argument('--stage', required=True, type=int, choices=[0, 1])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    spec = config['vendors'][args.vendor]
    node = spec['nodes'][args.stage]
    base = spec['ray_port']
    directory = Path('/workspace/artifacts/baselines/same-vendor-pp')/args.vendor
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/('stage'+str(args.stage)+'.json')
    if marker.exists():
        raise RuntimeError('Retained run state exists; inspect it before another start')
    if args.check:
        if not (Path(config['model_path'])/'DOWNLOAD_COMPLETE').exists():
            raise RuntimeError('Verified model completion marker missing')
        ports = ([base, spec['api_port']] if args.stage == 0 else []) if spec.get('executor') == 'mp' else list(range(base, base+10))+([spec['api_port']] if args.stage == 0 else [])
        for port in ports:
            with socket.socket() as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((node['host'], port))
        print(json.dumps(dict(preflight='passed', host=node['host'], devices=node['devices'], stage=args.stage)))
        return
    state = dict(status='starting', vendor=args.vendor, stage=args.stage, host=node['host'],
                 physical_devices=node['devices'], started_at=datetime.now(timezone.utc).isoformat(),
                 versions={name:importlib.metadata.version(name) for name in ['torch', 'vllm', 'ray']})
    def save(**values):
        state.update(values)
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(marker)
        print(json.dumps(values), flush=True)
    processes = []
    try:
        from network import ensure_peer_routes
        save(peer_routes=ensure_peer_routes(node, spec['nodes']))
        if spec.get('executor') == 'mp':
            run_native(config, spec, node, args.stage, directory, processes, save)
            return
        if args.stage == 1:
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection((spec['nodes'][0]['host'], base), timeout=3):
                        break
                except OSError:
                    time.sleep(3)
            else:
                raise TimeoutError('Ray head did not start')
        command = ['ray', 'start', '--node-ip-address='+node['host'], '--num-cpus=8',
                   '--num-gpus='+('0' if spec['resource'] == 'NPU' else '2'),
                   '--object-store-memory=268435456', '--node-manager-port='+str(base+1),
                   '--object-manager-port='+str(base+2), '--dashboard-agent-listen-port='+str(base+3),
                   '--dashboard-agent-grpc-port='+str(base+4), '--runtime-env-agent-port='+str(base+5),
                   '--metrics-export-port='+str(base+6), '--min-worker-port='+str(base+30),
                   '--max-worker-port='+str(base+129), '--disable-usage-stats', '--block']
        if spec['resource'] == 'NPU':
            command += ['--resources='+json.dumps({'NPU':2})]
        if args.stage == 0:
            command += ['--head', '--port='+str(base), '--include-dashboard=false',
                        '--dashboard-port='+str(base+8)]
        else:
            command += ['--address='+spec['nodes'][0]['host']+':'+str(base)]
        with (directory/('ray-stage'+str(args.stage)+'.log')).open('w') as log:
            ray_process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            processes.append(ray_process)
            save(status='ray_starting', ray_pid=ray_process.pid, ray_command=command)
            if args.stage == 1:
                raise RuntimeError('Ray worker exited with '+str(ray_process.wait()))
            deadline = time.monotonic()+600
            while time.monotonic() < deadline:
                if ray_process.poll() is not None:
                    raise RuntimeError('Ray head exited before readiness')
                try:
                    with socket.create_connection((node['host'], base+1), timeout=2):
                        break
                except OSError:
                    time.sleep(3)
            else:
                raise TimeoutError('Local Ray node manager did not start')
            import ray
            ray.init(address=node['host']+':'+str(base), _node_ip_address=node['host'], log_to_driver=False)
            while time.monotonic() < deadline:
                nodes = [n for n in ray.nodes() if n['Alive']]
                if ({n['NodeManagerAddress'] for n in nodes} == {n['host'] for n in spec['nodes']}
                        and len(nodes) == 2 and all(n['Resources'].get(spec['resource']) == 2 for n in nodes)):
                    break
                time.sleep(3)
            else:
                raise TimeoutError('Ray topology must have exactly two nodes, two devices each')
            save(status='loading_model', ray_nodes=[dict(host=n['NodeManagerAddress'], resources=n['Resources']) for n in nodes])
            ray.shutdown()
            run_native(config, spec, node, args.stage, directory, processes, save)
    except Exception as exc:
        save(status='failed', error=str(exc))
        raise
    finally:
        # Never call ray stop, killall or stop a shared container.
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == '__main__':
    main()
