"""Deploy/start/inspect/drain the retained cross-vendor experimental executor."""
import argparse
import json
import shlex
import subprocess
import hashlib
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/zhangjinhao/LLM-Serving-System'
ENTRY = '/workspace/heterogeneous_compute/pipeline_execution.py'
CONFIG = '/workspace/heterogeneous_compute/configs/ascend_musa.json'
LEGACY_ENTRY = '/workspace/src/heteroserve/pipeline_execution.py'
LEGACY_CONFIG = '/workspace/deploy/baselines/cross-vendor-pp.json'
PROBE_ENTRY = '/workspace/heterogeneous_compute/flagcx_transport.py'


def snapshot(ssh_config, config):
    """Retain source/config identity and reused container/package versions."""
    names = ['pipeline_execution.py', 'scheduling.py']
    if config['execution'].get('transport') == 'flagcx':
        names += ['manage.py', 'benchmark.py', 'flagcx_transport.py',
                  'platforms/flagcx/bridge.cc', 'platforms/flagcx/build.sh']
    sources = [ROOT/'heterogeneous_compute'/name for name in names]
    nodes = []
    for spec in config['stages']:
        identity = remote(ssh_config, spec['ssh_alias'], shlex.join([
            'docker','inspect','--format','{{.Id}} {{.Image}}',spec['container']])).strip().split()
        program = "import json,importlib.metadata as m; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name'].lower() in {'torch','torch-npu','torch-musa','vllm','transformers'}}))"
        versions = json.loads(remote(ssh_config, spec['ssh_alias'], shlex.join([
            'docker','exec',spec['container'],'python3','-c',program])))
        nodes.append(dict(ssh_alias=spec['ssh_alias'], container_id=identity[0], image_id=identity[1], versions=versions))
    directory = ROOT/config['execution']['result_dir'].removeprefix('/workspace/')
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(config=config, source_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), nodes=nodes)
    (directory/'deployment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def remote(ssh_config, alias, command):
    return subprocess.check_output(['ssh', '-F', str(ssh_config), '-o', 'BatchMode=yes',
                                   '-o', 'ConnectTimeout=10', alias, command], text=True)


def processes(ssh_config, spec, remote_config=CONFIG):
    command = shlex.join(['docker', 'exec', spec['container'], 'ps', '-eo', 'pid,args'])
    output = remote(ssh_config, spec['ssh_alias'], command)
    # Match the Python argv, not launcher shells or other model services.
    rows = []
    for row in output.splitlines()[1:]:
        pid, _, args = row.strip().partition(' ')
        argv = shlex.split(args)
        if (len(argv) > 1 and Path(argv[0]).name.startswith('python')
                and any(entry in argv for entry in {ENTRY, LEGACY_ENTRY, PROBE_ENTRY}) and '--config' in argv
                and any(c in argv for c in {remote_config} | ({LEGACY_CONFIG} if remote_config == CONFIG else set()))):
            rows.append(dict(pid=int(pid), argv=argv))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop', 'snapshot', 'probe', 'stop-probe', 'abort', 'benchmark'])
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'heterogeneous_compute/configs/ascend_musa.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    relative_config = args.config.resolve().relative_to(ROOT.resolve()).as_posix()
    remote_config = '/workspace/'+relative_config
    states = [processes(args.ssh_config, s, remote_config) for s in config['stages']]
    if args.action == 'snapshot':
        snapshot(args.ssh_config, config)
        print('Deployment identity retained in the task artifact directory.')
    elif args.action == 'status':
        print(json.dumps(dict(stages=states), indent=2))
        url = f"http://{config['master']}:{config['execution']['api_port']}/health"
        print(remote(args.ssh_config, config['stages'][0]['ssh_alias'],
                     shlex.join(['curl', '-sS', '--max-time', '5', url])))
    elif args.action == 'benchmark':
        if any(len(rows) != config['tp'] for rows in states):
            raise SystemExit('All model ranks must be running before benchmark acceptance.')
        client = config['stages'][-1]
        entry = '/workspace/heterogeneous_compute/benchmark.py'
        marker = config['execution']['result_dir']+'/performance/status.json'
        remote(args.ssh_config, client['ssh_alias'], shlex.join([
            'docker', 'exec', client['container'], 'test', '!', '-e', marker]))
        subprocess.run(['scp', '-O', '-F', str(args.ssh_config),
            str(ROOT/'heterogeneous_compute/benchmark.py'),
            client['ssh_alias']+':'+REMOTE+'/heterogeneous_compute/benchmark.py'], check=True)
        body = (shlex.join(['python3', '-u', entry, '--config', remote_config])+' > '+
                shlex.quote(config['execution']['result_dir']+'/benchmark.log')+' 2>&1')
        remote(args.ssh_config, client['ssh_alias'], shlex.join([
            'docker', 'exec', '-d', client['container'], 'bash', '-lc', body]))
        print('Model acceptance and official benchmark dispatched; inspect benchmark.log for progress.')
    elif args.action == 'abort':
        # Startup/native-library failures can prevent the cooperative drain from
        # making progress. Select only this config's exact Python processes.
        for spec, rows in zip(config['stages'], states):
            for row in rows:
                remote(args.ssh_config, spec['ssh_alias'], shlex.join([
                    'docker', 'exec', spec['container'], 'kill', '-TERM', str(row['pid'])]))
        time.sleep(1)
        for spec in config['stages']:
            for row in processes(args.ssh_config, spec, remote_config):
                remote(args.ssh_config, spec['ssh_alias'], shlex.join([
                    'docker', 'exec', spec['container'], 'kill', '-KILL', str(row['pid'])]))
        print('Only the selected configuration processes were stopped; containers retained.')
    elif args.action == 'stop-probe':
        for spec, rows in zip(config['stages'], states):
            for row in rows:
                if PROBE_ENTRY in row['argv']:
                    print(remote(args.ssh_config, spec['ssh_alias'], shlex.join([
                        'docker', 'exec', spec['container'], 'kill', '-TERM', str(row['pid'])])))
    elif args.action == 'stop':
        # Rank zero drains its admission queue, broadcasts termination and sends
        # stop on both PP lanes. Preserve containers and baseline model services.
        head = config['stages'][0]
        for row in states[0]:
            argv = row['argv']
            if argv[argv.index('--lane')+1] == '0':
                print(remote(args.ssh_config, head['ssh_alias'],
                             shlex.join(['docker', 'exec', head['container'], 'kill', '-TERM', str(row['pid'])])))
        print('Drain requested; inspect status to confirm all four ranks exit.')
    else:
        if any(states):
            raise SystemExit('Pipeline processes already exist; reuse or drain them before starting.')
        if args.action == 'start' and config['execution'].get('transport') == 'flagcx':
            expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            for stage, spec in enumerate(config['stages']):
                for lane in range(config['tp']):
                    report = json.loads(remote(args.ssh_config, spec['ssh_alias'], shlex.join([
                        'docker', 'exec', spec['container'], 'cat',
                        f"{config['execution']['result_dir']}/communication-stage{stage}-lane{lane}.json"])))
                    if not report.get('passed') or report.get('config_sha256') != expected:
                        raise SystemExit('FlagCX communication acceptance missing or stale; run probe first.')
        for stage, spec in enumerate(config['stages']):
            remote(args.ssh_config, spec['ssh_alias'],
                   shlex.join(['mkdir', '-p', REMOTE+'/heterogeneous_compute/configs']))
            for source, destination in [(ROOT/'heterogeneous_compute/pipeline_execution.py', 'heterogeneous_compute/pipeline_execution.py'),
                                        (ROOT/'heterogeneous_compute/scheduling.py', 'heterogeneous_compute/scheduling.py'),
                                        (ROOT/'heterogeneous_compute/flagcx_transport.py', 'heterogeneous_compute/flagcx_transport.py'),
                                        (args.config, relative_config)]:
                subprocess.run(['scp', '-O', '-F', str(args.ssh_config), str(source),
                    spec['ssh_alias']+':'+REMOTE+'/'+destination], check=True)
        for stage, spec in enumerate(config['stages']):
            for lane in range(config['tp']):
                directory = config['execution']['result_dir']
                entry = PROBE_ENTRY if args.action == 'probe' else ENTRY
                argv = ['python3', '-u', entry, '--config', remote_config, '--stage', str(stage), '--lane', str(lane)]
                if args.action == 'probe':
                    argv = ['timeout', '--signal=TERM', '--kill-after=5', '120', *argv]
                prefix = 'communication-' if args.action == 'probe' else ''
                body = ('mkdir -p '+shlex.quote(directory)+'; '+shlex.join(argv)
                        +' > '+shlex.quote(f'{directory}/{prefix}stage{stage}-lane{lane}.log')+' 2>&1')
                docker = ['docker', 'exec', '-d']
                for key, value in spec['env'].items():
                    docker += ['-e', f'{key}={value}']
                docker += [spec['container'], 'bash', '-lc', body]
                remote(args.ssh_config, spec['ssh_alias'], shlex.join(docker))
        snapshot(args.ssh_config, config)
        print('Four ranks dispatched; deployment identity retained; status reports PIDs and HTTP readiness.')


if __name__ == '__main__':
    main()
