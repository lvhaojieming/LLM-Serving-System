"""Deploy/start/inspect/drain the retained cross-vendor experimental executor."""
import argparse
import json
import shlex
import subprocess
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/zhangjinhao/LLM-Serving-System'
ENTRY = '/workspace/heterogeneous_compute/pipeline_execution.py'
CONFIG = '/workspace/heterogeneous_compute/configs/ascend_musa.json'
LEGACY_ENTRY = '/workspace/src/heteroserve/pipeline_execution.py'
LEGACY_CONFIG = '/workspace/deploy/baselines/cross-vendor-pp.json'


def snapshot(ssh_config, config):
    """Retain source/config identity and reused container/package versions."""
    sources = [ROOT/'heterogeneous_compute'/name for name in ['pipeline_execution.py','scheduling.py']]
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


def processes(ssh_config, spec):
    command = shlex.join(['docker', 'exec', spec['container'], 'ps', '-eo', 'pid,args'])
    output = remote(ssh_config, spec['ssh_alias'], command)
    # Match the Python argv, not launcher shells or other model services.
    rows = []
    for row in output.splitlines()[1:]:
        pid, _, args = row.strip().partition(' ')
        argv = shlex.split(args)
        if (len(argv) > 1 and Path(argv[0]).name.startswith('python')
                and any(entry in argv for entry in {ENTRY, LEGACY_ENTRY}) and '--config' in argv
                and any(c in argv for c in {CONFIG, LEGACY_CONFIG})):
            rows.append(dict(pid=int(pid), argv=argv))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop', 'snapshot'])
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'heterogeneous_compute/configs/ascend_musa.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    states = [processes(args.ssh_config, s) for s in config['stages']]
    if args.action == 'snapshot':
        snapshot(args.ssh_config, config)
        print('Deployment identity retained in the task artifact directory.')
    elif args.action == 'status':
        print(json.dumps(dict(stages=states), indent=2))
        url = f"http://{config['master']}:{config['execution']['api_port']}/health"
        print(remote(args.ssh_config, config['stages'][0]['ssh_alias'],
                     shlex.join(['curl', '-sS', '--max-time', '5', url])))
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
        for stage, spec in enumerate(config['stages']):
            remote(args.ssh_config, spec['ssh_alias'],
                   shlex.join(['mkdir', '-p', REMOTE+'/heterogeneous_compute/configs']))
            for source, destination in [(ROOT/'heterogeneous_compute/pipeline_execution.py', 'heterogeneous_compute/pipeline_execution.py'),
                                        (ROOT/'heterogeneous_compute/scheduling.py', 'heterogeneous_compute/scheduling.py'),
                                        (args.config, 'heterogeneous_compute/configs/ascend_musa.json')]:
                subprocess.run(['scp', '-O', '-F', str(args.ssh_config), str(source),
                    spec['ssh_alias']+':'+REMOTE+'/'+destination], check=True)
        for stage, spec in enumerate(config['stages']):
            for lane in range(config['tp']):
                directory = config['execution']['result_dir']
                argv = ['python3', '-u', ENTRY, '--config', CONFIG, '--stage', str(stage), '--lane', str(lane)]
                body = ('mkdir -p '+shlex.quote(directory)+'; '+shlex.join(argv)
                        +' > '+shlex.quote(f'{directory}/stage{stage}-lane{lane}.log')+' 2>&1')
                docker = ['docker', 'exec', '-d']
                for key, value in spec['env'].items():
                    docker += ['-e', f'{key}={value}']
                docker += [spec['container'], 'bash', '-lc', body]
                remote(args.ssh_config, spec['ssh_alias'], shlex.join(docker))
        snapshot(args.ssh_config, config)
        print('Four ranks dispatched; deployment identity retained; status reports PIDs and HTTP readiness.')


if __name__ == '__main__':
    main()
