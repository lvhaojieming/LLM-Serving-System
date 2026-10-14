"""Deploy/start/inspect/drain the retained cross-vendor experimental executor."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/zhangjinhao/LLM-Serving-System'
ENTRY = '/workspace/heterogeneous_compute/pipeline_execution.py'
CONFIG = '/workspace/heterogeneous_compute/configs/ascend_musa.json'
LEGACY_ENTRY = '/workspace/src/heteroserve/pipeline_execution.py'
LEGACY_CONFIG = '/workspace/deploy/baselines/cross-vendor-pp.json'


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
        if (len(argv) > 1 and argv[1] in {ENTRY, LEGACY_ENTRY} and '--config' in argv
                and any(c in argv for c in {CONFIG, LEGACY_CONFIG})):
            rows.append(dict(pid=int(pid), argv=argv))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'heterogeneous_compute/configs/ascend_musa.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    states = [processes(args.ssh_config, s) for s in config['stages']]
    if args.action == 'status':
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
                                        (args.config, 'heterogeneous_compute/configs/ascend_musa.json')]:
                subprocess.run(['scp', '-F', str(args.ssh_config), str(source),
                    spec['ssh_alias']+':'+REMOTE+'/'+destination], check=True)
        for stage, spec in enumerate(config['stages']):
            for lane in range(config['tp']):
                directory = config['execution']['result_dir']
                argv = ['python3', ENTRY, '--config', CONFIG, '--stage', str(stage), '--lane', str(lane)]
                body = ('mkdir -p '+shlex.quote(directory)+'; '+shlex.join(argv)
                        +' > '+shlex.quote(f'{directory}/stage{stage}-lane{lane}.log')+' 2>&1')
                docker = ['docker', 'exec', '-d']
                for key, value in spec['env'].items():
                    docker += ['-e', f'{key}={value}']
                docker += [spec['container'], 'bash', '-lc', body]
                remote(args.ssh_config, spec['ssh_alias'], shlex.join(docker))
        print('Four ranks dispatched; status reports PIDs and HTTP readiness.')


if __name__ == '__main__':
    main()
