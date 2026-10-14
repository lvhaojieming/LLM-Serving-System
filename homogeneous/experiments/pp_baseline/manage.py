"""Reuse vendor containers and cached images for native two-node TP2/PP2."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
REMOTE = '/home/zhangjinhao/LLM-Serving-System'
REL = 'homogeneous/experiments/pp_baseline'
TASK = 'moqe-native-pp-image'


class Manager:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(args.config.read_text(encoding='utf-8'))
        if self.config['tp'] != 2 or self.config['pp'] != 2 or self.config['resource_limits_enabled']:
            raise ValueError('This profile requires TP2/PP2 without container resource limits')
        self.artifacts = ROOT/'artifacts/baselines/same-vendor-pp'
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.client_upload_lock = threading.Lock()
        self.client_uploaded = False

    def ssh(self, node, command, check=True):
        result = subprocess.run(['ssh', '-F', str(self.args.ssh_config), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10', node['ssh_alias'], command], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        if check and result.returncode:
            raise RuntimeError(node['ssh_alias']+': '+result.stdout[-1000:]+result.stderr[-1500:])
        return result

    def upload(self, node, source, destination):
        parent = str(Path(destination).parent).replace('\\', '/')
        created = self.ssh(node, 'mkdir -p '+shlex.quote(parent), False)
        if created.returncode and node.get('container') and parent.startswith(REMOTE+'/'):
            # Existing bind roots can be root-owned. Create/chown only this task's leaf.
            leaf = '/workspace/'+parent[len(REMOTE)+1:]
            uid = self.ssh(node, 'id -u').stdout.strip()
            gid = self.ssh(node, 'id -g').stdout.strip()
            self.ssh(node, shlex.join(['docker', 'exec', node['container'], 'mkdir', '-p', leaf]))
            self.ssh(node, shlex.join(['docker', 'exec', node['container'], 'chown', uid+':'+gid, leaf]))
        elif created.returncode:
            raise RuntimeError(created.stderr)
        subprocess.run(['scp', '-O', '-F', str(self.args.ssh_config), str(source),
                        node['ssh_alias']+':'+destination], stdin=subprocess.DEVNULL, check=True,
                       stdout=subprocess.DEVNULL)

    def inspect(self, node, image):
        result = self.ssh(node, shlex.join(['docker', 'image', 'inspect', image, '--format', '{{.Id}}']), False)
        return result.stdout.strip() if result.returncode == 0 else None

    def prepare(self, name):
        spec = self.config['vendors'][name]
        source, target = spec['nodes']
        identity = self.inspect(source, spec['image'])
        if not identity:
            raise RuntimeError('Source image is missing')
        if self.inspect(target, spec['image']) == identity:
            return dict(status='complete', image_id=identity, reused=True)
        task = '/home/zhangjinhao/.cache/'+TASK
        report = REMOTE+'/artifacts/baselines/same-vendor-pp/'+name+'/image'
        # A known task is resumed by reading its state; never duplicate its key or sender.
        previous = self.ssh(source, 'cat '+shlex.quote(report+'/status.json'), False)
        if previous.returncode == 0:
            state = json.loads(previous.stdout)
            if state['status'] != 'failed':
                return state
            raise RuntimeError('Previous image transfer failed: '+state.get('error', ''))
        self.ssh(source, 'test ! -e '+shlex.quote(task)+' && mkdir -m 700 '+shlex.quote(task))
        self.ssh(source, shlex.join(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-C', TASK, '-f', task+'/id_ed25519']))
        public = self.ssh(source, 'cat '+shlex.quote(task+'/id_ed25519.pub')).stdout
        local = self.artifacts/name/'image'
        local.mkdir(parents=True, exist_ok=True)
        (local/'transfer.pub').write_text(public, encoding='utf-8', newline='\n')
        self.ssh(target, 'test ! -e '+shlex.quote(task)+' && mkdir -m 700 '+shlex.quote(task))
        for node in [source, target]:
            self.upload(node, local/'transfer.pub', task+'/transfer.pub')
        self.upload(target, ROOT/'shared/transfer_auth.py', task+'/transfer_auth.py')
        self.upload(source, Path.home()/'.ssh/known_hosts', task+'/known_hosts')
        authorized = 'MOQE_TRANSFER_TASK='+TASK+' MOQE_TRANSFER_SOURCE='+source['host']+' python3 '+task+'/transfer_auth.py add'
        self.ssh(target, authorized)
        peer = [target[k] for k in ['ssh_alias', 'host', 'ssh_port', 'host_key_alias']]
        transfer = dict(task_directory=task, report_directory=report, peer=peer, image=spec['image'],
                        image_id=identity, task_name=TASK, source_ip=source['host'])
        (local/'config.json').write_text(json.dumps(transfer, indent=2), encoding='utf-8')
        self.upload(source, local/'config.json', task+'/config.json')
        self.upload(source, ROOT/'shared/transfer_image.py', REMOTE+'/shared/transfer_image.py')
        launch = ('import pathlib,subprocess; p=pathlib.Path('+repr(report)+'); p.mkdir(parents=True,exist_ok=True); '
                  'f=(p/"run.log").open("wb"); child=subprocess.Popen(["python3","-u",'+repr(REMOTE+'/shared/transfer_image.py')+
                  ',"--config",'+repr(task+'/config.json')+'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True); print(child.pid)')
        body = shlex.join(['python3', '-c', launch])
        pid = self.ssh(source, body).stdout.strip()
        return dict(status='dispatched', pid=pid, image_id=identity)

    def docker(self, name, stage, command, detach=False):
        spec = self.config['vendors'][name]
        node = spec['nodes'][stage]
        devices = ','.join(map(str, node['devices']))
        env = dict(spec['env'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   VLLM_HOST_IP=node['host'], RAY_ADDRESS=spec['nodes'][0]['host']+':'+str(spec['ray_port']),
                   GLOO_SOCKET_IFNAME=node['interface'], NCCL_SOCKET_IFNAME=node['interface'],
                   MCCL_SOCKET_IFNAME=node['interface'], HCCL_SOCKET_IFNAME=node['interface'],
                   HCCL_IF_IP=node['host'],
                   OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                   RAY_TMPDIR='/tmp/moqe-pp/'+name+'/s'+str(stage))
        if name == 'ascend':
            # CANN numbers the mounted subset from zero; host cards remain node.devices.
            env['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        elif name == 'musa':
            env.update(MUSA_VISIBLE_DEVICES=devices, MTHREADS_VISIBLE_DEVICES=devices, CUDA_VISIBLE_DEVICES=devices)
        elif name == 'hygon':
            env.update(HIP_VISIBLE_DEVICES='0,1', ROCR_VISIBLE_DEVICES=devices, CUDA_VISIBLE_DEVICES='0,1')
        else:
            env.update(CUDA_VISIBLE_DEVICES=devices, IX_VISIBLE_DEVICES=devices)
        if spec.get('executor') == 'mp':
            env.pop('RAY_ADDRESS')
        argv = ['docker', 'exec']+(['-d'] if detach else [])
        for key, value in env.items():
            argv += ['-e', key+'='+str(value)]
        argv += [node['container'], 'bash', '-lc', command]
        return self.ssh(node, shlex.join(argv))

    def ensure_container(self, name, stage):
        spec = self.config['vendors'][name]
        node = spec['nodes'][stage]
        current = self.ssh(node, shlex.join(['docker', 'inspect', node['container']]), False)
        if current.returncode == 0:
            info = json.loads(current.stdout)[0]
            if not info['State']['Running']:
                raise RuntimeError('Existing container is paused/stopped; preserving its state')
            if info['Image'] != self.inspect(node, spec['image']):
                raise RuntimeError('Container image differs from configured image')
            return info['Id']
        if node['reuse']:
            raise RuntimeError('Required existing container is missing')
        image_id = self.inspect(node, spec['image'])
        if not image_id:
            raise RuntimeError('Image is not ready')
        argv = ['docker', 'run', '-d', '--name', node['container'], '--network', 'host', '--ipc', 'host',
                '--label', 'moqe.experiment=native-pp', '-v', REMOTE+':/workspace',
                '-v', '/home/zhangjinhao/models:/models:ro']
        if name == 'ascend':
            for device in node['devices']:
                argv += ['--device', '/dev/davinci'+str(device)]
            for device in ['davinci_manager', 'devmm_svm', 'hisi_hdc']:
                argv += ['--device', '/dev/'+device]
            argv += ['-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
                     '-v', '/etc/ascend_install.info:/etc/ascend_install.info:ro']
        elif name == 'hygon':
            for device in ['/dev/kfd', '/dev/mkfd', '/dev/dri']:
                argv += ['--device', device]
            argv += ['-v', '/opt/hyhal:/opt/hyhal:ro', '--group-add', 'video', '--cap-add', 'SYS_PTRACE',
                     '--security-opt', 'seccomp=unconfined']
        else:
            argv += ['--privileged']
        argv += ['--entrypoint', 'bash', spec['image'], '-lc', 'exec sleep infinity']
        return self.ssh(node, shlex.join(argv)).stdout.strip()

    def start(self, name):
        spec = self.config['vendors'][name]
        identity = self.inspect(spec['nodes'][0], spec['image'])
        if self.inspect(spec['nodes'][1], spec['image']) != identity:
            raise RuntimeError('Both stages must use the exact same image ID')
        directory = '/workspace/artifacts/baselines/same-vendor-pp/'+name
        local = self.artifacts/name
        local.mkdir(parents=True, exist_ok=True)
        (local/'config.json').write_bytes(self.args.config.read_bytes())
        snapshot = dict(vendor=name, image=spec['image'], image_id=identity, tp=2, pp=2,
                        config_sha256=hashlib.sha256(self.args.config.read_bytes()).hexdigest(), nodes=[])
        for stage, node in enumerate(spec['nodes']):
            container_id = self.ensure_container(name, stage)
            driver_hashes = []
            for library in node.get('driver_libraries', []):
                source = library['source']
                digest = self.ssh(node, shlex.join(['sha256sum', source])).stdout.split()[0]
                existing = self.ssh(node, shlex.join(['docker', 'exec', node['container'], 'sha256sum', library['target']]), False)
                if existing.returncode or existing.stdout.split()[0] != digest:
                    self.ssh(node, shlex.join(['docker', 'cp', source, node['container']+':'+library['target']]))
                self.ssh(node, shlex.join(['docker', 'exec', node['container'], 'ln', '-sf', Path(library['target']).name, library['link']]))
                driver_hashes.append(dict(library, sha256=digest))
            snapshot['nodes'].append(dict(node, container_id=container_id, driver_hashes=driver_hashes))
            self.upload(node, self.args.config, REMOTE+'/'+REL+'/config.json')
            self.upload(node, Path(__file__).with_name('run_node.py'), REMOTE+'/'+REL+'/run_node.py')
            self.upload(node, Path(__file__).with_name('network.py'), REMOTE+'/'+REL+'/network.py')
            check = self.docker(name, stage, 'python3 /workspace/'+REL+'/run_node.py --config /workspace/'+REL+'/config.json --vendor '+name+' --stage '+str(stage)+' --check')
            print(name, stage, check.stdout.strip(), flush=True)
        (local/'deployment.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        self.upload(spec['nodes'][0], local/'deployment.json', REMOTE+'/artifacts/baselines/same-vendor-pp/'+name+'/deployment.json')
        self.upload(spec['nodes'][0], local/'config.json', REMOTE+'/artifacts/baselines/same-vendor-pp/'+name+'/config.json')
        for stage in [0, 1]:
            body = 'mkdir -p '+directory+'; python3 -u /workspace/'+REL+'/run_node.py --config /workspace/'+REL+'/config.json --vendor '+name+' --stage '+str(stage)+' > '+directory+'/stage'+str(stage)+'.log 2>&1'
            self.docker(name, stage, body, detach=True)
        return dict(status='initializing', deployment=snapshot)

    def status(self, name):
        spec = self.config['vendors'][name]
        node = spec['nodes'][0]
        directory = REMOTE+'/artifacts/baselines/same-vendor-pp/'+name
        result = {}
        for label, path in [('image', directory+'/image/status.json'), ('head', directory+'/stage0.json')]:
            read = self.ssh(node, 'cat '+shlex.quote(path), False)
            if read.returncode == 0:
                try:
                    result[label] = json.loads(read.stdout)
                except ValueError:
                    result[label] = read.stdout.strip()
        worker = self.ssh(spec['nodes'][1], 'cat '+shlex.quote(directory+'/stage1.json'), False)
        if worker.returncode == 0:
            result['worker'] = json.loads(worker.stdout)
        read = self.ssh(self.config['client'], 'cat '+shlex.quote(directory+'/performance/status.json'), False)
        if read.returncode == 0:
            result['benchmark'] = json.loads(read.stdout)
        return result

    def benchmark(self, name):
        client = self.config['client']
        with self.client_upload_lock:
            if not self.client_uploaded:
                self.upload(client, self.args.config, REMOTE+'/'+REL+'/config.json')
                self.upload(client, Path(__file__).with_name('benchmark.py'), REMOTE+'/'+REL+'/benchmark.py')
                self.upload(client, ROOT/'heterogeneous_compute/platforms/iluvatar/benchmark.sh',
                            REMOTE+'/heterogeneous_compute/platforms/iluvatar/benchmark.sh')
                self.client_uploaded = True
        result = '/workspace/artifacts/baselines/same-vendor-pp/'+name+'/performance'
        body = ('mkdir -p '+result+'; exec python3 -u /workspace/'+REL+'/benchmark.py --config /workspace/'+REL+
                '/config.json --vendor '+name+' > '+result+'/client.log 2>&1')
        self.ssh(client, shlex.join(['docker', 'exec', '-d', client['container'], 'bash', '-lc', body]))
        return dict(status='benchmark_dispatched', client=client, result=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'start', 'status', 'benchmark'])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--vendor', default='all')
    args = parser.parse_args()
    manager = Manager(args)
    names = list(manager.config['vendors']) if args.vendor == 'all' else [args.vendor]
    def perform(name):
        try:
            return name, getattr(manager, args.action)(name)
        except Exception as exc:
            return name, dict(status='failed', error=str(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(perform, name) for name in names]
        for future in concurrent.futures.as_completed(tasks):
            name, state = future.result()
            print(json.dumps(dict(vendor=name, **state), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
