"""Stream a cached Docker image over pinned SSH; retain no image tar archive."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    task = Path(config['task_directory'])
    report = Path(config['report_directory'])
    report.mkdir(parents=True, exist_ok=True)
    peer = config['peer']
    options = ['-i', str(task/'id_ed25519'), '-o', 'IdentitiesOnly=yes',
               '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias='+peer[3],
               '-o', 'UserKnownHostsFile='+str(task/'known_hosts'),
               '-p', str(peer[2]), 'zhangjinhao@'+peer[1]]
    state = dict(status='starting', bytes=0, started_at=datetime.now(timezone.utc).isoformat(),
                 image=config['image'], expected_id=config['image_id'], peer=peer[0])
    started = time.monotonic()

    def save(**values):
        state.update(values)
        state['elapsed_seconds'] = time.monotonic()-started
        state['bytes_per_second'] = state['bytes']/max(1, state['elapsed_seconds'])
        temporary = report/'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(report/'status.json')

    def remote(command, **kwargs):
        return subprocess.run(['ssh', *options, command], stdin=subprocess.DEVNULL,
                              timeout=300, **kwargs)

    save()
    try:
        inspected = remote('docker image inspect '+shlex.quote(config['image'])+' --format "{{.Id}}"',
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if inspected.returncode or inspected.stdout.strip() != config['image_id']:
            save(status='streaming')
            with (report/'transfer.log').open('wb') as log:
                sender = subprocess.Popen(['docker', 'save', config['image']],
                                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log)
                receiver = subprocess.Popen(['ssh', *options, 'docker load'],
                                            stdin=subprocess.PIPE, stdout=log, stderr=log)
                try:
                    last = 0
                    while True:
                        block = sender.stdout.read(1024*1024)
                        if not block:
                            break
                        receiver.stdin.write(block)
                        state['bytes'] += len(block)
                        if time.monotonic()-last > 5:
                            save()
                            last = time.monotonic()
                    receiver.stdin.close()
                    if sender.wait() or receiver.wait():
                        raise RuntimeError('docker save / SSH docker load failed; see transfer.log')
                finally:
                    for process in (sender, receiver):
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=15)
            inspected = remote('docker image inspect '+shlex.quote(config['image'])+' --format "{{.Id}}"',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if inspected.returncode or inspected.stdout.strip() != config['image_id']:
                raise RuntimeError('Loaded image identity differs from cached source')
        cleanup = ('MOQE_TRANSFER_TASK='+shlex.quote(config['task_name'])+
                   ' MOQE_TRANSFER_SOURCE='+shlex.quote(config['source_ip'])+
                   ' python3 '+shlex.quote(str(task/'transfer_auth.py'))+' remove')
        removed = remote(cleanup, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if removed.returncode:
            raise RuntimeError('Image verified, but task authorization cleanup failed')
        # Delete only the files generated for this exact transfer task.
        for name in ['id_ed25519', 'id_ed25519.pub', 'transfer.pub', 'known_hosts', 'config.json']:
            (task/name).unlink()
        task.rmdir()
        save(status='complete', verified_id=inspected.stdout.strip())
        print('IMAGE_TRANSFER_COMPLETE', flush=True)
        return 0
    except Exception as exc:
        save(status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
