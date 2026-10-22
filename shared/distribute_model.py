"""SCP a verified downloaded or quantized checkpoint to configured peers."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--peer', action='append', help='Transfer only the named configured peer; repeat for several peers')
    parser.add_argument('--report-dir', type=Path, help='Retain this transfer report separately from earlier runs')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    completion_marker = config.get('completion_marker', 'DOWNLOAD_COMPLETE')
    manifest_name = config.get('manifest_name', 'download_manifest.json')
    metadata_files = config.get('metadata_files', ['SHA256SUMS', 'download_manifest.json', 'DOWNLOAD_COMPLETE'])
    for name in [completion_marker, manifest_name, *metadata_files]:
        if Path(name).name != name or name in {'.', '..'}:
            raise ValueError('Checkpoint metadata must be a direct child filename')
    if not metadata_files or metadata_files[-1] != completion_marker:
        raise ValueError('Publish the completion marker last')
    source, task, report = (Path(config[k]) for k in ['source_directory', 'task_directory', 'report_directory'])
    if args.report_dir is not None:
        report = args.report_dir
    report.mkdir(parents=True, exist_ok=True)
    peers = config['peers']
    if args.peer:
        unknown = set(args.peer) - {p[0] for p in peers}
        if unknown:
            parser.error('Unknown configured peer: ' + ', '.join(sorted(unknown)))
        peers = [p for p in peers if p[0] in args.peer]
    version = subprocess.run(['ssh', '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    match = re.search(r'OpenSSH_(\d+)\.(\d+)', version.stdout+version.stderr)
    legacy = ['-O'] if match and tuple(map(int, match.groups())) >= (8, 7) else []
    state = dict(source=str(source), status='waiting_for_download', nodes={p[0]:dict(status='queued') for p in peers},
                 started_at=datetime.now(timezone.utc).isoformat())
    guard = threading.Lock()

    def save():
        temp = report/'status.json.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(report/'status.json')

    def update(alias, **values):
        with guard:
            state['nodes'][alias].update(values)
            save()

    def options(peer):
        return ['-i', str(task/'id_ed25519'), '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias='+peer[3],
                '-o', 'UserKnownHostsFile='+str(task/'known_hosts')]

    def remote(peer, command, log, timeout=1200):
        return subprocess.run(['ssh', *options(peer), '-p', str(peer[2]), 'zhangjinhao@'+peer[1], command],
                              stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=timeout)

    if args.check:
        with (report/'preflight.log').open('w') as log:
            codes = []
            for peer in peers:
                r = remote(peer, 'true', log)
                print(peer[0], r.returncode, flush=True)
                codes.append(r.returncode)
        return 1 if any(codes) else 0

    save()
    try:
        while not (source/completion_marker).exists():
            path = Path(config['download_report'])
            if path.exists():
                download = json.loads(path.read_text())
                if download.get('error'):
                    raise RuntimeError('Download failed: '+download['error'])
            time.sleep(15)
        state['status'] = 'distributing'
        save()
        manifest = json.loads((source/manifest_name).read_text())
        files = manifest['files']
        expected = list(files.items()) if isinstance(files, dict) else [(item['path'], item['sha256']) for item in files]
        for name, digest in expected:
            if Path(name).is_absolute() or '..' in Path(name).parts or len(digest) != 64:
                raise ValueError('Invalid checkpoint manifest entry')
        destination = '/home/zhangjinhao/models/'+source.name
        ownership = config['task_name']

        def transfer(peer):
            alias = peer[0]
            with (report/(alias+'.log')).open('w') as log:
                try:
                    update(alias, status='preflight')
                    dst, marker = shlex.quote(destination), shlex.quote(ownership)
                    command = ('if [ -e '+dst+' ]; then test -f '+shlex.quote(destination+'/'+completion_marker)+' || '
                               'test "$(cat '+dst+'/TRANSFER_IN_PROGRESS 2>/dev/null)" = '+marker+'; '
                               'else mkdir -p '+dst+' && printf %s '+marker+' > '+dst+'/TRANSFER_IN_PROGRESS; fi')
                    if remote(peer, command, log).returncode:
                        raise RuntimeError('Destination ownership/preflight failed')
                    for name, digest in expected:
                        target = shlex.quote(destination+'/'+name)
                        verify = 'test -f '+target+' && test "$(sha256sum '+target+' | cut -d " " -f1)" = '+shlex.quote(digest)
                        if remote(peer, verify, log).returncode == 0:
                            continue
                        # A completed directory is immutable, even when its hashes differ.
                        allowed = 'test ! -e '+shlex.quote(destination+'/'+completion_marker)+' && test "$(cat '+dst+'/TRANSFER_IN_PROGRESS)" = '+marker
                        if remote(peer, allowed, log).returncode:
                            raise RuntimeError('Refusing to overwrite existing completed model')
                        remote(peer, 'mkdir -p '+shlex.quote(str(Path(destination+'/'+name).parent)), log)
                        update(alias, status='copying', file=name)
                        for attempt in range(3):
                            copied = subprocess.run(['scp', *legacy, *options(peer), '-P', str(peer[2]),
                                str(source/name), 'zhangjinhao@'+peer[1]+':'+destination+'/'+name],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=14400)
                            if copied.returncode == 0 and remote(peer, verify, log).returncode == 0:
                                break
                            time.sleep(5)
                        else:
                            raise RuntimeError('SCP/hash failed: '+name)
                    # Publish the completion marker only after all assets verify.
                    for name in metadata_files:
                        r = subprocess.run(['scp', *legacy, *options(peer), '-P', str(peer[2]), str(source/name),
                            'zhangjinhao@'+peer[1]+':'+destination+'/'], stdin=subprocess.DEVNULL, stdout=log, stderr=log, timeout=120)
                        if r.returncode:
                            raise RuntimeError('Metadata copy failed: '+name)
                    remote(peer, 'rm -f '+dst+'/TRANSFER_IN_PROGRESS', log)
                    update(alias, status='verified', completed_at=datetime.now(timezone.utc).isoformat())
                except Exception as error:
                    update(alias, status='failed', error=str(error))

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers']) as executor:
            list(executor.map(transfer, peers))
        state['status'] = 'complete' if all(n['status']=='verified' for n in state['nodes'].values()) else 'completed_with_errors'
    except Exception as error:
        state.update(status='failed', error=str(error))
    finally:
        # Remove only this task's restricted authorization and private transfer key.
        with (report/'authorization-cleanup.log').open('w') as log:
            for peer in peers:
                try:
                    cmd = 'MOQE_TRANSFER_TASK='+shlex.quote(config['task_name'])+' MOQE_TRANSFER_SOURCE='+shlex.quote(config['source_ip'])
                    cmd += ' python3 /home/zhangjinhao/.cache/'+config['task_name']+'/transfer_auth.py remove'
                    update(peer[0], authorization_removed=remote(peer, cmd, log).returncode==0)
                except Exception as error:
                    update(peer[0], cleanup_error=str(error))
        for name in ['id_ed25519', 'id_ed25519.pub', 'known_hosts']:
            (task/name).unlink()
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if state['status']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
