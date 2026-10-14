import os
from pathlib import Path
import sys

task_name = os.environ.get('MOQE_TRANSFER_TASK', 'moqe-model-transfer')
assert task_name in {'moqe-model-transfer', 'moqe-image-transfer', 'moqe-qwen3-8b-transfer', 'moqe-qwen3-32b-transfer'}
source_ip = os.environ.get('MOQE_TRANSFER_SOURCE', '192.168.162.23')
import ipaddress
ipaddress.ip_address(source_ip)
task = Path.home() / '.cache' / task_name
public = (task / 'transfer.pub').read_bytes().strip()
assert public.startswith(b'ssh-ed25519 ') and public.endswith(task_name.encode())
line = ('from="' + source_ip + '",restrict ').encode() + public
sshdir = Path.home() / '.ssh'
auth = sshdir / 'authorized_keys'
if sys.argv[1] == 'add':
    if not sshdir.exists(): sshdir.mkdir(mode=0o700)
    original = auth.read_bytes() if auth.exists() else b''
    if line not in original.splitlines():
        with auth.open('ab') as stream:
            if original and not original.endswith(b'\n'): stream.write(b'\n')
            stream.write(line+b'\n')
        if not original: os.chmod(auth,0o600)
    print('AUTHORIZED')
else:
    if auth.exists():
        cleaned = b''.join(x for x in auth.read_bytes().splitlines(keepends=True) if not x.rstrip(b'\r\n').endswith(public))
        auth.write_bytes(cleaned)
        assert not any(x.endswith(public) for x in auth.read_bytes().splitlines())
    (task/'transfer.pub').unlink()
    Path(__file__).unlink()
    task.rmdir()
    print('AUTHORIZATION_REMOVED')
