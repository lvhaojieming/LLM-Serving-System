"""Bounded, read-only telemetry for the retained heterogeneous system benchmark.

Host collectors reuse Docker and vendor SMI tools. A container-side sampler reads
only its own processes. The gateway-side packet observer keeps request IDs and
socket endpoints, never prompt/response bodies; it does not alter forwarding.
"""
import argparse
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import queue
import re
import signal
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heterogeneous_compute.serving import expand_config, generation_workers, router_configs


# Client event times supply latency distributions. Repeatedly retaining every
# backend histogram bucket adds substantial collector CPU and JSON traffic but
# does not improve that measurement. Keep the gauges/counters used by the run.
METRIC_NAMES = {
    'worker': frozenset({
        'vllm:num_requests_running', 'vllm:num_requests_waiting',
        'vllm:num_requests_swapped', 'vllm:request_success_total',
        'vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc',
        'vllm:cpu_cache_usage_perc', 'vllm:num_preemptions_total',
        'vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
        'vllm:engine_sleep_state',
    }),
    'gateway': frozenset({
        'heteroserve_gateway_inflight', 'heteroserve_gateway_router_calls_total',
        'heteroserve_gateway_rejected_total',
    }),
    'router': frozenset({
        'vllm_router_active_workers', 'vllm_router_worker_health',
        'vllm_router_processed_requests_total', 'vllm_router_requests_total',
        'vllm_router_policy_decisions_total', 'vllm_router_cb_outcomes_total',
        'vllm_router_cb_state',
    }),
}


def fetch(url, body=None, timeout=3):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
        text = r.read().decode()
    try:
        return json.loads(text)
    except ValueError:
        return text


def bounded(argv, timeout=3):
    """Do not block forever on a vendor ioctl, even after SIGKILL."""
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        return {"returncode": p.returncode, "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace")}
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        try:
            p.communicate(timeout=.2)
        except subprocess.TimeoutExpired:
            pass
        return {"timeout": True, "pid": p.pid, "stdout": "", "stderr": "bounded vendor query timed out; disabled for this run"}


def parse_metrics(raw, names=None):
    rows = []
    for line in raw.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([^\s{]+)(?:\{(.*)\})?\s+([^\s]+)', line)
        if match and (names is None or match[1] in names):
            try:
                value = float(match[3])
            except ValueError:
                continue
            rows.append({"name": match[1], "labels": dict(re.findall(r'(\w+)="([^"\\]*(?:\\.[^"\\]*)*)"', match[2] or "")), "value": value})
    return rows


def metric_sum(rows, name, exclude_reasons=()):
    selected = [r["value"] for r in rows if r["name"] == name and r["labels"].get("finished_reason") not in exclude_reasons]
    return sum(selected) if selected else None


def router_worker_snapshot(reports):
    """Retain pool identity while exposing the legacy combined workers list."""
    result={'workers':[],'pools':{},'errors':{}}
    for pool,report in reports.items():
        if not isinstance(report,dict) or not isinstance(report.get('workers'),list):
            result['errors'][pool]=report.get('error','Missing workers list') if isinstance(report,dict) else 'Invalid router workers response'
            continue
        result['pools'][pool]=report
        for worker in report['workers']:
            result['workers'].append({**worker,'router_pool':pool})
    return result


def process_sample():
    """CPU ticks, RSS and socket counts for processes in this container only."""
    procs = {}
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / 'stat').read_text().rsplit(')', 1)[1].split()
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            role = ('gateway' if 'heteroserve.gateway' in cmd else 'router' if ('vllm_router.launch_router' in cmd or cmd.strip() == 'vllm::router') else
                    'benchmark' if ('benchmark.py' in cmd or 'benchmark_telemetry.py' in cmd) else 'model' if 'serve ' in cmd else None)
            procs[int(d.name)] = dict(pid=int(d.name), ppid=int(stat[1]), role=role,
                start_ticks=int(stat[19]), cpu_seconds=(int(stat[11])+int(stat[12]))/os.sysconf('SC_CLK_TCK'),
                rss_bytes=int(stat[21])*os.sysconf('SC_PAGE_SIZE'), state=stat[0])
            if role=='model':
                argv=shlex.split(cmd);parameters={}
                for flag in ['--tensor-parallel-size','--pipeline-parallel-size','--port','--max-model-len']:
                    if flag in argv and argv.index(flag)+1<len(argv):parameters[flag]=argv[argv.index(flag)+1]
                parameters['--enforce-eager']='--enforce-eager' in argv
                parameters['--disable-custom-all-reduce']='--disable-custom-all-reduce' in argv
                procs[int(d.name)]['inference_parameters']=parameters
        except (OSError, ValueError, IndexError):
            continue
    for _ in range(8):
        for p in procs.values():
            if p['role'] is None and p['ppid'] in procs:
                p['role'] = procs[p['ppid']]['role']
    tcp = {}
    for file in ['/proc/net/tcp', '/proc/net/tcp6']:
        try:
            for line in Path(file).read_text().splitlines()[1:]:
                s = line.split()
                tcp[s[9]] = {'state': s[3], 'local': s[1], 'remote': s[2]}
        except OSError:
            pass
    for pid, p in procs.items():
        sockets = {}
        try:
            for fd in Path(f'/proc/{pid}/fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith('socket:[') and target[8:-1] in tcp:
                        sockets[target[8:-1]] = tcp[target[8:-1]]
                except OSError:
                    pass
        except OSError:
            pass
        p['tcp_connections'] = len(sockets)
        p['tcp_established'] = sum(s['state'] == '01' for s in sockets.values())
    return {'processes': list(procs.values()), 'time': time.time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def ipv4_tcp(frame):
    if len(frame) < 54 or frame[12:14] != b'\x08\x00':
        return None
    ihl = (frame[14] & 15) * 4
    if frame[14] >> 4 != 4 or ihl < 20 or frame[23] != 6 or len(frame) < 14+ihl+20:
        return None
    # A non-initial IP fragment has no TCP header at this offset.
    if struct.unpack_from('!H', frame, 20)[0] & 0x1fff:
        return None
    base = 14+ihl
    sport, dport, seq = struct.unpack_from('!HHI', frame, base)
    thl = (frame[base+12] >> 4)*4
    total = struct.unpack_from('!H', frame, 16)[0]
    if thl < 20 or total < ihl+thl or len(frame) < base+thl:
        return None
    size=total-ihl-thl
    payload=frame[base+thl:min(len(frame),14+total)]
    return dict(src=socket.inet_ntoa(frame[26:30]), dst=socket.inet_ntoa(frame[30:34]),
                sport=sport, dport=dport, seq=seq, flags=frame[base+13],size=size,
                payload=payload, truncated=len(payload)<size)


def packet_filter(ports, header_ports):
    """Copy headers for counting; copy request headers only toward workers.

    IPv4 total length still gives original TCP payload bytes when a response's
    copied snapshot is short. The larger request snapshot covers the complete
    supported HTTP header, including headers split across TCP segments.
    """
    ins=[(0x28,0,0,12),(0x15,0,0,0x800),(0x30,0,0,23),(0x15,0,0,6),
         (0xb1,0,0,14),(0x48,0,0,16)]
    header_start=len(ins)
    ins += [(0x15,0,0,p) for p in sorted(header_ports)]
    dst_start=len(ins)
    ins += [(0x15,0,0,p) for p in sorted(ports-header_ports)]
    ins += [(0x48,0,0,14)]
    src_start=len(ins)
    ins += [(0x15,0,0,p) for p in sorted(ports)]
    reject,metadata,header=len(ins),len(ins)+1,len(ins)+2
    ins += [(0x06,0,0,0),(0x06,0,0,256),(0x06,0,0,16384)]
    ins[1]=(0x15,0,reject-2,0x800)
    ins[3]=(0x15,0,reject-4,6)
    for i in range(header_start,header_start+len(header_ports)):
        code,_,_,port=ins[i];ins[i]=(code,header-i-1,0,port)
    for i in list(range(dst_start,dst_start+len(ports-header_ports)))+list(range(src_start,src_start+len(ports))):
        code,_,_,port=ins[i];ins[i]=(code,metadata-i-1,0,port)
    return ins


class PacketObserver:
    """Observe only the owned Gateway/Router/worker TCP ports with kernel BPF."""
    def __init__(self, config):
        self.workers = {w['base_url'].removesuffix('/v1') for w in generation_workers(config).values()}
        self.gateway_port = config['gateway']['port']
        self.router_ports={router['port'] for router in router_configs(config).values()}
        self.worker_endpoints={(u[7:].rsplit(':',1)[0],int(u.rsplit(':',1)[1])) for u in self.workers}
        self.worker_ports={port for _,port in self.worker_endpoints}
        embedding={b['base_url'].removesuffix('/v1') for b in config['backends'].values() if b.get('task')=='embed'}
        self.embedding_endpoints={(u[7:].rsplit(':',1)[0],int(u.rsplit(':',1)[1])) for u in embedding}
        self.ports = {self.gateway_port} | self.router_ports | self.worker_ports | {port for _,port in self.embedding_endpoints}
        self.routes = queue.SimpleQueue()
        self.stop = threading.Event()
        self.bytes = {'gateway_rx_tcp_payload': 0, 'gateway_tx_tcp_payload': 0,
                      'router_rx_tcp_payload': 0, 'router_tx_tcp_payload': 0}
        self.error = None
        self.flows = {}
        self.network_flows = {}
        self.gateway_address = config['gateway']['address']
        self.capture_info = {}

    def start(self):
        try:
            # ETH_P_ALL is necessary to observe locally transmitted packets;
            # the attached BPF immediately restricts this to owned IPv4/TCP ports.
            self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
            requested=16*1024*1024
            self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,requested)
            self.capture_info={'requested_receive_buffer_bytes':requested,
                               'metadata_snapshot_bytes':256,'request_snapshot_bytes':16384}
            # Linux reports twice the userspace request, for bookkeeping. Try
            # the privileged override only when the ordinary request was capped;
            # this changes this socket, not the host's shared sysctl settings.
            if self.sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)<2*requested:
                try:
                    self.sock.setsockopt(socket.SOL_SOCKET,getattr(socket,'SO_RCVBUFFORCE',33),requested)
                    self.capture_info['receive_buffer_forced']=True
                except OSError as exc:
                    self.capture_info['receive_buffer_override_error']=str(exc)
            self.capture_info['actual_receive_buffer_bytes']=self.sock.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF)
            ins=packet_filter(self.ports,self.worker_ports)
            class Filter(ctypes.Structure):
                _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
            class Program(ctypes.Structure):
                _fields_=[('length',ctypes.c_ushort),('filters',ctypes.POINTER(Filter))]
            array=(Filter*len(ins))(*(Filter(*r) for r in ins)); program=Program(len(ins),array)
            self.sock.setsockopt(socket.SOL_SOCKET,26,bytes(program))
            self.sock.settimeout(1)
            threading.Thread(target=self.loop, daemon=True).start()
        except Exception as exc:
            self.error = repr(exc)

    def request_header(self,p,flow,target):
        """Reassemble bounded HTTP headers, respecting TCP sequence/overlap.

        Payload snapshots may end inside a large request body; that is harmless
        once the header ends. A missing captured header byte is an error rather
        than an invented association. Response bodies are never accumulated.
        """
        data=p['payload']
        if data.startswith(b'POST ') and (flow not in self.flows or self.flows[flow]['seq']!=p['seq']):
            self.flows[flow]={'seq':p['seq'],'segments':{}}
        state=self.flows.get(flow)
        if state is None:return
        offset=(p['seq']-state['seq']) & 0xffffffff
        if offset>0x7fffffff:return  # segment before this HTTP request
        if offset>=16384:
            raise RuntimeError('Worker HTTP request header exceeds the 16384-byte capture limit')
        state['segments'][offset]=data[:16384-offset]
        header=bytearray()
        for position,segment in sorted(state['segments'].items()):
            if position>len(header):break  # wait for an out-of-order segment
            overlap=min(len(segment),len(header)-position)
            if header[position:position+overlap]!=segment[:overlap]:
                raise RuntimeError('Conflicting TCP retransmission in worker request header')
            header.extend(segment[overlap:])
            if b'\r\n\r\n' in header:break
        if b'\r\n\r\n' in header:
            header=header.split(b'\r\n\r\n',1)[0]
            match=re.search(br'(?im)^x-request-id:\s*([^\r\n]+)',header)
            if match:
                self.routes.put({'request_id':match[1].decode(errors='replace').strip(),
                                 'worker':target,'observed_at':time.time()})
            del self.flows[flow]
        elif p['truncated'] or len(header)>=16384:
            raise RuntimeError('Worker HTTP request header was truncated; route attribution is incomplete')
        if len(self.flows)>4096:
            raise RuntimeError('Worker request header reassembly capacity exceeded')

    def observe_packet(self,p):
        worker_out=(p['dst'],p['dport']) in self.worker_endpoints
        worker_in=(p['src'],p['sport']) in self.worker_endpoints
        embedding_out=(p['dst'],p['dport']) in self.embedding_endpoints
        embedding_in=(p['src'],p['sport']) in self.embedding_endpoints
        if not (worker_out or worker_in or embedding_out or embedding_in or p['sport'] in self.router_ports or p['sport']==self.gateway_port
                or p['dport'] in self.router_ports or p['dport']==self.gateway_port):
            return
        n=p['size'];flow=(p['src'],p['sport'],p['dst'],p['dport']);reverse=(p['dst'],p['dport'],p['src'],p['sport'])
        if (p['flags']&2 and not p['flags']&16) or p['payload'].startswith(b'GET '):
            self.network_flows.pop(flow,None);self.network_flows.pop(reverse,None)
            self.flows.pop(flow,None);self.flows.pop(reverse,None)
        if p['payload'].startswith(b'POST /v1/'):
            if p['dport']==self.gateway_port:self.network_flows[flow]='client_gateway'
            elif p['dport'] in self.router_ports:self.network_flows[flow]='gateway_router'
            elif embedding_out and p['src'] in {self.gateway_address,'127.0.0.1'}:self.network_flows[flow]='gateway_embedding'
            elif worker_out and p['src'] in {self.gateway_address,'127.0.0.1'}:self.network_flows[flow]='router_worker'
        kind=self.network_flows.get(flow);back=self.network_flows.get(reverse)
        if kind=='client_gateway' or back in {'gateway_router','gateway_embedding'}:self.bytes['gateway_rx_tcp_payload']+=n
        if back=='client_gateway' or kind in {'gateway_router','gateway_embedding'}:self.bytes['gateway_tx_tcp_payload']+=n
        if kind=='gateway_router' or back=='router_worker':self.bytes['router_rx_tcp_payload']+=n
        if back=='gateway_router' or kind=='router_worker':self.bytes['router_tx_tcp_payload']+=n
        if worker_out and n:
            self.request_header(p,flow,f"http://{p['dst']}:{p['dport']}")
        if p['flags']&5:
            self.network_flows.pop(flow,None);self.network_flows.pop(reverse,None)
            self.flows.pop(flow,None);self.flows.pop(reverse,None)

    def loop(self):
        while not self.stop.is_set():
            try:
                frame, addr = self.sock.recvfrom(65535)
                # Loopback is seen twice; retain the host-delivered copy only.
                if addr[0] == 'lo' and addr[2] == 4:
                    continue
                p = ipv4_tcp(frame)
                if p is None:
                    continue
                self.observe_packet(p)
            except socket.timeout:
                pass
            except Exception as exc:
                self.error=repr(exc)
                break

    def sample(self):
        rows=[]
        while not self.routes.empty():
            rows.append(self.routes.get())
        packet_stats={}
        try:
            received,dropped=struct.unpack_from('II',self.sock.getsockopt(263,6,12))
            packet_stats={'received':received,'dropped':dropped}
        except (OSError,AttributeError):
            pass
        return {'routes':rows,'tcp_payload_bytes':dict(self.bytes),'error':self.error,
                'packet_statistics':packet_stats,
                'capture':dict(self.capture_info),
                'network_scope':'Owned service TCP payload bytes on this host, including Gateway Embedding calls; includes retransmissions, excludes IP/TCP headers. No packet bodies retained.'}


def device_values(vendor, report):
    if report.get('timeout') or report.get('returncode') != 0:
        return []
    raw=report['stdout']; result=[]
    if vendor=='musa':
        try:
            for g in json.loads(raw)['GPU']:
                result.append(dict(device=int(g['Index']),utilization_percent=float(g['Utilization']['Gpu'].rstrip('%')),
                    memory_used_mib=float(g['FB Memory Usage']['Used'].replace('MiB','')),
                    memory_total_mib=float(g['FB Memory Usage']['Total'].replace('MiB',''))))
        except (ValueError,KeyError,TypeError):
            pass
    elif vendor=='ascend':
        for line in raw.splitlines():
            parts=[x.strip() for x in line.split('|')]
            if len(parts)>=5 and re.fullmatch(r'\d+\s+\d+',parts[1]) and re.search(r'[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d',parts[2]):
                nums=re.findall(r'\d+',parts[3]); mem=re.findall(r'\d+',parts[3])
                # Usage columns share the same pipe-delimited cell in 25.5.5.
                if len(mem)>=5:
                    result.append(dict(device=int(parts[1].split()[1]),utilization_percent=float(nums[0]),
                        memory_used_mib=float(mem[-2]),memory_total_mib=float(mem[-1])))
    elif vendor=='iluvatar':
        for line in raw.splitlines():
            values=[v.strip() for v in line.split(',')]
            if len(values)==4:
                try:
                    result.append(dict(device=int(values[0]),utilization_percent=float(values[1]),memory_used_mib=float(values[2]),memory_total_mib=float(values[3])))
                except ValueError:
                    pass
    return result


def collect(config_path,node,central,max_seconds=86400):
    config=expand_config(json.loads(Path(config_path).read_text()))
    specs=[s for s in config['services'].values() if s['node']==node]
    container=specs[0]['container']
    vendor='ascend' if node.startswith('ascend') else 'musa' if node.startswith('s5000') else 'iluvatar'
    script='/workspace/heterogeneous_compute/benchmark_telemetry.py'
    metrics={name:w for name,w in generation_workers(config).items() if w['node']==node}
    gateway_node=config['services']['gateway']['node']
    disabled_smi=False; started=time.monotonic(); failures=0
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=12)
    while time.monotonic()-started<max_seconds:
        before=time.monotonic(); sample={'node':node,'vendor':vendor,'time':time.time(),'workers':{}}
        try:
            status=fetch(central+'/status')
            if status.get('finished'):
                break
            failures=0
        except Exception:
            failures+=1
            if failures>24:
                break
            time.sleep(5);continue
        jobs={}
        jobs['container']=pool.submit(bounded,['docker','inspect','--format','{{json .}}',container],3)
        jobs['process']=pool.submit(bounded,['docker','exec',container,'python3',script,'process'],3)
        if not disabled_smi:
            cmd=(['npu-smi','info'] if vendor=='ascend' else ['mthreads-gmi','-q','--json'] if vendor=='musa' else
                 ['docker','exec',container,'ixsmi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'])
            jobs['device']=pool.submit(bounded,cmd,3)
        for name,w in metrics.items():
            jobs['worker:'+name]=pool.submit(fetch,w['base_url'][:-3]+'/metrics')
        router_reports={}
        if node==gateway_node:
            sample['router_metrics']=[]
            for name,router in router_configs(config).items():
                label=router.get('pool',name)
                host=router.get('host','127.0.0.1')
                jobs['router_workers:'+label]=pool.submit(fetch,f"http://{host}:{router['port']}/workers")
                jobs['router_metrics:'+label]=pool.submit(fetch,f"http://{host}:{router['metrics_port']}/metrics")
            jobs['gateway_metrics']=pool.submit(fetch,f"http://127.0.0.1:{config['gateway']['port']}/metrics")
        for name,future in jobs.items():
            try:
                value=future.result(timeout=5)
                if name.startswith('worker:'):
                    rows=parse_metrics(value,METRIC_NAMES['worker'])
                    sample['workers'][name[7:]]={'metrics':rows,'running':metric_sum(rows,'vllm:num_requests_running'),
                        'waiting':metric_sum(rows,'vllm:num_requests_waiting'),
                        'completed':metric_sum(rows,'vllm:request_success_total',('abort','error'))}
                elif name.startswith('router_workers:'):
                    router_reports[name.split(':',1)[1]]=value
                elif name.startswith('router_metrics:'):
                    label=name.split(':',1)[1]
                    rows=parse_metrics(value,METRIC_NAMES['router'])
                    sample['router_metrics'].extend({**row,'labels':{**row['labels'],'router_pool':label}} for row in rows)
                elif name=='gateway_metrics':
                    sample[name]=parse_metrics(value,METRIC_NAMES['gateway'])
                elif name=='container':
                    data=json.loads(value['stdout']); state=data['State']
                    sample['container']={'id':data['Id'],'image':data['Image'],'restart_count':data['RestartCount'],
                        'running':state['Running'],'oom_killed':state['OOMKilled'],'started_at':state['StartedAt'],
                        'limits':{k:data['HostConfig'].get(k) for k in ['Memory','NanoCpus','CpuQuota','PidsLimit']}}
                elif name=='process':
                    sample['process']=json.loads(value['stdout'])
                elif name=='device':
                    disabled_smi=bool(value.get('timeout'))
                    sample['device']={'values':device_values(vendor,value),'raw':value,'disabled_after_timeout':disabled_smi}
                else:
                    sample[name]=value
            except Exception as exc:
                if name.startswith('router_workers:'):
                    router_reports[name.split(':',1)[1]]={'error':repr(exc)}
                else:
                    sample[name]={'error':repr(exc)}
        if node==gateway_node:
            sample['router_workers']=router_worker_snapshot(router_reports)
        if disabled_smi and 'device' not in sample:
            sample['device']={'values':[],'unavailable':'First bounded SMI query timed out; not retried during this run'}
        try:
            fetch(central+'/telemetry',sample)
        except Exception:
            pass
        time.sleep(max(0,5-(time.monotonic()-before)))
    pool.shutdown(wait=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['collect','process','observe'])
    parser.add_argument('--config',default=str(ROOT/'heterogeneous_compute/configs/serving.json'))
    parser.add_argument('--node');parser.add_argument('--central');parser.add_argument('--max-seconds',type=int,default=86400)
    args=parser.parse_args()
    if args.action=='process':
        print(json.dumps(process_sample()));return
    if args.action=='collect':
        collect(args.config,args.node,args.central,args.max_seconds);return
    config=expand_config(json.loads(Path(args.config).read_text()))
    observer=PacketObserver(config);observer.start();failures=0
    while True:
        try:
            finished=fetch(args.central+'/status').get('finished')
            fetch(args.central+'/trace',observer.sample());failures=0
            if finished:break
        except Exception:
            failures+=1
            if failures>24:break
        time.sleep(2)
    observer.stop.set()


if __name__=='__main__':
    main()
