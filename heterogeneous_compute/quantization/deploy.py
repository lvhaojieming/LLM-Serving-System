"""Reuse the existing container, INT4 layout adapter and official benchmark.

One configuration drives AWQ/GPTQ and the matched FP16 TP1 control. The local
controller waits for each official benchmark process, then drains only its server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import sys
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RELATIVE = 'heterogeneous_compute/quantization/deploy.py'
REMOTE = '/home/zhangjinhao/LLM-Serving-System'
CONFIG = '/workspace/heterogeneous_compute/quantization/deployment.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def profile_dir(config, algorithm):
    return ROOT/config['result_dir']/algorithm


def native_layers(config, algorithm):
    """Compare actual checkpoint Q/MLP down projections to CPU FP32 math."""
    import numpy as np
    import torch
    import torch_npu
    from safetensors import safe_open
    from shared.ascend_int4.layout import decode, repack, matmul
    torch.set_num_threads(8)
    torch.manual_seed(config['benchmark']['seed'])
    model = Path(config['profiles'][algorithm]['model'])
    index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    quant = json.loads((model/'config.json').read_text())['quantization_config']
    if quant['quant_method'] != algorithm or quant['bits'] != 4 or quant['group_size'] != 128:
        raise ValueError('Checkpoint quantization configuration differs')
    results = []
    for projection in ['self_attn.q_proj','mlp.down_proj']:
        prefix = 'model.layers.0.'+projection
        tensors = {}
        for suffix in ['qweight','qzeros','scales','g_idx']:
            key = prefix+'.'+suffix
            if key in index:
                with safe_open(model/index[key], framework='pt', device='cpu') as source:
                    tensors[suffix] = source.get_tensor(key)
        q,z = decode(tensors['qweight'],tensors['qzeros'],algorithm)
        if algorithm == 'gptq' and not torch.equal(tensors['g_idx'].long(),torch.arange(q.shape[0])//128):
            raise ValueError('GPTQ g_idx is not sequential')
        scale = tensors['scales']
        if scale.dtype != torch.float16:
            raise ValueError('Preserving scales requires FP16 for this checkpoint')
        weight = (torch.from_numpy(q.astype(np.float32))-torch.from_numpy(z).repeat_interleave(128,dim=0))*scale.float().repeat_interleave(128,dim=0)
        packed,scales,offset = repack(q,z,scale,torch.device('npu:0'))
        checks = []
        for tokens in [1,32]:
            x = (torch.randn(tokens,q.shape[0])*.1).half()
            reference = x.float()@weight
            output = matmul(x.npu(),packed,scales,offset,128).cpu().float()
            relative = float(torch.linalg.vector_norm(output-reference)/torch.linalg.vector_norm(reference))
            finite = bool(torch.isfinite(output).all())
            checks.append(dict(tokens=tokens,finite=finite,relative_l2_error=relative))
            if not finite or relative >= .005:
                raise ValueError('Native INT4 operator numerical check failed')
        results.append(dict(projection=projection,k=q.shape[0],n=q.shape[1],checks=checks,passed=True))
        del packed,scales,offset,weight,tensors,q,z
        torch.npu.empty_cache()
    report = dict(algorithm=algorithm,passed=True,operator='torch_npu.npu_weight_quant_batchmatmul',
                  activation='FP16',results=results,scope='Layer correctness only, not inference latency or model PPL')
    save(profile_dir(config,algorithm)/'layers.json',report)
    print(json.dumps(report),flush=True)


def start_server(config, algorithm):
    directory = profile_dir(config,algorithm)
    marker = directory/'server.json'
    spec = config['profiles'][algorithm]
    directory.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ,ASCEND_RT_VISIBLE_DEVICES=str(config['device']),
               VLLM_WORKER_MULTIPROC_METHOD='spawn',OMP_NUM_THREADS='4',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1')
    env['PYTHONPATH'] = '/workspace/shared/ascend_int4:/workspace'+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['MOQE_ASCEND_INT4_ADAPTER'] = '0' if algorithm == 'fp16' else '1'
    argv = ['python3','-m','vllm.entrypoints.cli.main','serve',spec['model'],
        '--served-model-name',spec['served_model'],'--host',config['host'],'--port',str(spec['port']),
        '--dtype',config['dtype'],'--tensor-parallel-size','1','--pipeline-parallel-size','1',
        '--max-model-len',str(config['max_model_len']),'--max-num-seqs',str(config['max_num_seqs']),
        '--max-num-batched-tokens',str(config['max_num_batched_tokens']),
        '--gpu-memory-utilization',str(config['gpu_memory_utilization']),
        '--enforce-eager','--no-enable-prefix-caching','--seed',str(config['benchmark']['seed'])]
    if algorithm != 'fp16':
        argv += ['--quantization','moqe_ascend_int4']
    if marker.exists():
        previous = json.loads(marker.read_text())
        stat = Path('/proc/'+str(previous['pid'])+'/stat')
        if stat.exists() and stat.read_text().split()[2] != 'Z':
            actual = Path('/proc/'+str(previous['pid'])+'/cmdline').read_bytes().split(b'\0')
            actual = [a.decode() for a in actual if a]
            if actual[1:] != argv[1:]:
                raise RuntimeError('Existing profile PID or serving configuration differs')
            print(json.dumps(dict(pid=previous['pid'],algorithm=algorithm,reused=True)),flush=True)
            return
    with socket.socket() as sock:
        sock.bind((config['host'],spec['port']))
    with (directory/'server.log').open('a') as sink:
        sink.write(json.dumps(dict(event='w4a16_serve_begin',algorithm=algorithm,argv=argv))+'\n')
        sink.flush()
        process = subprocess.Popen(argv,env=env,stdout=sink,stderr=sink,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
    save(marker,dict(pid=process.pid,argv=argv,algorithm=algorithm,started_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(pid=process.pid,algorithm=algorithm,port=spec['port'])),flush=True)


def stop_server(config, algorithm):
    marker = profile_dir(config,algorithm)/'server.json'
    if not marker.exists():
        return
    previous = json.loads(marker.read_text())
    pid = previous['pid']
    command = Path('/proc/'+str(pid)+'/cmdline')
    if command.exists():
        argv = command.read_bytes().split(b'\0')
        spec = config['profiles'][algorithm]
        if spec['model'].encode() not in argv or spec['served_model'].encode() not in argv or b'vllm.entrypoints.cli.main' not in argv:
            raise ValueError('PID ownership changed; refusing to stop it')
        os.killpg(pid,signal.SIGTERM)
        deadline = time.monotonic()+30
        while command.exists() and time.monotonic()<deadline:
            # A reparented zombie cannot hold GPU resources.
            if Path('/proc/'+str(pid)+'/stat').read_text().split()[2] == 'Z':
                break
            time.sleep(.5)
        if command.exists() and Path('/proc/'+str(pid)+'/stat').read_text().split()[2] != 'Z':
            os.killpg(pid,signal.SIGKILL)
    print(json.dumps(dict(algorithm=algorithm,stopped_owned_pid=pid)),flush=True)


def probe(config, algorithm):
    spec = config['profiles'][algorithm]
    base = 'http://'+config['host']+':'+str(spec['port'])
    marker=profile_dir(config,algorithm)/'server.json'
    if marker.exists():
        pid=json.loads(marker.read_text())['pid']
        stat=Path('/proc/'+str(pid)+'/stat')
        if not stat.exists() or stat.read_text().split()[2]=='Z':
            print(json.dumps(dict(algorithm=algorithm,status='server_exited',log=str(marker.with_name('server.log')))))
            raise SystemExit(2)
    health = urllib.request.urlopen(base+'/health',timeout=5).status
    results = []
    for prompt,expected in [('2+3等于多少？只输出数字。','5'),('法国的首都是哪里？只回答城市名。','巴黎'),
                            ('What is 7 minus 4? Reply with only the digit.','3')]:
        body = dict(model=spec['served_model'],messages=[dict(role='user',content=prompt)],
                    max_tokens=16,temperature=0,chat_template_kwargs=dict(enable_thinking=False))
        request = urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),
                                         headers={'Content-Type':'application/json'})
        response = json.load(urllib.request.urlopen(request,timeout=120))
        text = response['choices'][0]['message']['content'].strip()
        results.append(dict(prompt=prompt,text=text,expected=expected,passed=text==expected))
    log = (profile_dir(config,algorithm)/'server.log').read_text(errors='replace')
    log = log[log.rfind('"event": "w4a16_serve_begin"'):]
    prepared = [json.loads(line[line.index('{'):]) for line in log.splitlines() if '"event": "w4a16_layer_prepared"' in line]
    executed = sum('"event": "w4a16_operator_executed"' in line for line in log.splitlines())
    proof = algorithm=='fp16' or (len(prepared)==160 and executed==160 and
            all(r['packed_dtype']=='torch.int32' and r['scale_dtype']=='torch.float16' and
                r['packed_bytes']==r['k']*r['n']//2 for r in prepared))
    report = dict(algorithm=algorithm,passed=health==200 and all(r['passed'] for r in results) and proof,
                  requests=results,prepared_fused_linears=len(prepared),executed_int4_linears=executed,
                  operator_proof_passed=proof,scope='Limited real-decode acceptance; no PPL or general quality claim')
    save(profile_dir(config,algorithm)/'acceptance.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    if not report['passed']:
        raise ValueError('Real decode or INT4 execution proof failed')


def ssh(ssh_config,node,argv):
    return subprocess.run(['ssh','-F',str(ssh_config),'-o','BatchMode=yes','-o','ConnectTimeout=10',
                           '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4',node,shlex.join(argv)],
                          text=True,encoding='utf-8',capture_output=True)


def orchestrate(config,args):
    """Local detached controller; official benchmark waits are blocking, not polling."""
    output = ROOT/config['result_dir']
    state = dict(status='starting',controller_pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),order=config['order'],profiles={})
    marker = output/'controller.json'
    if marker.exists():
        raise RuntimeError('Refusing duplicate controller or overwrite of retained task status')
    def publish(**values):
        state.update(values);save(marker,state);print(json.dumps(values),flush=True)
    def remote_action(action,algorithm,required=True):
        result=ssh(args.ssh_config,config['node'],['docker','exec',config['container'],'python3',
                    '/workspace/'+RELATIVE,'--config',CONFIG,'--action',action,'--algorithm',algorithm])
        if result.returncode and required:
            raise RuntimeError(action+': '+result.stderr[-2000:]+'\n'+result.stdout[-1500:])
        return result
    publish()
    try:
        for algorithm in config['order']:
            publish(status='verifying_layers',current=algorithm)
            if algorithm!='fp16':
                remote_action('layers',algorithm)
            remote_action('start',algorithm)
            try:
                publish(status='loading_model',current=algorithm)
                deadline=time.monotonic()+900
                while time.monotonic()<deadline:
                    result=remote_action('probe',algorithm,required=False)
                    if result.returncode==0:
                        break
                    if result.returncode==2:
                        raise RuntimeError('Model process exited; see '+result.stdout[-1000:])
                    # Readiness only; benchmark progress is never continuously polled.
                    time.sleep(5)
                else:
                    raise TimeoutError('W4A16 model readiness timeout; see server.log')
                spec,client=config['profiles'][algorithm],config['client']
                result_dir='/workspace/'+config['result_dir']+'/'+algorithm+'/performance'
                values=dict(MODEL_PATH='/models/Qwen3-14B',SERVED_MODEL=spec['served_model'],
                    DATASET_PATH='/workspace/artifacts/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json',
                    SERVER_CONFIG=CONFIG,BASE_URL='http://'+config['host']+':'+str(spec['port']),RESULT_DIR=result_dir,
                    CONCURRENCIES=' '.join(map(str,config['benchmark']['concurrencies'])),
                    NUM_PROMPTS=config['benchmark']['num_prompts'],NUM_WARMUPS=config['benchmark']['num_warmups'],
                    OUTPUT_LEN=config['benchmark']['output_len'],BENCHMARK_SEED=config['benchmark']['seed'],
                    TEMPERATURE=0,CAPTURE_DEVICE_STATS='false')
                body='from pathlib import Path\np=Path('+repr(result_dir)+')\np.mkdir(parents=True,exist_ok=True)\n'
                body+='import hashlib\nassert hashlib.sha256(Path('+repr(values['DATASET_PATH'])+').read_bytes()).hexdigest()=="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4", "Dataset differs from retained baseline"\n'
                body+='(p/"benchmark.env").write_text('+repr(''.join(k+'='+shlex.quote(str(v))+'\n' for k,v in values.items()))+')\n'
                body+='Path('+repr(CONFIG)+').write_text('+repr(json.dumps(config))+')\n'
                installed=ssh(args.ssh_config,client['node'],['docker','exec',client['container'],'python3','-c',body])
                if installed.returncode:
                    raise RuntimeError(installed.stderr)
                publish(status='benchmark_running',current=algorithm,result_dir=result_dir)
                command='bash /workspace/heterogeneous_compute/platforms/iluvatar/benchmark.sh '+result_dir+'/benchmark.env > '+result_dir+'/run.log 2>&1; code=$?; printf "%s\\n" "$code" > '+result_dir+'/run.exit; exit "$code"'
                completed=ssh(args.ssh_config,client['node'],['docker','exec',client['container'],'bash','-lc',command])
                if completed.returncode:
                    raise RuntimeError('Official benchmark failed: '+completed.stderr[-1200:])
                verify="""from pathlib import Path
import json,importlib.metadata as m
p=Path(RESULT)
cases={}
lengths=None
for concurrency in [1,4,8]:
 r=json.loads((p/('sharegpt-c%s.json'%concurrency)).read_text())
 assert r['completed']==200 and r.get('failed',0)==0,'Failed benchmark requests'
 assert all(n==64 for n in r['output_lens']),'Output budget changed'
 assert lengths is None or lengths==r['input_lens'],'Input sequence differs across cases'
 lengths=r['input_lens']
 assert abs(sum(r['output_lens'])/r['duration']-r['output_throughput'])<1e-6,'Throughput recomputation failed'
 keys=['completed','failed','duration','output_throughput','mean_ttft_ms','p95_ttft_ms','p99_ttft_ms','mean_tpot_ms','p95_tpot_ms','p99_tpot_ms']
 cases[str(concurrency)]={k:r[k] for k in keys if k in r}
report=dict(status='complete',algorithm=ALGORITHM,client_vllm=m.version('vllm'),config=CONFIG_VALUE,cases=cases,
 validation='600/600 successful; output budgets, identical inputs across cases, throughput recomputation verified')
(p/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
"""
                verify='RESULT='+repr(result_dir)+'\nALGORITHM='+repr(algorithm)+'\nCONFIG_VALUE='+repr(config)+'\n'+verify
                checked=ssh(args.ssh_config,client['node'],['docker','exec',client['container'],'python3','-c',verify])
                if checked.returncode:
                    raise RuntimeError('Benchmark integrity verification failed: '+checked.stderr[-1200:])
                save(output/algorithm/'performance-summary.json',json.loads(checked.stdout))
                state['profiles'][algorithm]=dict(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
                publish()
            finally:
                remote_action('stop',algorithm,required=False)
        publish(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        publish(status='failed',error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('deployment.json'))
    parser.add_argument('--action',choices=['layers','start','stop','probe','run'],required=True)
    parser.add_argument('--algorithm',choices=['awq','gptq','fp16'])
    parser.add_argument('--ssh-config',type=Path)
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.action=='run':
        if args.ssh_config is None:
            parser.error('--ssh-config required for local controller')
        orchestrate(config,args)
    else:
        if args.algorithm is None:
            parser.error('--algorithm required')
        {'layers':native_layers,'start':start_server,'stop':stop_server,'probe':probe}[args.action](config,args.algorithm)


if __name__=='__main__':
    main()
