"""Open-loop SLO capacity, overload recovery and stability of the current system.

No service start/stop, model restart or container change. The standard-library
telemetry receiver is read-only apart from appending this experiment's results.
Run through benchmark.py --system-config; use the existing container's aiohttp.
"""
import asyncio
from collections import Counter
import csv
from functools import reduce
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from heterogeneous_compute.serving import expand_config, generation_workers, gateway_config, router_configs, router_workers
from heterogeneous_compute.benchmark_telemetry import admission_waiting, metric_sum


SLO_FIELDS={'ttft_seconds':'ttft','tpot_seconds':'tpot','e2e_seconds':'e2e'}


def save(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def validate_cross_vendor_ranks(serving,snapshots):
    """Match configured placements to real executor rank processes, not vLLM CLI defaults."""
    expected=set()
    for worker in generation_workers(serving).values():
        if worker.get('executor')!='cross-vendor':continue
        port=urlsplit(worker['base_url']).port
        for stage,placement in enumerate(worker['placements']):
            for lane,device in enumerate(placement['devices']):
                expected.add((placement['node'],port,stage,lane,device))
    if not expected:return
    observed=[]
    for node,snapshot in snapshots.items():
        for proc in snapshot.get('process',{}).get('processes',[]):
            executor=proc.get('executor',{})
            if executor.get('kind')!='cross-vendor':continue
            params=proc.get('inference_parameters',{})
            observed.append((node,int(params.get('--port',0)),executor['stage'],
                             executor['lane'],executor['physical_device']))
            if params.get('--tp-communicator')!='vllm':
                raise RuntimeError('Cross-vendor executor is not using the configured official local TP')
    if len(observed)!=len(set(observed)) or set(observed)!=expected:
        raise RuntimeError('Cross-vendor rank/device placement differs from the configured deployment')


def percentile(values,q):
    if not values:return None
    values=sorted(values);x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(values[j]-values[i])*(x-i)


def resolved_cohort_window(start,now,timeout_seconds,window_seconds=60):
    """Select only sent cohorts whose complete total deadline has elapsed."""
    end=now-timeout_seconds
    return (end-window_seconds,end) if end-window_seconds>=start else None


def stage_passes(config,result):
    require_queue=config.get('require_queue_stability',True)
    if type(require_queue) is not bool:raise ValueError('require_queue_stability must be boolean')
    return bool(result['load_valid'] and result['resource_stable']
        and result['success_ratio']>=config['minimum_success_ratio']
        and result['slo_ratio']>=config['minimum_slo_ratio']
        and (not require_queue or result['queue_stable']))


def validated_search_reference(config,serving,frozen,workload):
    """Reinterpret one complete stage only after checking frozen comparability."""
    spec=config.get('search_reference')
    if not spec:return None
    directory=Path(spec['result_dir'])
    if not directory.is_absolute():directory=ROOT/directory
    old_config=json.loads((directory/'config.json').read_text())
    allowed={'require_queue_stability','search_reference','coarse_rates','capacity_definition','run_mode'}
    if {k:v for k,v in config.items() if k not in allowed}!={k:v for k,v in old_config.items() if k not in allowed}:
        raise RuntimeError('Search reference workload/SLO/configuration mismatch')
    if serving!=json.loads((directory/'serving-snapshot.json').read_text()):
        raise RuntimeError('Search reference service configuration mismatch')
    previous=json.loads((directory/'preflight.json').read_text())
    normalize=lambda value:json.loads(json.dumps(value))
    for key in ['healthy_workers','router_pools','identities','service_processes']:
        if normalize(frozen.get(key))!=previous.get(key):raise RuntimeError('Search reference frozen identity mismatch: '+key)
    if workload!=json.loads((directory/'workload.json').read_text()):
        raise RuntimeError('Search reference token workload mismatch')
    stages=json.loads((directory/'stages.json').read_text())
    stage=next((s for s in stages if s['case']==spec['case']),None)
    if not stage or not stage_passes(config,stage):raise RuntimeError('Search reference stage does not pass the current criteria')
    rate=stage['rate']
    if rate<config.get('minimum_search_rate',0) or rate>=min(config['coarse_rates']):
        raise RuntimeError('Search reference must be a lower bracket below the first new test rate')
    return {'source_directory':str(directory),'source_case':stage['case'],'rate':rate,
        'original_passed':stage['passed'],'reinterpreted_passed':True,
        'require_queue_stability':config.get('require_queue_stability',True),'source_stage':stage,
        'scope':'Search bracket only; source raw requests/stages are not rewritten or merged. New candidate repeats and long stability tests remain required.'}


def meets_slo(row,slo):
    if not slo or set(slo)-set(SLO_FIELDS):raise ValueError('Configure at least one supported latency SLO')
    return bool(row.get('success') and all(row.get(SLO_FIELDS[key]) is not None and row[SLO_FIELDS[key]]<=limit
        for key,limit in slo.items()))


def workload_profiles(config):
    profiles=config.get('workload',{}).get('profiles')
    if profiles is None:
        return [{'name':'fixed','input_tokens':config['input_tokens'],'output_tokens':config['output_tokens'],'weight':100}]
    if not isinstance(profiles,list) or not profiles or any(not isinstance(p,dict) for p in profiles):
        raise ValueError('Workload profiles must be a nonempty list')
    names=[p.get('name') for p in profiles]
    if any(not isinstance(n,str) or not n for n in names) or len(set(names))!=len(names):
        raise ValueError('Workload profile names must be distinct')
    if any(type(p.get(k)) is not int or p[k]<minimum for p in profiles
           for k,minimum in [('input_tokens',1),('output_tokens',2),('weight',1)]):
        raise ValueError('Workload lengths and percentage weights must be positive integers')
    if sum(p['weight'] for p in profiles)!=100:raise ValueError('Workload weights must sum to 100')
    return profiles


def build_workload(config,tokenizer):
    """Reuse canonical synthetic chat padding for each configured length class."""
    profiles=workload_profiles(config);rng=random.Random(config['seed'])
    divisor=reduce(math.gcd,(p['weight'] for p in profiles))
    block=[p for p in profiles for _ in range(p['weight']//divisor)]
    if config['prompt_count']%len(block):raise ValueError('prompt_count must preserve exact workload proportions')
    prefixes=['quiet garden','mountain village','coastal forest','winter sunrise','ancient library','flowing river','distant island','summer orchard']
    workload=[]
    for start in range(0,config['prompt_count'],len(block)):
        order=list(block);rng.shuffle(order)
        for profile in order:
            index=len(workload)
            content=f"Benchmark request {index:08d}. Please write a detailed description of a {rng.choice(prefixes)}. "
            base_length=len(tokenizer.apply_chat_template([{'role':'user','content':content}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False))
            padding=max(0,profile['input_tokens']-base_length);prompt=None
            for n in [padding,*range(max(0,padding-4),padding+5)]:
                messages=[{'role':'user','content':content+' garden'*n}]
                ids=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                if len(ids)==profile['input_tokens']:prompt=messages;break
            if prompt is None:raise RuntimeError('Cannot construct exact input length for '+profile['name'])
            workload.append({'messages':prompt,'workload_name':profile['name'],
                'input_tokens':profile['input_tokens'],'output_tokens':profile['output_tokens']})
    return workload


def summarize(rows,start,end,slo,by_workload=True):
    """Completion-time goodput and sent-cohort ratios deliberately differ."""
    duration=end-start
    sent=[r for r in rows if start<=r['sent_at']<end]
    completed=[r for r in rows if start<=r['completed_at']<end and r['success']]
    good=[r for r in completed if meets_slo(r,slo)]
    successful=[r for r in sent if r['success']]
    cohort_good=[r for r in sent if meets_slo(r,slo)]
    result=dict(duration_seconds=duration,sent=len(sent),actual_rate=len(sent)/duration,
        completed_in_window=len(completed),completed_rate=len(completed)/duration,
        goodput=len(good)/duration,good_completed_in_window=len(good),
        success_ratio=len(successful)/len(sent) if sent else 0,
        slo_ratio=len(cohort_good)/len(sent) if sent else 0,
        cohort_good=len(cohort_good),cohort_success=len(successful),
        drain_completions=sum(r['success'] and r['completed_at']>=end for r in sent),
        errors=dict(Counter(r.get('error','unknown') for r in sent if not r['success'])),
        send_lag_p99=percentile([r['send_lag'] for r in sent],.99),
        output_tokens=dict(Counter(str(r.get('output_tokens')) for r in successful)),
        input_tokens=dict(Counter(str(r.get('input_tokens')) for r in successful)))
    if by_workload:
        names=sorted({r.get('workload_name','fixed') for r in rows})
        result['by_workload']={name:summarize([r for r in rows if r.get('workload_name','fixed')==name],start,end,slo,False) for name in names}
    return result


class Receiver:
    def __init__(self,directory,url,serving):
        self.directory=directory;self.latest={};self.history=[];self.routes={};self.network={};self.lock=threading.Lock()
        self.state={'phase':'waiting_for_collectors','finished':False,'time':time.time()}
        nodes={placement['node'] for w in generation_workers(serving).values()
               for placement in w.get('placements', [w])} | {serving['services']['gateway']['node']}
        self.nodes=nodes
        self.gateway_node=serving['services']['gateway']['node']
        self.admission_enabled=bool(serving.get('admission'))
        self.router_members={router.get('pool',name):{w['base_url'].removesuffix('/v1')
            for w in router_workers(serving,name).values()} for name,router in router_configs(serving).items()}
        receiver=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                if self.path!='/status':self.send_error(404);return
                payload=json.dumps(receiver.state).encode();self.send_response(200);self.end_headers();self.wfile.write(payload)
            def do_POST(self):
                # Port is bound to the internal interface. Only known collector
                # schema is accepted; bodies are bounded and never executed.
                length=int(self.headers.get('Content-Length','0'))
                if self.path not in {'/telemetry','/trace'} or not 0<length<=4*1024*1024:
                    self.send_error(400);return
                try:
                    body=json.loads(self.rfile.read(length));body['received_at']=time.time()
                    with receiver.lock:
                        if self.path=='/telemetry':
                            if body['node'] not in receiver.nodes:raise ValueError('unknown node')
                            receiver.latest[body['node']]=body;receiver.history.append(body)
                            if len(receiver.history)>50000:receiver.history=receiver.history[-40000:]
                            file='telemetry.jsonl'
                        else:
                            for route in body.get('routes',[]):receiver.routes[route['request_id']]=route
                            body['packet_drops_total']=receiver.network.get('packet_drops_total',0)+body.get('packet_statistics',{}).get('dropped',0)
                            receiver.network=body;file='routes.jsonl'
                        with (directory/file).open('a',encoding='utf-8') as out:
                            out.write(json.dumps(body,ensure_ascii=False)+'\n')
                    self.send_response(200);self.end_headers();self.wfile.write(b'{}')
                except (ValueError,KeyError,TypeError):self.send_error(400)
        parsed=urlsplit(url)
        self.server=ThreadingHTTPServer((parsed.hostname,parsed.port),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()

    def phase(self,phase,**extra):
        self.state={'phase':phase,'finished':False,'time':time.time(),**extra}
        save(self.directory/'status.json',self.state)
        print(json.dumps(self.state),flush=True)

    def workers(self):
        latest=self.latest.get(self.gateway_node,{})
        return {r['url'] for r in latest.get('router_workers',{}).get('workers',[]) if r.get('is_healthy')}

    def router_health(self):
        snapshot=self.latest.get(self.gateway_node,{}).get('router_workers',{})
        if snapshot.get('errors'):raise RuntimeError('Router pool telemetry failed: '+str(snapshot['errors']))
        pools=snapshot.get('pools')
        # Legacy single-router collectors expose only the combined workers list.
        if pools is None and len(self.router_members)==1:
            pools={next(iter(self.router_members)):snapshot}
        if set(pools or {})!=set(self.router_members):
            raise RuntimeError('Router pool telemetry is incomplete')
        healthy={}
        for name,report in pools.items():
            listed={w['url'] for w in report.get('workers',[])}
            if not listed.issubset(self.router_members[name]):
                raise RuntimeError('Worker registered in the wrong Router pool: '+name)
            healthy[name]={w['url'] for w in report.get('workers',[]) if w.get('is_healthy')}
            if not healthy[name]:raise RuntimeError('Router pool has no healthy workers: '+name)
        return healthy

    def queue(self):
        running=waiting=0;missing=[]
        for node in self.latest.values():
            for name,worker in node.get('workers',{}).items():
                if worker.get('running') is None or worker.get('waiting') is None:missing.append(name)
                else:running+=worker['running'];waiting+=worker['waiting']
        snapshot=self.latest.get(self.gateway_node,{})
        gw=metric_sum(snapshot.get('gateway_metrics',[]),'heteroserve_gateway_inflight')
        queued=admission_waiting(snapshot) if getattr(self,'admission_enabled',False) else 0
        return {'running':running,'waiting':waiting,'gateway_inflight':gw,'gateway_waiting':queued,'missing_workers':missing}


class Experiment:
    def __init__(self,config,serving,directory):
        self.config=config;self.serving=serving;self.directory=directory
        self.receiver=Receiver(directory,config['telemetry_url'],serving)
        self.expected=set();self.identities={};self.active=set();self.completed=[];self.abort=None;self.case=0
        self.rows_file=(directory/'requests.jsonl').open('a',encoding='utf-8',buffering=1)
        self.sample_file=(directory/'timeline.jsonl').open('a',encoding='utf-8',buffering=1)
        self.results=[];self.frozen=None;self.tokenizer=None;self.prompts=[]
        self.embedding_required=bool(gateway_config(serving).get('route_embeddings',{}).get(config['model']))
        self.start=time.time();self.deadline=self.start+23*3600

    def check(self):
        if (self.directory/'STOP').exists():raise RuntimeError('User stop file exists; load stopped')
        if time.time()>self.deadline:raise RuntimeError('Experiment lifetime exceeded')
        if self.abort:raise RuntimeError(self.abort)
        if not self.frozen:return
        if getattr(self,'serving',{}).get('admission'):
            queue=self.receiver.queue()
            if queue['gateway_inflight'] is None:raise RuntimeError('Gateway inflight telemetry is unavailable; no valid capacity claim')
            if queue['gateway_waiting'] is None:raise RuntimeError('Admission queue telemetry is missing')
            if queue['gateway_waiting']>self.serving['admission']['max_queue']:
                raise RuntimeError('Admission queue exceeded its frozen capacity')
        if self.receiver.network.get('error'):raise RuntimeError('Request route observer failed')
        if self.receiver.network.get('packet_drops_total',0):raise RuntimeError('Route observer dropped packets; stop before claiming per-worker attribution')
        if self.receiver.workers()!=self.expected:raise RuntimeError('Frozen healthy worker set changed; no data concatenation across configurations')
        if self.receiver.router_health()!=self.router_identities:raise RuntimeError('Frozen per-pool healthy worker set changed')
        for node in self.receiver.nodes:
            row=self.receiver.latest.get(node,{})
            if time.time()-row.get('received_at',0)>self.config['telemetry_max_age_seconds']:
                raise RuntimeError('Stale telemetry: '+node)
            c=row.get('container',{})
            identity=(c.get('id'),c.get('restart_count'),c.get('started_at'))
            if identity!=self.identities[node] or not c.get('running') or c.get('oom_killed'):
                raise RuntimeError('Container identity/restart/OOM event: '+node)
            current={p['pid']:p['start_ticks'] for p in row.get('process',{}).get('processes',[]) if p.get('role') in {'admission','gateway','router','model'}}
            for pid,stamp in self.process_identities[node].items():
                if current.get(pid)!=stamp:raise RuntimeError('Frozen service process exited or restarted: '+node+'/'+str(pid))

    async def wait(self,seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.check();await asyncio.sleep(min(1,max(0,until-time.monotonic())))

    async def timeline(self):
        while True:
            row={'time':time.time(),'phase':self.receiver.state.get('phase'),'case':self.case,'client_active':len(self.active),
                 'target_rate':self.receiver.state.get('rate'),
                 'healthy_workers':len(self.receiver.workers()),**self.receiver.queue()}
            self.sample_file.write(json.dumps(row)+'\n')
            await asyncio.sleep(1)

    async def one(self,session,prompt,index,scheduled,phase,endpoint=None,model=None,expected_text=None,output_tokens=None):
        import aiohttp
        c=self.config
        profile=prompt if isinstance(prompt,dict) else {}
        input_tokens=profile.get('input_tokens',c.get('input_tokens'))
        output_tokens=output_tokens if output_tokens is not None else profile.get('output_tokens',c.get('output_tokens'))
        if isinstance(prompt,dict):prompt=profile['messages']
        sent=time.time();row={'index':index,'case':self.case,'phase':phase,'scheduled_at':scheduled,'sent_at':sent,
            'send_lag':max(0,sent-scheduled),'success':False,'request_id':None,'vendor':None,'worker':None,'selected_pool':None,'embedding_ms':None,
            'tpot':None,'input_tokens':None,'output_tokens':None,'gateway_queue_ms':None,
            'workload_name':profile.get('workload_name','fixed'),
            'requested_input_tokens':input_tokens,'requested_output_tokens':output_tokens}
        if 'ttft_seconds' in c['slo']:row['ttft']=None
        if 'e2e_seconds' in c['slo']:row['e2e']=None
        body={'model':model or c['model'],'messages':prompt,'temperature':0,'max_tokens':output_tokens,
              'ignore_eos':expected_text is None,'stream':True,'stream_options':{'include_usage':True},
              'chat_template_kwargs':{'enable_thinking':False}}
        content=[];first=last=None;done=False;finish=None;usage=None;raw_bytes=0;start=time.monotonic()
        try:
            async with session.post((endpoint or c['endpoint'])+'/v1/chat/completions',json=body,
                timeout=aiohttp.ClientTimeout(total=c['timeout_seconds'],connect=5),headers={'x-benchmark-request-id':str(index)}) as response:
                row['http_status']=response.status;row['request_id']=response.headers.get('x-request-id')
                row['selected_pool']=response.headers.get('x-moqe-expert')
                queue_header=response.headers.get('x-heteroserve-queue-ms')
                if queue_header is not None:
                    try:
                        elapsed=float(queue_header)
                        if math.isfinite(elapsed) and elapsed>=0:row['gateway_queue_ms']=elapsed
                    except ValueError:pass
                embedding_header=response.headers.get('x-heteroserve-embedding-ms')
                if embedding_header is not None:
                    try:
                        elapsed=float(embedding_header)
                        if math.isfinite(elapsed) and elapsed>=0:row['embedding_ms']=elapsed
                    except ValueError:pass
                if response.status!=200:
                    await response.read();row['error']='429' if response.status==429 else '5xx' if response.status>=500 else 'http_'+str(response.status)
                else:
                    async for raw in response.content:
                        raw_bytes+=len(raw)
                        if not raw.startswith(b'data:'):continue
                        data=raw[5:].strip()
                        if data==b'[DONE]':done=True;continue
                        event=json.loads(data)
                        if 'error' in event:row['error']='stream_error';continue
                        if event.get('usage'):usage=event['usage']
                        if event.get('system_fingerprint'):row['system_fingerprint']=event['system_fingerprint']
                        for choice in event.get('choices',[]):
                            delta=choice.get('delta',{}).get('content') or ''
                            if delta:
                                now=time.monotonic();first=first if first is not None else now;last=now;content.append(delta)
                            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
                    if usage:
                        row['input_tokens']=usage.get('prompt_tokens');row['output_tokens']=usage.get('completion_tokens')
                    text=''.join(content)
                    row['output_text_sha256']=hashlib.sha256(text.encode()).hexdigest()
                    row['content_bytes']=len(text.encode());row['finish_reason']=finish;row['done']=done
                    if not row.get('error'):
                        if not done or not finish or not content:row['error']='abnormal_truncation'
                        elif expected_text is not None and text.strip()!=expected_text:row['error']='smoke_answer'
                        elif usage is None:row['error']='missing_usage'
                        elif expected_text is None and (row['input_tokens']!=input_tokens or row['output_tokens']!=output_tokens or finish!='length'):row['error']='length_mismatch'
                        elif finish in {'abort','error'}:row['error']='backend_abort'
                        elif endpoint is None and getattr(self,'embedding_required',False) and row['embedding_ms'] is None:
                            row['error']='missing_embedding_measurement' if embedding_header is None else 'invalid_embedding_measurement'
                        elif endpoint is None and c.get('gateway_admission') and row['gateway_queue_ms'] is None:
                            row['error']='missing_admission_measurement'
                        else:row['success']=True
        except asyncio.TimeoutError:row['error']='timeout'
        except aiohttp.ClientConnectionError as exc:row.update(error='connection_error',detail=type(exc).__name__)
        except (aiohttp.ClientPayloadError,ValueError,KeyError) as exc:row.update(error='abnormal_truncation',detail=type(exc).__name__)
        except aiohttp.ClientError as exc:row.update(error='client_http_error',detail=type(exc).__name__)
        row['completed_at']=time.time();row['wire_response_bytes']=raw_bytes
        if 'e2e_seconds' in c['slo']:row['e2e']=time.monotonic()-start
        if first is not None:
            if 'ttft_seconds' in c['slo']:row['ttft']=first-start
            if row.get('output_tokens',0) and row['output_tokens']>1:row['tpot']=(last-first)/(row['output_tokens']-1)
        row['slo_pass']=meets_slo(row,c['slo']);self.rows_file.write(json.dumps(row,ensure_ascii=False)+'\n')
        self.completed.append(row)
        return row

    async def preflight(self,session):
        r=self.receiver;r.phase('waiting_for_collectors')
        for _ in range(120):
            if set(r.latest)==r.nodes and r.workers() and r.network.get('error') is None and r.network:break
            await asyncio.sleep(1)
        else:raise RuntimeError('Collectors/route observer did not become ready')
        self.expected=r.workers();all_workers=generation_workers(self.serving)
        self.router_identities=r.router_health()
        routers=router_configs(self.serving)
        routing=gateway_config(self.serving)
        required_embedding=self.config.get('required_embedding_route')
        if required_embedding and routing.get('route_embeddings',{}).get(self.config['model'])!=required_embedding:
            raise RuntimeError('Frozen workload requires Embedding before hardware-pool selection')
        multi_pool=bool(self.serving['router'].get('pools'))
        if multi_pool:
            selected=routing.get('random_routes',{}).get(self.config['model'],[])
            expected_backends={router['backend'] for router in routers.values()}
            if len(selected)!=len(expected_backends) or set(selected)!=expected_backends:
                raise RuntimeError('Gateway does not uniformly select the configured hardware pools')
            expected_router=self.config.get('required_router',{'policy':'round_robin'})
            if any(any(router.get(key)!=value for key,value in expected_router.items()) for router in routers.values()):
                raise RuntimeError('L2 Router policy/parameters differ from the frozen experiment configuration')
        vendors=set();direct=[]
        validate_cross_vendor_ranks(self.serving,r.latest)
        self.identities={};self.process_identities={}
        for node in r.nodes:
            snapshot=r.latest[node];c=snapshot['container'];self.identities[node]=(c['id'],c['restart_count'],c['started_at'])
            self.process_identities[node]={p['pid']:p['start_ticks'] for p in snapshot.get('process',{}).get('processes',[]) if p.get('role') in {'admission','gateway','router','model'}}
            if c['limits']['Memory'] or c['limits']['CpuQuota'] not in {0,-1,None} or c['limits']['PidsLimit'] not in {0,-1,None}:
                raise RuntimeError('Unexpected resource limits: '+node)
        for name,w in all_workers.items():
            url=w['base_url'].removesuffix('/v1')
            if url not in self.expected:continue
            vendor='ascend' if w['node'].startswith('ascend') else 'musa' if w['node'].startswith('s5000') else 'iluvatar'
            placements=w.get('placements',[{'node':w['node']}])
            vendors.update('ascend' if p['node'].startswith('ascend') else 'musa'
                           if p['node'].startswith('s5000') else 'iluvatar' for p in placements)
            row=await self.one(session,[{'role':'user','content':'What is 2+3? Reply with only the digit.'}],-1,time.time(),'direct-smoke',url,w['model'],'5',16)
            row['worker']=url;row['vendor']=vendor;direct.append(row)
            if not row['success']:raise RuntimeError('Direct generation failed: '+name+' '+str(row.get('error')))
        if vendors!={'ascend','musa','iluvatar'}:raise RuntimeError('All three vendors are required')
        if self.serving['gateway']['max_inflight']!=self.config['gateway_max_inflight']:raise RuntimeError('Gateway budget configuration mismatch')
        minimum_files=self.serving['gateway'].get('min_open_files')
        if minimum_files:
            gateway_procs=[p for p in r.latest[r.gateway_node].get('process',{}).get('processes',[]) if p.get('role')=='gateway']
            if not gateway_procs or any(p.get('open_files_limit',{}).get('soft')!='unlimited'
                    and int(p.get('open_files_limit',{}).get('soft',0))<minimum_files for p in gateway_procs):
                raise RuntimeError('Live Gateway open-file limit is below its frozen requirement')
        if self.serving.get('admission'):
            expected=self.config.get('gateway_admission',{})
            if not expected or any(self.serving['admission'].get(key)!=value for key,value in expected.items()):
                raise RuntimeError('Admission configuration is not frozen consistently')
            if r.queue()['gateway_waiting'] is None:raise RuntimeError('Admission queue telemetry is missing')
        required_context=max(p['input_tokens']+p['output_tokens'] for p in workload_profiles(self.config))
        for node,snapshot in r.latest.items():
            for proc in snapshot.get('process',{}).get('processes',[]):
                parameters=proc.get('inference_parameters')
                if not parameters:continue
                if parameters.get('--runner')=='pooling':continue
                if int(parameters.get('--tensor-parallel-size',1))!=self.config['required_tp'] or int(parameters.get('--pipeline-parallel-size',1))!=self.config['required_pp']:
                    raise RuntimeError('Running model process does not use the configured TP/PP: '+node)
                if int(parameters.get('--max-model-len',0))<required_context:
                    raise RuntimeError('Running model context is too short for frozen workload: '+node)
                if (node.startswith('s5000') and proc.get('executor',{}).get('kind')!='cross-vendor'
                        and not parameters.get('--disable-custom-all-reduce')):
                    raise RuntimeError('MUSA process is not using the required standard MCCL path')
        r.phase('building_workload',workload=self.config.get('workload',{'name':'fixed'}))
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(self.config['tokenizer'],local_files_only=True)
        workload=build_workload(self.config,self.tokenizer)
        self.prompts=workload;save(self.directory/'workload.json',workload)
        semaphore=asyncio.Semaphore(4)
        async def pooled_check(i):
            async with semaphore:
                return await self.one(session,workload[i%len(workload)],i,time.time(),'pooled-preflight')
        pooled=[];mapped=set();limit=8*len(self.expected) if multi_pool else 2*len(self.expected)
        backend_by_worker={w['base_url'].removesuffix('/v1'):router.get('backend')
            for name,router in routers.items() for w in router_workers(self.serving,name).values()}
        require_all_workers=all(router['policy']=='round_robin' for router in routers.values())
        required_pools={backend_by_worker[w] for w in self.expected}
        def pooled_covered():
            if require_all_workers:return self.expected.issubset(mapped)
            # Low-concurrency adaptive routing need not visit every idle replica.
            # Each replica already passed direct generation; require genuine L1
            # requests through every pool, and retain actual worker coverage.
            return len(pooled)>=2*len(self.expected) and {backend_by_worker[w] for w in mapped}==required_pools
        # A random L1 does not visit each hardware pool equally in a short sample.
        # Bound additional genuine alias requests rather than mistaking chance
        # undercoverage for a broken worker, or bypassing L1 to force coverage.
        while len(pooled)<limit and not pooled_covered():
            count=min(2*len(self.expected),limit-len(pooled));offset=len(pooled)
            batch=await asyncio.gather(*(pooled_check(i) for i in range(offset,offset+count)))
            pooled.extend(batch)
            if not all(row['success'] for row in batch):
                raise RuntimeError('Unified workload streaming check failed: '+str(Counter(row.get('error') for row in batch if not row['success'])))
            await asyncio.sleep(5)
            for row in batch:
                worker=r.routes.get(row['request_id'],{}).get('worker')
                if not row['request_id'] or not worker:
                    raise RuntimeError('Successful pooled request has no request-ID-to-worker association')
                if worker not in self.expected:raise RuntimeError('Pooled request reached a worker outside the frozen healthy set')
                if multi_pool and row['selected_pool']!=backend_by_worker.get(worker):
                    raise RuntimeError('L1 selected pool and observed L2 worker disagree')
                mapped.add(worker)
            if r.workers()!=self.expected or r.router_health()!=self.router_identities:
                raise RuntimeError('Healthy Router pool set changed during preflight')
        if not pooled_covered():raise RuntimeError('Random-pool preflight exhausted its request budget before meeting policy-specific coverage')
        self.frozen={'healthy_workers':sorted(self.expected),'expected_workers':len(all_workers),'missing_workers':sorted({w['base_url'].removesuffix('/v1') for w in all_workers.values()}-self.expected),
            'identities':self.identities,'service_processes':self.process_identities,'direct_smoke':direct,
            'pooled_requests':len(pooled),'pooled_worker_coverage':sorted(mapped),'samples':dict(r.latest),
            'pooled_coverage_requirement':'every healthy worker' if require_all_workers else 'every pool; every healthy worker separately checked by direct generation',
            'router_pools':{name:sorted(workers) for name,workers in self.router_identities.items()},
            'routing':{'random_routes':routing.get('random_routes',{}),'routers':routers},
            'admission':self.serving.get('admission'),
            'workload':{'profiles':workload_profiles(self.config),'counts':dict(Counter(p['workload_name'] for p in workload)),
                'maximum_required_context':required_context,'prompt_count':len(workload)},
            'pipeline':{'route_embeddings':routing.get('route_embeddings',{}),
                'embedding_timing':'x-heteroserve-embedding-ms, finite nonnegative milliseconds',
                'recorded_slo_metrics':[SLO_FIELDS[key] for key in self.config['slo']],
                'latency_scope':'Only configured latency SLO metrics are persisted. TPOT measures generated-token intervals; Embedding duration is a separate observation.'},
            'execution_mode':self.config['execution_mode'],'all_devices_accepted':len(self.expected)==len(all_workers)}
        save(self.directory/'preflight.json',self.frozen)
        self.search_reference_result=validated_search_reference(self.config,self.serving,self.frozen,self.prompts)
        if self.search_reference_result:save(self.directory/'search-reference.json',self.search_reference_result)
        self.check();self.completed.clear()

    async def drain(self):
        self.receiver.phase('draining',case=self.case)
        budget=math.ceil(max(self.config['timeout_seconds'],self.serving['gateway'].get('timeout_seconds',180)))+30
        for _ in range(budget):
            self.check();q=self.receiver.queue()
            if not self.active and q['gateway_inflight']==0 and q['gateway_waiting']==0 and q['running']==0 and q['waiting']==0:return
            await asyncio.sleep(1)
        raise RuntimeError('Residual server requests did not drain; subsequent load stopped')

    async def send_window(self,session,rate,seconds,phase,offset=0):
        """No concurrency semaphore: local saturation invalidates the stage."""
        start_mono=time.monotonic();wall=time.time();n=math.ceil(seconds*rate)
        profiles=workload_profiles(self.config) if 'workload' in self.config else None
        block=100//reduce(math.gcd,(p['weight'] for p in profiles)) if profiles else 1
        for i in range(n):
            scheduled=wall+i/rate
            if i/rate>=seconds:break
            self.check()
            await asyncio.sleep(max(0,start_mono+i/rate-time.monotonic()))
            self.check()
            if len(self.active)>=self.config['max_client_tasks']:raise RuntimeError('Load generator task limit reached; not a server capacity result')
            # Keep complete mixture blocks aligned in every stage. A partial
            # final block has integer counts; report the actual sent mixture.
            prompt=self.prompts[(i+offset//block*block)%len(self.prompts)]
            task=asyncio.create_task(self.one(session,prompt,i+offset,scheduled,phase))
            self.active.add(task);task.add_done_callback(self.active.discard)
        await self.wait(max(0,start_mono+seconds-time.monotonic()))
        return wall,wall+seconds,n

    def backlog_stable(self,start,end):
        points=[]
        for row in self.receiver.history:
            t=row.get('received_at',0)
            if not start<=t<end:continue
            waiting=sum(w['waiting'] for w in row.get('workers',{}).values() if w.get('waiting') is not None)
            points.append((t,waiting))
        # Compare per-node means; a flat total must not hide one vendor's growth.
        changes={}
        for node in self.receiver.nodes:
            values=[r for r in self.receiver.history if r['node']==node and start<=r.get('received_at',0)<end]
            first=[sum(w['waiting'] for w in r.get('workers',{}).values() if w.get('waiting') is not None) for r in values if r['received_at']<start+(end-start)/3]
            last=[sum(w['waiting'] for w in r.get('workers',{}).values() if w.get('waiting') is not None) for r in values if r['received_at']>=end-(end-start)/3]
            changes[node]=statistics.mean(last)-statistics.mean(first) if first and last else None
        if self.serving.get('admission'):
            values=[r for r in self.receiver.history if r['node']==self.receiver.gateway_node and start<=r.get('received_at',0)<end]
            first=[admission_waiting(r) for r in values if r['received_at']<start+(end-start)/3]
            last=[admission_waiting(r) for r in values if r['received_at']>=end-(end-start)/3]
            changes['gateway_queue']=statistics.mean(last)-statistics.mean(first) if first and last and None not in first+last else None
        return all(v is not None and v<=2 for v in changes.values()),changes

    def queue_statistics(self,start,end):
        values={}
        for snapshot in self.receiver.history:
            if not start<=snapshot.get('received_at',0)<end:continue
            for name,worker in snapshot.get('workers',{}).items():
                if worker.get('waiting') is not None:values.setdefault(name,[]).append(worker['waiting'])
            if self.serving.get('admission') and snapshot['node']==self.receiver.gateway_node:
                value=admission_waiting(snapshot)
                if value is not None:values.setdefault('gateway',[]).append(value)
        return {name:{'sample_mean':statistics.mean(samples),'sample_peak':max(samples),'samples':len(samples)}
                for name,samples in values.items()}

    def resources_stable(self,start,end):
        report={};stable=True
        for node in self.receiver.nodes:
            for role in ['admission','gateway','router','model']:
                points=[]
                for r in self.receiver.history:
                    t=r.get('received_at',0)
                    if start+(end-start)/2<=t<end and r['node']==node:
                        processes=[p for p in r.get('process',{}).get('processes',[]) if p.get('role')==role]
                        if processes:points.append((t,sum(p['rss_bytes'] for p in processes)))
                if len(points)<3:continue
                t0=points[0][0];xs=[t-t0 for t,_ in points];ys=[v for _,v in points]
                xm=statistics.mean(xs);ym=statistics.mean(ys)
                slope=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/max(1e-9,sum((x-xm)**2 for x in xs))
                growth=statistics.mean(ys[-max(1,len(ys)//4):])-statistics.mean(ys[:max(1,len(ys)//4)])
                threshold=max(128*1024**2/3600,ys[0]*.01/3600)
                excessive=end-start>=1800 and slope>threshold and growth>max(256*1024**2,ys[0]*.02)
                report[node+'/'+role]={'rss_sum_slope_bytes_per_second':slope,'rss_sum_change_bytes':growth,'excessive_sustained_growth':excessive,
                    'scope':'Sum of process RSS; shared pages may be counted more than once. Growth is a diagnostic, not proof of a memory leak.'}
                stable &= not excessive
        return stable,report

    async def stage(self,session,rate,seconds,label,repeat=0,transition=True,drain=True):
        self.case+=1;self.completed.clear();r=self.receiver
        r.phase(label,rate=rate,seconds=seconds,case=self.case,repeat=repeat)
        if transition:
            await self.send_window(session,rate,self.config['transition_seconds'],label+'-transition',repeat*100003)
        r.phase(label,rate=rate,seconds=seconds,case=self.case,repeat=repeat,measurement=True)
        start,end,planned=await self.send_window(session,rate,seconds,label,repeat*100003+1000)
        if self.active:await asyncio.gather(*list(self.active))
        stable,changes=self.backlog_stable(start,end)
        resources_ok,resources=self.resources_stable(start,end)
        result=summarize(self.completed,start,end,self.config['slo'])
        result.update(label=label,rate=rate,repeat=repeat,case=self.case,start=start,end=end,planned=planned,
            queue_stable=stable,queue_changes=changes,queue_statistics=self.queue_statistics(start,end),healthy_workers=len(self.expected))
        result['resource_trends']=resources;result['resource_stable']=resources_ok
        result['load_valid']=result['sent']>=planned*self.config['minimum_send_ratio'] and (result['send_lag_p99'] or 0)<=self.config['max_send_lag_seconds']
        result['queue_stability_required']=self.config.get('require_queue_stability',True)
        result['passed']=stage_passes(self.config,result)
        if seconds>=1800:
            rolling=[summarize(self.completed,t,min(t+300,end),self.config['slo']) for t in range(int(start),int(end)-299,300)]
            result['five_minute_cohorts']=rolling
            result['passed'] &= all(w['success_ratio']>=self.config['minimum_success_ratio'] and w['slo_ratio']>=self.config['minimum_slo_ratio'] for w in rolling)
        self.results.append(result);save(self.directory/'stages.json',self.results)
        export_results(self.directory,self.config,self.results,self.receiver.routes,plots=False)
        print(json.dumps({'event':'stage_finished',**result}),flush=True)
        if drain:await self.drain()
        return result

    async def recovery(self,session,candidate,multiplier):
        c=self.config;r=self.receiver
        await self.stage(session,candidate*.5,c['overload_baseline_seconds'],'recovery-baseline')
        q=r.queue();baseline=q['waiting'];gateway_baseline=q['gateway_waiting']
        self.case+=1;self.completed.clear()
        r.phase('overload',rate=candidate*multiplier,multiplier=multiplier,case=self.case)
        overload_start,overload_end,_=await self.send_window(session,candidate*multiplier,c['overload_seconds'],'overload')
        # Deliberately do not gather/drain overload requests before lowering the
        # arrival rate. The continuous completion timeline retains carryover.
        recovery_start=time.time();r.phase('recovery',rate=candidate*.5,multiplier=multiplier,case=self.case)
        good_since=None;restored=None;windows=[]
        while time.time()-recovery_start<c['recovery_max_seconds']:
            start,end,planned=await self.send_window(session,candidate*.5,60,'recovery')
            # Total deadlines vary with the workload. A short fixed lookback
            # would omit still-pending long requests and overstate recovery.
            cohort=resolved_cohort_window(recovery_start,end,c['timeout_seconds'])
            window=summarize(self.completed,*cohort,c['slo']) if cohort else {}
            windows.append({'start':start,'end':end,'resolved_cohort':window,'queue':r.queue()})
            normal=bool(window and window['success_ratio']>=c['minimum_success_ratio'] and window['slo_ratio']>=c['minimum_slo_ratio']
                and window['send_lag_p99']<=c['max_send_lag_seconds'] and r.queue()['waiting']<=baseline+2
                and r.queue()['gateway_waiting'] is not None and r.queue()['gateway_waiting']<=gateway_baseline+2)
            if normal:
                good_since=good_since or start
                if end-good_since>=c['recovery_hold_seconds']:restored=good_since-recovery_start;break
            else:good_since=None
        if self.active:await asyncio.gather(*list(self.active))
        save(self.directory/f'recovery-{multiplier}.json',{'multiplier':multiplier,'restored':restored is not None,'recovery_seconds':restored,
             'observation_seconds':time.time()-recovery_start,'remaining':r.queue(),'windows':windows,
             'overload':summarize(self.completed,overload_start,overload_end,c['slo'])})
        await self.drain()
        if restored is None:raise RuntimeError('Overload did not recover within the frozen observation window')

    async def execute(self):
        import aiohttp
        connector=aiohttp.TCPConnector(limit=0,limit_per_host=0,ttl_dns_cache=300,
                                      keepalive_timeout=self.config.get('client_keepalive_seconds',15))
        async with aiohttp.ClientSession(connector=connector,trust_env=False) as session:
            timeline=asyncio.create_task(self.timeline())
            try:
                await self.preflight(session);await self.drain()
                if self.config.get('run_mode')=='preflight':
                    save(self.directory/'summary.json',{'status':'preflight_complete',
                         'healthy_workers':len(self.expected),'all_devices_accepted':self.frozen['all_devices_accepted'],
                         'scope':'Deployment, live rank identity, mixed workload, monitoring and drain checks only; no capacity claim'})
                    return
                reference=getattr(self,'search_reference_result',None)
                lower=reference['rate'] if reference else 0;upper=None
                sweep=self.config.get('run_mode','full')=='sweep'
                for rate in self.config['coarse_rates']:
                    result=await self.stage(session,rate,self.config['coarse_seconds'],'coarse')
                    if not result['load_valid']:raise RuntimeError('Load generator failed to produce the requested arrival process')
                    if sweep:continue
                    if not result['passed']:upper=rate;break
                    lower=rate
                if sweep:
                    passed=[s['rate'] for s in self.results if s['passed']]
                    failed=[s['rate'] for s in self.results if not s['passed']]
                    peak=max(self.results,key=lambda s:s['goodput'])
                    report={'status':'capacity_sweep_complete','run_mode':'sweep',
                        'scope':'Short measured points only; no repeated or long-duration capacity validation.',
                        'require_queue_stability':self.config.get('require_queue_stability',True),
                        'reference_rate':reference['rate'] if reference else None,
                        'highest_passing_measured_rate':max(passed) if passed else None,
                        'first_failing_measured_rate':failed[0] if failed else None,
                        'lowest_failing_measured_rate':min(failed) if failed else None,
                        'highest_requested_rate':max(s['rate'] for s in self.results),
                        'peak_goodput':{k:peak[k] for k in ['rate','actual_rate','goodput','success_ratio','slo_ratio','passed']},
                        'long_stability_validated':False,'stages':self.results}
                    save(self.directory/'sweep-summary.json',report);save(self.directory/'summary.json',report)
                    return
                if upper is None:raise RuntimeError('No failing upper bound found; verified lower bound only')
                if lower==0:
                    # Determine whether the SLO is already infeasible at sparse
                    # arrivals rather than silently relaxing it or reporting 0.
                    for rate in [r for r in [self.config['coarse_rates'][0]/2,self.config['coarse_rates'][0]/4]
                                 if r>=self.config.get('minimum_search_rate',0)]:
                        result=await self.stage(session,rate,self.config['fine_seconds'],'low-rate-slo-check')
                        if result['passed']:lower=rate;break
                    if lower==0:raise RuntimeError('No tested rate at or above the configured minimum passes all experiment conditions; no capacity claim')
                for step in range(self.config['fine_max_steps']):
                    if upper-lower<=self.config['fine_resolution']:break
                    rate=round((upper+lower)/2,4)
                    result=await self.stage(session,rate,self.config['fine_seconds'],'fine',step)
                    if result['passed']:lower=rate
                    else:upper=rate
                minimum=self.config.get('minimum_search_rate',0)
                while lower>0 and lower>=minimum:
                    repeats=[await self.stage(session,lower,self.config['fine_seconds'],'boundary-repeat',i) for i in range(self.config['fine_repeats'])]
                    if all(x['passed'] for x in repeats):
                        steady=await self.stage(session,lower,self.config['steady_seconds'],'steady-2h')
                        if steady['passed']:break
                    upper=lower;lower=round(lower-self.config['fine_resolution'],4)
                if lower<=0 or lower<minimum:raise RuntimeError('No candidate at or above the minimum search rate passed repeated/long stability verification')
                save(self.directory/'capacity.json',{'max_stable_rate':lower,'first_failing_or_unverified_upper':upper,
                    'require_queue_stability':self.config.get('require_queue_stability',True),
                    'capacity_definition':self.config.get('capacity_definition','SLO, resource and queue stability constrained rate'),
                    'slo':self.config['slo'],'validated_seconds':self.config['steady_seconds'],'healthy_workers':len(self.expected),
                    'workload':self.config.get('workload',{'name':'fixed'}),
                    'deployment_scope':self.config['scope'],'execution_mode':self.config['execution_mode']})
                for multiplier in self.config['overload_multipliers']:await self.recovery(session,lower,multiplier)
                if self.config.get('extended_steady_seconds'):
                    await self.stage(session,lower,self.config['extended_steady_seconds'],'extended-steady')
                save(self.directory/'summary.json',{'status':'main_workload_complete','capacity':lower,'upper_bound':upper,'stages':self.results,
                    'supplementary':'Not executed: each supplementary workload requires its own frozen SLO and context compatibility.',
                    'all_devices_accepted':self.frozen['all_devices_accepted']})
            finally:
                timeline.cancel()
                if self.active:
                    for task in self.active:task.cancel()
                    await asyncio.gather(*list(self.active),return_exceptions=True)


def pool_device_inventory(serving,healthy_workers):
    """Include every PP stage of the frozen, healthy generation workers."""
    healthy=set(healthy_workers);inventory={};matched=set()
    for name,router in router_configs(serving).items():
        pool=router.get('pool',name);devices=set()
        for worker in router_workers(serving,name).values():
            if worker['base_url'].removesuffix('/v1') not in healthy:continue
            matched.add(worker['base_url'].removesuffix('/v1'))
            for placement in worker.get('placements', [worker]):
                if not placement.get('devices'):raise ValueError('Frozen worker has no device inventory')
                devices.update((placement['node'],int(device)) for device in placement['devices'])
        if devices:inventory[pool]=devices
    if not healthy or matched!=healthy:raise ValueError('Frozen healthy worker device coverage is incomplete')
    return inventory


def pool_utilization(inventory,samples,at,max_age=25):
    """Device mean within a pool; incomplete coverage never implies idle cards."""
    rows=[]
    for pool,expected in inventory.items():
        values=[];missing=[]
        for node,device in sorted(expected):
            sample=samples.get(node,{})
            age=at-sample.get('received_at',-float('inf'))
            devices={d['device']:d for d in sample.get('device',{}).get('values',[])}
            value=devices.get(device,{}).get('utilization_percent')
            if (0<=age<=max_age and type(value) in (int,float)
                    and math.isfinite(value) and 0<=value<=100):values.append(value)
            else:missing.append(f'{node}:{device}')
        observed=sum(values)/len(values) if values else None
        rows.append({'time':at,'pool':pool,'expected_devices':len(expected),'sampled_devices':len(values),
            'coverage_ratio':len(values)/len(expected),'complete':not missing,
            'utilization_mean_percent':observed if not missing else None,
            'observed_utilization_mean_percent':observed,
            'observed_device_min_percent':min(values) if values else None,
            'observed_device_max_percent':max(values) if values else None,'missing_devices':missing})
    return rows


def export_pool_utilization(directory):
    """Derive experiment results from retained telemetry, without new SMI calls."""
    directory=Path(directory)
    serving=json.loads((directory/'serving-snapshot.json').read_text())
    frozen=json.loads((directory/'preflight.json').read_text())
    config=json.loads((directory/'config.json').read_text())
    inventory=pool_device_inventory(serving,frozen['healthy_workers'])
    stages=json.loads((directory/'stages.json').read_text()) if (directory/'stages.json').exists() else []
    status=json.loads((directory/'status.json').read_text())
    gateway_node=serving['services']['gateway']['node'];samples={};latest=[];aggregates={}
    source=directory/'telemetry.jsonl';remaining=source.stat().st_size;source_bytes=remaining
    target=directory/'pool-utilization.csv';temporary=target.with_suffix('.csv.tmp');points=0
    fields=['time','case','phase','rate','pool','expected_devices','sampled_devices','coverage_ratio','complete',
        'utilization_mean_percent','observed_utilization_mean_percent','observed_device_min_percent',
        'observed_device_max_percent','missing_devices']
    with source.open('rb') as stream,temporary.open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader()
        while remaining:
            line=stream.readline(remaining);remaining-=len(line)
            if not line or not line.endswith(b'\n'):break  # live receiver may be writing its last record
            sample=json.loads(line);samples[sample['node']]=sample
            if sample['node']!=gateway_node:continue
            at=sample['received_at'];latest=pool_utilization(inventory,samples,at,config.get('telemetry_max_age_seconds',25))
            stage=next((s for s in stages if s['start']<=at<s['end']),None)
            for row in latest:
                tagged={**row,'case':stage['case'] if stage else None,
                    'phase':stage['label'] if stage else 'outside-completed-measurement',
                    'rate':stage['rate'] if stage else None}
                writer.writerow({**tagged,'missing_devices':';'.join(row['missing_devices'])});points+=1
                if not stage:continue
                key=(stage['case'],row['pool'])
                agg=aggregates.setdefault(key,{'case':stage['case'],'phase':stage['label'],'rate':stage['rate'],
                    'pool':row['pool'],'expected_devices':row['expected_devices'],'samples':0,
                    'complete_samples':0,'coverage_sum':0,'coverage_min':1,'observed_sum':0,
                    'observed_samples':0,'observed_pool_mean_peak_percent':None,'observed_device_peak_percent':None})
                agg['samples']+=1;agg['complete_samples']+=row['complete'];agg['coverage_sum']+=row['coverage_ratio']
                agg['coverage_min']=min(agg['coverage_min'],row['coverage_ratio'])
                observed=row['observed_utilization_mean_percent']
                if observed is not None:
                    agg['observed_sum']+=observed;agg['observed_samples']+=1
                    agg['observed_pool_mean_peak_percent']=max(agg['observed_pool_mean_peak_percent'] or 0,observed)
                    agg['observed_device_peak_percent']=max(agg['observed_device_peak_percent'] or 0,row['observed_device_max_percent'])
    temporary.replace(target);summaries=[]
    for agg in aggregates.values():
        n=agg['samples'];observed_n=agg.pop('observed_samples');total=agg.pop('observed_sum')
        agg['coverage_mean']=agg.pop('coverage_sum')/n
        agg['observed_utilization_mean_percent']=total/observed_n if observed_n else None
        complete=agg['complete_samples']==n
        agg['utilization_mean_percent']=agg['observed_utilization_mean_percent'] if complete else None
        agg['pool_mean_peak_percent']=agg['observed_pool_mean_peak_percent'] if complete else None
        summaries.append(agg)
    report={'analysis':'Derived from existing experiment telemetry; load and collection unchanged.',
        'processor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'generated_at':time.time(),
        'experiment_finished_at_export':status.get('finished',False),'telemetry_bytes_read':source_bytes,'rows':points,
        'scope':'Healthy AWQ logical devices only; excludes failed workers, Embedding and reserved/other-service devices.',
        'semantics':'Arithmetic device mean within each vendor pool, sampled at Gateway telemetry times (about 5s), using latest node samples no older than the frozen telemetry age. Stage means/peaks use completed measurement windows only. Missing/stale/invalid values are not zero. Partial-coverage observed values cannot establish whole-pool utilization. Vendor SMI utilization definitions may differ.',
        'latest':latest,'stages':summaries}
    save(directory/'pool-utilization-summary.json',report)
    return report


def export_results(directory,config,stages,routes,plots=True):
    """Standalone CSV and scientific PNG/PDF figures; no latency main plots."""
    fields=['label','case','repeat','rate','actual_rate','goodput','completed_rate','slo_ratio','success_ratio','sent','planned','send_lag_p99','queue_stable','queue_stability_required','load_valid','passed','start','end']
    with (directory/'pressure.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(stages)
    with (directory/'workload-pressure.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=['workload_name',*fields],extrasaction='ignore');writer.writeheader()
        for stage in stages:
            for name,metrics in stage.get('by_workload',{}).items():
                writer.writerow({**stage,**metrics,'workload_name':name})
    if not plots:return
    if all((directory/name).exists() for name in ['preflight.json','serving-snapshot.json','config.json','status.json','telemetry.jsonl']):
        export_pool_utilization(directory)
    bins={};requests=0;associated=0
    snapshot=directory/'serving-snapshot.json'
    serving=json.loads(snapshot.read_text()) if snapshot.exists() else None
    vendors_by_worker={};vendors_by_backend={}
    if serving:
        for name,router in router_configs(serving).items():
            pool=router.get('pool')
            if pool in {'ascend','musa','iluvatar'}:
                vendors_by_backend[router['backend']]=pool
                vendors_by_worker.update({w['base_url'].removesuffix('/v1'):pool for w in router_workers(serving,name).values()})
        for w in generation_workers(serving).values():
            node=w.get('node','')
            vendor='ascend' if node.startswith('ascend') else 'musa' if node.startswith('s5000') else 'iluvatar' if node.startswith('iluvatar') else None
            if vendor:vendors_by_worker.setdefault(w['base_url'].removesuffix('/v1'),vendor)
    request_fields=['index','case','phase','workload_name','requested_input_tokens','requested_output_tokens','request_id','selected_pool','worker','vendor','embedding_ms','gateway_queue_ms','scheduled_at','sent_at','send_lag','completed_at','input_tokens','output_tokens','http_status','error','success','slo_pass',
                    *[field for key,field in SLO_FIELDS.items() if key in config['slo']]]
    with (directory/'requests.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=request_fields,extrasaction='ignore');writer.writeheader()
        if (directory/'requests.jsonl').exists():
            for line in (directory/'requests.jsonl').open(encoding='utf-8'):
                row=json.loads(line);route=routes.get(row.get('request_id'),{})
                row['worker']=route.get('worker',row.get('worker'))
                if row['worker']:
                    row['vendor']=vendors_by_worker.get(row['worker'],row.get('vendor'))
                elif row.get('selected_pool') in vendors_by_backend:
                    row['vendor']=vendors_by_backend[row['selected_pool']]
                writer.writerow(row)
                if row['phase'] in {'direct-smoke','pooled-preflight'}:continue
                requests+=1;associated+=bool(row['worker'])
                sent=bins.setdefault(int(row['sent_at']),Counter());sent['sent']+=1
                done=bins.setdefault(int(row['completed_at']),Counter());done['completed']+=row['success'];done['goodput']+=row['slo_pass']
                done['errors']+=not row['success']
                if row.get('vendor'):
                    done[row['vendor']+'_completed']+=row['success'];sent[row['vendor']+'_sent']+=1
    save(directory/'metric-coverage.json',{'requests':requests,'requests_associated_with_worker':associated,
        'association_scope':'Forwarded requests joined by Gateway request ID; Gateway rejections have no worker.',
        'selected_pool_scope':'x-moqe-expert records the L1 hardware-pool choice when returned; absence is kept missing, not assigned by guess.',
        'network_scope':'TCP payload accounting includes retransmissions and excludes IP/TCP headers.',
        'limitations':['Ascend .23 device SMI may be unavailable; missing samples are not zero.',
            'Router internal queue is not exposed by the installed metrics; do not substitute backend waiting or Gateway inflight for it.',
            'Device utilization includes any other activity sharing the physical device.',
            'Current deployment retains mixed execution modes and a load generator sharing the .96 compute host.']})
    time_fields=['time','sent','completed','goodput','errors','ascend_sent','musa_sent','iluvatar_sent','ascend_completed','musa_completed','iluvatar_completed']
    with (directory/'request-timeline.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=time_fields);writer.writeheader()
        for t in sorted(bins):writer.writerow({'time':t,**{k:bins[t].get(k,0) for k in time_fields[1:]}})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        save(directory/'plot-error.json',{'reason':'matplotlib unavailable; CSV retained for rendering in an existing plotting environment'});return
    def finish(fig,name):
        fig.tight_layout();fig.savefig(directory/(name+'.png'),dpi=180);fig.savefig(directory/(name+'.pdf'));plt.close(fig)
    selected=[s for s in stages if s['label'] in {'coarse','fine','boundary-repeat','low-rate-slo-check'}]
    if selected:
        selected=sorted(selected,key=lambda s:s['actual_rate']);x=[s['actual_rate'] for s in selected]
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[s['goodput'] for s in selected],'o-',label='Goodput@SLO');ax.plot(x,[s['completed_rate'] for s in selected],'s-',label='Successful completion');ax.plot(x,x,'--',color='gray',label='y=x');ax.set(xlabel='Actual arrival rate (req/s)',ylabel='Requests per second');ax.legend();ax.grid(alpha=.25);finish(fig,'capacity')
        fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,[s['slo_ratio'] for s in selected],'o-',label='SLO attainment');ax.plot(x,[s['success_ratio'] for s in selected],'s-',label='Success ratio');ax.axhline(config['minimum_slo_ratio'],linestyle='--',color='gray');ax.set(xlabel='Actual arrival rate (req/s)',ylabel='Fraction',ylim=(0,1.02));ax.legend();ax.grid(alpha=.25);finish(fig,'attainment')
    if bins:
        times=list(range(min(bins),max(bins)+1));origin=times[0];x=[(t-origin)/60 for t in times]
        def smoothed(key,width=30):return [sum(bins.get(t-j,{}).get(key,0) for j in range(width))/width for t in times]
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        for key in ['sent','completed','goodput']:axes[0].plot(x,smoothed(key),label=key)
        axes[0].set(ylabel='req/s (30 s mean)');axes[0].legend()
        for vendor in ['ascend','musa','iluvatar']:axes[1].plot(x,smoothed(vendor+'_completed'),label=vendor)
        axes[1].set(xlabel='Time (minutes)',ylabel='Successful req/s');axes[1].legend();finish(fig,'request-time')
    timeline=[]
    if (directory/'timeline.jsonl').exists():
        timeline=[json.loads(line) for line in (directory/'timeline.jsonl').open()]
    if timeline:
        t0=timeline[0]['time'];x=[(r['time']-t0)/60 for r in timeline]
        fig,ax=plt.subplots(figsize=(10,4))
        for key in ['client_active','gateway_inflight','gateway_waiting','waiting','running']:ax.plot(x,[r.get(key) for r in timeline],label=key)
        ax.set(xlabel='Time (minutes)',ylabel='Requests');ax.legend();finish(fig,'backlog-time')
    resources={};devices={};health={}
    if (directory/'telemetry.jsonl').exists():
        for line in (directory/'telemetry.jsonl').open():
            r=json.loads(line);t=r['received_at'];node=r['node']
            for role in ['admission','gateway','router','model']:
                ps=[p for p in r.get('process',{}).get('processes',[]) if p.get('role')==role]
                if ps:resources.setdefault(node+'/'+role,[]).append((t,sum(p['rss_bytes'] for p in ps)/1024**2,sum(p['cpu_seconds'] for p in ps),sum(p['tcp_established'] for p in ps)))
            for d in r.get('device',{}).get('values',[]):devices.setdefault(node+'/'+str(d['device']),[]).append((t,d['utilization_percent'],d['memory_used_mib']))
            for name,w in r.get('workers',{}).items():health.setdefault(node+'/'+name,[]).append((t,w.get('running'),w.get('waiting')))
    if resources:
        origin=min(v[0][0] for v in resources.values());fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for key,values in resources.items():
            xs=[(v[0]-origin)/60 for v in values]
            axes[0].plot(xs,[v[1] for v in values],label=key);axes[2].plot(xs,[v[3] for v in values],label=key)
            axes[1].plot(xs[1:],[(b[2]-a[2])/(b[0]-a[0])*100 for a,b in zip(values,values[1:])],label=key)
        axes[0].set(ylabel='Sum RSS (MiB)');axes[0].legend(fontsize=6,ncol=2);axes[1].set(ylabel='CPU % (one core=100)');axes[2].set(ylabel='Established TCP sockets',xlabel='Time (minutes)');finish(fig,'process-stability')
    if devices:
        origin=min(v[0][0] for v in devices.values());fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        for key,values in devices.items():
            xs=[(v[0]-origin)/60 for v in values];axes[0].plot(xs,[v[1] for v in values],label=key);axes[1].plot(xs,[v[2] for v in values],label=key)
        axes[0].set(ylabel='Device utilization %');axes[1].set(ylabel='Device memory (MiB)',xlabel='Time (minutes)');finish(fig,'device-stability')


def main(config_path,directory):
    config_path=Path(config_path).resolve()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'experiment.lock').open('x') as out:out.write(str(time.time()))
    config=json.loads(Path(config_path).read_text());serving=expand_config(json.loads((ROOT/config['serving_config']).read_text()))
    save(directory/'config.json',config);save(directory/'serving-snapshot.json',serving)
    sources=[Path(__file__),Path(__file__).with_name('benchmark_telemetry.py'),Path(config_path),ROOT/config['serving_config']]
    save(directory/'sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    experiment=Experiment(config,serving,directory)
    try:
        asyncio.run(experiment.execute())
        experiment.receiver.state={'phase':'complete','finished':True,'time':time.time()}
    except Exception as exc:
        experiment.receiver.state={'phase':'stopped','finished':True,'time':time.time(),'reason':str(exc)}
        save(directory/'failure.json',experiment.receiver.state)
        print(json.dumps(experiment.receiver.state),flush=True)
    finally:
        save(directory/'status.json',experiment.receiver.state)
        # Let all collectors deliver their final sample and stop themselves.
        time.sleep(12)
        try:export_results(directory,config,experiment.results,experiment.receiver.routes)
        except Exception as exc:save(directory/'plot-error.json',{'reason':repr(exc)})
        experiment.receiver.server.shutdown()
        experiment.rows_file.close();experiment.sample_file.close()
    return 0 if experiment.receiver.state['phase']=='complete' else 1


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config');parser.add_argument('--result-dir',required=True)
    parser.add_argument('--export-pool-utilization',action='store_true',help='Only derive pool utilization artifacts from retained experiment telemetry')
    parser.add_argument('--follow',action='store_true',help='Export now, then refresh once after this experiment and its collectors finish')
    args=parser.parse_args()
    if args.follow and not args.export_pool_utilization:parser.error('--follow requires --export-pool-utilization')
    if args.export_pool_utilization:
        report=export_pool_utilization(args.result_dir)
        print(json.dumps({k:report[k] for k in ['rows','experiment_finished_at_export','latest','stages']}),flush=True)
        if args.follow and not report['experiment_finished_at_export']:
            while not json.loads((Path(args.result_dir)/'status.json').read_text()).get('finished'):
                time.sleep(15)
            time.sleep(15)  # allow the retained collectors' final samples to arrive
            report=export_pool_utilization(args.result_dir)
            print(json.dumps({k:report[k] for k in ['rows','experiment_finished_at_export','latest','stages']}),flush=True)
    else:
        if not args.config:parser.error('--config is required when starting an experiment')
        raise SystemExit(main(args.config,args.result_dir))
