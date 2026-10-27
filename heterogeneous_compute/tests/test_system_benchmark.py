import asyncio
import csv
import json
import io
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from heterogeneous_compute.system_benchmark import Experiment, Receiver, build_workload, export_results, meets_slo, summarize, workload_profiles
from heterogeneous_compute.benchmark_telemetry import (
    METRIC_NAMES, PacketObserver, admission_waiting, device_values, ipv4_tcp, metric_sum, packet_filter, parse_metrics,
    router_worker_snapshot,
)


SLO={'ttft_seconds':2,'tpot_seconds':.1,'e2e_seconds':20}


def test_mixed_workload_is_reproducible_exact_per_block_and_validates_proportions():
    from collections import Counter
    import pytest
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/system_benchmark.json').read_text())
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return list(range(40+messages[0]['content'].count(' garden')))
    workload=build_workload(config,Tokenizer())
    assert build_workload(config,Tokenizer())==workload
    assert Counter(p['workload_name'] for p in workload)=={'short':120,'long-input':90,'long-output':90}
    for start in range(0,len(workload),10):
        assert Counter(p['workload_name'] for p in workload[start:start+10])=={'short':4,'long-input':3,'long-output':3}
    assert {(p['input_tokens'],p['output_tokens']) for p in workload}=={(256,64),(2048,64),(256,1024)}
    assert max(p['input_tokens']+p['output_tokens'] for p in workload)==2112
    bad={**config,'workload':{'profiles':[{'name':'bad','input_tokens':256,'output_tokens':64,'weight':99}]}}
    with pytest.raises(ValueError,match='sum to 100'):workload_profiles(bad)
    with pytest.raises(ValueError,match='exact workload proportions'):build_workload({**config,'prompt_count':256},Tokenizer())
    short=row(0,1,workload_name='short');long=row(0,2,False,workload_name='long-output',error='timeout')
    result=summarize([short,long],0,3,SLO)
    assert result['by_workload']['long-output']['success_ratio']==0
    assert result['by_workload']['short']['success_ratio']==1
    assert result['success_ratio']==.5


def test_admission_samples_do_not_double_count_server_queue_or_hide_missing_metrics():
    metrics=parse_metrics('''haproxy_backend_current_queue{proxy="gateway"} 3
haproxy_server_current_queue{proxy="gateway",server="app"} 3
haproxy_backend_current_queue{proxy="control"} 1
''',METRIC_NAMES['admission'])
    assert admission_waiting({'admission_metrics':metrics})==3
    assert admission_waiting({'admission_metrics':{'error':'timeout'}}) is None
    assert admission_waiting({}) is None
    experiment=Experiment.__new__(Experiment)
    experiment.serving={'admission':{}}
    experiment.receiver=SimpleNamespace(gateway_node='g',history=[
        {'node':'g','received_at':1,'workers':{'worker':{'waiting':2}}},
        {'node':'g','received_at':2,'workers':{'worker':{'waiting':6}}}])
    assert experiment.queue_statistics(0,3)['worker']=={'sample_mean':4,'sample_peak':6,'samples':2}


def row(sent,finished,success=True,**extra):
    return dict(sent_at=sent,completed_at=finished,success=success,ttft=.5,tpot=.05,e2e=finished-sent,send_lag=.01,
                input_tokens=256,output_tokens=128,**extra)


def test_runtime_probe_requires_done_exact_budgets_and_no_backend_error():
    from heterogeneous_compute.benchmark import fixed_stream
    import urllib.request
    import pytest

    def payload(done=True, input_tokens=256, error=False):
        events = [{'choices': [{'delta': {'content': 'a'}, 'finish_reason': None}]},
                  {'choices': [{'delta': {'content': 'b'}, 'finish_reason': 'length'}]},
                  {'choices': [], 'usage': {'prompt_tokens': input_tokens, 'completion_tokens': 128}}]
        if error:
            events.append({'error': {'message': 'worker failed'}})
        data = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
        return (data + ('data: [DONE]\n\n' if done else '')).encode()

    def run(data):
        opener = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(data))
        with patch.object(urllib.request, 'build_opener', return_value=opener):
            return fixed_stream('http://unused', 'model', [], 256, 128, 1)

    assert run(payload())['passed']
    for data in [payload(done=False), payload(input_tokens=255), payload(error=True)]:
        with pytest.raises(RuntimeError):
            run(data)


def test_goodput_completion_window_and_sent_cohort_are_not_confused():
    rows=[row(8,11),row(12,15),row(19,21),row(14,15,False,error='429')]
    result=summarize(rows,10,20,SLO)
    assert result['goodput']==.2  # includes carry-in, excludes drain
    assert result['sent']==3
    assert result['slo_ratio']==2/3  # late success belongs to the sent cohort
    assert result['success_ratio']==2/3
    assert result['drain_completions']==1
    assert result['errors']=={'429':1}


def test_success_but_slow_or_failed_request_never_contributes_goodput():
    slow=row(0,19);slow['ttft']=2.01
    failed=row(0,1,False,error='abnormal_truncation')
    result=summarize([slow,failed],0,20,SLO)
    assert result['completed_rate']==.05
    assert result['goodput']==0
    assert result['slo_ratio']==0
    assert not meets_slo({**row(0,1),'tpot':None},SLO)


def test_tpot_only_slo_ignores_removed_ttft_and_e2e_metrics_but_requires_valid_tpot():
    sample={**row(0,999),'ttft':999,'tpot':.05,'e2e':999}
    assert meets_slo(sample,{'tpot_seconds':.1})
    assert not meets_slo(sample,SLO)  # original three-SLO configuration remains valid
    assert not meets_slo({**sample,'tpot':.101},{'tpot_seconds':.1})
    assert not meets_slo({**sample,'tpot':None},{'tpot_seconds':.1})
    assert not meets_slo({**sample,'success':False},{'tpot_seconds':.1})
    result=summarize([sample],0,1000,{'tpot_seconds':.1})
    assert result['goodput']==.001 and result['slo_ratio']==1


def test_tpot_only_export_omits_cancelled_latency_columns():
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        sample={**row(0,1),'phase':'coarse','request_id':'r','worker':None,'slo_pass':True}
        (directory/'requests.jsonl').write_text(json.dumps(sample)+'\n')
        # CSV production is verified without loading a plotting environment.
        with patch.dict(sys.modules,{'matplotlib':None}):
            export_results(directory,{'slo':{'tpot_seconds':.1}},[],{})
        with (directory/'requests.csv').open(newline='',encoding='utf-8') as handle:
            reader=csv.DictReader(handle);rows=list(reader)
        assert len(rows)==1 and 'tpot' in reader.fieldnames
        assert 'ttft' not in reader.fieldnames and 'e2e' not in reader.fieldnames


def test_metrics_respect_labels_and_exclude_backend_aborts():
    raw='''# comment
vllm:num_requests_waiting{engine="0"} 3
vllm:request_success_total{finished_reason="length"} 10
vllm:request_success_total{finished_reason="abort"} 2
'''
    metrics=parse_metrics(raw)
    assert metric_sum(metrics,'vllm:num_requests_waiting')==3
    assert metric_sum(metrics,'vllm:request_success_total',('abort','error'))==10
    assert metric_sum(metrics,'nonexistent') is None


def test_collector_retains_required_metrics_without_backend_histogram_payload():
    raw='''vllm:num_requests_running{engine="0"} 2
vllm:num_requests_waiting{engine="0"} 3
vllm:request_success_total{finished_reason="length"} 10
vllm:request_success_total{finished_reason="abort"} 2
vllm:time_to_first_token_seconds_bucket{le="2",engine="0"} 20
vllm:time_to_first_token_seconds_sum{engine="0"} 1.2
process_resident_memory_bytes 99999
'''
    metrics=parse_metrics(raw,METRIC_NAMES['worker'])
    assert len(metrics)==4
    assert metric_sum(metrics,'vllm:num_requests_running')==2
    assert metric_sum(metrics,'vllm:num_requests_waiting')==3
    assert metric_sum(metrics,'vllm:request_success_total',('abort','error'))==10
    assert metric_sum(metrics,'vllm:kv_cache_usage_perc') is None


def frame(body,sport=45000,dport=18741,seq=100,src='192.168.162.60',dst='192.168.162.97',options=False):
    ethernet=b'\0'*12+b'\x08\x00'
    ip=bytearray(24 if options else 20);ip[0]=0x40+len(ip)//4;ip[9]=6
    tcp=bytearray(32 if options else 20)
    struct.pack_into('!H',ip,2,len(ip)+len(tcp)+len(body))
    ip[12:16]=socket.inet_aton(src);ip[16:20]=socket.inet_aton(dst)
    struct.pack_into('!HHI',tcp,0,sport,dport,seq);tcp[12]=(len(tcp)//4)<<4;tcp[13]=0x18
    return ethernet+ip+tcp+body


def run_filter(instructions,packet):
    """Execute the small classic-BPF subset used by the socket filter."""
    accumulator=index=pc=0
    while pc<len(instructions):
        code,jt,jf,k=instructions[pc]
        if code==0x28:accumulator=int.from_bytes(packet[k:k+2],'big')
        elif code==0x30:accumulator=packet[k]
        elif code==0xb1:index=(packet[k]&15)*4
        elif code==0x48:accumulator=int.from_bytes(packet[k+index:k+index+2],'big')
        elif code==0x15:pc+=jt if accumulator==k else jf
        elif code==0x06:return k
        else:raise AssertionError('Unexpected BPF instruction')
        pc+=1
    raise AssertionError('BPF did not return')


def test_kernel_filter_copies_full_request_headers_but_short_response_snapshots():
    ins=packet_filter({18600,18601,18741},{18741})
    assert run_filter(ins,frame(b'POST /v1/chat/completions',options=True))==16384
    assert run_filter(ins,frame(b'data: token',sport=18741,dport=45000))==256
    assert run_filter(ins,frame(b'POST /v1/chat/completions',dport=18600))==256
    assert run_filter(ins,frame(b'POST /v1/chat/completions',dport=18601))==256
    assert run_filter(ins,frame(b'unrelated',sport=45001,dport=45002))==0
    udp=bytearray(frame(b'unrelated'));udp[23]=17
    assert run_filter(ins,udp)==0


def test_short_snapshots_preserve_original_tcp_byte_counts_and_strip_padding():
    original=frame(b'x'*5000,sport=18741,dport=45000,options=True)
    packet=ipv4_tcp(original[:256])
    assert packet['size']==5000 and packet['truncated']
    assert len(packet['payload'])==256-14-24-32
    padded=ipv4_tcp(frame(b'abc')+b'ethernet padding')
    assert padded['payload']==b'abc' and not padded['truncated']


def observer():
    return PacketObserver({'backends':{'worker':{'base_url':'http://192.168.162.97:18741/v1'}},
        'gateway':{'address':'192.168.162.60','port':18600},'router':{'port':18601}})


def pool_serving():
    return {'backends':{vendor+'-awq':{'base_url':f'http://192.168.162.{host}:18741/v1','node':node}
            for vendor,host,node in [('ascend',24,'ascend910c-02'),('musa',97,'s5000-01'),('iluvatar',58,'iluvatar150-01')]},
        'gateway':{'address':'192.168.162.60','port':18600},
        'router':{'host':'127.0.0.1','policy':'round_robin','pools':{
            vendor:{'port':port,'metrics_port':port+1,'backend':vendor+'-awq','nodes':[node]}
            for vendor,port,node in [('ascend',18601,'ascend910c-02'),('musa',18603,'s5000-01'),('iluvatar',18605,'iluvatar150-01')]}}}


def test_three_router_ports_preserve_aggregate_gateway_router_traffic_counts():
    capture=PacketObserver(pool_serving())
    assert capture.router_ports=={18601,18603,18605}
    request=b'POST /v1/chat/completions HTTP/1.1\r\n\r\n{}'
    response=b'HTTP/1.1 200 OK\r\n\r\nresponse'
    ins=packet_filter(capture.ports,capture.worker_ports)
    for offset,port in enumerate(sorted(capture.router_ports)):
        outgoing=frame(request,sport=45000+offset,dport=port,src='127.0.0.1',dst='127.0.0.1')
        incoming=frame(response,sport=port,dport=45000+offset,src='127.0.0.1',dst='127.0.0.1')
        assert run_filter(ins,outgoing)==256 and run_filter(ins,incoming)==256
        capture.observe_packet(ipv4_tcp(outgoing));capture.observe_packet(ipv4_tcp(incoming))
    assert capture.bytes=={'gateway_rx_tcp_payload':3*len(response),'gateway_tx_tcp_payload':3*len(request),
                          'router_rx_tcp_payload':3*len(request),'router_tx_tcp_payload':3*len(response)}


def test_embedding_network_bytes_belong_to_gateway_without_becoming_generation_routes():
    config=pool_serving()
    config['backends']['embedding']={'base_url':'http://192.168.162.58:18458/v1','task':'embed'}
    capture=PacketObserver(config)
    request=b'POST /v1/embeddings HTTP/1.1\r\nx-request-id: same-awq-id\r\n\r\n{}'
    response=b'HTTP/1.1 200 OK\r\n\r\n'+b'large embedding array'*100
    assert 18458 in capture.ports and 18458 not in capture.worker_ports
    capture.observe_packet(ipv4_tcp(frame(request,dport=18458,dst='192.168.162.58')))
    capture.observe_packet(ipv4_tcp(frame(response,sport=18458,dport=45000,src='192.168.162.58',dst='192.168.162.60')[:256]))
    assert capture.bytes=={'gateway_rx_tcp_payload':len(response),'gateway_tx_tcp_payload':len(request),
                          'router_rx_tcp_payload':0,'router_tx_tcp_payload':0}
    assert capture.routes.empty() and capture.flows=={}


def test_router_snapshot_keeps_pool_identity_and_does_not_hide_failed_collection():
    report=router_worker_snapshot({'ascend':{'workers':[{'url':'a','is_healthy':True}]},
        'musa':{'workers':[{'url':'m','is_healthy':False}]},'iluvatar':{'error':'connection refused'}})
    assert report['workers']==[{'url':'a','is_healthy':True,'router_pool':'ascend'},
                               {'url':'m','is_healthy':False,'router_pool':'musa'}]
    assert report['errors']=={'iluvatar':'connection refused'}
    receiver=Receiver.__new__(Receiver);receiver.gateway_node='gateway-node'
    receiver.router_members={'ascend':{'a'},'musa':{'m'},'iluvatar':{'i'}}
    receiver.latest={'gateway-node':{'router_workers':report}}
    try:receiver.router_health()
    except RuntimeError as exc:assert 'telemetry failed' in str(exc)
    else:raise AssertionError('A missing Router pool cannot be treated as a complete snapshot')


def test_pool_freeze_detects_cross_pool_worker_registration_even_when_union_is_unchanged():
    receiver=Receiver.__new__(Receiver);receiver.gateway_node='gateway-node'
    receiver.router_members={'ascend':{'a'},'musa':{'m'},'iluvatar':{'i'}}
    reports={pool:{'workers':[{'url':url,'is_healthy':True}]} for pool,url in [('ascend','a'),('musa','m'),('iluvatar','i')]}
    receiver.latest={'gateway-node':{'router_workers':router_worker_snapshot(reports)}}
    assert receiver.workers()=={'a','m','i'}
    assert receiver.router_health()=={'ascend':{'a'},'musa':{'m'},'iluvatar':{'i'}}
    reports['ascend'],reports['musa']=reports['musa'],reports['ascend']
    receiver.latest['gateway-node']['router_workers']=router_worker_snapshot(reports)
    assert receiver.workers()=={'a','m','i'}
    try:receiver.router_health()
    except RuntimeError as exc:assert 'wrong Router pool' in str(exc)
    else:raise AssertionError('Equal union does not imply valid hardware-pool placement')
    # A legacy collector's single workers list remains supported.
    receiver.router_members={'router':{'a','m'}}
    receiver.latest['gateway-node']['router_workers']={'workers':[{'url':'a','is_healthy':True},{'url':'m','is_healthy':False}]}
    assert receiver.router_health()=={'router':{'a'}}


def test_header_reassembly_handles_overlap_and_out_of_order_segments_without_body_storage():
    capture=observer()
    header=b'POST /v1/chat/completions HTTP/1.1\r\nx-request-id: abc\r\n\r\n'
    # Receive the tail before the middle; concatenate-by-arrival would corrupt
    # the header. The overlapping retransmission must not duplicate bytes.
    for start,end in [(0,20),(35,len(header)),(12,35)]:
        capture.observe_packet(ipv4_tcp(frame(header[start:end],seq=100+start)))
    route=capture.routes.get_nowait()
    assert route['request_id']=='abc' and route['worker']=='http://192.168.162.97:18741'
    assert capture.flows=={} and capture.routes.empty()
    huge=header.replace(b'abc',b'def')+b'secret body'*2000
    capture.observe_packet(ipv4_tcp(frame(huge,seq=10000)[:16384]))
    assert capture.routes.get_nowait()['request_id']=='def'
    assert capture.flows=={}
    response=frame(b'data: secret response'*1000,sport=18741,dport=45000,
                   src='192.168.162.97',dst='192.168.162.60')
    capture.observe_packet(ipv4_tcp(response[:256]))
    assert capture.bytes['router_rx_tcp_payload']==len(b'data: secret response'*1000)
    assert capture.flows=={}


def test_truncated_request_header_fails_instead_of_claiming_route_coverage():
    capture=observer()
    packet=ipv4_tcp(frame(b'POST /v1/chat/completions HTTP/1.1\r\nlong-header: '+b'x'*20000)[:16384])
    try:capture.observe_packet(packet)
    except RuntimeError as exc:assert 'truncated' in str(exc)
    else:raise AssertionError('Truncated headers must stop route attribution')
    assert capture.routes.empty()


def test_experiment_stops_when_observer_reports_any_packet_drop():
    experiment=Experiment.__new__(Experiment)
    experiment.deadline=float('inf');experiment.abort=None;experiment.frozen=True
    experiment.receiver=SimpleNamespace(network={'packet_drops_total':1})
    with tempfile.TemporaryDirectory() as directory:
        experiment.directory=Path(directory)
        try:experiment.check()
        except RuntimeError as exc:assert 'dropped packets' in str(exc)
        else:raise AssertionError('Packet loss must invalidate per-worker attribution')


def test_packet_metadata_decodes_owned_request_without_retaining_body():
    body=b'POST /v1/chat/completions HTTP/1.1\r\nx-request-id: abc\r\n\r\nsecret'
    ethernet=b'\0'*12+b'\x08\x00'
    ip=bytearray(20);ip[0]=0x45;ip[9]=6
    struct.pack_into('!H',ip,2,40+len(body));ip[12:16]=socket.inet_aton('192.168.162.60');ip[16:20]=socket.inet_aton('192.168.162.97')
    tcp=bytearray(20);struct.pack_into('!HHI',tcp,0,45000,18741,100);tcp[12]=0x50
    result=ipv4_tcp(ethernet+ip+tcp+body)
    assert result['dst']=='192.168.162.97' and result['dport']==18741
    assert result['size']==len(body) and result['payload']==body
    assert ipv4_tcp(b'bad') is None


def test_vendor_device_parsers_keep_missing_values_missing():
    raw={'GPU':[{'Index':'0','Utilization':{'Gpu':'82%'},'FB Memory Usage':{'Used':'40000MiB','Total':'81920MiB'}}]}
    values=device_values('musa',{'returncode':0,'stdout':json.dumps(raw)})
    assert values[0]['utilization_percent']==82
    assert device_values('ascend',{'timeout':True})==[]
    ascend='| 0     2 | 0000:99:00.0 | 15          0    / 0          12793/ 65536 |'
    assert device_values('ascend',{'returncode':0,'stdout':ascend})[0]['memory_used_mib']==12793


def test_fixed_rate_sender_does_not_wait_for_responses():
    experiment=Experiment.__new__(Experiment)
    experiment.config={'max_client_tasks':100}
    experiment.prompts=[[{'role':'user','content':'x'}]]
    experiment.active=set();experiment.check=lambda:None
    arrivals=[]
    async def one(*args,**kwargs):
        arrivals.append(time.monotonic())
        await asyncio.sleep(.15)
    async def wait(seconds):await asyncio.sleep(seconds)
    experiment.one=one;experiment.wait=wait
    async def run():
        await experiment.send_window(None,40,.1,'test')
        assert len(arrivals)==4
        assert max(arrivals)-min(arrivals)<.13
        await asyncio.gather(*experiment.active)
    asyncio.run(run())


def test_stream_parser_counts_429_truncation_errors_and_total_deadline():
    import aiohttp
    from aiohttp import web
    async def run():
        async def handler(request):
            body=await request.json();mode=body['model']
            if mode=='429':return web.Response(status=429)
            if mode=='slow':
                await asyncio.sleep(.08)
                return web.Response(text='late')
            payload='data: '+json.dumps({'choices':[{'delta':{'role':'assistant','content':''},'finish_reason':None}]})+'\n\n'
            if mode=='stream-error':payload+='data: '+json.dumps({'error':{'message':'engine failed'}})+'\n\n'
            else:payload+='data: '+json.dumps({'choices':[{'delta':{'content':'hello'},'finish_reason':'length'}]})+'\n\n'
            input_length=2048 if body['messages'][0]['content']=='long-input' else 256
            output_length=body['max_tokens'] if mode!='wrong-budget' else 64
            payload+='data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':input_length,'completion_tokens':output_length}})+'\n\n'
            if mode!='truncated':payload+='data: [DONE]\n\n'
            headers={'x-moqe-expert':'musa-awq','x-request-id':'req-1','x-heteroserve-embedding-ms':'nan' if mode=='bad-embedding' else '1.25'}
            if mode=='missing-embedding':headers.pop('x-heteroserve-embedding-ms')
            return web.Response(text=payload,content_type='text/event-stream',headers=headers)
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        experiment=Experiment.__new__(Experiment);experiment.case=0;experiment.completed=[];experiment.rows_file=io.StringIO()
        experiment.embedding_required=True
        experiment.config={'output_tokens':128,'input_tokens':256,'model':'normal','endpoint':f'http://127.0.0.1:{port}','timeout_seconds':.03,'slo':SLO}
        try:
            async with aiohttp.ClientSession() as session:
                for mode,error in [('normal',None),('429','429'),('truncated','abnormal_truncation'),('stream-error','stream_error'),('slow','timeout'),('bad-embedding','invalid_embedding_measurement'),('missing-embedding','missing_embedding_measurement')]:
                    row=await experiment.one(session,[{'role':'user','content':'x'}],0,time.time(),'test',model=mode)
                    assert row.get('error')==error
                    assert row['success']==(error is None)
                    if mode=='normal':assert row['selected_pool']=='musa-awq' and row['request_id']=='req-1' and row['embedding_ms']==1.25
                experiment.config['slo']={'tpot_seconds':.1}
                sample=await experiment.one(session,[{'role':'user','content':'x'}],0,time.time(),'test',model='normal')
                assert sample['success'] and sample['tpot'] is not None
                assert 'ttft' not in sample and 'e2e' not in sample
                persisted=json.loads(experiment.rows_file.getvalue().splitlines()[-1])
                assert 'ttft' not in persisted and 'e2e' not in persisted
                experiment.config.pop('input_tokens');experiment.config.pop('output_tokens')
                for name,input_length,output_length in [('short',256,64),('long-input',2048,64),('long-output',256,1024)]:
                    prompt={'messages':[{'role':'user','content':name}],'workload_name':name,'input_tokens':input_length,'output_tokens':output_length}
                    sample=await experiment.one(session,prompt,0,time.time(),'mixed',model='normal')
                    assert sample['success'] and sample['workload_name']==name
                    assert sample['input_tokens']==input_length and sample['requested_output_tokens']==output_length
                bad=await experiment.one(session,prompt,0,time.time(),'mixed',model='wrong-budget')
                assert not bad['success'] and bad['error']=='length_mismatch'
        finally:await runner.cleanup()
    asyncio.run(run())
