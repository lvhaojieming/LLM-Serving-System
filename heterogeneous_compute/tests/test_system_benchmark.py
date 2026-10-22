import asyncio
import json
import io
from pathlib import Path
import socket
import struct
import tempfile
import time
from types import SimpleNamespace

from heterogeneous_compute.system_benchmark import Experiment, meets_slo, summarize
from heterogeneous_compute.benchmark_telemetry import (
    METRIC_NAMES, PacketObserver, device_values, ipv4_tcp, metric_sum, packet_filter, parse_metrics,
)


SLO={'ttft_seconds':2,'tpot_seconds':.1,'e2e_seconds':20}


def row(sent,finished,success=True,**extra):
    return dict(sent_at=sent,completed_at=finished,success=success,ttft=.5,tpot=.05,e2e=finished-sent,send_lag=.01,
                input_tokens=256,output_tokens=128,**extra)


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
            mode=(await request.json())['model']
            if mode=='429':return web.Response(status=429)
            if mode=='slow':
                await asyncio.sleep(.08)
                return web.Response(text='late')
            payload='data: '+json.dumps({'choices':[{'delta':{'role':'assistant','content':''},'finish_reason':None}]})+'\n\n'
            if mode=='stream-error':payload+='data: '+json.dumps({'error':{'message':'engine failed'}})+'\n\n'
            else:payload+='data: '+json.dumps({'choices':[{'delta':{'content':'hello'},'finish_reason':'length'}]})+'\n\n'
            payload+='data: '+json.dumps({'choices':[],'usage':{'prompt_tokens':256,'completion_tokens':128}})+'\n\n'
            if mode!='truncated':payload+='data: [DONE]\n\n'
            return web.Response(text=payload,content_type='text/event-stream')
        app=web.Application();app.router.add_post('/v1/chat/completions',handler)
        runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        experiment=Experiment.__new__(Experiment);experiment.case=0;experiment.completed=[];experiment.rows_file=io.StringIO()
        experiment.config={'output_tokens':128,'input_tokens':256,'model':'normal','endpoint':f'http://127.0.0.1:{port}','timeout_seconds':.03,'slo':SLO}
        try:
            async with aiohttp.ClientSession() as session:
                for mode,error in [('normal',None),('429','429'),('truncated','abnormal_truncation'),('stream-error','stream_error'),('slow','timeout')]:
                    row=await experiment.one(session,[{'role':'user','content':'x'}],0,time.time(),'test',model=mode)
                    assert row.get('error')==error
                    assert row['success']==(error is None)
        finally:await runner.cleanup()
    asyncio.run(run())
