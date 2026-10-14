"""Check numerical output, continuous refill, streaming and pipeline overlap."""
import argparse
import json
import subprocess
from pathlib import Path

PROGRAM = r'''
import concurrent.futures, json, time, urllib.request, urllib.error
from pathlib import Path
urls = dict(CONFIG['validation']['references'])
urls['pipeline'] = 'http://%s:%s' % (CONFIG['master'], CONFIG['execution']['api_port'])
def request(item, name):
    body = dict(model='Qwen3-14B', messages=[dict(role='user',content=item['prompt'])],
                temperature=0, max_tokens=item['max_tokens'],
                chat_template_kwargs=dict(enable_thinking=False))
    started = time.monotonic()
    query = urllib.request.Request(urls[name]+'/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    response = json.load(urllib.request.urlopen(query, timeout=150))
    return dict(id=item['id'], backend=name, response=response,
                client_seconds=time.monotonic()-started)
items = CONFIG['execution']['requests']
with concurrent.futures.ThreadPoolExecutor(max_workers=len(items)*len(urls)) as pool:
    futures = [pool.submit(request, item, name) for item in items for name in urls]
    rows = [f.result() for f in concurrent.futures.as_completed(futures)]
comparisons = []
for item in items:
    texts = {r['backend']:r['response']['choices'][0]['message']['content']
             for r in rows if r['id']==item['id']}
    comparisons.append(dict(id=item['id'], texts=texts,
         matches={name:texts[name]==texts['pipeline'] for name in urls if name!='pipeline'}))
invalid = []
for change in [{'stream':True}, {'temperature':0.5}, {'max_tokens':0},
               {'max_tokens':129}, {'model':'wrong'},
               {'messages':[{'role':'user','content':'这是一段验证上下文限制的文本。'*600}]}]:
    body = dict(model='Qwen3-14B', messages=[dict(role='user',content='hi')],
                temperature=0, max_tokens=8)
    body.update(change)
    query = urllib.request.Request(urls['pipeline']+'/v1/chat/completions',
                data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    try:
        urllib.request.urlopen(query, timeout=15)
        status = 200
    except urllib.error.HTTPError as error:
        status = error.code
    invalid.append(status)
health = json.load(urllib.request.urlopen(urls['pipeline']+'/health', timeout=10))
continuous = None
if CONFIG['execution'].get('scheduler') == 'continuous':
    prompt = 'Write the integers from 1 to 30, separated by commas. Start with 1,2,3'
    def streaming(index, maximum):
        body = dict(model='Qwen3-14B', prompt=prompt, max_tokens=maximum,
                    temperature=0, ignore_eos=True, stream=True,
                    stream_options=dict(include_usage=True))
        query = urllib.request.Request(urls['pipeline']+'/v1/completions', data=json.dumps(body).encode(),
                                      headers={'Content-Type':'application/json'})
        started, events, usage, done, identifier = time.monotonic(), [], None, False, None
        with urllib.request.urlopen(query, timeout=180) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                packet = json.loads(payload)
                identifier = packet['id'].removeprefix('cmpl-')
                if packet['choices']:
                    events.append(dict(seconds=time.monotonic()-started, **packet['choices'][0]))
                elif 'usage' in packet:
                    usage = packet['usage']
        return dict(index=index, id=identifier, maximum=maximum, events=events, usage=usage, done=done,
                    text=''.join(e['text'] for e in events), seconds=time.monotonic()-started)
    admitted_before = health['admitted_requests']
    snapshots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(streaming,index,maximum) for index,maximum in enumerate([4,8,16,24,32,64]*2)]
        waiting, streams = set(futures), []
        while waiting:
            completed, waiting = concurrent.futures.wait(waiting,timeout=.05,return_when=concurrent.futures.FIRST_COMPLETED)
            streams.extend(f.result() for f in completed)
            snapshots.append(json.load(urllib.request.urlopen(urls['pipeline']+'/health',timeout=10)))
    for _ in range(100):
        health = json.load(urllib.request.urlopen(urls['pipeline']+'/health', timeout=10))
        if not health['active_requests'] and not health['inflight_batches'] and not health['kv_owners']:
            break
        time.sleep(.02)
    long_item = dict(id='chunked-context', prompt='背景资料：今天气温适宜，实验正在进行。'*80+
                    '请忽略背景资料，只回答9加8的结果，输出数字。', max_tokens=8)
    chunked_response = request(long_item,'pipeline')
    chunked_native = request(long_item,'ascend')
    cancel_query = urllib.request.Request(urls['pipeline']+'/v1/completions',
        data=json.dumps(dict(model='Qwen3-14B',prompt=prompt,max_tokens=128,temperature=0,
                             ignore_eos=True,stream=True)).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(cancel_query,timeout=30) as response:
        for line in response:
            if line.startswith(b'data: '):
                canceled_identifier=json.loads(line[6:])['id'].removeprefix('cmpl-')
                break
    for _ in range(300):
        health=json.load(urllib.request.urlopen(urls['pipeline']+'/health',timeout=10))
        if not health['active_requests'] and not health['kv_owners']:
            break
        time.sleep(.02)
    after_cancel=request(CONFIG['execution']['requests'][0],'pipeline')
    directory = Path('/home/zhangjinhao/LLM-Serving-System')/CONFIG['execution']['result_dir'].removeprefix('/workspace/')
    lane_rows = [[json.loads(line) for line in (directory/('stage0-lane%s.jsonl'%lane)).read_text().splitlines()] for lane in range(2)]
    lane_tokens = [{r['id']:r['generated_token_ids'] for r in rs} for rs in lane_rows]
    canceled_row=next(r for r in lane_rows[0] if r['id']==canceled_identifier)
    chunked_prompt=json.dumps([dict(role='user',content=long_item['prompt'])],ensure_ascii=False)
    chunked_row=next(r for r in reversed(lane_rows[0]) if r['prompt']==chunked_prompt)
    chunked_ok = (len(chunked_row['prompt_token_ids'])>512 and chunked_row['text']=='17' and
                  chunked_row['text']==chunked_native['response']['choices'][0]['message']['content'])
    cancel_ok = (len(canceled_row['generated_token_ids'])<128 and health['kv_owners']==0 and
                 after_cancel['response']['choices'][0]['message']['content']=='5')
    stream_ids = {r['id'] for r in streams}
    baseline_tokens = lane_tokens[0][next(r['id'] for r in streams if r['maximum']==64)]
    token_prefixes_match = all(lane_tokens[0][r['id']]==baseline_tokens[:r['maximum']] for r in streams)
    native_query = urllib.request.Request(urls['ascend']+'/v1/completions',
        data=json.dumps(dict(model='Qwen3-14B', prompt=prompt, max_tokens=64, temperature=0, ignore_eos=True)).encode(),
        headers={'Content-Type':'application/json'})
    native_text = json.load(urllib.request.urlopen(native_query, timeout=180))['choices'][0]['text']
    overlaps = []
    for lane in range(2):
        timeline = [json.loads(line) for line in (directory/('timeline-stage0-lane%s.jsonl'%lane)).read_text().splitlines()]
        session_start=max(i for i,r in enumerate(timeline) if r['event']=='clock_calibration')
        timeline=timeline[session_start:]
        calibration = timeline[0]
        local = [r for r in timeline if r['event']=='execution']
        peer = [r for r in timeline if r['event']=='peer_execution']
        lower_bounds = [max(0,min(a['end'], b['end']-calibration['offset']-calibration['uncertainty'])-
                             max(a['start'], b['start']-calibration['offset']+calibration['uncertainty']))
                        for a in local for b in peer if a['batch_id']!=b['batch_id']]
        overlaps.append(dict(lane=lane, clock_uncertainty_ms=calibration['uncertainty']*1000,
            clock_assumption='Stable monotonic offset during this short acceptance; RTT uncertainty, drift not measured',
            overlapping_windows=sum(x>0 for x in lower_bounds), conservative_max_overlap_ms=max(lower_bounds,default=0)*1000,
            max_packed_batch_size=max(r['batch_size'] for r in local)))
    stream_ok = all(r['done'] and len(r['events'])==r['maximum'] and r['usage']['completion_tokens']==r['maximum']
                    and r['events'][-1]['finish_reason']=='length' and r['events'][0]['seconds']<r['events'][-1]['seconds']
                    for r in streams)
    lane_match = all(lane_tokens[0][identifier]==lane_tokens[1][identifier] for identifier in stream_ids)
    early = min(streams,key=lambda r:r['maximum'])
    latest = max(streams,key=lambda r:r['maximum'])
    refilled = any(s['admitted_requests']>admitted_before+8 and s['active_requests']>0 for s in snapshots)
    continuous_passed = (stream_ok and lane_match and refilled and chunked_ok and cancel_ok and
        all(r['overlapping_windows']>0 and r['max_packed_batch_size']>1 for r in overlaps) and
        health['max_inflight_batches']==2 and not health['active_requests'] and not health['kv_owners'])
    continuous = dict(passed=continuous_passed, requests=12, streams=streams, tp_lane_tokens_match=lane_match,
        token_prefixes_match=token_prefixes_match, native_64token_text_match=all(r['text']==native_text for r in streams if r['maximum']==64),
        overlap=overlaps, scope='Stage execution window overlap, including dispatch and native TP; not a GPU kernel profiler',
        health_after=health, refilled_before_all_requests_finished=refilled,
        max_observed_active=max(s['active_requests'] for s in snapshots),
        max_observed_queued=max(s['queue'] for s in snapshots),
        chunked_prefill_passed=chunked_ok, chunked_prompt_tokens=len(chunked_row['prompt_token_ids']),
        cancellation_passed=cancel_ok, canceled_generated_tokens=len(canceled_row['generated_token_ids']),
        shorter_request_seconds=early['seconds'], longer_request_seconds=latest['seconds'])
native_match = all(all(r['matches'].values()) for r in comparisons)
functional_passed = all(x==400 for x in invalid) and health['ready'] and (continuous is None or continuous['passed'])
batch_invariance = native_match and (continuous is None or continuous['token_prefixes_match'] and continuous['native_64token_text_match'])
passed = functional_passed and batch_invariance
print(json.dumps(dict(passed=passed, functional_passed=functional_passed, batch_invariance_passed=batch_invariance,
     native_text_matched_cases=sum(all(r['matches'].values()) for r in comparisons), concurrent_pipeline_requests=len(items),
     comparisons=comparisons, rejected_statuses=invalid, health=health, rows=rows, continuous=continuous), ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent/'configs/ascend_musa.json')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parents[1]/'artifacts/baselines/cross-vendor-pp/scheduled/acceptance.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    program = 'import json\nCONFIG=json.loads('+repr(json.dumps(config))+')\n'+PROGRAM
    result = subprocess.run(['ssh','-F',str(args.ssh_config),'-o','BatchMode=yes',
                             config['stages'][0]['ssh_alias'],'python3 -'],
                            input=program, text=True, encoding='utf-8', capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr)
    report = json.loads(result.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], cases=len(report['comparisons']),
                         functional_passed=report['functional_passed'], batch_invariance_passed=report['batch_invariance_passed'],
                         rejected_statuses=report['rejected_statuses'], output=str(args.output))))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
