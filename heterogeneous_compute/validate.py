"""Reproduce concurrent FIFO acceptance against native vendor references."""
import argparse
import json
import subprocess
from pathlib import Path

PROGRAM = r'''
import concurrent.futures, json, time, urllib.request, urllib.error
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
passed = all(all(r['matches'].values()) for r in comparisons) and all(x==400 for x in invalid) and health['ready']
print(json.dumps(dict(passed=passed, concurrent_pipeline_requests=len(items),
     comparisons=comparisons, rejected_statuses=invalid, health=health, rows=rows), ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ssh-config', required=True, type=Path)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).parent/'configs/ascend_musa.json')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parents[1]/'artifacts/baselines/cross-vendor-pp/reorganized-acceptance.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    program = 'import json\nCONFIG=json.loads('+repr(json.dumps(config))+')\n'+PROGRAM
    result = subprocess.run(['ssh','-F',str(args.ssh_config),'-o','BatchMode=yes',
                             config['stages'][0]['ssh_alias'],'python3 -'],
                            input=program, text=True, encoding='utf-8', capture_output=True, check=True)
    report = json.loads(result.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], cases=len(report['comparisons']),
                         rejected_statuses=report['rejected_statuses'], output=str(args.output))))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
