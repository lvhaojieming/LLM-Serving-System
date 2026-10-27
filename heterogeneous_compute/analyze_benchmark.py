"""Offline comparison of frozen system-benchmark artifacts; never starts services."""
import argparse
import base64
import bisect
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heterogeneous_compute.serving import generation_workers
from heterogeneous_compute.system_benchmark import percentile, summarize
from heterogeneous_compute.benchmark_telemetry import metric_sum


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), mean=statistics.mean(values), min=min(values),
                p50=percentile(values, .5), p95=percentile(values, .95),
                p99=percentile(values, .99), max=max(values)) if values else None


def interpolate(points, at, max_gap=25):
    times = [p[0] for p in points]
    i = bisect.bisect_left(times, at)
    if i < len(points) and times[i] == at:
        return points[i][1]
    if i == 0 or i == len(points):
        return None
    a, b = points[i-1], points[i]
    if b[0]-a[0] > max_gap or b[1] < a[1]:
        return None
    return a[1]+(b[1]-a[1])*(at-a[0])/(b[0]-a[0])


def analyze(directory):
    config, serving, frozen = [read(directory/n) for n in
                              ['config.json', 'serving-snapshot.json', 'preflight.json']]
    workers = generation_workers(serving)
    pool_by_url = {url: pool for pool, urls in frozen['router_pools'].items() for url in urls}
    pool_by_name = {name: pool_by_url[w['base_url'].removesuffix('/v1')]
                    for name, w in workers.items() if w['base_url'].removesuffix('/v1') in pool_by_url}
    routes = {}
    different_worker_routes = 0
    drops = 0
    observer_errors = []
    for trace in lines(directory/'routes.jsonl'):
        drops = max(drops, trace.get('packet_drops_total', 0))
        if trace.get('error'):
            observer_errors.append(trace['error'])
        for route in trace['routes']:
            key = route['request_id']
            if key in routes and routes[key]['worker'] != route['worker']:
                different_worker_routes += 1
            if key not in routes or route['observed_at'] < routes[key]['observed_at']:
                routes[key] = route
    requests = list(lines(directory/'requests.jsonl'))
    for row in requests:
        route = routes.get(row.get('request_id'))
        row['_worker'] = route['worker'] if route else None
        row['_pool'] = pool_by_url.get(row['_worker'])
    counters = defaultdict(list)
    snapshots = defaultdict(list)
    snapshot_times = {}
    device_nodes = defaultdict(set)
    for name, pool in pool_by_name.items():
        for device in workers[name]['devices']:
            device_nodes[workers[name]['node']].add(device)
    clocks = []
    gateway_processes = []
    for sample in lines(directory/'telemetry.jsonl'):
        compact = {'time': sample['received_at'], 'workers': {}}
        for name, worker in sample.get('workers', {}).items():
            if name not in pool_by_name:
                continue
            metrics = worker.get('metrics', [])
            item = {k: worker.get(k) for k in ['running', 'waiting']}
            item['kv_fraction'] = metric_sum(metrics, 'vllm:gpu_cache_usage_perc')
            if item['kv_fraction'] is None:
                item['kv_fraction'] = metric_sum(metrics, 'vllm:kv_cache_usage_perc')
            compact['workers'][name] = item
            for counter in ['vllm:generation_tokens_total', 'vllm:num_preemptions_total']:
                value = metric_sum(metrics, counter)
                if value is not None:
                    counters[(name, counter)].append((sample['time'], value))
        snapshots[sample['node']].append(compact)
        for proc in sample.get('process', {}).get('processes', []):
            if proc.get('role') == 'gateway':
                gateway_processes.append({'time': sample['received_at'],
                    'cpu': proc['cpu_seconds'], 'rss': proc['rss_bytes'],
                    'tcp': proc['tcp_established']})
        if sample['node'].startswith('s5000'):
            raw = sample.get('device', {}).get('raw', {})
            if raw.get('returncode') == 0:
                for gpu in json.loads(raw['stdout'])['GPU']:
                    if int(gpu['Index']) in device_nodes[sample['node']]:
                        clocks.append({'time': sample['received_at'], 'node': sample['node'],
                            'graphics_mhz': float(gpu['Clocks']['Graphics'].removesuffix('MHz')),
                            'memory_mhz': float(gpu['Clocks']['Memory'].removesuffix('MHz')),
                            'thermal': gpu.get('Thermal Slowdown Stats')})
    for node, values in snapshots.items():
        values.sort(key=lambda r:r['time'])
        snapshot_times[node] = [s['time'] for s in values]
    for points in counters.values():
        points.sort()
    timeline = list(lines(directory/'timeline.jsonl'))
    utilization = read(directory/'pool-utilization-summary.json')['stages']
    results = []
    for stage in read(directory/'stages.json'):
        start, end = stage['start'], stage['end']
        duration = end-start
        case_rows = [r for r in requests if r['case'] == stage['case']]
        cohort = [r for r in case_rows if start <= r['sent_at'] < end]
        check = summarize(case_rows, start, end, config['slo'])
        for key in ['sent', 'cohort_success', 'cohort_good', 'goodput', 'success_ratio', 'slo_ratio', 'errors']:
            assert check[key] == stage[key], (directory, stage['rate'], key)
        assert len(cohort) == stage['planned']
        r = {k:stage[k] for k in ['case','rate','sent','cohort_success','cohort_good','actual_rate',
             'goodput','success_ratio','slo_ratio','passed','load_valid','queue_stable','errors','drain_completions']}
        r.update(run_id=directory.name, start=start, end=end,
            successful_tpot_ms=stats([x['tpot']*1000 for x in cohort if x['success']]),
            http_statuses=dict(Counter(str(x.get('http_status')) for x in cohort)),
            unknown_worker_requests=sum(x['_worker'] is None for x in cohort),
            error_attribution={}, pools={})
        for error in sorted(stage['errors']):
            failed = [x for x in cohort if not x['success'] and x.get('error') == error]
            r['error_attribution'][error] = {
                'http_statuses': dict(Counter(str(x.get('http_status')) for x in failed)),
                'associated_pool': dict(Counter(x['_pool'] or 'unknown' for x in failed)),
                'selected_pool': dict(Counter(x.get('selected_pool') or 'unknown' for x in failed)),
                'without_request_id': sum(not x.get('request_id') for x in failed)}
        for pool, urls in frozen['router_pools'].items():
            names = [n for n,p in pool_by_name.items() if p == pool]
            nodes = set(workers[n]['node'] for n in names)
            arrivals = [x for x in routes.values() if x['worker'] in urls and start <= x['observed_at'] < end]
            sent = [x for x in cohort if x['_pool'] == pool]
            completed = [x for x in case_rows if x['_pool'] == pool and x['success']
                         and start <= x['completed_at'] < end]
            pooled_points = []
            individual = defaultdict(list)
            for point in timeline:
                if not start <= point['time'] < end:
                    continue
                active = {}
                for node in nodes:
                    index = bisect.bisect_right(snapshot_times[node], point['time'])-1
                    if index < 0:
                        continue
                    snap = snapshots[node][index]
                    if point['time']-snap['time'] <= config['telemetry_max_age_seconds']:
                        active.update({n:v for n,v in snap['workers'].items() if n in names})
                if len(active) != len(names) or any(v['running'] is None or v['waiting'] is None for v in active.values()):
                    continue
                pooled_points.append({'seconds':point['time']-start,
                    'running':sum(v['running'] for v in active.values()),
                    'waiting':sum(v['waiting'] for v in active.values())})
                for name,v in active.items():
                    individual[name].append(v)
            tokens = []
            preemptions = []
            for name in names:
                for metric, target in [('vllm:generation_tokens_total',tokens),('vllm:num_preemptions_total',preemptions)]:
                    points=counters[(name,metric)]
                    if any(right[1]<left[1] for left,right in zip(points,points[1:])
                           if left[0]<=end and right[0]>=start):
                        continue
                    a,b = [interpolate(points, t) for t in [start,end]]
                    if a is not None and b is not None and b >= a:
                        target.append(b-a)
            first = [x['waiting'] for x in pooled_points if x['seconds'] < duration/3]
            last = [x['waiting'] for x in pooled_points if x['seconds'] >= duration*2/3]
            by_worker = {}
            for name, values in individual.items():
                cap = 8 if pool == 'ascend' else 16
                by_worker[name] = {'running':stats([v['running'] for v in values]),
                    'waiting':stats([v['waiting'] for v in values]),
                    'at_max_sequences_fraction':sum(v['running']>=cap for v in values)/len(values),
                    'kv_usage_raw':stats([v['kv_fraction'] for v in values]),
                    'routed_cohort':sum(x['_worker'] == workers[name]['base_url'].removesuffix('/v1') for x in sent),
                    'long_output_cohort':sum(x['_worker'] == workers[name]['base_url'].removesuffix('/v1') and x['workload_name']=='long-output' for x in sent)}
            req_status = Counter('success' if x['success'] else x.get('error','unknown') for x in sent)
            u = next(x for x in utilization if x['case']==stage['case'] and x['pool']==pool)
            r['pools'][pool] = {'healthy_workers':len(names), 'arrivals_at_workers':len(arrivals),
                'arrival_rps':len(arrivals)/duration, 'successful_completion_rps':len(completed)/duration,
                'cohort_associated_requests':len(sent), 'cohort_statuses':dict(req_status),
                'cohort_workloads':dict(Counter(x['workload_name'] for x in sent)),
                'backend_output_tokens_per_second':sum(tokens)/duration if len(tokens)==len(names) else None,
                'token_counter_workers_covered':len(tokens),
                'preemption_counter_delta':sum(preemptions) if len(preemptions)==len(names) else None,
                'running':stats([x['running'] for x in pooled_points]),
                'waiting':stats([x['waiting'] for x in pooled_points]),
                'waiting_first_third_mean':statistics.mean(first) if first else None,
                'waiting_last_third_mean':statistics.mean(last) if last else None,
                'queue_time_series':pooled_points, 'workers':by_worker,
                'successful_tpot_ms':stats([x['tpot']*1000 for x in sent if x['success']]),
                'utilization':u}
        points = [x for x in timeline if start <= x['time'] < end]
        r['gateway'] = {k:stats([x.get(k) for x in points]) for k in ['gateway_inflight','gateway_waiting']}
        gp = [x for x in gateway_processes if start <= x['time'] < end]
        cpu = [(b['cpu']-a['cpu'])/(b['time']-a['time'])*100 for a,b in zip(gp,gp[1:])]
        r['gateway']['one_core_cpu_percent'] = stats(cpu)
        r['gateway']['established_tcp'] = stats([x['tcp'] for x in gp])
        r['musa_clocks'] = {node:{key:stats([x[key] for x in clocks if x['node']==node and start<=x['time']<end])
                                 for key in ['graphics_mhz','memory_mhz']}
                            for node in ['s5000-01','s5000-02']}
        r['routed_arrival_rps_total'] = sum(x['arrival_rps'] for x in r['pools'].values())
        r['successful_completion_rps_pool_sum'] = sum(x['successful_completion_rps'] for x in r['pools'].values())
        assert abs(r['successful_completion_rps_pool_sum']-stage['completed_rate']) < 1e-9
        assert sum(r['errors'].values())+r['cohort_success'] == r['sent']
        assert sum(len(x['queue_time_series']) for x in r['pools'].values()) > 0
        results.append(r)
    return {'run_id':directory.name,'route_observer_packet_drops':drops,
            'route_observer_errors':observer_errors,'multi_worker_route_conflicts':different_worker_routes,
            'request_rows':len(requests),'stages':results}


def render_report(result, output):
    stages=sorted([s for run in result['runs'] for s in run['stages']],key=lambda s:s['rate'])
    if {r['run_id'] for r in result['runs']} != {'20261027T105055Z','20261027T113010Z'}:
        rows=''.join('<tr>'+''.join('<td>'+html.escape(str(s[k]))+'</td>' for k in
                     ['run_id','rate','sent','success_ratio','slo_ratio','goodput','passed'])+'</tr>' for s in stages)
        (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Benchmark analysis</title>'
            '<h1>Offline benchmark analysis</h1><table border="1"><tr><th>Run</th><th>Rate</th><th>Sent</th>'
            '<th>Success ratio</th><th>SLO ratio</th><th>Goodput</th><th>Passed</th></tr>'+rows+'</table>'
            '<p>Detailed evidence and metric semantics: <a href="analysis.json">analysis.json</a>; '
            '<a href="pool-analysis.csv">pool-analysis.csv</a>.</p>',encoding='utf-8')
        return
    single=next(s for s in stages if s['rate']==18)
    def table(headers,rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    overall=table(['请求/秒','请求数','完整批次成功率','SLO达标率','窗口Goodput','429','5xx','超时','短测判定'],[
        [s['rate'],s['sent'],f"{s['success_ratio']:.2%}",f"{s['slo_ratio']:.2%}",f"{s['goodput']:.2f}",
         s['errors'].get('429',0),s['errors'].get('5xx',0),s['errors'].get('timeout',0),'通过' if s['passed'] else '未通过'] for s in stages])
    pool_rows=[]
    for name,label in [('ascend','昇腾'),('musa','摩尔'),('iluvatar','天数')]:
        p=single['pools'][name]
        pool_rows.append([label,p['healthy_workers'],f"{p['arrival_rps']:.3f}",f"{p['successful_completion_rps']:.3f}",
            f"{p['backend_output_tokens_per_second']:.0f}",f"{p['running']['mean']:.1f}",
            f"{p['waiting']['mean']:.1f} / {p['waiting']['max']:.0f}",
            f"{p['successful_tpot_ms']['p99']:.2f}"])
    pools=table(['厂商池','健康实例','到达AWQ req/s','成功完成 req/s','后端生成 token/s','平均running','平均/峰值waiting','请求平均TPOT的p99(ms)'],pool_rows)
    musa=single['pools']['musa'];fraction=statistics.mean(w['at_max_sequences_fraction'] for w in musa['workers'].values())
    queue_start=musa['waiting_first_third_mean'];queue_end=musa['waiting_last_third_mean']
    utilization=table(['厂商池','观测设备覆盖','利用率均值','解释'],[
        [label,f"{p['utilization']['coverage_mean']:.1%}",f"{p['utilization']['observed_utilization_mean_percent']:.2f}%",
         '仅.24的16/26个设备，不代表整个昇腾池' if name=='ascend' else '完整健康AWQ设备覆盖；厂商口径可能不同']
        for name,label in [('ascend','昇腾'),('musa','摩尔'),('iluvatar','天数')] for p in [single['pools'][name]]])
    workers=table(['摩尔实例','测量批次请求数','其中长输出','running达到16的时间比例','平均waiting','waiting峰值','最大KV使用比例'],[
        [name,w['routed_cohort'],w['long_output_cohort'],f"{w['at_max_sequences_fraction']:.1%}",
         f"{w['waiting']['mean']:.2f}",f"{w['waiting']['max']:.0f}",f"{w['kv_usage_raw']['max']:.2%}" if w['kv_usage_raw'] else '缺失']
        for name,w in musa['workers'].items()])
    chart=base64.b64encode((output.parent/'20261027T113010Z/backlog-time.png').read_bytes()).decode()
    image=('<h2>原实验队列图：18请求/秒</h2><p>复用原实验导出的Matplotlib图，包含预检、过渡、测量与排空阶段。入口waiting为0，模型waiting增长；具体厂商归因见下表和原始JSON。</p>'
           '<img src="data:image/png;base64,'+chart+'" alt="18请求每秒原始队列时序图">')
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六节点 mixed 压测分析</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1180px;margin:36px auto;padding:0 24px;color:#203047;line-height:1.65}}h1,h2{{line-height:1.3}}h2{{margin-top:32px}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:16px 0}}td,th{{border:1px solid #d6dee8;padding:9px;text-align:left}}th{{background:#eef3f8}}img{{max-width:100%}}.lead{{padding:18px;background:#eaf3fa;border-left:5px solid #2468ad}}.caution{{padding:15px;background:#fff4db}}code{{background:#eef1f4;padding:2px 5px}}a{{color:#2468ad}}</style>
<h1>六节点 mixed 压测分析</h1><p>数据：20261027T105055Z（64→48→32→24）与 20261027T113010Z（18）。每档60秒过渡、300秒测量，完整排空。分析仅使用本机已校验原始数据，不恢复服务。</p>
<div class="lead"><strong>核心结论：</strong>18 req/s通过本轮短测，24及以上未通过。当前主要约束集中在摩尔池：其到达约6 req/s、测量窗口完成约5.17 req/s，等待持续增加。不能把18称为长期稳定容量，也不能把16～18的窗口Goodput称为硬件理论极限。</div>
<h2>1. 五个完整测量档位</h2>{overall}<p>所有档位均产生了目标请求速率，失败不是客户端没有发够请求。总计55,800个正式测量请求；预检及过渡另存原始日志。成功率和SLO达标率在这些档位相同，因为成功完成的请求平均TPOT均低于100ms；拒绝、超时、HTTP错误仍使整体SLO达标率下降。</p>
{image}<h2>2. 18 req/s为什么成功，却不能称为队列稳定</h2>{pools}
<p>全窗口观察到的AWQ到达率合计{single['routed_arrival_rps_total']:.3f} req/s，接近客户端18 req/s。按完整请求ID和观测时间去重；不是用完成响应数冒充到达率。摩尔到达与成功完成差约{musa['arrival_rps']-musa['successful_completion_rps']:.3f} req/s，其他两池基本平衡。</p>
<p>摩尔池模型waiting的均值约{musa['waiting']['mean']:.1f}、峰值{musa['waiting']['max']:.0f}；前100秒均值{queue_start:.2f}，后100秒均值{queue_end:.2f}。入口Gateway等待始终为0，但在途均值{single['gateway']['gateway_inflight']['mean']:.1f}、峰值{single['gateway']['gateway_inflight']['max']:.0f}。这说明请求已经通过入口，随后积压在模型侧。</p>
<p>本轮按用户要求将平均队列增长设为观察项，因此18档可以判“通过”。5,400个发送批次请求最终全部完成，但有{single['drain_completions']}个在测量窗口结束后才成功完成；“全部最终成功”与“测量窗口瞬时吞吐”是两个口径。</p>
<h2>3. 主要瓶颈及证据强度</h2><p><strong>较强证据：</strong>7个摩尔实例平均running总量{musa['running']['mean']:.2f}，接近7×16=112；各实例约{fraction:.1%}的1秒对齐采样达到16。队列分布已较均匀，当前数据不支持“仅有一个摩尔实例分流过多”作为主要解释。</p>{workers}
<p>健康摩尔实例最大KV使用比例不超过{max(w['kv_usage_raw']['max'] for w in musa['workers'].values()):.2%}，三类池的抢占计数差分均为0。这不支持KV耗尽/抢占是本轮排队主因。摩尔后端生成约1,829 token/s，在五档间基本不变，显示当前配置已处于吞吐平台。16并发上限、eager执行路径与计算/通信开销共同相关；仅凭本实验无法证明调高16必然提高吞吐。</p>
<p><strong>跨池分配是下一步应验证的因素：</strong>L1按厂商池均匀随机，健康实例数却是13/7/8，而且服务能力不同。L2 cache_aware只在池内选择，不能自动修正池间比例。高压档中，摩尔running+waiting约占Gateway在途的大部分，共享512在途和128等待预算被占满，引发全局拒绝；这是依据队列和在途的关联推断，不是一次因果对照实验。</p>
<h2>4. 失败发生在哪里</h2><p>64/48/32/24四档共29,389个429，全部没有Gateway请求ID或厂商选择头；结合入口队列长期接近128，证据指向入口准入拒绝。196个5xx中有194个503也没有请求ID，另外为1个502和1个500；不能把全部5xx都归咎于模型计算。</p>
<p>55个总超时全部已返回HTTP200响应头，并能关联到摩尔实例，说明“健康接口200”或“流式响应已经开始”不能保证在150秒总截止时间内完成。队列、长输出及执行时间共同消耗截止预算。</p>
<h2>5. 利用率与频率应怎样读</h2>{utilization}
<p>摩尔GPU Graphics均值：.97约{single['musa_clocks']['s5000-01']['graphics_mhz']['mean']:.0f}MHz、.96约{single['musa_clocks']['s5000-02']['graphics_mhz']['mean']:.0f}MHz，Memory均为2500MHz。频率差存在，但不能据此断言热降频或硬件故障。昇腾.23设备SMI缺失；没有把它填成0，也不能用.24的高利用率代表整个昇腾池。厂商利用率口径和图模式不同，不能直接当作硬件效率排名。</p>
<h2>6. 容量结论与下一步</h2><div class="caution">已验证的是“18短测通过、24短测失败”，不是“稳定容量在18～24”。18已有队列增长，持续更久可能先触及超时或入口容量。按摩尔窗口处理约5.2 req/s且L1份额为1/3粗略推算，均匀分配下的可持续量级约15～16 req/s；这是工作假设，仍需较低速率长测验证，不能替代实测稳定容量。</div>
<ol><li>优先做池间容量分配对照：先测各池持续处理能力，再按能力分配入口流量；保留官方Router作池内选择。</li><li>在统一28实例配置下做低于18的较长验证，检查waiting斜率接近0，而不仅看最终成功率。</li><li>摩尔max_num_seqs调整应单独做受控对照，观察输出token/s、TPOT和队列；不要同时改路由、并发和执行模式。</li><li>将“短测成功/SLO”和“队列平稳”分别报告。若业务关心等待体验，另行定义排队/首token/总时延目标；当前实验没有恢复TTFT/E2E指标。</li></ol>
<h2>7. 数据可信范围与复现</h2><p>两个实验的健康集合、容器/模型进程身份、精确workload与服务配置相同，实验配置只改变扫描速率；每档一次、固定seed。所有正式请求都已结束再统计，没有丢弃在途请求。路由观测丢包0，多worker关联冲突0。各池成功完成率相加与原始阶段完成率一致；各档成功数加错误数等于发送数。</p>
<p>本实验使用重复的300个合成prompt，不能代表所有真实流量。Ascend为ACL图、Iluvatar为FULL decode图、MUSA为V0 eager；不构成同执行模式的厂商对比。.96还承担发压。正式负载强制固定输出长度，仅验收了协议、长度及完成状态；没有对每条完整生成内容做语义评分。没有用高利用率直接推导可扩展容量，也没有证明所有配置最优。</p>
<p>原始数据目录保持完整，当前暂停状态不变。<a href="analysis.json">详细JSON</a> · <a href="stage-analysis.csv">各档CSV</a> · <a href="pool-analysis.csv">各池CSV</a> · <a href="worker-analysis.csv">各实例CSV</a> · <a href="analysis-inputs.json">输入校验清单</a>。复现命令：<code>python heterogeneous_compute/analyze_benchmark.py --runs artifacts/serving/system-benchmark/20261027T105055Z artifacts/serving/system-benchmark/20261027T113010Z --output artifacts/serving/system-benchmark/analysis</code>。报告复用原实验已导出的Matplotlib队列图，无需额外安装绘图库。</p></html>'''
    (output/'report.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result={'semantics':{
        'arrival_rps':'Distinct Gateway request IDs observed arriving at AWQ workers during the full measurement window; not completed-response counts.',
        'completion_rps':'Successful completions during the same window, including transition carryover and excluding subsequent drain.',
        'backend_token_rps':'Interpolated generation_tokens_total differences; includes generated tokens of requests that may later fail. 25-second maximum sample interpolation gap.',
        'cohort':'All requests sent in the measurement window, including eventual drain/timeout outcomes.',
        'tpot':'Per-request mean interval (last content time - first content time)/(output_tokens-1); excludes waiting/prefill/first-token time, not a percentile of individual token gaps.',
        'queue':'1-second resampling of latest per-node healthy-worker telemetry no older than 25s. running and waiting are separate; no backend/server queue double counting.',
        'scope':'Same 28-worker deployment, short fixed mixed workload. Not theoretical hardware capacity, semantic quality evaluation, or long-term stable capacity.'},
        'runs':[analyze(p) for p in args.runs]}
    baseline_config=read(args.runs[0]/'config.json');baseline_config.pop('coarse_rates')
    baseline_frozen=read(args.runs[0]/'preflight.json')
    comparisons=[]
    for p in args.runs[1:]:
        current=read(p/'config.json');current.pop('coarse_rates')
        frozen=read(p/'preflight.json')
        comparisons.append({'run_id':p.name,'same_configuration_except_rates':current==baseline_config,
            'same_workload':read(p/'workload.json')==read(args.runs[0]/'workload.json'),
            'same_serving_snapshot':read(p/'serving-snapshot.json')==read(args.runs[0]/'serving-snapshot.json'),
            'same_frozen_deployment':{k:frozen[k]==baseline_frozen[k] for k in
                                      ['healthy_workers','router_pools','identities','service_processes']}})
    result['comparability']=comparisons
    if {p.name for p in args.runs}=={'20261027T105055Z','20261027T113010Z'}:
        assert all(c['same_configuration_except_rates'] and c['same_workload'] and c['same_serving_snapshot']
                   and all(c['same_frozen_deployment'].values()) for c in comparisons)
        assert all(not r['route_observer_packet_drops'] and not r['route_observer_errors']
                   and not r['multi_worker_route_conflicts'] for r in result['runs'])
    (args.output/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    render_report(result,args.output)
    fields=['run_id','rate','pool','healthy_workers','arrivals_at_workers','arrival_rps','successful_completion_rps',
            'backend_output_tokens_per_second','waiting_mean','waiting_peak','waiting_first_third_mean','waiting_last_third_mean','running_mean']
    with (args.output/'pool-analysis.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for run in result['runs']:
            for stage in run['stages']:
                for pool,p in stage['pools'].items():
                    row={k:p.get(k) for k in fields};row.update(run_id=run['run_id'],rate=stage['rate'],pool=pool,
                        waiting_mean=p['waiting']['mean'],waiting_peak=p['waiting']['max'],running_mean=p['running']['mean'])
                    writer.writerow(row)
    stage_fields=['run_id','rate','sent','actual_rate','success_ratio','slo_ratio','goodput','passed',
                  'load_valid','queue_stable','429','5xx','timeout','tpot_p99_ms']
    with (args.output/'stage-analysis.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=stage_fields);writer.writeheader()
        for run in result['runs']:
            for s in run['stages']:
                row={k:s.get(k) for k in stage_fields}
                row.update({k:s['errors'].get(k,0) for k in ['429','5xx','timeout']})
                row['tpot_p99_ms']=s['successful_tpot_ms']['p99'];writer.writerow(row)
    worker_fields=['run_id','rate','pool','worker','cohort_requests','long_output_requests','running_mean',
                   'at_max_sequences_fraction','waiting_mean','waiting_peak','kv_usage_max_fraction']
    with (args.output/'worker-analysis.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=worker_fields);writer.writeheader()
        for run in result['runs']:
            for s in run['stages']:
                for pool,p in s['pools'].items():
                    for name,w in p['workers'].items():
                        writer.writerow(dict(run_id=run['run_id'],rate=s['rate'],pool=pool,worker=name,
                            cohort_requests=w['routed_cohort'],long_output_requests=w['long_output_cohort'],
                            running_mean=w['running']['mean'],at_max_sequences_fraction=w['at_max_sequences_fraction'],
                            waiting_mean=w['waiting']['mean'],waiting_peak=w['waiting']['max'],
                            kv_usage_max_fraction=w['kv_usage_raw']['max'] if w['kv_usage_raw'] else None))
    manifest={'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs':{str(p/n):hashlib.sha256((p/n).read_bytes()).hexdigest() for p in args.runs for n in
                        ['config.json','serving-snapshot.json','preflight.json','stages.json','requests.jsonl','timeline.jsonl','routes.jsonl','telemetry.jsonl','workload.json','pool-utilization-summary.json','local-sync-manifest.json','launch.json']}}
    (args.output/'analysis-inputs.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'runs':len(result['runs']),'stages':sum(len(r['stages']) for r in result['runs'])}))


if __name__ == '__main__':
    main()
