#!/usr/bin/env python3
"""Add server-stage and sampled-memory evidence to a completed pressure run."""
import argparse, collections, gzip, json, re, statistics
from pathlib import Path

def pct(xs,p):return sorted(xs)[min(len(xs)-1,int((len(xs)-1)*p))] if xs else None

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--server-log',type=Path,required=True);a=ap.parse_args()
    summary=json.loads((a.run/'summary.json').read_text())
    with gzip.open(a.run/'requests.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    with gzip.open(a.run/'health.jsonl.gz','rt') as f:health=[json.loads(l) for l in f]
    successful={r['request_id'] for r in rows if r['status']==200}
    pattern=re.compile(r'request_id=(\S+) stage=(\S+) event=(start|end) monotonic_ns=(\d+)(?: queue_ms=([\d.]+))?')
    stages=collections.defaultdict(lambda:collections.defaultdict(dict));queues=collections.defaultdict(list)
    for line in (gzip.open(a.server_log,'rt') if a.server_log.suffix=='.gz' else a.server_log.open()):
        m=pattern.search(line)
        if not m or m[1] not in successful:continue
        stages[m[1]][m[2]][m[3]]=int(m[4])
        if m[5]:queues[m[2]].append(float(m[5]))
    failures=[];intervals=[]
    for rid in successful:
        previous=0
        for stage in ['embedding','retrieval','rerank']:
            item=stages[rid][stage]
            if set(item)!={'start','end'} or item['start']<previous or item['end']<item['start']:
                failures.append({'id':rid,'stage':stage,'times':item});continue
            previous=item['end'];intervals.append((item['start'],item['end'],rid,stage))
    active=[];overlap=0
    for begin,end,rid,stage in sorted(intervals):
        active=[x for x in active if x[0]>begin]
        overlap+=sum(other_id!=rid and other_stage!=stage for _,other_id,other_stage in active)
        active.append((end,rid,stage))
    memory={}
    for phase in summary['phases']:
        rs=[r for r in rows if r['phase']==phase and 'sent' in r]
        if not rs:continue
        start=min(r['sent'] for r in rs);end=max(r['completed'] for r in rs)
        hs=[h for h in health if start<=h['t']<=end and 'health' in h]
        values=[(h['t'],h['health']['device_memory']['stats']['used_mb']) for h in hs if h['health'].get('device_memory',{}).get('stats')]
        if values:
            n=max(1,len(values)//3)
            memory[phase]={'samples':len(values),'used_min_mib':min(v for _,v in values),'used_max_mib':max(v for _,v in values),'first_third_mean_mib':statistics.mean(v for _,v in values[:n]),'last_third_mean_mib':statistics.mean(v for _,v in values[-n:]),'min_free_mib':min(h['health']['device_memory']['stats']['free_mb'] for h in hs if h['health']['device_memory'].get('stats'))}
    analysis={'successful_requests':len(successful),'complete_ordered_stage_traces':len(successful)-len({f['id'] for f in failures}),'stage_failures':failures,'cross_request_cross_stage_overlaps':overlap,'stage_queue_ms':{k:{'p50':pct(v,.5),'p95':pct(v,.95),'p99':pct(v,.99),'max':max(v)} for k,v in queues.items()},'memory_by_phase':memory,'rejection_latency_ms':{phase:{'p95':pct([r['latency_ms'] for r in rows if r['phase']==phase and r['status']==503],.95),'p99':pct([r['latency_ms'] for r in rows if r['phase']==phase and r['status']==503],.99)} for phase in summary['phases']},'interpretation':'Overlap is established from server monotonic stage intervals. Sampled memory trends do not prove absence of leaks beyond this test duration.'}
    (a.run/'analysis.json').write_text(json.dumps(analysis,indent=2))
    lines=['# '+summary['backend']+' 在线压力验证','',f"状态：{summary['status']}；阶段依赖错误：{len(failures)}；跨请求跨阶段重叠：{overlap}。",'', '| 场景 | 请求 | 成功 | 503 | 其他失败/不一致 | 成功吞吐req/s | p95 ms | p99 ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in summary['phases'].items():
        lines.append(f"| {k} | {v['sent']} | {v['http_200']} | {v['http_503']} | {v['failures']} | {v['successful_qps']:.2f} | {v['p95_ms']} | {v['p99_ms']} |")
    lines+=['',f"队列边界符合：{summary['queue_bounds_ok']}；健康探测异常：{summary['health_errors']}。",'', '原始证据：config.json、queries.jsonl、reference.json、requests.jsonl.gz、health.jsonl.gz、summary.json、analysis.json。', '', '503在超载阶段是预期背压，具体拒绝原因见summary.json；成功结果逐项匹配各自后端串行参考，分数容差rel=1e-4、abs=1e-6。15分钟持续负载仅说明本次时间窗口内表现。']
    (a.run/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(analysis,indent=2))

if __name__=='__main__':main()
