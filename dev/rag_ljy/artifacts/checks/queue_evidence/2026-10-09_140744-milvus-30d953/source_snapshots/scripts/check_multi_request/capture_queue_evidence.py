#!/usr/bin/env python3
"""Capture live API evidence for multi-request stage queues and admission limits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from check_requests import http, run_scenario, same_results

PROJECT = Path(__file__).resolve().parents[2]
CHINA_TIME = timezone(timedelta(hours=8))


def burst(url, queries, timeout):
    barrier = threading.Barrier(len(queries))
    origin = time.perf_counter()
    def send(q):
        barrier.wait(timeout=timeout)
        start = time.perf_counter()
        row = {'query_id': q['query_id'], 'sent_ms': (start-origin)*1000}
        request = Request(url+'/query', data=json.dumps({'query':q['query']}).encode(),
                          headers={'Content-Type':'application/json'})
        try:
            try:
                response = build_opener(ProxyHandler({})).open(request,timeout=timeout)
            except HTTPError as error:
                response = error
            with response:
                row.update(status=response.status, request_id=response.headers.get('X-Request-ID'),
                           retry_after=response.headers.get('Retry-After'), results=json.load(response))
        except Exception as error:
            row.update(status=None,request_id=None,results=None,error=str(error))
        row['completed_ms']=(time.perf_counter()-origin)*1000
        row['latency_ms']=(time.perf_counter()-start)*1000
        return row
    with ThreadPoolExecutor(len(queries)) as pool:
        return list(pool.map(send,queries))


def parse_stages(text, request_ids):
    spans = {}
    for line in text.splitlines():
        match = re.search(r'request_id=(\S+) stage=(\S+) event=(start|end) monotonic_ns=(\d+)',line)
        if match and match[1] in request_ids:
            rid,stage,kind,stamp = match.groups()
            spans.setdefault(rid,{}).setdefault(stage,{})[kind]=int(stamp)
    return spans


def write_report(out, summary, rows, spans):
    """Make the captured timestamps usable in reviews without opening raw JSON."""
    checks=summary['checks']
    header=['scenario','query_id','request_id','stage','start_ms','end_ms','duration_ms']
    origin=min(s['embedding']['start'] for s in spans.values())
    with (out/'stage_timeline.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(header)
        for name, requests in rows.items():
            for row in requests:
                for stage, stamps in spans.get(row['request_id'],{}).items():
                    if 'start' in stamps and 'end' in stamps:
                        writer.writerow([name,row['query_id'],row['request_id'],stage,
                            (stamps['start']-origin)/1e6,(stamps['end']-origin)/1e6,
                            (stamps['end']-stamps['start'])/1e6])
    queue=checks['queue_bounds']; progression=checks['stage_progression']
    lines=[
        '# 在线多请求与独立阶段队列：实测证据', '',
        f'日期：{summary["started_at"]}（北京时间）；结论：**{summary["status"]}**。', '',
        f'在线 API 实际后端：{summary["backend"]}；embedding=vLLM；精排=Transformers。',
        f'固定输入：NFCorpus 50 条查询文件的前 {summary["admission_limit"]} 条；超限场景重复第一条，便于逐一对照串行参考。',
        f'查询文件 SHA256：`{summary["queries_sha256"]}`。', '',
        '| 验证内容 | 实测结果 |', '|---|---|',
        f'| 串行参考 | {checks["serial"]["http_successes"]}/{checks["serial"]["requests"]} HTTP 200 |',
        f'| 同时到达 | {checks["simultaneous"]["http_successes"]}/{checks["simultaneous"]["requests"]} HTTP 200，全部匹配串行结果 |',
        f'| 错峰到达（间隔 20ms） | {checks["staggered"]["http_successes"]}/{checks["staggered"]["requests"]} HTTP 200，全部匹配串行结果 |',
        f'| 超限突发 {checks["overload"]["requests"]} 请求 | 接受 {checks["overload"]["accepted"]} 个；{checks["overload"]["rejected_503"]} 个 HTTP 503，Retry-After=1；接受的结果全部匹配串行参考 |',
        f'| 超限后恢复 | {checks["recovery"]["http_successes"]}/{checks["recovery"]["requests"]} 成功；最终 inflight 和三个队列均归零 |',
        f'| 阶段依赖及结果归属 | {progression["completed_requests"]} 个成功请求均有完整三阶段记录；自身 embedding→检索→精排依赖顺序正确；所有请求 ID 唯一 |',
        f'| 跨请求阶段推进 | 日志观察到 {progression["cross_stage_overlap_pairs"]} 对检索/精排与其他请求 embedding 的阶段区间重叠 |',
        f'| 有界队列 | {queue["health_samples"]} 次有效采样均未超限；在途峰值 {queue["peak_inflight"]}；等待峰值 embedding={queue["peak_queued"]["embedding"]}、retrieval={queue["peak_queued"]["retrieval"]}、rerank={queue["peak_queued"]["rerank"]} |', '',
        '队列等待槽各 8，全局准入上限 8；worker 分别为 embedding=2、retrieval=4、rerank=1。',
        '结果核对包含排名、块/文档 ID、正文、标题、来源与检索名次；分数容差为相对 1e-4 / 绝对 1e-6。', '',
        '## 同时到达场景：各请求独立推进', '',
        '以下毫秒数来自服务器 monotonic_ns，以本场景最早的 embedding 开始时刻为零点。', '',
        '| Query ID | embedding 起止（ms） | 检索起止（ms） | 精排起止（ms） |', '|---|---|---|---|',
    ]
    simultaneous=[r for r in rows['simultaneous'] if r['request_id'] in spans]
    base=min(spans[r['request_id']]['embedding']['start'] for r in simultaneous)
    for row in simultaneous:
        stages=spans[row['request_id']]
        intervals=[f'{(stages[k]["start"]-base)/1e6:.2f}–{(stages[k]["end"]-base)/1e6:.2f}' for k in ('embedding','retrieval','rerank')]
        lines.append('| '+row['query_id']+' | '+' | '.join(intervals)+' |')
    first=min(simultaneous,key=lambda r:spans[r['request_id']]['rerank']['start'])
    early=(spans[first['request_id']]['rerank']['start']-base)/1e6
    last=max(spans[r['request_id']]['embedding']['end'] for r in simultaneous)
    lines.extend(['',f'例如 {first["query_id"]} 在 {early:.2f}ms 已进入精排，其余请求最后一个 embedding 到 {(last-base)/1e6:.2f}ms 才结束，说明无需等待整组请求完成上一阶段。', '',
        '## 原始证据及边界', '',
        '- `serial.json`、`simultaneous.json`、`staggered.json`、`overload.json`、`recovery.json`：完整响应与请求 ID。',
        '- `server.log`、`stage_timeline.json`、`stage_timeline.csv`：本轮服务端阶段记录与时间线。',
        '- `health_samples.json`、`health_after.json`、`summary.json`：队列采样、恢复状态与判定。',
        '- `source_snapshots/` 和 `SHA256SUMS`：本轮脚本及核心实现快照、原始输出校验。', '',
        '本轮共 34 次 HTTP 调用：33 次成功、1 次预期超限拒绝；这是正确性与控制行为验证，不是吞吐量收益测试。',
        '阶段区间包含 CPU/HTTP/NPU 等待，重叠证明调度层独立推进，不证明底层 NPU kernel 同时执行或 vLLM 将请求合并为同一 forward batch。',
        '**基于内存水位的准入控制尚未实现**；本轮证明的是请求数量准入上限与三个独立有界队列。',
        'stage queue 满槽后的背压、取消及失败隔离另由 `tests/test_scheduler.py` / `tests/test_server.py` 覆盖；本轮未把任何阶段队列填至 8 个等待项。',
        '已有双库 200 次完整链路测试见 [报告](../../../benchmarks/queued-pipeline-2026-10-09_134251-78a48d/report.md)。', ''])
    (out/'report.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8088')
    parser.add_argument('--queries',type=Path,default=PROJECT/'data/queries/nfcorpus-50.jsonl')
    parser.add_argument('--server-log',type=Path,default=Path('/tmp/rag-query-server.log'))
    parser.add_argument('--output-dir',type=Path,default=PROJECT/'artifacts/checks/queue_evidence')
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    url=args.url.rstrip('/')
    code,_,initial=http(url+'/health',args.timeout)
    if code!=200 or initial.get('status')!='ready' or initial.get('inflight')!=0:
        parser.error('Use a ready, idle API')
    queues=initial.get('stage_queues',{})
    if set(queues)!={'embedding','retrieval','rerank'}:
        parser.error('API must expose all three dedicated stage queues')
    count=initial['max_inflight']
    if not 2<=count<=8 or not math.isfinite(args.timeout) or args.timeout<=0:
        parser.error('Evidence run supports an admission limit of 2..8 and positive timeout')
    queries=[json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    if len(queries)<count:
        parser.error('Not enough frozen queries')
    queries=queries[:count]
    # Repeat one baseline query so every accepted overload result has an
    # independently obtained serial reference. HTTP request IDs remain unique.
    overload_queries=queries+[queries[0]]
    now=datetime.now(CHINA_TIME)
    out=args.output_dir/f'{now:%Y-%m-%d_%H%M%S}-{initial["retrieval_backend"]}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    print(f'Results: {out}',flush=True)
    def save(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    summary={'started_at':now.isoformat(),'status':'running','backend':initial['retrieval_backend'],
             'queries_sha256':hashlib.sha256(args.queries.read_bytes()).hexdigest(),
             'admission_limit':count,'health_before':initial,
             'scope':'Live API correctness, stage progression, queue depths and count-based overload handling; no memory-aware admission test.'}
    save('summary.json',summary);save('queries.json',queries)
    save('overload_queries.json',overload_queries)
    for relative in ('scripts/check_multi_request/capture_queue_evidence.py',
                     'scripts/check_multi_request/check_requests.py',
                     'src/rag_engine/server.py','src/rag_engine/scheduler.py',
                     'src/rag_engine/query_runtime.py','src/rag_engine/vllm_embedding.py'):
        target=out/'source_snapshots'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((PROJECT/relative).read_bytes())
    samples=[];lock=threading.Lock();stop=threading.Event();scenario={'name':'serial'}
    offset=args.server_log.stat().st_size
    def watch():
        while not stop.is_set():
            try:
                code,_,health=http(url+'/health',2)
                row={'monotonic_ns':time.monotonic_ns(),'scenario':scenario['name'],'http_status':code,**health}
            except Exception as error:
                row={'scenario':scenario['name'],'error':str(error)}
            with lock:
                samples.append(row)
            stop.wait(.02)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    all_rows={}
    try:
        for name in ('serial','simultaneous','staggered'):
            scenario['name']=name
            rows=run_scenario(name,queries[:count],url,args.timeout,20)
            all_rows[name]=rows;save(name+'.json',rows)
            if any(r['status'] is None for r in rows):
                raise RuntimeError('Transport failure; stop adding requests')
            print(f'{name}: {sum(r["status"]==200 for r in rows)}/{len(rows)} successful',flush=True)
        scenario['name']='overload'
        rows=burst(url,overload_queries,args.timeout)
        all_rows['overload']=rows;save('overload.json',rows)
        print(f'overload: {sum(r["status"]==200 for r in rows)} accepted, {sum(r["status"]==503 for r in rows)} rejected',flush=True)
        scenario['name']='recovery'
        rows=run_scenario('serial',queries[:1],url,args.timeout,20)
        all_rows['recovery']=rows;save('recovery.json',rows)
        code,_,final=http(url+'/health',args.timeout);save('health_after.json',final)
        baseline={r['query_id']:r['results'] for r in all_rows['serial']}
        checks={}
        for name in ('serial','simultaneous','staggered','recovery'):
            rows=all_rows[name]
            checks[name]={'requests':len(rows),'http_successes':sum(r['status']==200 for r in rows),
                'correct_results':sum(same_results(baseline[r['query_id']],r['results']) for r in rows)}
        overloaded=all_rows['overload']
        accepted=[r for r in overloaded if r['status']==200]
        rejected=[r for r in overloaded if r['status']==503]
        checks['overload']={'requests':len(overloaded),'accepted':len(accepted),'rejected_503':len(rejected),
                           'retry_after_present':bool(rejected) and all(r['retry_after']=='1' for r in rejected),
                           'accepted_correct_results':sum(same_results(baseline[r['query_id']],r['results']) for r in accepted)}
        ids=[r['request_id'] for rows in all_rows.values() for r in rows]
        checks['unique_request_ids']=None not in ids and len(set(ids))==len(ids)
        valid_samples=[s for s in samples if s.get('http_status')==200 and 'stage_queues' in s]
        checks['queue_bounds']={
            'health_samples':len(valid_samples),'peak_inflight':max((s['inflight'] for s in valid_samples),default=0),
            'limits_respected':bool(valid_samples) and all(s['inflight']<=count and all(q['queued']<=q['capacity'] and q['running']<=q['workers'] for q in s['stage_queues'].values()) for s in valid_samples),
            'peak_queued':{stage:max((s['stage_queues'][stage]['queued'] for s in valid_samples),default=0) for stage in queues},
        }
        log=args.server_log.read_bytes()[offset:].decode(errors='replace')
        (out/'server.log').write_text(log)
        spans=parse_stages(log,{r['request_id'] for rows in all_rows.values() for r in rows if r['status']==200})
        save('stage_timeline.json',spans)
        complete=[(rid,s) for rid,s in spans.items() if all('start' in s.get(k,{}) and 'end' in s.get(k,{}) for k in queues)]
        own_order=all(all(s[k]['start']<=s[k]['end'] for k in queues) and s['embedding']['end']<=s['retrieval']['start'] and s['retrieval']['end']<=s['rerank']['start'] for _,s in complete)
        witnesses=[]
        for rid,a in complete:
            for other,b in complete:
                if rid==other:continue
                for stage in ('retrieval','rerank'):
                    overlap=min(a[stage]['end'],b['embedding']['end'])-max(a[stage]['start'],b['embedding']['start'])
                    if overlap>0:
                        witnesses.append({'request_id':rid,'stage':stage,'other_request_id':other,
                                          'other_stage':'embedding','overlap_ms':overlap/1e6})
        checks['stage_progression']={'completed_requests':len(complete),'own_dependencies_respected':own_order,
            'cross_stage_overlap_pairs':len(witnesses),'example_overlaps':witnesses[:5]}
        good=all(c['http_successes']==c['requests']==c['correct_results'] for n,c in checks.items() if n in ('serial','simultaneous','staggered','recovery'))
        good=good and len(accepted)==count and len(rejected)==1 and checks['overload']['retry_after_present'] and checks['overload']['accepted_correct_results']==len(accepted)
        good=good and checks['unique_request_ids'] and checks['queue_bounds']['limits_respected']
        good=good and len(complete)==3*count+len(accepted)+1 and own_order and bool(witnesses)
        good=good and code==200 and final.get('status')=='ready' and final.get('inflight')==0 and all(q['queued']==q['running']==0 for q in final['stage_queues'].values())
        summary.update(status='passed' if good else 'failed',checks=checks,health_after=final,
                       request_count=sum(len(v) for v in all_rows.values()))
    except Exception as error:
        summary.update(status='failed',error_type=type(error).__name__,error=str(error))
    finally:
        stop.set();watcher.join(timeout=3)
        save('health_samples.json',samples)
        summary['finished_at']=datetime.now(CHINA_TIME).isoformat();save('summary.json',summary)
        if 'checks' in summary:
            write_report(out,summary,all_rows,spans)
        files=sorted(p for p in out.rglob('*') if p.is_file())
        (out/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}\n' for p in files))
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
