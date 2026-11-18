#!/usr/bin/env python3
"""Real four-request RAG integration check, using the frozen NFCorpus input set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from check_generation_50 import ROOT, http, frozen_queries, validate_service, telemetry_error, response_checks, priority_response_valid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--backend', choices=['elasticsearch','milvus','mixed'], default='elasticsearch')
    args=parser.parse_args()
    url='http://127.0.0.1:8088'
    code,_,before=http(url+'/health')
    assert code==200
    for backend in (('elasticsearch','milvus') if args.backend=='mixed' else (args.backend,)):
        validate_service(before,backend,64)
    if before['generation']['execution_profile'].get('backend')!='vllm':
        raise RuntimeError('Full RAG must use vLLM generation')
    if telemetry_error(before): raise RuntimeError(telemetry_error(before))
    print('PREFLIGHT=passed',flush=True)
    if not args.run: return
    out=ROOT/'experiment_outputs/checks/vllm_rag_live'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out),flush=True)
    queries=frozen_queries(ROOT/'data/queries/nfcorpus-50.jsonl')[:4]
    (out/'health_before.json').write_text(json.dumps(before,indent=2))
    (out/'queries.jsonl').write_text(''.join(json.dumps({'query_id':q.query_id,'query':q.query})+'\n' for q in queries))
    stop=threading.Event(); readings=[]; failures=[]
    def observe():
        with (out/'memory.jsonl').open('w') as log:
            while not stop.is_set():
                try:
                    code,_,h=http(url+'/health')
                    if code!=200: raise RuntimeError('Health unavailable')
                    readings.append(h)
                    log.write(json.dumps({'time':time.time(),'health':h})+'\n'); log.flush()
                    if telemetry_error(h): failures.append(telemetry_error(h))
                except Exception as error: failures.append(repr(error))
                stop.wait(.1)
    watcher=threading.Thread(target=observe); watcher.start()
    barrier=threading.Barrier(4)
    def send(item):
        i,q=item
        priority=('low','normal','high','high')[i]
        backend=('elasticsearch','milvus')[i % 2] if args.backend=='mixed' else args.backend
        payload={'query':q.query,'generate':True,'priority':priority,'max_new_tokens':64, 'retrieval_backend':backend}
        barrier.wait()
        started=time.monotonic()
        row={'query_id':q.query_id,'request':payload}
        try:
            row['status'],row['headers'],row['response']=http(url+'/query',payload,timeout=300)
            row['passed']=bool(response_checks(row,64,4096) and priority_response_valid(row,priority)
                and row['response'].get('generation_backend')=='vllm'
                and row['response'].get('retrieval_backend')==backend
                and row['headers'].get('x-retrieval-backend')==backend
                and row['response'].get('execution_memory_estimate',{}).get('reserved_mib')==0)
        except Exception as error: row.update(passed=False,error=repr(error))
        row['latency_ms']=(time.monotonic()-started)*1000
        return row
    try:
        with ThreadPoolExecutor(4) as pool: rows=list(pool.map(send,enumerate(queries)))
    finally:
        stop.set(); watcher.join(10)
    code,_,after=http(url+'/health')
    (out/'requests.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'health_after.json').write_text(json.dumps(after,indent=2))
    summary={'status':'passed' if all(r['passed'] for r in rows) and not failures else 'failed',
        'backend':args.backend, 'submitted':4,'successful':sum(r['passed'] for r in rows), 'errors':failures,
        'successful_by_backend':{b:sum(r['passed'] and r['request']['retrieval_backend']==b for r in rows) for b in ('elasticsearch','milvus')},
        'max_inflight':max((h['inflight'] for h in readings),default=0),
        'request_budgets_mib':[r.get('headers',{}).get('x-memory-estimate-mib') for r in rows],
        'generation_reserved_mib':0,
        'scope':'real selected-backend RAG with vLLM generation; not 50-query or 85/75 acceptance'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if summary['status']!='passed': raise SystemExit(1)


if __name__=='__main__':main()
