#!/usr/bin/env python3
"""Reproducible open-loop RAG load: fixed queries, bursts, rate ramp, soak, recovery.

No retries; bounded client concurrency; client drops/lateness are reported, not
hidden. HTTP 503 is expected under overload; other errors or changed successful
results fail validation. Run one backend at a time against a configured server.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/check_multi_request'))
from check_requests import same_results


def http(url, body=None, timeout=120):
    req = Request(url, data=None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    try:
        r = build_opener(ProxyHandler({})).open(req, timeout=timeout)
    except HTTPError as e:
        r = e
    with r:
        return r.status, dict(r.headers), json.load(r)


def percentile(values, p):
    return sorted(values)[max(0, math.ceil(len(values)*p)-1)] if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', required=True, choices=['milvus', 'elasticsearch'])
    ap.add_argument('--url', default='http://127.0.0.1:8088')
    ap.add_argument('--queries', type=Path, default=ROOT/'data/queries/nfcorpus-50.jsonl')
    ap.add_argument('--output-dir', type=Path, default=ROOT/'experiment_outputs/benchmarks')
    ap.add_argument('--phase-seconds', type=int, default=60)
    ap.add_argument('--soak-seconds', type=int, default=900)
    ap.add_argument('--rates', default='1,4,12')
    ap.add_argument('--soak-rate', type=float, default=4)
    ap.add_argument('--workers', type=int, default=64)
    a=ap.parse_args()
    rates=[float(x) for x in a.rates.split(',')]
    if min(rates+[a.soak_rate,a.phase_seconds,a.soak_seconds,a.workers])<=0: ap.error('positive values required')
    run=a.output_dir/(datetime.now().strftime('%Y%m%dT%H%M%S')+'-'+a.backend+'-pressure')
    run.mkdir(parents=True,exist_ok=False)
    print('OUTPUT='+str(run),flush=True)
    queries=[json.loads(l) for l in a.queries.read_text().splitlines() if l.strip()]
    assert len(queries)==50 and len({q['query_id'] for q in queries})==50
    shutil.copyfile(a.queries,run/'queries.jsonl'); shutil.copyfile(__file__,run/'checker_snapshot.py')
    status,_,initial=http(a.url+'/health',timeout=10)
    assert status==200 and initial['status']=='ready' and initial['retrieval_backend']==a.backend
    (run/'config.json').write_text(json.dumps({'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'query_sha256':hashlib.sha256(a.queries.read_bytes()).hexdigest(),'health_before':initial,'started_at':datetime.now(timezone.utc).isoformat()},indent=2))
    baseline={}; rows=[]; health=[]; ids=set(); lock=threading.Lock(); stop=threading.Event(); slots=threading.BoundedSemaphore(a.workers); phase_times={}
    request_file=gzip.open(run/'requests.jsonl.gz','wt'); health_file=gzip.open(run/'health.jsonl.gz','wt')
    def record(row):
        with lock:
            rows.append(row); request_file.write(json.dumps(row)+'\n'); request_file.flush()
    def request(phase,index,target):
        sent=time.monotonic(); q=queries[index%50]
        row={'phase':phase,'index':index,'query_id':q['query_id'],'scheduled':target,'sent':sent,'schedule_lag_ms':(sent-target)*1000}
        try:
            code,headers,data=http(a.url+'/query',{'query':q['query']})
            rid=headers.get('X-Request-ID') or headers.get('x-request-id')
            row.update(status=code,request_id=rid,retry_after=headers.get('Retry-After') or headers.get('retry-after'))
            with lock:
                row['unique_id']=bool(rid) and rid not in ids
                if rid: ids.add(rid)
            if code==200:
                row['matches_reference']=phase=='reference' or same_results(baseline[q['query_id']],data)
                row['result_ids']=[r['chunk_id'] for r in data]
                if phase=='reference': baseline[q['query_id']]=data
                if not row['matches_reference']: row['mismatched_results']=data
            else: row['error_body']=data
        except Exception as e: row.update(status=None,error=repr(e))
        row.update(completed=time.monotonic(),latency_ms=(time.monotonic()-sent)*1000)
        record(row)
    def worker(*args):
        try: request(*args)
        finally: slots.release()
    def monitor():
        while not stop.is_set():
            t=time.monotonic()
            try:
                code,_,data=http(a.url+'/health',timeout=5); item={'t':t,'status':code,'health':data}
            except Exception as e: item={'t':t,'error':repr(e)}
            health.append(item); health_file.write(json.dumps(item)+'\n'); health_file.flush(); stop.wait(max(0,1-(time.monotonic()-t)))
    monitor_thread=threading.Thread(target=monitor,daemon=True); monitor_thread.start()
    def drain():
        deadline=time.monotonic()+150
        while time.monotonic()<deadline:
            _,_,h=http(a.url+'/health',timeout=10)
            if h['inflight']==0 and h['memory_admission'].get('reserved_mib',0)==0:return
            time.sleep(.2)
        raise RuntimeError('Server did not drain within 150 seconds')
    def serial(name,n):
        start=time.monotonic()
        for i in range(n): request(name,i,time.monotonic())
        drain(); phase_times[name]=[start,time.monotonic()]
    def arrivals(name, rate=None, seconds=None, count=None):
        start=time.monotonic(); n=count if count is not None else math.ceil(rate*seconds)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i in range(n):
                target=start if count is not None else start+i/rate
                time.sleep(max(0,target-time.monotonic()))
                if slots.acquire(blocking=False): pool.submit(worker,name,i,target)
                else: record({'phase':name,'index':i,'status':'client_drop','scheduled':target})
        drain(); phase_times[name]=[start,time.monotonic()]
        print(name+': '+str(dict(Counter(r['status'] for r in rows if r['phase']==name))),flush=True)
    fatal=None
    try:
        serial('reference',50)
        if len(baseline)!=50: raise RuntimeError('Reference queries failed')
        (run/'reference.json').write_text(json.dumps(baseline))
        for n in [2,4,8,16,32]: arrivals('burst_'+str(n),count=n)
        arrivals('stagger_2s',rate=.5,seconds=20)
        for rate in rates: arrivals('rate_'+str(rate),rate=rate,seconds=a.phase_seconds)
        arrivals('soak',rate=a.soak_rate,seconds=a.soak_seconds)
        arrivals('final_overload',rate=max(rates)*2,seconds=30)
        serial('recovery',50)
    except Exception as e: fatal=repr(e)
    finally:
        stop.set(); monitor_thread.join(10); request_file.close(); health_file.close()
    summaries={}
    for phase in dict.fromkeys(r['phase'] for r in rows):
        rs=[r for r in rows if r['phase']==phase]; good=[r for r in rs if r['status']==200]; errors=[r for r in rs if r['status'] not in [200,503] or not r.get('unique_id') or (r['status']==200 and not r.get('matches_reference')) or (r['status']==503 and r.get('retry_after')!='1')]
        duration=phase_times.get(phase,[0,0]); duration=duration[1]-duration[0]
        lat=[r['latency_ms'] for r in good]
        summaries[phase]={'sent':len(rs),'http_200':len(good),'http_503':sum(r['status']==503 for r in rs),'failures':len(errors),'duration_including_drain_s':duration,'successful_qps':len(good)/duration if duration else 0,'p50_ms':percentile(lat,.5),'p95_ms':percentile(lat,.95),'p99_ms':percentile(lat,.99),'max_client_lag_ms':max((r.get('schedule_lag_ms',0) for r in rs),default=0),'rejection_reasons':dict(Counter(str(r.get('error_body')) for r in rs if r['status']==503))}
    hs=[x['health'] for x in health if 'health' in x]
    bounds=all(h['inflight']<=h['max_inflight'] and all(q['queued']<=q['capacity'] and q['running']<=q['workers'] for q in h['stage_queues'].values()) for h in hs)
    final=http(a.url+'/health',timeout=10)[2]
    failed=fatal or any('error' in h or h.get('status')!=200 for h in health) or not bounds or not hs or any(s['failures'] for s in summaries.values()) or any(r['status']!=200 for r in rows if r['phase'] in ['reference','recovery']) or final['inflight']!=0
    summary={'status':'failed' if failed else 'passed','fatal':fatal,'backend':a.backend,'phases':summaries,'health_samples':len(health),'health_errors':sum('error' in h or h.get('status')!=200 for h in health),'queue_bounds_ok':bounds,'max_inflight':max((h['inflight'] for h in hs),default=0),'memory_used_mib_min_max':[min((h['device_memory']['stats']['used_mb'] for h in hs),default=0),max((h['device_memory']['stats']['used_mb'] for h in hs),default=0)],'health_after':final,'note':'Capacity and memory rejection reasons are recorded separately; no automatic retry. Memory plateau and recovery must be assessed from the time series; HTTP overlap alone is not stage-overlap proof.'}
    (run/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2),flush=True)
    return int(bool(failed))

if __name__=='__main__': sys.exit(main())
