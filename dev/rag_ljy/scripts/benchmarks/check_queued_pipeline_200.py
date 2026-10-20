#!/usr/bin/env python3
"""Test 50 frozen queries x two databases x two modes through real queued APIs.

Sequential mode sends requests serially; concurrent mode sends bounded simultaneous
waves and overlaps BM25/dense. Both configurations use dedicated stage queues.
This is a full-pipeline correctness test; it is not the old retrieval-only timer.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'check_multi_request'))

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
from uuid import uuid4

from _bootstrap import PROJECT_ROOT, load_project_env
from check_requests import http, same_results, send


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(values):
    return {'mean': sum(values)/len(values), 'p50': percentile(values, .5), 'p95': percentile(values, .95)}


def run_requests(url, queries, parallel, timeout):
    rows = []
    origin = time.perf_counter()
    for start in range(0, len(queries), parallel):
        wave = queries[start:start + parallel]
        if len(wave) == 1:
            rows.append(send(url, wave[0], timeout, origin))
        else:
            barrier = threading.Barrier(len(wave))
            def worker(q):
                barrier.wait(timeout=timeout)
                return send(url, q, timeout, origin)
            with ThreadPoolExecutor(len(wave)) as pool:
                rows.extend(pool.map(worker, wave))
        if len(rows) % 10 == 0 or len(rows) == len(queries):
            print(f'  completed {len(rows)}/{len(queries)}', flush=True)
    return rows


def stage_evidence(log, rows):
    ids = {r['request_id'] for r in rows}
    starts, intervals, timings = {}, {s: [] for s in ('embedding', 'retrieval', 'rerank')}, {}
    for line in log.read_text().splitlines():
        match = re.search(r'request_id=(\S+) stage=(\S+) event=(start|end) monotonic_ns=(\d+)', line)
        if match:
            rid, stage, kind, stamp = match.groups()
            if rid not in ids:
                continue
            if kind == 'start':
                starts[rid, stage] = int(stamp)
            elif (rid, stage) in starts:
                intervals[stage].append((rid, starts[rid, stage], int(stamp)))
        match = re.search(r'request_id=(\S+) event=finished status=(\S+) timings=(\{.*\})', line)
        if match and match[1] in ids:
            timings[match[1]] = json.loads(match[3])
    overlaps = {}
    for stage, spans in intervals.items():
        overlaps[stage] = sum(max(a[1], b[1]) < min(a[2], b[2]) for i,a in enumerate(spans) for b in spans[i+1:])
    stage_metrics = {key: stats([t[key] for t in timings.values() if key in t])
                     for key in sorted({k for t in timings.values() for k in t})}
    return {'completed_stage_records': len(timings), 'overlapping_stage_pairs': overlaps,
            'stage_ms': stage_metrics, 'request_timings': timings}


def main():
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT/'data/queries/nfcorpus-50.jsonl')
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'artifacts/benchmarks')
    args = parser.parse_args()
    if not 2 <= args.parallel <= 8 or not 1 <= args.port <= 65535 or args.timeout <= 0:
        parser.error('parallel must be 2..8; use a valid port and positive timeout')
    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    if len(queries) != 50 or len({q['query_id'] for q in queries}) != 50:
        parser.error('Use exactly the saved 50 unique queries')
    # Fail before launching if another service owns the requested port.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    now = datetime.now(timezone(timedelta(hours=8)))
    out = args.output_dir/f'queued-pipeline-{now:%Y-%m-%d_%H%M%S}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    (out/'queries.jsonl').write_bytes(args.queries.read_bytes())
    summary = {'started_at': now.isoformat(), 'status': 'running', 'cases': 200,
               'scope': 'full pipeline; serial/sequential vs bounded simultaneous/concurrent',
               'queries_sha256': hashlib.sha256(args.queries.read_bytes()).hexdigest(),
               'parallel_requests': args.parallel, 'embedding_backend': 'vllm', 'checks': {},
               'note': 'Within-engine result comparison, not qrels relevance. HTTP embedding overlap does not prove batching within one vLLM forward pass.'}
    def save():
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save(); print(f'Results: {out}', flush=True)
    try:
        for backend in ('milvus', 'elasticsearch'):
            by_mode = {}
            for mode in ('sequential','concurrent'):
                directory = out/backend/mode;directory.mkdir(parents=True)
                log = directory/'server.log'
                env = dict(os.environ, DC_BACKEND='ascend', RAG_EMBEDDING_BACKEND='vllm',
                           RETRIEVAL_BACKEND=backend, MILVUS_COLLECTION='nfcorpus_chunks_v1',
                           ELASTICSEARCH_INDEX='nfcorpus-chunks-v1')
                cmd = [sys.executable, '-u', str(PROJECT_ROOT/'scripts/runtime/serve_queries.py'),
                       '--device','npu:0','--port',str(args.port),'--execution-mode',mode,
                       '--embedding-workers',str(args.parallel),'--stage-queue-capacity','8','--max-inflight','8']
                with log.open('w') as output:
                    server = subprocess.Popen(cmd,cwd=PROJECT_ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        url = f'http://127.0.0.1:{args.port}'
                        deadline = time.monotonic()+args.timeout
                        while True:
                            if server.poll() is not None:
                                raise RuntimeError(f'{backend}/{mode}: API exited; inspect {log}')
                            try:
                                code,_,health = http(url+'/health',2)
                                if code==200:
                                    break
                            except OSError:
                                pass
                            if time.monotonic()>=deadline:
                                raise TimeoutError(f'{backend}/{mode}: API startup timed out')
                            time.sleep(.25)
                        if health['retrieval_backend']!=backend or health['execution_mode']!=mode:
                            raise RuntimeError('API backend/mode does not match test label')
                        print(f'{backend}/{mode}: ready, embedding workers={health["stage_queues"]["embedding"]["workers"]}',flush=True)
                        rows = run_requests(url,queries,1 if mode=='sequential' else args.parallel,args.timeout)
                        (directory/'responses.json').write_text(json.dumps(rows,indent=2)+'\n')
                        (directory/'health.json').write_text(json.dumps(health,indent=2)+'\n')
                        by_mode[mode]=rows
                    finally:
                        server.terminate()
                        try:
                            server.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            server.kill();server.wait();raise RuntimeError('Test API did not drain on shutdown')
                evidence = stage_evidence(log,rows)
                (directory/'stages.json').write_text(json.dumps(evidence,indent=2)+'\n')
                span=max(r['completed_ms'] for r in rows)-min(r['sent_ms'] for r in rows)
                summary['checks'][backend+'/'+mode]={'measured_wall_ms':span,
                    'observed_requests_per_second':len(rows)*1000/span,'http_successes':sum(r['status']==200 for r in rows),
                    'requests':len(rows),'latency_ms':stats([r['latency_ms'] for r in rows]),
                    'embedding_overlap_pairs':evidence['overlapping_stage_pairs']['embedding'],
                    'completed_stage_records':evidence['completed_stage_records']}
                save()
            baseline={r['query_id']:r for r in by_mode['sequential']}
            matches=sum(same_results(baseline[r['query_id']]['results'],r['results']) for r in by_mode['concurrent'])
            summary['checks'][backend+'/consistency']={'matched':matches,'total':50,
                'mismatched_query_ids':[r['query_id'] for r in by_mode['concurrent'] if not same_results(baseline[r['query_id']]['results'],r['results'])]}
            save()
        ids=[r['request_id'] for b in ('milvus','elasticsearch') for m in ('sequential','concurrent')
             for r in json.loads((out/b/m/'responses.json').read_text())]
        summary['unique_request_ids']=len(set(ids))==200 and None not in ids
        cases_ok=all(c['http_successes']==50 and c['completed_stage_records']==50 for k,c in summary['checks'].items() if not k.endswith('/consistency'))
        same=all(summary['checks'][b+'/consistency']['matched']==50 for b in ('milvus','elasticsearch'))
        overlap=all(summary['checks'][b+'/concurrent']['embedding_overlap_pairs']>0 for b in ('milvus','elasticsearch'))
        summary['status']='passed' if cases_ok and same and overlap and summary['unique_request_ids'] else 'failed'
    except Exception as exc:
        summary.update(status='failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        summary['finished_at']=datetime.now(timezone(timedelta(hours=8))).isoformat();save()
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
