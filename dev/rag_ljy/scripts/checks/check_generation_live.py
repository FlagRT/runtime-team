#!/usr/bin/env python3
"""Finite real RAG generation check for burst/staggered/continuous arrivals.

Default is read-only preflight. This is a functional/concurrency check; 85/75%
acceptance remains a separate measured-watermark experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rag_engine.benchmark_data import load_queries
from memory_backend import require_backend


def http(url, body=None, *, timeout=300):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type':'application/json'})
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        return response.code, dict(response.headers.items()), json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--backend', choices=['milvus','elasticsearch'], required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8088')
    parser.add_argument('--queries', type=Path, default=ROOT/'data/queries/nfcorpus-50.jsonl')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--spacing', type=float, default=.5)
    parser.add_argument('--continuous-seconds', type=float, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if not 2 <= args.count <= 16 or not .05 <= args.spacing <= 10 or not 1 <= args.continuous_seconds <= 30:
        parser.error('count 2..16, spacing .05..10s, continuous duration 1..30s required')
    queries = load_queries(args.queries)
    def health():
        code, _, body = http(args.url+'/health', timeout=5)
        if code != 200:
            raise RuntimeError('API health unavailable')
        require_backend(body, args.backend)
        return body
    before = health()
    if not before.get('generation',{}).get('enabled'):
        raise RuntimeError('Generation is not enabled')
    cfg = before['memory_admission']['config']
    if not (cfg['enabled'] and cfg['estimate_requests'] and cfg['request_budget_mib'] is None
            and cfg['safety_margin_mib']==5120 and cfg['high_watermark']==.85 and cfg['low_watermark']==.75
            and before['max_inflight']==0):
        raise RuntimeError('Required dynamic measured-memory admission configuration is missing')
    if before['inflight'] or before['memory_admission']['reserved_mib']:
        raise RuntimeError('API must be drained')
    if before['memory_admission']['status'] != 'ready':
        raise RuntimeError('Memory admission must be ready before the functional check')
    if not 1 <= args.max_new_tokens <= before['generation']['execution_profile']['max_new_tokens']:
        raise ValueError('max-new-tokens exceeds API configuration')
    print(json.dumps({'preflight':'passed','run_requested':args.run,'backend':args.backend}), flush=True)
    if not args.run:
        return 0
    lock = open('/tmp/rag-watermarks-modes.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = ROOT/'experiment_outputs/checks/generation_integration'/args.backend/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out), flush=True)
    source_files = [Path(__file__), *(ROOT/'src/rag_engine').glob('*.py')]
    from rag_engine.generation import checkpoint_manifest
    for path in source_files:
        shutil.copyfile(path, out/path.name)
    shutil.copyfile(args.queries, out/'queries.jsonl')
    (out/'manifest.json').write_text(json.dumps({'options':vars(args), 'initial_health':before,
        'generation_model':checkpoint_manifest(before['generation']['execution_profile']['model_path']),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'query_sha256':hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        'scope':'real RAG generation and arrival modes; not 85% acceptance',
        'compute_utilization':'not measured', 'pressure_source':'real requests only'}, indent=2, default=str))
    rows, samples, stop = [], [], threading.Event()
    result = {'status':'incomplete', 'backend':args.backend, 'modes':{}, 'errors':[],
              'watermark_acceptance':'not assessed by this functional check'}
    phase = {'mode':'setup'}
    unsafe = threading.Event()
    def watch():
        while not stop.is_set():
            try:
                h = health()
                samples.append({'t':time.monotonic(),'mode':phase['mode'],'health':h})
                if (h['device_memory']['status'] != 'ok' or h['device_memory'].get('stale')
                        or h['memory_admission'].get('reason') == 'stale_telemetry'
                        or h['memory_estimation']['fault'] or h['device_memory']['stats']['free_mb'] < 5120):
                    unsafe.set()
            except Exception as error:
                result['errors'].append(repr(error)); unsafe.set()
            stop.wait(.1)
    def send(index, gate=None, generate=True):
        if gate is not None:
            gate.wait()
        query = queries[index % len(queries)]
        body = {'query':query.query, 'generate':generate, 'max_new_tokens':args.max_new_tokens}
        row = {'mode':phase['mode'],'query_id':query.query_id,'body':body,'sent':time.monotonic()}
        try:
            if unsafe.is_set():
                raise RuntimeError('Client stopped before submitting another request')
            row['status'], row['headers'], row['response'] = http(args.url+'/query',body)
        except Exception as error:
            row['error'] = repr(error)
        row['completed'] = time.monotonic()
        rows.append(row)
        return row
    def valid(row):
        if row.get('status') == 200:
            if row['body']['generate']:
                body = row['response']
                return (isinstance(body,dict) and bool(body.get('answer')) and bool(body.get('sources'))
                    and bool(body.get('hits')) and body['usage']['output_tokens']>0
                    and body['usage']['output_tokens'] <= args.max_new_tokens
                    and body['usage']['input_tokens'] <= before['generation']['execution_profile']['max_input_tokens'])
            return isinstance(row['response'],list) and bool(row['response'])
        detail = row.get('response',{}).get('detail',{})
        return (row.get('status') == 503 and isinstance(detail,dict)
                and detail.get('reason') in ('request_budget','memory_pressure')
                and bool({k.lower():v for k,v in row.get('headers',{}).items()}.get('retry-after')))
    def drain():
        end = time.monotonic()+300
        while time.monotonic()<end:
            h = health()
            if not h['inflight'] and not h['memory_admission']['reserved_mib']:
                return h
            time.sleep(.1)
        raise RuntimeError('Accepted requests or completed reservations did not drain')
    watcher = threading.Thread(target=watch,daemon=True)
    watcher.start()
    try:
        # Preserve the old response contract with the generation model resident.
        old = send(0, generate=False)
        if not valid(old) or old['status'] != 200:
            raise RuntimeError('Retrieval-only compatibility check failed')
        drain()
        for mode in ('burst','staggered','continuous'):
            phase['mode'] = mode
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = []
                gate = threading.Event() if mode=='burst' else None
                count = args.count if mode!='continuous' else min(16, int(args.continuous_seconds/args.spacing)+1)
                try:
                    for i in range(count):
                        if unsafe.is_set():
                            raise RuntimeError('Telemetry/fault/5GiB guard stopped further submissions')
                        if mode!='burst':
                            time.sleep(max(0,start+i*args.spacing-time.monotonic()))
                        futures.append(pool.submit(send,i,gate))
                finally:
                    if gate is not None:
                        gate.set()
                responses = [future.result(305) for future in futures]
            drain()
            checks = {'all_http_expected':all(valid(r) for r in responses),
                      'some_real_generation_completed':any(r.get('status')==200 for r in responses),
                      'telemetry_and_safety_ok':not unsafe.is_set()}
            result['modes'][mode] = {'status':'passed' if all(checks.values()) else 'failed',
                'checks':checks, 'sent':len(responses), 'successful':sum(r.get('status')==200 for r in responses),
                'arrival_times':[r['sent'] for r in responses]}
            if not all(checks.values()):
                raise RuntimeError(mode+' functional check failed')
        result['status'] = 'passed'
    except BaseException as error:
        result['status']='failed'; result['errors'].append(repr(error))
    finally:
        try:
            result['health_after']=drain()
        except Exception as error:
            result['status']='failed'; result['errors'].append(repr(error))
        stop.set(); watcher.join(10)
        (out/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (out/'memory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
        (out/'summary.json').write_text(json.dumps(result, indent=2))
    print('RESULT='+result['status'], flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
