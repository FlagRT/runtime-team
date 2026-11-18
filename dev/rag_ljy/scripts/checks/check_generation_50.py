#!/usr/bin/env python3
"""Replay all 50 frozen NFCorpus queries through the existing generation API.

Default: read-only preflight. --run sends each query exactly once. Serial by
default; --parallel uses bounded waves. No retry, service restart or model load.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from rag_engine.benchmark_data import load_queries
from memory_backend import require_backend

QUERY_SHA256 = '9b0a898ba4e50ccf0081591ba4a2a1ddccc960c112582692a52e90aca09e7429'


def http(url, body=None, *, timeout=5):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type':'application/json'})
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode('utf-8', errors='replace')
        try:
            content = json.loads(raw)
        except ValueError:
            content = {'invalid_json':raw}
        return response.code, {k.lower():v for k,v in response.headers.items()}, content


def frozen_queries(path):
    queries = load_queries(path)
    if len(queries)!=50 or len({q.query_id for q in queries})!=50:
        raise ValueError('Exactly the 50 saved unique query IDs are required')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != QUERY_SHA256:
        raise ValueError('Query bytes differ from the frozen NFCorpus-50 baseline')
    return queries


def validate_service(health, backend, max_new_tokens):
    if backend not in health.get("retrieval_backends", []):
        require_backend(health, backend)
    cfg = health.get('memory_admission',{}).get('config',{})
    if (health.get('status')!='ready' or not health.get('generation',{}).get('enabled')
            or not health.get('memory_estimation',{}).get('enabled')):
        raise RuntimeError('A ready generation API with dynamic memory estimation is required')
    if not (cfg.get('enabled') and cfg.get('estimate_requests') and cfg.get('request_budget_mib') is None
            and cfg.get('safety_margin_mib')==5120 and cfg.get('high_watermark')==.85
            and cfg.get('low_watermark')==.75 and health.get('max_inflight')==0):
        raise RuntimeError('Keep measured 85/75%, 5GiB safety and dynamic budgets; max-inflight must be 0')
    if health['device_memory'].get('source')!='device-context.runtime.memory_stats':
        raise RuntimeError('Unexpected device memory source')
    if max_new_tokens > health['generation']['execution_profile']['max_new_tokens']:
        raise ValueError('Requested output limit exceeds the running API limit')
    if health['generation']['execution_profile'].get('backend') == 'vllm':
        if set(health.get('device_memory_by_stage', {})) != {'embedding_rerank', 'generation'}:
            raise RuntimeError('vLLM RAG must expose both device memory samples')
        if set(health.get('memory_admission_by_stage', {})) != {'embedding_rerank', 'generation'}:
            raise RuntimeError('vLLM RAG must protect both devices')
        for state in health['memory_admission_by_stage'].values():
            policy = state.get('config', {})
            if (policy.get('high_watermark') != .85 or policy.get('low_watermark') != .75
                    or policy.get('safety_margin_mib') != 5120):
                raise RuntimeError('Both devices must retain 85/75 watermarks and 5GiB headroom')


def telemetry_error(health):
    for stage, reading in health.get('device_memory_by_stage', {}).items():
        if reading.get('status') != 'ok' or reading.get('stale') or reading.get('age_seconds', 99) > 3:
            return stage + ':invalid_or_stale_telemetry'
        if reading.get('stats', {}).get('free_mb', -1) < 5120:
            return stage + ':less_than_5GiB_measured_free'
    memory = health['device_memory']
    if health['memory_estimation'].get('fault'):
        return 'memory_estimator_fault'
    if memory.get('status')!='ok' or memory.get('stale') or health['memory_admission'].get('reason') in (
            'stale_telemetry','invalid_telemetry','telemetry_unavailable'):
        return 'invalid_or_stale_telemetry'
    if memory.get('stats',{}).get('free_mb',-1)<5120:
        return 'less_than_5GiB_measured_free'
    return None


def response_checks(row, max_new_tokens, max_input_tokens):
    body = row.get('response')
    headers = row.get('headers',{})
    if row.get('status')!=200 or not isinstance(body,dict):
        return False
    usage = body.get('usage',{})
    if not isinstance(usage,dict):
        return False
    def count(value, upper):
        return isinstance(value,int) and not isinstance(value,bool) and 1<=value<=upper
    timings=body.get('timings_ms',{})
    required=('embedding_ms','rerank_ms','generation_ms','total_ms',
              'embedding_queue_ms','retrieval_queue_ms','rerank_queue_ms','generation_queue_ms')
    stage_timings=(isinstance(timings,dict) and all(isinstance(timings.get(key),(int,float))
        and not isinstance(timings[key],bool) and math.isfinite(timings[key]) and timings[key]>=0 for key in required))
    try:
        budget=float(headers.get('x-memory-estimate-mib',0))
    except (TypeError,ValueError):
        return False
    return (isinstance(body.get('answer'),str) and bool(body['answer'].strip())
            and isinstance(body.get('sources'),list) and bool(body['sources'])
            and isinstance(body.get('hits'),list) and bool(body['hits'])
            and count(usage.get('input_tokens'),max_input_tokens)
            and count(usage.get('output_tokens'),max_new_tokens)
            and usage.get('max_new_tokens')==max_new_tokens and stage_timings
            and body.get('finish_reason') in ('eos','length')
            and bool(headers.get('x-request-id')) and math.isfinite(budget) and budget>0)


def validate_vllm_workflow(health):
    if health.get('generation', {}).get('execution_profile', {}).get('backend') != 'vllm':
        raise RuntimeError('This test requires the complete RAG-to-vLLM generation workflow')
    readings = health.get('device_memory_by_stage', {})
    for stage, die in (('embedding_rerank', 7), ('generation', 6)):
        reading = readings.get(stage, {})
        if reading.get('physical_die') != die or reading.get('source') != 'device-context.runtime.memory_stats':
            raise RuntimeError(f'{stage} must have verified device-context telemetry for physical die {die}')


def validate_priority_service(health):
    memory = health.get('device_memory', {})
    if memory.get('backend') != 'ascend' or not str(memory.get('device', '')).startswith('npu:'):
        raise RuntimeError('This priority experiment requires real Ascend NPU telemetry')
    scheduling = health.get('scheduling', {})
    if scheduling.get('policy') != 'priority_with_aging' or scheduling.get('dispatch_evidence') is not True:
        raise RuntimeError('Restart the drained API with the new priority code before this test')


def priority_response_valid(row, priority):
    body = row['response'].get('scheduling', {})
    if row['headers'].get('x-request-priority') != priority or body.get('priority') != priority:
        return False
    stages = body.get('stages', {})
    for name in ('embedding', 'retrieval', 'rerank', 'generation'):
        records = stages.get(name, [])
        if not isinstance(records, list) or len(records) != 1:
            return False
        decision = records[0]
        if (decision.get('priority') != priority or decision.get('effective_priority') not in ('high','normal','low')
                or not isinstance(decision.get('overtook_lower_priority'), int)
                or decision['overtook_lower_priority'] < 0):
            return False
    return True


def replay(args, queries, before, out):
    priority_cycle = getattr(args, 'priority_cycle', False)
    require_vllm = getattr(args, 'require_vllm_generation', False)
    assigned = {q.query_id: ('low', 'normal', 'high')[i % 3] for i, q in enumerate(queries)}
    stop, unsafe = threading.Event(), threading.Event()
    health_errors, rows = [], []
    def health():
        code, _, body = http(args.url+'/health')
        if code!=200:
            raise RuntimeError('API health unavailable: HTTP '+str(code))
        validate_service(body,args.backend,args.max_new_tokens)
        if require_vllm:
            validate_vllm_workflow(body)
        if priority_cycle:
            validate_priority_service(body)
        return body
    def observe():
        with (out/'memory.jsonl').open('w') as log:
            while not stop.is_set():
                try:
                    h=health()
                    log.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
                                          'monotonic':time.monotonic(),'health':h})+'\n')
                    log.flush()
                    reason=telemetry_error(h)
                    if reason:
                        health_errors.append(reason); unsafe.set()
                except Exception as error:
                    health_errors.append(repr(error)); unsafe.set()
                stop.wait(.2)
    def wait_ready(*, admission_required):
        end=time.monotonic()+args.ready_timeout
        while time.monotonic()<end:
            if unsafe.is_set():
                raise RuntimeError('Telemetry/fault/5GiB guard stopped new submissions')
            h=health()
            if telemetry_error(h):
                raise RuntimeError(telemetry_error(h))
            if h['inflight']==0 and h['memory_admission']['reserved_mib']==0 and (
                    not admission_required or h['memory_admission']['status']=='ready'):
                return h
            time.sleep(.1)
        raise RuntimeError('Timed out waiting for admitted work, reservations or memory admission')
    def send(query, gate):
        gate.wait()
        row={'query_id':query.query_id,'query':query.query,'sent_at':datetime.now(timezone.utc).isoformat(),
             'sent_monotonic':time.monotonic(),'http_submitted':False,
             'request':{'query':query.query,'generate':True,'max_new_tokens':args.max_new_tokens}}
        if before.get('retrieval_backends'):
            row['request']['retrieval_backend'] = args.backend
        if priority_cycle:
            row['request']['priority'] = assigned[query.query_id]
        try:
            if unsafe.is_set():
                raise RuntimeError('Stopped before HTTP submission')
            row['http_submitted']=True
            row['status'],row['headers'],row['response']=http(
                args.url+'/query',row['request'],timeout=args.request_timeout)
        except Exception as error:
            row['error']=repr(error)
        row['completed_monotonic']=time.monotonic()
        row['latency_ms']=(row['completed_monotonic']-row['sent_monotonic'])*1000
        row['complete_generation']=bool(response_checks(row,args.max_new_tokens,
            before['generation']['execution_profile']['max_input_tokens']))
        if before.get('retrieval_backends') and row['complete_generation']:
            row['complete_generation'] = (row['headers'].get('x-retrieval-backend') == args.backend
                and row['response'].get('retrieval_backend') == args.backend)
        if priority_cycle and row['complete_generation']:
            row['complete_generation'] = priority_response_valid(row, assigned[query.query_id])
        body=row.get('response',{})
        detail=body.get('detail',{}) if isinstance(body,dict) else {}
        row['rejection_reason']=detail.get('reason') if isinstance(detail,dict) else None
        row['expected_rejection']=(row.get('status')==503
            and row['rejection_reason'] in ('request_budget','memory_pressure')
            and bool(row.get('headers',{}).get('retry-after')))
        if isinstance(body,dict) and isinstance(body.get('answer'),str):
            sources=body.get('sources',[])
            sources=sources if isinstance(sources,list) else []
            ids={s['citation_id'] for s in sources if isinstance(s,dict) and isinstance(s.get('citation_id'),int)}
            cites={int(n) for n in re.findall(r'\[(\d+)\]',body['answer'])}
            row['quality_observations']={
                'answer_echoes_query':body['answer'].strip().casefold()==query.query.strip().casefold(),
                'numeric_citation_ids':sorted(cites),'unknown_citation_ids':sorted(cites-ids),
                'output_limit_reached':body.get('finish_reason')=='length',
                'note':'Observations only; factual correctness and groundedness are not certified'}
        return row
    summary={'status':'incomplete','backend':args.backend,'expected_queries':50,
             'parallel':args.parallel,'priority_cycle':priority_cycle,'errors':[],
             'scope':'50 frozen queries, complete generation coverage; not 85/75% acceptance or answer-quality certification',
             'automatic_retries':False,'watermark_acceptance':'not assessed'}
    watcher=threading.Thread(target=observe,daemon=True)
    watcher.start()
    try:
        with (out/'requests.jsonl').open('w') as log, ThreadPoolExecutor(args.parallel) as pool:
            recording=threading.Lock()
            def record(query,gate):
                row=send(query,gate)
                with recording:
                    rows.append(row); log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
                    print(f'{len(rows)}/50 query_id={row["query_id"]} HTTP={row.get("status")} generation={row["complete_generation"]}',flush=True)
                return row
            for start in range(0,50,args.parallel):
                # Explicit client pacing between waves. A refused query is never
                # retried; each saved input has at most one POST in this run.
                wait_ready(admission_required=True)
                gate=threading.Event()
                futures=[]
                try:
                    for query in queries[start:start+args.parallel]:
                        futures.append(pool.submit(record,query,gate))
                finally:
                    gate.set()
                for future in futures:
                    future.result()
        summary['health_after']=wait_ready(admission_required=False)
    except BaseException as error:
        summary['errors'].append(repr(error))
    finally:
        stop.set(); watcher.join(10)
        if 'health_after' not in summary:
            try:
                summary['health_after']=health()
            except Exception as error:
                summary['errors'].append('final health: '+repr(error))
    submitted={r['query_id'] for r in rows if r['http_submitted']}
    summary.update(attempted=len(submitted),recorded_queries=len(rows),successful_generation=sum(r['complete_generation'] for r in rows),
        http_503=sum(r.get('status')==503 for r in rows),
        failed_query_ids=[r['query_id'] for r in rows if not r['complete_generation']],
        unsubmitted_query_ids=[q.query_id for q in queries if q.query_id not in submitted],
        telemetry_errors=health_errors)
    successful=[r for r in rows if r['complete_generation']]
    def metrics(values):
        values=sorted(values)
        if not values:
            return None
        return {'mean':sum(values)/len(values),'p50':values[(len(values)-1)//2],
                'p95':values[math.ceil(.95*len(values))-1],'max':values[-1]}
    summary['client_latency_ms']=metrics([r['latency_ms'] for r in successful])
    summary['stage_timing_ms']={stage:metrics([r['response']['timings_ms'][stage] for r in successful])
        for stage in ('embedding_ms','rerank_ms','generation_ms','total_ms')}
    summary['token_usage']={kind:sum(r['response']['usage'][kind] for r in successful)
                            for kind in ('input_tokens','output_tokens')}
    summary['quality_observations']={
        'answer_echoes_query':sum(r.get('quality_observations',{}).get('answer_echoes_query',False) for r in successful),
        'output_limit_reached':sum(r.get('quality_observations',{}).get('output_limit_reached',False) for r in successful),
        'note':'Answer quality is not certified by workflow execution checks'}
    if len(rows)==50 and all(r['complete_generation'] for r in rows) and not (summary['errors'] or health_errors):
        summary['status']='passed'
    elif summary['errors'] or health_errors or any(r.get('status') not in (200,503) or
            (r.get('status')==200 and not r['complete_generation']) or
            (r.get('status')==503 and not r['expected_rejection']) for r in rows):
        summary['status']='failed'
    if priority_cycle:
        decisions = [decision for row in successful
                     for stage in row['response']['scheduling']['stages'].values() for decision in stage]
        overtakes = sum(d['overtook_lower_priority'] for d in decisions)
        promotions = sum(d['priority'] != d['effective_priority'] for d in decisions)
        summary['priority_validation'] = {
            'assigned_counts': {p: sum(v == p for v in assigned.values()) for p in ('high','normal','low')},
            'dispatch_decisions': len(decisions), 'overtook_lower_priority': overtakes,
            'aging_promotions': promotions,
            'overtaking': 'observed' if overtakes else 'not_observed',
            'aging': 'observed' if promotions else 'not_observed',
            'note': 'Generation coverage is separate from priority contention. No contention is not a priority acceptance pass.'}
    with (out/'generated_responses.txt').open('w') as readable:
        by_id = {r['query_id']: r for r in rows}
        for query in queries:
            row = by_id.get(query.query_id, {})
            readable.write(f"{query.query_id} | {query.query}\nHTTP={row.get('status', 'not submitted')} "
                           f"priority={row.get('request', {}).get('priority', 'normal')}\n")
            readable.write(str(row.get('response', {}).get('answer', row.get('error', row.get('rejection_reason', '')))) + '\n\n')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--backend',choices=['milvus','elasticsearch'],required=True)
    parser.add_argument('--url',default='http://127.0.0.1:8088')
    parser.add_argument('--queries',type=Path,default=ROOT/'data/queries/nfcorpus-50.jsonl')
    parser.add_argument('--parallel',type=int,default=1)
    parser.add_argument('--require-vllm-generation',action='store_true')
    parser.add_argument('--priority-cycle',action='store_true',
                        help='assign low/normal/high in frozen input order; require dispatch evidence')
    parser.add_argument('--max-new-tokens',type=int,default=256)
    parser.add_argument('--request-timeout',type=float,default=300)
    parser.add_argument('--ready-timeout',type=float,default=90)
    args=parser.parse_args()
    if not 1<=args.parallel<=50 or not 1<=args.max_new_tokens<=8192 or any(
            not math.isfinite(v) or v<=0 for v in (args.request_timeout,args.ready_timeout)):
        parser.error('parallel 1..50, max-new-tokens 1..8192 and positive finite timeouts required')
    queries=frozen_queries(args.queries)
    code,_,before=http(args.url+'/health')
    if code!=200:
        raise RuntimeError('API health unavailable')
    validate_service(before,args.backend,args.max_new_tokens)
    if args.require_vllm_generation:
        validate_vllm_workflow(before)
    if args.priority_cycle:
        validate_priority_service(before)
    print(json.dumps({'preflight':'passed','queries':50,'queries_sha256':QUERY_SHA256,
        'backend':args.backend,'parallel':args.parallel,'max_new_tokens':args.max_new_tokens,
        'run_requested':args.run}),flush=True)
    if not args.run:
        return 0
    with open('/tmp/rag-watermarks-modes.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        out=ROOT/'experiment_outputs/benchmarks'/('priority_generation_50' if args.priority_cycle else 'generation_50')/args.backend/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out.mkdir(parents=True)
        print('OUTPUT='+str(out),flush=True)
        shutil.copyfile(args.queries,out/'queries.jsonl')
        files=[p for sub in ('src','scripts') for p in (ROOT/sub).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.cpp')]
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for sub in ('src','scripts'):
            shutil.copytree(ROOT/sub,out/'source'/sub,ignore=shutil.ignore_patterns('__pycache__'))
        from rag_engine.generation import checkpoint_manifest
        model=checkpoint_manifest(before['generation']['execution_profile']['model_path'])
        packages={}
        for name in ('torch','torch-npu','transformers','vllm','vllm-ascend','elasticsearch','pymilvus'):
            try:
                packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name]='not_installed_in_client_environment'
        (out/'manifest.json').write_text(json.dumps({'command':sys.argv,'options':vars(args),
            'query_sha256':QUERY_SHA256,'initial_health':before,'generation_model':model,
            'client_packages':packages,'python':sys.version,
            'visible_device_nodes':[str(p) for p in Path('/dev').glob('davinci[0-9]*')],
            'source_sha256':hashes,'compute_utilization':'not measured',
            'input_policy':'original query bytes; no query rewriting or automatic retry'},indent=2,default=str))
        summary=replay(args,queries,before,out)
    if args.priority_cycle:
        print('GENERATION_COVERAGE='+summary['status'],flush=True)
        print('PRIORITY_OVERTAKING='+summary['priority_validation']['overtaking'],flush=True)
        print('PRIORITY_AGING='+summary['priority_validation']['aging'],flush=True)
    print('RESULT='+summary['status'],flush=True)
    return {'passed':0,'failed':1,'incomplete':2}[summary['status']]


if __name__=='__main__':
    sys.exit(main())
