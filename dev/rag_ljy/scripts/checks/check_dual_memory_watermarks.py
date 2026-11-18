#!/usr/bin/env python3
"""Real RAG 85/75 watermark trials on both databases and both physical dies.

No allocator filler, engine reconfiguration, cache clearing, restart or retries.
Default: read-only preflight. --run executes a bounded, recorded load matrix.
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
import shutil
import signal
import sys
import threading
import time
from collections import Counter

from check_generation_50 import ROOT, http, validate_service, validate_vllm_workflow, telemetry_error

STAGES = {'embedding_rerank': 7, 'generation': 6}


def ratio(memory):
    s = memory['stats']
    return max(s['used_mb'], s['total_mb'] - s['free_mb']) / s['total_mb']


def phase(memory, admission):
    if (memory.get('status') != 'ok' or memory.get('stale')
            or admission.get('watermark_basis') != 'device_measured_usage'):
        return None
    used = ratio(memory)
    if used >= .85 and admission.get('reason') == 'memory_pressure': return 'high'
    if .75 < used < .85 and admission.get('reason') == 'memory_pressure': return 'middle'
    if used <= .75 and admission.get('status') == 'ready': return 'low'
    return None


def stage_phase(health, stage):
    return phase(health['device_memory_by_stage'][stage], health['memory_admission_by_stage'][stage])


def decision(row, stage):
    """Only the server's admission snapshot proves which die caused a 503."""
    response = row.get('response')
    detail = response.get('detail', {}) if isinstance(response, dict) else {}
    if not isinstance(detail, dict) or row.get('status') != 503 or not row.get('headers', {}).get('retry-after'):
        return None
    snapshot = detail.get('admission_snapshot', {})
    memory = snapshot.get('device_memory', {})
    admission = snapshot.get('memory_admission', {})
    if memory.get('physical_die') != STAGES[stage] or detail.get('reason') != 'memory_pressure':
        return None
    observed_phase = phase(memory, admission)
    if observed_phase is None: return None
    return {'phase': observed_phase, 'used_percent': ratio(memory)*100,
            'time': detail.get('decision_monotonic', row['sent']), 'request_id': row.get('headers', {}).get('x-request-id')}


def assess_stage(samples, rows, stage):
    readings = [(r['time'], r['health']['device_memory_by_stage'][stage]) for r in samples]
    peak = max((ratio(m) for _, m in readings), default=0)
    decisions = [d for r in rows if (d := decision(r, stage))]
    peak = max([peak] + [d['used_percent']/100 for d in decisions])
    highs = [d for d in decisions if d['phase'] == 'high']
    observed_high = [t for t,m in readings if ratio(m) >= .85]
    first = min(observed_high + [d['time'] for d in highs], default=None)
    middle = [d for d in decisions if d['phase'] == 'middle' and first is not None and d['time'] > first]
    low_after_middle = False
    for r in rows:
        if r.get('status') == 200 and r.get('complete_generation') and middle and r['sent'] > min(d['time'] for d in middle):
            low_after_middle |= (stage_phase(r['before'], stage) == 'low' and stage_phase(r['after'], stage) == 'low')
    crossed = False
    for r in rows:
        if r.get('role') == 'load' and r.get('complete_generation') and first is not None:
            trace = r.get('response', {}).get('scheduling', {}).get('stages', {}).get('embedding', [])
            # A stage dispatch is evidence of admission, unlike client send time.
            crossed |= bool(trace and trace[0]['dispatched_monotonic'] < first < r['completed'])
    checks = {'actual_reached_85': first is not None, 'high_memory_503': bool(highs),
              'middle_band_memory_503': bool(middle), 'low_recovery_http_200': low_after_middle,
              'accepted_generation_completed_across_pressure': crossed}
    return {'physical_die': STAGES[stage], 'peak_actual_percent': round(peak*100, 3),
            'minimum_free_mib': min((m['stats']['free_mb'] for _,m in readings), default=None),
            'checks': checks, 'decision_evidence': decisions,
            'status': 'passed' if all(checks.values()) else ('threshold_not_reached' if first is None else 'incomplete')}


def validate_health(h, before, backends, max_new_tokens):
    for backend in backends: validate_service(h, backend, max_new_tokens)
    validate_vllm_workflow(h)
    reason = telemetry_error(h)
    if reason: raise RuntimeError(reason)
    for stage in STAGES:
        m = h['device_memory_by_stage'][stage]
        if m.get('sampling_process') != before['device_memory_by_stage'][stage].get('sampling_process'):
            raise RuntimeError('Sampler changed during trial; restart cannot count as recovery')
        if h['memory_admission_by_stage'][stage]['config'] != before['memory_admission_by_stage'][stage]['config']:
            raise RuntimeError('Admission configuration changed during trial')
    if h['generation']['execution_profile'] != before['generation']['execution_profile']:
        raise RuntimeError('Generation engine changed during trial')
    if h['memory_admission_by_stage']['generation'].get('reserved_mib') != 0:
        raise RuntimeError('Unexpected per-request generation reservation')


class Trial:
    def __init__(self, args, before, backend, mode, inputs, out, cancelled):
        self.a, self.before, self.backend, self.mode = args, before, backend, mode
        self.inputs, self.out, self.cancelled = inputs, out, cancelled
        self.samples, self.rows, self.errors = [], [], []
        self.stop = threading.Event(); self.unsafe = threading.Event(); self.high = threading.Event()
        self.lock = threading.Lock(); self.probe_count = 0
        self.events = (out/'events.jsonl').open('w')

    def emit(self, kind, **data):
        with self.lock:
            self.events.write(json.dumps({'kind':kind,'utc':datetime.now(timezone.utc).isoformat(),**data},ensure_ascii=False)+'\n')
            self.events.flush()

    def health(self):
        code,_,h = http(self.a.url+'/health',timeout=5)
        if code != 200: raise RuntimeError('Health unavailable')
        validate_health(h,self.before,[self.backend],self.a.max_new_tokens)
        return h

    def stopped(self): return self.cancelled.is_set() or self.unsafe.is_set()

    def body(self, profile, index):
        tier,batch = profile
        rows = self.inputs[tier]
        return {'query':rows[index % len(rows)]['query'], 'retrieval_backend':self.backend,
                'generate':True, 'priority':'normal', 'max_new_tokens':self.a.max_new_tokens,
                'retriever_top_k':100,'coarse_top_k':50,'fine_top_k':5,'reranker_batch_size':batch}

    def send(self, body, role, gate=None, profile=None):
        if gate is not None: gate.wait()
        if self.stopped(): return
        row = {'sent':time.monotonic(), 'role':role, 'profile':profile, 'request':body}
        try:
            row['before'] = self.health()
            if self.stopped(): return
            row['sent'] = time.monotonic()
            row['status'],row['headers'],row['response'] = http(self.a.url+'/query',body,timeout=self.a.request_timeout)
            row['completed'] = time.monotonic()
            row['after'] = self.health()
            response = row['response']
            row['complete_generation'] = bool(row['status']==200 and isinstance(response,dict)
                and response.get('generation_backend')=='vllm' and response.get('retrieval_backend')==self.backend
                and row['headers'].get('x-retrieval-backend')==self.backend
                and response.get('answer') and response.get('usage',{}).get('output_tokens',0)>0)
            detail = response.get('detail',{}) if isinstance(response,dict) else {}
            row['reason'] = detail.get('reason') if isinstance(detail,dict) else str(detail)
            expected_reject = row['status']==503 and row['reason'] in ('memory_pressure','request_budget') and row['headers'].get('retry-after')
            if not (row['complete_generation'] or expected_reject):
                raise RuntimeError('Unexpected HTTP/response: '+str(row['status'])+' '+str(row['reason']))
        except Exception as error:
            row['error'] = repr(error); self.unsafe.set()
            self.errors.append(repr(error))
        finally:
            row.setdefault('completed',time.monotonic())
            with self.lock: self.rows.append(row)
            self.emit('request',**row)

    def watch(self):
        seen_high = set(); last_probe = 0; probe = None; last_phases = None; last_progress = 0
        with ThreadPoolExecutor(1) as probes:
            while not self.stop.is_set():
                try:
                    h=self.health(); now=time.monotonic()
                    row={'time':now,'health':h}
                    with self.lock:self.samples.append(row)
                    self.emit('health',**row)
                    if now-last_progress >= 5:
                        print(json.dumps({'backend':self.backend,'mode':self.mode,'inflight':h['inflight'],
                            'actual_percent':{s:round(ratio(h['device_memory_by_stage'][s])*100,3) for s in STAGES},
                            'completed_requests':len(self.rows)},ensure_ascii=False),flush=True)
                        last_progress=now
                    for stage in STAGES:
                        if ratio(h['device_memory_by_stage'][stage])>=.85:
                            seen_high.add(stage); self.high.set()
                    phases = tuple(stage_phase(h, stage) for stage in STAGES)
                    if (seen_high and not self.stopped() and self.probe_count<self.a.max_probes
                            and (now-last_probe>=1 or phases != last_phases) and (probe is None or probe.done())):
                        # Probe is still a full RAG request. It does not allocate filler memory.
                        body=self.body(('original',1),0)
                        body.update(max_new_tokens=16,retriever_top_k=20,coarse_top_k=20,fine_top_k=3)
                        probe=probes.submit(self.send,body,'probe')
                        self.probe_count+=1; last_probe=now; last_phases=phases
                except Exception as error:
                    self.errors.append(repr(error)); self.unsafe.set(); self.emit('monitor_error',error=repr(error)); break
                self.stop.wait(self.a.sample_interval)

    def drain(self):
        end=time.monotonic()+self.a.drain_timeout
        while time.monotonic()<end:
            h=self.health()
            if h['inflight']==0 and h['memory_admission_by_stage']['embedding_rerank']['reserved_mib']==0:return h
            time.sleep(.1)
        raise RuntimeError('Accepted requests/reservations did not drain; no services were killed')

    def run(self, profiles):
        deadline=time.monotonic()+self.a.trial_seconds
        watcher=threading.Thread(target=self.watch,daemon=True); watcher.start()
        futures=[]
        try:
            with ThreadPoolExecutor(self.a.client_limit) as clients:
                for profile in profiles:
                    if self.stopped() or self.high.is_set() or time.monotonic()>=deadline: break
                    # A prior profile is fully drained before escalation.
                    self.drain()
                    preview=http(self.a.url+'/memory/estimate',self.body(profile,0),timeout=15)
                    self.emit('profile',profile=profile,estimate_status=preview[0],estimate=preview[2])
                    if preview[0]!=200:raise RuntimeError('Profile estimate unavailable')
                    print(f'{self.backend}/{self.mode}: query_tokens={profile[0]} rerank_batch={profile[1]}',flush=True)
                    futures=[]; gate=threading.Event()
                    count=self.a.count if self.mode!='continuous' else min(self.a.max_load_requests,math.ceil(self.a.continuous_seconds/self.a.spacing))
                    started=time.monotonic()
                    try:
                        for index in range(count):
                            if self.stopped() or time.monotonic()>=deadline or self.high.is_set():break
                            if sum(not f.done() for f in futures)>=self.a.client_limit:
                                self.emit('client_saturation',limit=self.a.client_limit)
                                break  # Do not hide an unbounded client-side backlog.
                            futures.append(clients.submit(self.send,self.body(profile,index),'load',gate,profile))
                            if self.mode!='burst':
                                gate.set()
                                wait=max(0,started+(index+1)*self.a.spacing-time.monotonic())
                                self.cancelled.wait(wait)
                    finally:gate.set()
                    for future in futures:future.result()
                    if self.high.is_set():break
                # Observe natural release and probe high/middle/low without restarting.
                while self.high.is_set() and not self.stopped() and time.monotonic()<deadline:
                    h=self.health()
                    if (all(stage_phase(h,s)=='low' for s in STAGES)
                            and h['inflight']==0 and any(r.get('role')=='probe' and r.get('complete_generation')
                                and all(stage_phase(r['before'],s)=='low' and stage_phase(r['after'],s)=='low' for s in STAGES)
                                for r in list(self.rows))):break
                    self.cancelled.wait(.1)
        except Exception as error:
            self.errors.append(repr(error)); self.unsafe.set()
        finally:
            self.stop.set(); watcher.join(self.a.request_timeout+10)
            if watcher.is_alive():self.errors.append('Monitor/probe did not stop')
            try:after=self.drain()
            except Exception as error:after=None; self.errors.append(repr(error))
            results={stage:assess_stage(self.samples,self.rows,stage) for stage in STAGES}
            status=('failed' if self.errors else 'passed' if all(r['status']=='passed' for r in results.values()) else 'incomplete')
            summary={'backend':self.backend,'mode':self.mode,'status':status,'devices':results,'errors':self.errors,
                'requests':len(self.rows),'http_counts':dict(Counter(str(r.get('status')) for r in self.rows)),
                'rejection_reasons':dict(Counter(r.get('reason') for r in self.rows if r.get('status')==503)),
                'health_after':after, 'client_limit':self.a.client_limit, 'generation_reserved_mib':0,
                'scope':'real full RAG; each die assessed independently; no engine/config/restart changes'}
            (self.out/'summary.json').write_text(json.dumps(summary,indent=2))
            self.events.close()
        print(json.dumps({'backend':self.backend,'mode':self.mode,'result':status,
                          'devices':{k:{'status':v['status'],'peak_actual_percent':v['peak_actual_percent']} for k,v in results.items()}}),flush=True)
        return summary


def prepare_inputs(path, profiles):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    groups={tier:[r for r in rows if r.get('tier','original')==tier] for tier in ('original','1024','2048','4096','6144')}
    if not groups['original'] or not groups['6144']:raise ValueError('Missing original/6144 corpus-derived pressure inputs')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B/tokenizer.json')
    tokenizer.no_padding(); tokenizer.no_truncation()
    for tier,_ in profiles:
        if tier in groups:continue
        groups[tier]=[{**r,'query':tokenizer.decode(tokenizer.encode(r['query'],add_special_tokens=False).ids[:int(tier)],skip_special_tokens=False),
                      'tier':tier,'constructed':True,'parent_tier':'6144'} for r in groups['6144'][:8]]
    return {tier:groups[tier][:8] for tier in {'original'}|{t for t,_ in profiles}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--backends',nargs='+',choices=['milvus','elasticsearch'],default=['milvus','elasticsearch'])
    p.add_argument('--modes',nargs='+',choices=['burst','staggered','continuous'],default=['burst','staggered','continuous'])
    p.add_argument('--profiles',default='1024:8,1024:16,2048:16,2112:32,2304:32,3072:32')
    p.add_argument('--count',type=int,default=4)
    p.add_argument('--client-limit',type=int,default=16)
    p.add_argument('--max-load-requests',type=int,default=16)
    p.add_argument('--spacing',type=float,default=.2)
    p.add_argument('--continuous-seconds',type=float,default=3)
    p.add_argument('--trial-seconds',type=float,default=180)
    p.add_argument('--request-timeout',type=float,default=180)
    p.add_argument('--drain-timeout',type=float,default=180)
    p.add_argument('--sample-interval',type=float,default=.02)
    p.add_argument('--max-probes',type=int,default=256)
    p.add_argument('--max-new-tokens',type=int,default=256)
    p.add_argument('--url',default='http://127.0.0.1:8088')
    p.add_argument('--inputs',type=Path,default=ROOT/'experiment_outputs/inputs/request_pressure/queries.jsonl')
    a=p.parse_args()
    if not (2<=a.count<=a.client_limit<=32 and 1<=a.max_load_requests<=64 and 1<=a.max_probes<=512 and 1<=a.max_new_tokens<=1024):p.error('Invalid finite request/client bounds')
    if any(not math.isfinite(v) or v<=0 for v in (a.spacing,a.continuous_seconds,a.trial_seconds,a.request_timeout,a.drain_timeout,a.sample_interval)):p.error('Intervals must be finite and positive')
    try:profiles=[(t,int(b)) for t,b in (item.split(':') for item in a.profiles.split(','))]
    except ValueError:p.error('profiles format: token_count:rerank_batch,...')
    if any(not t.isdecimal() or not 128<=int(t)<=3072 or not 1<=b<=32 for t,b in profiles):p.error('Query token tiers 128..3072; batches 1..32')
    code,_,before=http(a.url+'/health')
    if code!=200:raise RuntimeError('RAG API unavailable')
    validate_health(before,before,a.backends,a.max_new_tokens)
    if before['inflight'] or before['memory_admission']['reserved_mib']:raise RuntimeError('API must be idle before this isolated suite')
    inputs=prepare_inputs(a.inputs,profiles)
    print(json.dumps({'preflight':'passed','backends':a.backends,'modes':a.modes,'profiles':profiles,
        'run_requested':a.run,'pressure_source':'real RAG requests with corpus-derived long queries',
        'generation_note':'vLLM preallocated KV pool may prevent real queries from reaching 85%; report threshold_not_reached, never substitute reservation pressure'}),flush=True)
    if not a.run:return 0
    cancelled=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:cancelled.set())
    with open('/tmp/rag-watermarks-modes.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=ROOT/'experiment_outputs/checks/dual_memory_watermarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out.mkdir(parents=True); print('OUTPUT='+str(out),flush=True)
        (out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False))
        for sub in ('src','scripts'):
            shutil.copytree(ROOT/sub,out/'source'/sub,ignore=shutil.ignore_patterns('__pycache__'))
        sys.path.insert(0,str(ROOT/'src'))
        from rag_engine.generation import checkpoint_manifest
        packages={}
        for name in ('torch','torch-npu','vllm','vllm-ascend','transformers','pymilvus','elasticsearch'):
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]='not_installed'
        manifests={}
        for name in ('Qwen3-Embedding-0.6B','Qwen3-Reranker-0.6B'):
            directory=Path('/mnt/raid/jliu171/models/Qwen')/name
            manifests[name]={f:hashlib.sha256((directory/f).read_bytes()).hexdigest() for f in ('config.json','tokenizer.json')}
        manifests['OneRec-8B']=checkpoint_manifest(before['generation']['execution_profile']['model_path'])
        (out/'manifest.json').write_text(json.dumps({'command':sys.argv,'options':vars(a),'health_before':before,
            'packages':packages,'models':manifests,'input_sha256':hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
            'pressure_source':'real full RAG requests only; constructed long queries explicitly labeled',
            'compute_utilization':'not measured','sampling':'device-context.runtime.memory_stats on each die',
            'cross_trial_state':'natural warm/cache state; no claim of independent cold starts'},indent=2,default=str))
        summaries=[]
        for backend in a.backends:
            for mode in a.modes:
                target=out/backend/mode;target.mkdir(parents=True)
                if cancelled.is_set():
                    summaries.append({'backend':backend,'mode':mode,'status':'not_run','reason':'cancelled'});continue
                try:
                    code,_,h=http(a.url+'/health')
                    if code!=200: raise RuntimeError('Health unavailable before trial')
                    validate_health(h,before,[backend],a.max_new_tokens)
                    if any(stage_phase(h,s)!='low' for s in STAGES) or h['inflight'] or h['memory_admission']['reserved_mib']:
                        summaries.append({'backend':backend,'mode':mode,'status':'not_run','reason':'no idle low baseline'});continue
                except Exception as error:
                    summaries.append({'backend':backend,'mode':mode,'status':'failed','reason':repr(error)})
                    cancelled.set();continue
                trial=Trial(a,before,backend,mode,inputs,target,cancelled)
                result=trial.run(profiles);summaries.append(result)
                if result['status']=='failed':cancelled.set()
        status='passed' if summaries and all(r['status']=='passed' for r in summaries) else 'failed' if any(r['status']=='failed' for r in summaries) else 'incomplete'
        summary={'status':status,'trials':summaries,'cancelled':cancelled.is_set(),
                 'acceptance':'Both dies must independently pass for every requested database/arrival mode'}
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('RESULT='+status,flush=True)
    return {'passed':0,'incomplete':2,'failed':1}[status]


if __name__=='__main__':sys.exit(main())
