#!/usr/bin/env python3
"""Finite real-request test of the measured-only 85/75% and 5GiB policy.

No external allocations, coefficient changes, cache clearing, service restart,
or linear peak extrapolation. Default is read-only preflight.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import shutil
import signal
import sys
import threading
import time
from check_watermarks_modes import ROOT, http, conflicting_processes, sha256
from check_request_memory_modes import assess, measured_phase_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['burst','staggered','continuous'], default=['burst','staggered','continuous'])
    parser.add_argument('--profiles', default='original:4,1024:4,1024:8,1024:16,2048:4,2048:8,4096:1,4096:2,6144:1')
    parser.add_argument('--warmup-profile', help='Optional two-request burst of the same corpus input (index 1) before each mode, recorded separately')
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--spacing', type=float, default=.2)
    parser.add_argument('--continuous-seconds', type=float, default=5)
    args = parser.parse_args()
    if not 2 <= args.count <= 32 or args.spacing <= 0 or not 1 <= args.continuous_seconds <= 30:
        parser.error('count 2..32, positive spacing, continuous duration 1..30s required')
    lock = open('/tmp/rag-watermarks-modes.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError('Other test running: '+json.dumps(conflicts))
    url = 'http://127.0.0.1:8088'
    def health():
        code, _, h = http(url+'/health', timeout=10)
        if code != 200: raise RuntimeError('health failed')
        return h
    before = health()
    cfg = before['memory_admission']['config']
    assert before['inflight'] == 0 and before['memory_admission']['reserved_mib'] == 0
    assert before['max_inflight'] == 0 and cfg['estimate_requests'] and cfg['request_budget_mib'] is None
    assert cfg['high_watermark'] == .85 and cfg['low_watermark'] == .75 and cfg['safety_margin_mib'] == 5120
    assert before['memory_admission']['watermark_basis'] == 'device_measured_usage'
    assert before['device_memory']['source'] == 'device-context.runtime.memory_stats'
    assert before['memory_estimation']['fault'] is None
    print('Measured-only policy preflight passed', flush=True)
    if not args.run: return 0
    out = ROOT/'experiment_outputs/checks/measured_watermarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out), flush=True)
    source = ROOT/'experiment_outputs/inputs/request_pressure/queries.jsonl'
    records = [json.loads(line) for line in source.read_text().splitlines()]
    groups = {tier:[r for r in records if r.get('tier','original')==tier] for tier in ('original','1024','2048','4096','6144')}
    profiles=[(t,int(b)) for t,b in (p.split(':') for p in args.profiles.split(','))]
    if args.warmup_profile:
        wt,wb=args.warmup_profile.split(':');wb=int(wb)
        if wt not in groups or not 1<=wb<=before['reranker_batch_limit']:raise ValueError('Unsupported warmup profile')
    if any(not 1 <= b <= before['reranker_batch_limit'] for t,b in profiles):
        raise ValueError('Unsupported batch')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file('/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B/tokenizer.json')
    for tier,_ in profiles:
        if tier in groups: continue
        n=int(tier)
        if not 128 <= n <= 6144: raise ValueError('Derived token tiers must be between 128 and 6144')
        groups[tier]=[{**r, 'query':tokenizer.decode(tokenizer.encode(r['query'],add_special_tokens=False).ids[:n],skip_special_tokens=False),
                       'tier':tier,'parent_tier':'6144','token_clip':n,'constructed':True} for r in groups['6144']]
    (out/'derived_inputs.json').write_text(json.dumps(groups,ensure_ascii=False))
    shutil.copyfile(source, out/'inputs.jsonl')
    for part in ('src','scripts'):
        shutil.copytree(ROOT/part,out/'source'/part,ignore=shutil.ignore_patterns('__pycache__'))
    hashes = {}
    for model in ('Qwen3-Embedding-0.6B','Qwen3-Reranker-0.6B'):
        root = __import__('pathlib').Path('/mnt/raid/jliu171/models/Qwen')/model
        hashes[model] = {str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts}
    (out/'manifest.json').write_text(json.dumps({'command':sys.argv,'options':vars(args),'initial_health':before,'models':hashes,
        'input_sha256':sha256(source),'pressure_source':'real RAG requests only',
        'protocol':'measured-only-v1; replaces unsafe linear escalation, preserves old safety_hold evidence',
        'compute_utilization':'not measured'},indent=2))
    events = (out/'events.jsonl').open('w')
    mutex, stop, unsafe = threading.Lock(), threading.Event(), threading.Event()
    state = {'mode':'setup','profile':None}
    probing = threading.Event()
    samples, rows, pending = [], [], []
    pool = ThreadPoolExecutor(max_workers=32)
    summary = {'status':'incomplete','modes':{},'errors':[]}
    def emit(kind, **data):
        with mutex:
            events.write(json.dumps({'kind':kind,'utc':datetime.now(timezone.utc).isoformat(),**data})+'\n');events.flush()
    def sample():
        h = health()
        row = {'t':time.monotonic(),'mode':state['mode'],'profile':state['profile'],'health':h}
        with mutex: samples.append(row)
        emit('health',**row)
        if h['device_memory']['status']!='ok' or h['device_memory'].get('stale') or h['memory_admission']['reason']=='stale_telemetry' or h['memory_estimation']['fault'] or h['device_memory']['stats']['free_mb']<5120:
            unsafe.set()
        return h
    def watch():
        last = 0
        last_phase, high_seen = None, False
        while not stop.is_set():
            try:
                h = sample()
                phase = next((p for p in ('high','middle','recovery') if measured_phase_ok(h,p)), None)
                high_seen |= phase == 'high'
                if high_seen and phase and phase != last_phase:
                    probing.set()
                    try: send({'query':groups['original'][0]['query'], 'reranker_batch_size':1},role='probe')
                    finally: probing.clear()
                last_phase = phase
                if phase == 'recovery': high_seen = False
                if time.monotonic()-last >= 5:
                    print(f"{state['mode']}: actual={h['device_memory']['stats']['memory_used_percent']}% reserved={h['memory_admission']['reserved_mib']}MiB inflight={h['inflight']}",flush=True)
                    last=time.monotonic()
            except Exception as e:
                unsafe.set();emit('monitor_error',error=repr(e));return
            stop.wait(.001)
    def check():
        if unsafe.is_set(): raise RuntimeError('Read/fault/5GiB guard stopped further submissions')
    def send(body, gate=None, role="load"):
        if gate is not None: gate.wait()
        row={'mode':state['mode'],'profile':state['profile'],'role':role,'payload':body,'sent':time.monotonic()}
        try:
            row['before']=health()
            row['sent']=time.monotonic()
            row['status'],row['headers'],row['body']=http(url+'/query',body,timeout=120)
            row['response_received']=time.monotonic()
            row['after']=health()
            detail=row['body'].get('detail') if isinstance(row['body'],dict) else None
            row['reason']=detail.get('reason') if isinstance(detail,dict) else detail
            if row['status']!=200 and not (row['status']==503 and row['reason'] in ('memory_pressure','request_budget') and row['headers'].get('retry-after')):unsafe.set()
        except Exception as e:
            row['error']=repr(e);unsafe.set()
        finally:
            row['completed']=time.monotonic()
            with mutex:rows.append(row)
            emit('request',**row)
    def drain():
        for f in pending:f.result(timeout=125)
        pending.clear()
        end=time.monotonic()+30
        while time.monotonic()<end:
            h=sample()
            if h['inflight']==0 and h['memory_admission']['reserved_mib']==0 and not probing.is_set():return h
            time.sleep(.1)
        raise RuntimeError('Accepted work did not drain')
    def interrupted(*_): raise KeyboardInterrupt('Interrupted; drain accepted work')
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(600)
    watcher=threading.Thread(target=watch,daemon=True);watcher.start()
    try:
        send({'query':groups['original'][0]['query'],'reranker_batch_size':1},role='probe')
        drain();check()
        for mode in args.modes:
            state.update(mode=mode,profile=None)
            if not measured_phase_ok(drain(),'recovery'):raise RuntimeError('No natural low baseline for next mode')
            if args.warmup_profile:
                state.update(mode='warmup:'+mode,profile={'tier':wt,'batch':wb})
                gate=threading.Event()
                for i in range(2):
                    body={'query':groups[wt][1]['query'],'reranker_batch_size':wb,'coarse_top_k':50,'retriever_top_k':100,'fine_top_k':5}
                    pending.append(pool.submit(send,body,gate,'warmup'))
                gate.set();drain();check()
                if any(measured_phase_ok(s['health'],'high') for s in samples if s['mode']=='warmup:'+mode):
                    raise RuntimeError('Warmup reached high; cannot attribute transition to arrival-mode load')
                state.update(mode=mode,profile=None)
                emit('mode_baseline',mode=mode,health=sample())
            for tier,batch in profiles:
                check();state['profile']={'tier':tier,'batch':batch}
                def payload(i):return {'query':groups[tier][i%len(groups[tier])]['query'],'reranker_batch_size':batch,'coarse_top_k':50,'retriever_top_k':100,'fine_top_k':5}
                code,_,estimate=http(url+'/memory/estimate',payload(0))
                if code!=200:raise RuntimeError('Estimate preview failed')
                emit('profile',mode=mode,profile=state['profile'],estimate=estimate)
                print(f'{mode}: {tier} tokens, batch {batch}',flush=True)
                if mode=='burst':
                    gate=threading.Event()
                    for i in range(args.count):pending.append(pool.submit(send,payload(i),gate))
                    gate.set()
                else:
                    count=args.count if mode=='staggered' else math.ceil(args.continuous_seconds/args.spacing)
                    start=time.monotonic()
                    for i in range(count):
                        check()
                        if any(s['mode']==mode and measured_phase_ok(s['health'],'high') for s in samples):break
                        if len([f for f in pending if not f.done()])>=32:raise RuntimeError('Client limit reached; no hidden backlog')
                        pending.append(pool.submit(send,payload(i)))
                        time.sleep(max(0,start+(i+1)*args.spacing-time.monotonic()))
                # Observe/probe while accepted work is still running, never wait for
                # drain before looking for high/middle transitions.
                end=time.monotonic()+120
                while time.monotonic()<end:
                    check();h=sample()
                    high_seen=any(s['mode']==mode and measured_phase_ok(s['health'],'high') for s in samples)
                    if high_seen:
                        send({'query':groups['original'][0]['query'],'reranker_batch_size':1},role='probe')
                    if all(f.done() for f in pending) and (not high_seen or measured_phase_ok(h,'recovery')):break
                    time.sleep(.1)
                drain();check()
                if high_seen:break
            ss=[s for s in samples if s['mode']==mode];rr=[r for r in rows if r['mode']==mode]
            result=assess(ss,rr)
            result['sent']=len(rr)
            result['successful']=sum(r.get('status')==200 for r in rr)
            summary['modes'][mode]=result
            print(json.dumps({'mode':mode,'result':result['status'],'peak_actual_percent':result['peak_actual_percent']}),flush=True)
        summary['status']='passed' if all(v['status']=='passed' for v in summary['modes'].values()) else 'incomplete'
    except BaseException as e:
        summary['status']='failed';summary['errors'].append(repr(e))
    finally:
        signal.alarm(0)
        pool.shutdown(wait=True)
        try:summary['health_after']=drain()
        except Exception as e:summary['errors'].append('drain:'+repr(e));summary['status']='failed'
        stop.set();watcher.join(15)
        for mode in args.modes:
            if mode not in summary['modes']:
                ss=[s for s in samples if s['mode']==mode];rr=[r for r in rows if r['mode']==mode]
                summary['modes'][mode]=assess(ss,rr) if ss else {'status':'not_run'}
        summary['minimum_free_mib']=min((s['health']['device_memory']['stats']['free_mb'] for s in samples if s['health']['device_memory'].get('stats')),default=None)
        summary['total_requests']=len(rows)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));events.close()
    print('RESULT='+summary['status'],flush=True)
    return 0 if summary['status']=='passed' else 1 if summary['status']=='failed' else 2

if __name__=='__main__':sys.exit(main())
