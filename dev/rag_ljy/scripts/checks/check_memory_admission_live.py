#!/usr/bin/env python3
"""Real-device admission verification. Run with NO other load generator.

Uses actual device-context reads and bounded allocations on the selected device.
Raises usage above the unchanged 85% watermark, retains an 8 GiB free guard,
checks hysteresis at 80%, then releases all test allocations. It never intends
to cause OOM. Separate temporary API instances test budget/stale-read refusal.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import load_project_env, PROJECT_ROOT
load_project_env()
sys.path.insert(0,str(PROJECT_ROOT/'scripts/benchmarks'))
from check_online_pressure import http
from rag_engine.device import prepare_torch_device
import argparse, gc, json, os, shutil, socket, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


def main():
    raise SystemExit('External-allocation experiment retired. Use scripts/checks/check_request_memory_modes.py --run; no extra device allocations are permitted for the revised experiment.')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device',default='npu:0')
    ap.add_argument('--url',default='http://127.0.0.1:8088')
    ap.add_argument('--aux-port',type=int,default=8087)
    a=ap.parse_args()
    root=PROJECT_ROOT/'experiment_outputs/checks/memory_admission'
    out=root/(datetime.now().strftime('%Y%m%dT%H%M%S')+'-real-pressure'); out.mkdir(parents=True)
    shutil.copyfile(__file__,out/'checker_snapshot.py')
    print('OUTPUT='+str(out),flush=True)
    events=[]; checks={}; buffers=[]; child=None; log=None; drain_pool=None; drain_futures=[]
    queries=[json.loads(x) for x in (PROJECT_ROOT/'data/queries/nfcorpus-50.jsonl').read_text().splitlines()][:3]
    def save(kind,**data):
        events.append({'t':time.monotonic(),'kind':kind,**data})
        (out/'events.json').write_text(json.dumps(events,indent=2))
    def health(url=a.url):return http(url+'/health',timeout=10)[2]
    def wait(predicate,url=a.url,timeout=30):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            h=health(url)
            if predicate(h):return h
            time.sleep(.2)
        raise RuntimeError('health condition timeout: '+json.dumps(h))
    def send(label,url=a.url,expected=200,reason=None,n=3):
        for q in queries[:n]:
            code,headers,body=http(url+'/query',{'query':q['query']})
            save(label,status=code,headers=headers,body=body)
            assert code==expected,(label,code,body)
            if reason:
                assert body['detail']['reason']==reason,(label,body)
                assert {k.lower():v for k,v in headers.items()}.get('retry-after')=='1'
        checks[label]=True
    torch,device=prepare_torch_device(a.device)
    import runtime
    module=getattr(torch,device.type)
    def read():
        runtime.synchronize(); return runtime.memory_stats(0)
    def used(s):return max(s['used_mb'],s['total_mb']-s['free_mb'])
    def release_all():
        buffers.clear(); gc.collect(); module.empty_cache(); runtime.synchronize()
    def start_aux(name,extra):
        nonlocal child,log
        with socket.socket() as sock:sock.bind(('127.0.0.1',a.aux_port))
        log=(out/(name+'.log')).open('w')
        command=[sys.executable,'-u',str(PROJECT_ROOT/'scripts/runtime/serve_queries.py'),'--device',a.device,'--port',str(a.aux_port),'--memory-admission','--memory-allow-unverified-scope']+extra
        child=subprocess.Popen(command,cwd=PROJECT_ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        url='http://127.0.0.1:'+str(a.aux_port)
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError('aux server exited; see '+name+'.log')
            try:
                h=health(url)
                if h.get('status')=='ready' and h['device_memory'].get('status')=='ok':return url
            except Exception:pass
            time.sleep(.5)
        raise RuntimeError('aux readiness timeout')
    def stop_aux():
        nonlocal child,log
        if child is not None:
            child.terminate()
            try: child.wait(30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
            child=None
        if log:log.close();log=None
    error=None
    try:
        h=wait(lambda h:h['inflight']==0 and h['memory_admission']['reserved_mib']==0)
        assert h['memory_admission']['config']['high_watermark']==.85
        assert h['memory_admission']['config']['low_watermark']==.75
        send('normal_acceptance')
        wait(lambda h:h['inflight']==0 and h['memory_admission']['reserved_mib']==0)
        baseline=read(); save('allocation_baseline',stats=baseline,health=health())
        assert used(baseline)/baseline['total_mb']<.70,'Need low initial usage'
        before=health()['memory_admission']['accepted']
        drain_pool=ThreadPoolExecutor(max_workers=6)
        drain_futures=[drain_pool.submit(http,a.url+'/query',{'query':queries[i%3]['query']}) for i in range(6)]
        h=wait(lambda h:h['memory_admission']['accepted']>=before+6,timeout=15)
        save('requests_accepted_before_pressure',health=h)
        overlap_seen=False
        # Actual device pressure; cap each allocation and reserve 8 GiB free.
        while True:
            s=read(); ratio=used(s)/s['total_mb']
            if s['free_mb']<8192:raise RuntimeError('Free memory fell below 8 GiB guard')
            if ratio>=.857:break
            if s['free_mb']<8192+256:raise RuntimeError('8 GiB guard reached before target')
            if len(buffers)>=200:raise RuntimeError('allocation cap reached')
            buffers.append(torch.empty(256*1024*1024,dtype=torch.uint8,device=device))
            save('allocation_step',stats=read(),held_mib=256*len(buffers))
            if len(buffers)%16==0:
                h=health();save('pressure_drain_observation',health=h)
                overlap_seen |= h['memory_admission']['reason']=='memory_pressure' and h['inflight']>0
        h=wait(lambda h:h['memory_admission']['reason']=='memory_pressure' and h['device_memory']['stats']['used_mb']/h['device_memory']['stats']['total_mb']>.85)
        save('above_actual_85_percent',health=h,stats=read())
        for future in drain_futures:
            code,headers,body=future.result(timeout=120)
            save('accepted_request_drained',status=code,headers=headers,body=body)
            assert code==200
        drain_pool.shutdown();drain_pool=None
        checks['six_preaccepted_requests_completed']=True
        save('pressure_drain_overlap',observed=overlap_seen)
        send('high_watermark_rejection',expected=503,reason='memory_pressure')
        while used(read())/read()['total_mb']>.80 and buffers:
            buffers.pop(); gc.collect(); module.empty_cache(); runtime.synchronize()
        h=wait(lambda h:.75<h['device_memory']['stats']['used_mb']/h['device_memory']['stats']['total_mb']<.85)
        save('middle_band',health=h,stats=read())
        assert h['memory_admission']['reason']=='memory_pressure'
        send('hysteresis_still_paused',expected=503,reason='memory_pressure')
        release_all()
        h=wait(lambda h:h['memory_admission']['status']=='ready' and h['device_memory']['stats']['used_mb']/h['device_memory']['stats']['total_mb']<=.75)
        save('below_low_watermark',health=h,stats=read())
        send('recovery_acceptance')
        # No large allocations here: budget accounting alone must reject.
        url=start_aux('budget',['--memory-request-mib','60000'])
        wait(lambda h:h['memory_admission']['reason']=='request_budget',url)
        save('budget_health',health=health(url))
        send('request_budget_rejection',url,503,'request_budget'); stop_aux()
        # Slow sampling / short max age intentionally forces stale-read windows.
        url=start_aux('stale',['--memory-request-mib','2048','--memory-probe-interval','8','--memory-max-sample-age','1.5'])
        wait(lambda h:h['memory_admission']['reason']=='stale_telemetry',url)
        send('stale_read_rejection',url,503,'stale_telemetry',n=1)
        wait(lambda h:h['memory_admission']['status']=='ready' and h['device_memory']['age_seconds']<.3,url,timeout=15)
        send('fresh_read_recovery',url,n=1); stop_aux()
    except Exception as e:error=repr(e)
    finally:
        release_all();stop_aux()
        if drain_pool is not None:drain_pool.shutdown(wait=True)
        save('final',stats=read(),health=health())
        summary={'status':'failed' if error else 'passed','error':error,'checks':checks,'device':a.device,'fixed_watermarks':[.85,.75],'minimum_free_guard_mib':8192,'notes':['This exercises actual 85% device usage; no lowered watermarks.','60000 MiB is a deliberate refusal-test budget, not a recommended request budget.','Auxiliary stale/budget configurations do not change the main API.','Probe failures and reservations also require offline regression tests.']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return int(error is not None)

if __name__=='__main__':sys.exit(main())
