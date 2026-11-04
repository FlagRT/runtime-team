#!/usr/bin/env python3
"""Pause only the isolated sampler, verify stale-read 503, and always resume it."""
import argparse, fcntl, json, os, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from check_watermarks_modes import ROOT, http, conflicting_processes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    lock=open('/tmp/rag-watermarks-modes.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if conflicting_processes():raise RuntimeError('Another test is running')
    url='http://127.0.0.1:8088'
    def health():
        code,_,h=http(url+'/health');assert code==200;return h
    initial=health()
    assert initial['inflight']==0 and initial['memory_admission']['reserved_mib']==0
    assert initial['memory_admission']['status']=='ready'
    assert initial['device_memory']['stats']['memory_used_percent']<=75
    pid=initial['device_memory']['sampling_process']
    assert pid!=os.getpid() and Path(f'/proc/{pid}').exists()
    print('Isolated sampler preflight passed',flush=True)
    if not args.run:return 0
    out=ROOT/'experiment_outputs/checks/stale_memory'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True);print('OUTPUT='+str(out),flush=True)
    result={'status':'failed','command':sys.argv,'initial_health':initial,
            'fault_injection':'SIGSTOP isolated telemetry process; no NPU workload or memory allocations'}
    body={'query':'How does exercise affect blood pressure?','reranker_batch_size':1}
    identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
    # Independent failsafe resumes telemetry even if this client is killed.
    watchdog_code="import os,signal,time;from pathlib import Path;time.sleep(8);p=Path('/proc/%d/stat');os.kill(%d,signal.SIGCONT) if p.exists() and p.read_text().split()[21]==%r else None"%(pid,pid,identity)
    watchdog=subprocess.Popen([sys.executable,'-c',watchdog_code],start_new_session=True)
    try:
        os.kill(pid,signal.SIGSTOP)
        time.sleep(initial['memory_admission']['config']['max_sample_age_s']+.4)
        result['stale_health']=health()
        code,headers,response=http(url+'/query',body)
        result['rejected_request']={'payload':body,'status':code,'headers':headers,'body':response}
        assert code==503 and headers.get('retry-after') and response['detail']['reason']=='stale_telemetry'
        assert health()['memory_admission']['accepted']==initial['memory_admission']['accepted']
    finally:
        if Path(f'/proc/{pid}/stat').exists() and Path(f'/proc/{pid}/stat').read_text().split()[21]==identity:
            os.kill(pid,signal.SIGCONT)
        (out/'result.json').write_text(json.dumps(result,indent=2))
    end=time.monotonic()+15
    while time.monotonic()<end:
        if health()['memory_admission']['status']=='ready':break
        time.sleep(.05)
    code,headers,response=http(url+'/query',body)
    result['recovery_request']={'payload':body,'status':code,'headers':headers,'body':response}
    assert code==200
    end=time.monotonic()+10
    while time.monotonic()<end:
        h=health()
        if h['inflight']==0 and h['memory_admission']['reserved_mib']==0:break
        time.sleep(.05)
    result['health_after']=h
    assert h['memory_admission']['status']=='ready' and h['inflight']==0 and h['memory_admission']['reserved_mib']==0
    result['status']='passed'
    (out/'result.json').write_text(json.dumps(result,indent=2));print('RESULT=passed')
    return 0

if __name__=='__main__':sys.exit(main())
