"""Bounded worker-loss regression, not device recovery or checkpoint resume.

Launch: python failure_restart.py --backend kunlun --runtime-root ... --output NEW
One healthy group, one injected rank loss, then an independently started group.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def worker(args):
    import torch
    import torch.distributed as dist
    from deepfm_train import bind_runtime_device
    from distributed import TrainingGroup
    rank=int(os.environ['RANK'])
    if args.backend=='cpu':
        device=torch.device('cpu'); sync=lambda:None; backend='gloo'
    else:
        sys.path.insert(0,str(args.runtime_root.resolve()))
        import runtime
        device=bind_runtime_device(runtime,args.backend,rank)
        sync=lambda:runtime.synchronize(rank,timeout_ms=10000)
        backend='flagcx'
    record=dict(rank=rank,pid=os.getpid(),scenario=args.scenario,backend=backend,
                device=str(device),baseline=False,closed=False,verdict='ERROR')
    target=args.output/f'rank{rank}.json'
    def save():target.write_text(json.dumps(record,indent=2)+'\n')
    save()
    group=TrainingGroup(backend,rank,2,sync,10)
    try:
        value=torch.ones(4,device=device)
        group.collective(dist.all_reduce,value)
        record['baseline']=value.cpu().tolist()==[2.,2.,2.,2.];save()
        if not record['baseline']:raise AssertionError('Bad baseline AllReduce')
        if args.scenario=='loss' and rank==0:
            record['injected']=True;save()
            raise RuntimeError('Injected rank loss before next collective')
        value.fill_(1)
        group.collective(dist.all_reduce,value)
        if value.cpu().tolist()!=[2.,2.,2.,2.]:raise AssertionError('Bad second AllReduce')
        record['verdict']='PASS'
    except BaseException as error:
        record.update(verdict='FAIL',error=repr(error));save();raise
    finally:
        group.close();record['closed']=True;save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['cpu','ascend','kunlun'],required=True)
    p.add_argument('--runtime-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',choices=['healthy','loss'],default='healthy')
    p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:return worker(args)
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(scope=args.backend,runs=[],verdict='ERROR',
                  boundary='Worker-process loss only; no driver reset, checkpoint resume or atomic update guarantee')
    try:
        for name,scenario in [('before','healthy'),('injected','loss'),('restart','healthy')]:
            output=args.output/name;output.mkdir()
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            cmd=[sys.executable,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
                 '--rdzv_backend=static','--master_addr=127.0.0.1',f'--master_port={port}',
                 __file__,'--worker','--backend',args.backend,'--scenario',scenario,'--output',str(output)]
            if args.runtime_root:cmd+=['--runtime-root',str(args.runtime_root)]
            start=time.monotonic();timed_out=False
            with (args.output/(name+'.log')).open('x') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                    try:code=process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
            records=[json.loads((output/f'rank{rank}.json').read_text()) for rank in range(2)]
            alive=[]
            for r in records:
                try:os.kill(r['pid'],0);alive.append(r['pid'])
                except ProcessLookupError:pass
            passed=(not alive and not timed_out and all(r['baseline'] for r in records)
                    and ((code!=0 and records[0].get('injected')) if scenario=='loss' else
                         (code==0 and all(r['verdict']=='PASS' and r['closed'] for r in records))))
            manifest['runs'].append(dict(name=name,argv=cmd,exit_code=code,timed_out=timed_out,
                                          seconds=time.monotonic()-start,remaining_worker_pids=alive,pass_check=bool(passed)))
            if not passed:raise AssertionError('Scenario acceptance failed: '+name)
        manifest['verdict']='PASS'
    except BaseException as error:manifest['error']=repr(error);raise
    finally:(args.output/'execution.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
