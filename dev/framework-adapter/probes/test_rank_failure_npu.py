"""Bounded torchrun supervisor: expected worker failures are not training success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def assess(rows, scenario, returncode, elapsed_after_injection, guarded=False):
    assert set(rows) == {0,1}
    for rank, events in rows.items():
        starts=[e for e in events if e['event']=='start']
        assert len(starts)==1 and starts[0]['restart_count']=='0', 'worker restarted'
        assert any(e['event']=='group_ready' and e['backend']=='hccl' for e in events), f'rank {rank}: HCCL group not ready; no linkage verdict'
        assert not any(e['event']=='adapter' and e['backend']=='native' for e in events), 'native retry/routing'
    if scenario=='control':
        assert returncode==0
        for events in rows.values():
            assert sum(e['event']=='optimizer_done' for e in events)==3
            assert any(e['event']=='completed' for e in events)
            assert any(e['event']=='finally' and e['dispatcher_restored'] for e in events)
            assert any(e['event']=='adapter' and e['stage']=='backward' and e['reason']=='completed' for e in events)
    else:
        assert returncode not in (0,124,137), 'must fail by worker error, not outer watchdog'
        injected=[e for e in rows[0] if e['event']=='injected']
        assert len(injected)==1 and injected[0]['stage']==scenario
        assert any((e['event']=='step_gate_enter' and e.get('phase')=='backward' if guarded
                    else e['event']=='gradient_collective_enter')
                   and e['monotonic']<=injected[0]['monotonic'] for e in rows[1])
        assert 0 <= elapsed_after_injection < 90, 'peer exit exceeded bound'
        assert not any(e['event'] in ('optimizer_enter','optimizer_done','completed') for ev in rows.values() for e in ev)
        assert any(e['event']=='adapter' and e['reason']=='error_no_retry' for e in rows[0])
        assert any(e['event']=='finally' and e['dispatcher_restored'] for e in rows[0])
    return {'status':'pass','scenario':scenario,'worker_returncode':returncode,
            'elapsed_after_injection_s':elapsed_after_injection,
            'optimizer_steps':{str(r):sum(e['event']=='optimizer_done' for e in ev) for r,ev in rows.items()},
            'scope':'torchrun supervision of injected Python failure; not driver recovery or rollback'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=('control','forward','backward'),required=True)
    p.add_argument('--config',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--guarded',action='store_true')
    a=p.parse_args(); result={'status':'fail'}
    if a.out.exists(): p.error('New output required')
    output=Path(os.environ['DC_OUT_DIR'])
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
         '--max-restarts=0','--monitor-interval=0.2',str(Path(__file__).with_name('rank_failure_worker.py')),
         '--config',a.config,'--scenario',a.scenario]
    if a.guarded: cmd.append('--guarded')
    result['command']=cmd
    try:
        model=Path(os.environ['DC_MODEL'])/'model.safetensors'
        h=hashlib.sha256()
        with model.open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024),b''): h.update(data)
        result['model_sha256']=h.hexdigest()
        assert h.hexdigest()=='0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
        with (a.out.parent/'torchrun.log').open('x') as log:
            process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=process.wait(timeout=480)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                raise RuntimeError('Outer watchdog fired; not a successful failure linkage')
        end=time.monotonic()
        result['worker_returncode']=code
        rows={r:[json.loads(line) for line in (output/f'events-rank{r}.jsonl').read_text().splitlines()] for r in (0,1)}
        result['group_ready']={str(r):any(e['event']=='group_ready' for e in ev) for r,ev in rows.items()}
        injected=[e['monotonic'] for e in rows[0] if e['event']=='injected']
        result['elapsed_after_injection_s']=end-injected[0] if injected else None
        result['optimizer_calls']={str(r):sum(e['event']=='optimizer_enter' for e in ev) for r,ev in rows.items()}
        result.update(assess(rows,a.scenario,code,result['elapsed_after_injection_s'],a.guarded))
        if a.guarded:
            guards={str(r):json.loads((output/f'step-control/rank{r}/report.json').read_text()) for r in (0,1)}
            assert all(g['hooks_restored'] for g in guards.values())
            assert all(g['steps']==(3 if a.scenario=='control' else 0) for g in guards.values())
            assert all(g['status']==('PASS' if a.scenario=='control' else 'FAIL') for g in guards.values())
            if a.scenario!='control':
                assert not any(e['event']=='gradient_collective_enter' for ev in rows.values() for e in ev)
            result['step_guard']=guards
        # torchrun may have killed the peer without Python finally; check OS exit,
        # never infer peer dispatcher restoration from lack of a final report.
        pids=[e['pid'] for ev in rows.values() for e in ev if e['event']=='start']
        assert not any(Path(f'/proc/{pid}').exists() for pid in pids), 'worker still present'
        result['worker_pids']=pids; result['workers_gone']=True
        result['peer_finally_observed']=any(e['event']=='finally' for e in rows[1])
        if a.scenario=='control':
            for rank in (0,1):
                assert json.loads((output/f'adapter_rank{rank}.json').read_text())['status']=='PASS'
                assert json.loads((output/f'train_leg_result_rank{rank}.json').read_text())['verdict']=='TRAIN_LEG_PASS'
    except BaseException:
        result['status']='fail'; result['error']=traceback.format_exc(); print(result['error'])
    a.out.write_text(json.dumps(result,indent=2))
    print('overall: '+result['status'].upper())
    return 0 if result['status']=='pass' else 1


if __name__=='__main__': raise SystemExit(main())
