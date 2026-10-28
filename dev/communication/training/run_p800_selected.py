"""Host-side bounded idle-card selection and strict-environment DeepFM tests.

No resets, no foreign container operations, no installs. P800's physical index
is NOT /dev/xpu suffix; query UUID/minor before granting device access.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from summarize_repeated import summarize


def validate_single(root, mode):
    """A clean launcher alone does not establish model or teardown success."""
    r=json.loads((root/mode/'rank0.json').read_text())
    if not (r['rank']==0 and r['world_size']==1 and r['backend']=='flagcx'
            and r['evidence_scope']=='P800 integration' and r['device']=='cuda:0'
            and r['runtime_info']['name']=='kunlun' and r['comm_mode']==mode
            and r['verdict']=='PASS' and r['group_state']=='CLOSED'
            and r['teardown']=='closed' and r['measured_steps']==50
            and r['warmup_steps']==5 and r['global_batch']==64 and r['local_batch']==64):
        raise ValueError('Unexpected single-card model metadata')
    oracle=r['oracle']; training=r['training']
    if not (len(oracle['gradient_checks'])==17 and all(c['pass'] for c in oracle['gradient_checks'])
            and oracle['update_pass'] and oracle['rank_parameter_max_abs']==0
            and training['rank_parameter_max_abs']==0 and training['parameters_finite']
            and len(training['losses'])==50 and len(training['step_seconds'])==50):
        raise ValueError('Single-card oracle/training checks failed')
    if not (all(math.isfinite(v) for v in training['losses'])
            and all(math.isfinite(v) and v>0 for v in training['step_seconds'])
            and training['gradient_communication']['gradient_allreduce_calls']==(850 if mode=='manual' else 50)
            and training['gradient_communication']['gradient_payload_bytes']==training['parameter_count']*4*50):
        raise ValueError('Single-card samples or gradient accounting failed')


def snapshot():
    raw=subprocess.check_output(['xpu-smi','-m'],text=True)
    rows=[shlex.split(line) for line in raw.splitlines() if re.match(r'^[0-9a-fA-F]{8}:',line)]
    # xpu-smi --help defines zero-based fields 17=memory used, 19=utilization.
    free={int(r[2]):r for r in rows if len(r)>=20 and int(r[17])==0 and int(r[19])==0}
    return raw,free


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mode',choices=['single','pair'],required=True)
    p.add_argument('--wait-seconds',type=int,default=180)
    p.add_argument('--poll-seconds',type=int,default=5)
    p.add_argument('--prototype-revision',required=True)
    p.add_argument('--output-name',required=True)
    a=p.parse_args()
    if a.wait_seconds<1 or not 1<=a.poll_seconds<=60 or not re.fullmatch(r'[a-zA-Z0-9_-]+',a.output_name):
        p.error('positive wait, polling in [1,60] and simple output name required')
    root=a.workspace.resolve()
    lease=(root/'.communication-p800-test.lock').open('a')
    # Prevent two of our own monitors from racing for the same devices.
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output=root/'evidence'/a.output_name
    output.mkdir(parents=True,exist_ok=False)
    sources=['training/deepfm_train.py','training/distributed.py','training/run_repeated.py',
             'training/summarize_repeated.py','training/run_p800_selected.py',
             'probes/p800_identity.py','env/p800-single/docker-compose.yml']
    if a.mode=='pair':sources.append('env/p800-pair/docker-compose.yml')
    record=dict(mode=a.mode,prototype_revision=a.prototype_revision,observations=[],runs=[],verdict='WAITING',
                source_sha256={name:hashlib.sha256((root/'dev/communication'/name).read_bytes()).hexdigest()
                               for name in sources})
    def save():
        temporary=output/'selection-execution.tmp'
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(output/'selection-execution.json')
    deadline=time.monotonic()+a.wait_seconds
    selected=None
    while time.monotonic()<deadline:
        raw,free=snapshot()
        record['observations'].append(dict(time=time.time(),raw=raw,free_ids=sorted(free)))
        candidates=[(4,5),(6,7),(2,3)] if a.mode=='pair' else [(5,),(7,),(4,),(6,),(0,),(2,),(3,)]
        selected=next((pair for pair in candidates if all(card in free for card in pair)),None)
        save()
        if selected: break
        print(f'waiting for {a.mode} idle healthy devices; free={sorted(free)}',flush=True)
        time.sleep(a.poll_seconds)
    if not selected:
        record.update(verdict='NOT_RUN',reason='No eligible idle device set in observation window');save()
        print('overall: NOT_RUN (resources unavailable)',flush=True)
        return 3
    identities=[]
    for card in selected:
        query=subprocess.check_output(['xpu-smi','-i',str(card),'-q'],text=True)
        identities.append(dict(physical_card=card,
            node='/dev/xpu'+re.search(r'Minor Number\s*:\s*(\d+)',query).group(1),
            uuid=re.search(r'XPU UUID\s*:\s*(\S+)',query).group(1)))
    record.update(selected=identities,verdict='RUNNING');save()
    env={**os.environ,'FLAGRT_WORKSPACE_ROOT':str(root),'WORKSPACE_ROOT':str(root),
         'COMM_XPU_NODE_A':identities[0]['node']}
    if len(identities)==2: env['COMM_XPU_NODE_B']=identities[1]['node']
    envdir=root/'dev/communication/env'/('p800-'+a.mode)
    container='flagrt-communication-p800-'+a.mode+'-1028'
    existing=subprocess.run(['docker','inspect',container],capture_output=True,text=True)
    if existing.returncode==0:
        state=json.loads(existing.stdout)[0]
        project=(state.get('Config',{}).get('Labels') or {}).get('com.docker.compose.project')
        if project!='flagrt-communication-p800-'+a.mode or state['State']['Running']:
            record.update(verdict='ERROR',reason='Container name occupied or already running; left untouched')
            save();return 2
    py='/root/miniconda/envs/python310_torch29_cuda/bin/python3'
    try:
        with (output/'startup.log').open('x') as log:
            subprocess.run([sys.executable,str(root/'dev/lib/baseline.py'),'export',str(envdir)],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run(['bash',str(root/'dev/lib/up.sh'),str(envdir),'--strict'],
                           env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        cmd=[py,'dev/communication/probes/p800_identity.py',
             '--output','/workspace/evidence/'+a.output_name+'/identity.json']
        for identity in identities: cmd+=['--expected-uuid',identity['uuid']]
        with (output/'identity.log').open('x') as log:
            subprocess.run(['docker','exec',container,*cmd],stdout=log,
                           stderr=subprocess.STDOUT,timeout=30,check=True)
        if a.mode=='pair':
            cmd=[py,'dev/communication/training/run_repeated.py','--backend','kunlun',
                 '--collective','flagcx','--runtime-root','/workspace/dev/device-context/prototype',
                 '--prototype-revision',a.prototype_revision,'--model-root','/workspace/dev/model-defs/deepfm/v1',
                 '--output','/workspace/evidence/'+a.output_name+'/training','--launch-timeout','240']
            with (output/'training-driver.log').open('x') as log:
                code=subprocess.run(['docker','exec',container,*cmd],stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            record['runs'].append(dict(argv=cmd,exit_code=code));save()
            if code==0:
                summary=summarize(output/'training')
                (output/'training/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        else:
            for mode in ('manual','fused'):
                cmd=[py,'-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1',
                     '--rdzv_backend=static','--master_addr=127.0.0.1','--master_port=29541',
                     'dev/communication/training/deepfm_train.py','--backend','kunlun','--collective','flagcx',
                     '--comm-mode',mode,'--runtime-root','/workspace/dev/device-context/prototype',
                     '--prototype-revision',a.prototype_revision,'--model-root','/workspace/dev/model-defs/deepfm/v1',
                     '--output','/workspace/evidence/'+a.output_name+'/'+mode,'--steps','50','--warmup','5']
                with (output/(mode+'.log')).open('x') as log:
                    code=subprocess.run(['docker','exec',container,*cmd],stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
                record['runs'].append(dict(argv=cmd,exit_code=code));save()
                if code: break
                validate_single(output,mode)
        record['verdict']='PASS' if all(r['exit_code']==0 for r in record['runs']) else 'FAIL'
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
        record.update(verdict='ERROR',reason=repr(error))
    finally:
        # Only our explicitly named test container. Logs and baseline stay on disk.
        subprocess.run(['docker','stop','-t','10',container],capture_output=True)
        save()
    print('overall: '+record['verdict'],flush=True)
    return {'PASS':0,'FAIL':1,'ERROR':2}[record['verdict']]


if __name__=='__main__': raise SystemExit(main())
