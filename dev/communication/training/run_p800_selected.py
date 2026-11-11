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


def validate_health(path, expected_uuid):
    record=json.loads(path.read_text())
    if (record.get('verdict')!='PASS'
            or record.get('uuid','').removeprefix('GPU-')!=expected_uuid.removeprefix('GPU-')
            or record.get('checks')!={'identity':True,'fill_readback':True,
                                     'matmul':True,'event_completed':True}):
        raise ValueError('Native device probe incomplete; no training permitted')


def snapshot():
    raw=subprocess.check_output(['xpu-smi','-m'],text=True)
    rows=[shlex.split(line) for line in raw.splitlines() if re.match(r'^[0-9a-fA-F]{8}:',line)]
    # xpu-smi --help defines zero-based fields 17=memory used, 19=utilization.
    free={int(r[2]):r for r in rows if len(r)>=20 and int(r[17])==0 and int(r[19])==0}
    return raw,free


def choose_idle_devices(free, mode, excluded=()):
    candidates=[(4,5),(6,7),(2,3)] if mode=='pair' else [(5,),(7,),(4,),(6,),(0,),(2,),(3,)]
    return next((pair for pair in candidates
                 if all(card in free and card not in excluded for card in pair)),None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--mode',choices=['single','pair'],required=True)
    p.add_argument('--wait-seconds',type=int,default=180)
    p.add_argument('--poll-seconds',type=int,default=5)
    p.add_argument('--prototype-revision',required=True)
    p.add_argument('--output-name',required=True)
    p.add_argument('--include-native',action='store_true')
    p.add_argument('--include-observed',action='store_true')
    p.add_argument('--failure-restart',action='store_true')
    p.add_argument('--parallel-minimal',action='store_true')
    p.add_argument('--exclude-card',type=int,action='append',default=[],
                   help='Exclude a physical device after a failed diagnostic; never broadens candidate topology')
    a=p.parse_args()
    if any(card not in range(8) for card in a.exclude_card):
        p.error('Excluded physical card must be in [0,7]')
    if a.parallel_minimal and (a.mode!='pair' or a.include_native or a.include_observed or a.failure_restart):
        p.error('Parallel examples require pair mode without other experiments')
    if (a.failure_restart and (a.include_native or a.include_observed)) or (a.mode!='pair' and (a.include_native or a.include_observed or a.failure_restart)):
        p.error('Choose one pair-only experiment: native matrix or failure/restart')
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
             'probes/p800_identity.py','probes/p800_health.py','env/p800-single/docker-compose.yml']
    if a.mode=='pair':sources.append('env/p800-pair/docker-compose.yml')
    if a.parallel_minimal:sources+=['training/parallel_minimal.py','training/run_parallel_matrix.py']
    record=dict(mode=a.mode,prototype_revision=a.prototype_revision,observations=[],runs=[],verdict='WAITING',
                excluded_physical_cards=sorted(set(a.exclude_card)),
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
        selected=choose_idle_devices(free,a.mode,a.exclude_card)
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
        # Identity/idle counters do not establish that a device can execute.
        # Probe each visible UUID natively before entering any collective. Use
        # the actual container order, not host indices or /dev/xpu suffixes.
        visible=json.loads((output/'identity.json').read_text())['visible_uuid_order']
        record['native_preflight']=[]
        for index,uuid in enumerate(visible):
            target=f'health-local{index}'
            cmd=['docker','exec','-e',f'CUDA_VISIBLE_DEVICES={index}',container,
                 'timeout','-k','5','35',py,'dev/communication/probes/p800_health.py',
                 '--expected-uuid',uuid,'--output',f'/workspace/evidence/{a.output_name}/{target}.json']
            with (output/(target+'.log')).open('x') as log:
                code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=55).returncode
            record['native_preflight'].append(dict(local_index=index,uuid=uuid,exit_code=code))
            save()
            if code!=0:raise ValueError('Native probe failed/timed out before communication; see '+target)
            validate_health(output/(target+'.json'),uuid)
        if a.mode=='pair':
            cmd=[py,'dev/communication/training/run_repeated.py','--backend','kunlun',
                 '--collective','flagcx','--runtime-root','/workspace/dev/device-context/prototype',
                 '--prototype-revision',a.prototype_revision,'--model-root','/workspace/dev/model-defs/deepfm/v1',
                 '--output','/workspace/evidence/'+a.output_name+'/training','--launch-timeout','240']
            if a.include_native:cmd.append('--include-native')
            if a.include_observed:cmd.append('--include-observed')
            if a.failure_restart:
                cmd=[py,'dev/communication/training/failure_restart.py','--backend','kunlun',
                     '--runtime-root','/workspace/dev/device-context/prototype',
                     '--output','/workspace/evidence/'+a.output_name+'/failure']
            if a.parallel_minimal:
                cmd=[py,'dev/communication/training/run_parallel_matrix.py','--backend','kunlun',
                     '--collective','flagcx','--runtime-root','/workspace/dev/device-context/prototype',
                     '--prototype-revision',a.prototype_revision,
                     '--output','/workspace/evidence/'+a.output_name+'/parallel']
            with (output/'training-driver.log').open('x') as log:
                code=subprocess.run(['docker','exec',container,*cmd],stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
            record['runs'].append(dict(argv=cmd,exit_code=code));save()
            if code==0:
                summary=(json.loads((output/'parallel/summary.json').read_text()) if a.parallel_minimal else
                         json.loads((output/'failure/execution.json').read_text()) if a.failure_restart
                         else summarize(output/'training',a.include_native,a.include_observed))
                if summary['verdict']!='PASS':raise ValueError('Incomplete acceptance')
                # Container-created training/ is root-owned; publish to the
                # host-owned execution directory instead of chmod/chown trees.
                (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
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
