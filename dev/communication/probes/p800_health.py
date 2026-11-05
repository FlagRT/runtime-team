"""Small native-vendor health probe used only for selecting a test device.

Run under an external process timeout. This is not device conformance or a
hardware root-cause test; it does not reset devices or enable FlagGems/KL3.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-uuid',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result=dict(torch=torch.__version__,checks={},verdict='ERROR')
    def save(): a.output.write_text(json.dumps(result,indent=2)+'\n')
    def stage(name):
        result['stage']=name;save();print(name,flush=True)
    try:
        if os.environ.get('XPU_EVENT_KL3_ENABLE') is not None:
            raise ValueError('K-3: XPU_EVENT_KL3_ENABLE must be unset')
        torch.set_num_threads(1)
        stage('device_identity')
        if torch.cuda.device_count()!=1: raise ValueError('Exactly one visible card required')
        prop=torch.cuda.get_device_properties(0)
        result['uuid']=str(prop.uuid)
        if result['uuid'].removeprefix('GPU-')!=a.expected_uuid.removeprefix('GPU-'):
            raise ValueError('Physical card identity does not match expected UUID')
        result['checks']['identity']=True
        start=time.monotonic()
        stage('fill_and_synchronize')
        x=torch.ones(4,device='cuda:0')
        torch.cuda.synchronize()
        result['checks']['fill_readback']=x.cpu().tolist()==[1.,1.,1.,1.]
        stage('matmul')
        matrix=torch.ones(32,32,device='cuda:0')
        result['checks']['matmul']=bool((matrix@matrix==32).all().item())
        stage('event_query')
        event=torch.cuda.Event()
        event.record()
        deadline=time.monotonic()+5
        while not event.query() and time.monotonic()<deadline: time.sleep(.001)
        result['checks']['event_completed']=event.query()
        result['memory_allocated_bytes']=torch.cuda.memory_allocated(0)
        result['seconds']=time.monotonic()-start
        result['verdict']='PASS' if all(result['checks'].values()) else 'FAIL'
    except BaseException as e:
        result['error']=repr(e)
        raise
    finally: save()
    print('overall: '+result['verdict'],flush=True)
    return 0 if result['verdict']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
