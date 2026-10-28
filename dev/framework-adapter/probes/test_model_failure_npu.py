"""Isolated model failure controls with real NPU tensors and scoped injection.

Injection raises in provider context entry/exit, never corrupts a driver. Each
failed worker exits; dispatcher restoration does not certify device recovery.
"""
import argparse
from contextlib import contextmanager
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def worker(a):
    import torch
    import flag_gems
    import triton
    import _adapter_package
    from flagrt_adapter import Environment, OperatorPolicy, load_records
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.startup import package_identity
    sys.path.insert(0,a.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count()==1
    torch.manual_seed(23)
    revision,_=package_identity(flag_gems.__file__)
    env=Environment(torch.npu.get_device_name(0),torch.__version__,metadata.version('torch-npu'),
        revision,os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,'triton-ascend=='+metadata.version('triton-ascend'))
    policy=OperatorPolicy(env,((32,64),),('torch.float32',),True,load_records(a.records),operator='tanh')
    calls=[]
    class Injected(RuntimeError): pass
    class Provider:
        @contextmanager
        def use_gems(self,include):
            calls.append(list(include))
            with flag_gems.use_gems(include=include):
                if (a.failure=='forward' and len(calls)==1) or (a.failure=='backward' and len(calls)==2):
                    raise Injected(a.failure)
                yield
            if a.failure=='restore': raise Injected('restore-after-real-unregister')
    model=torch.nn.Sequential(torch.nn.Linear(16,64),torch.nn.Tanh(),torch.nn.Linear(64,2)).to('npu')
    before=[p.detach().cpu().clone() for p in model.parameters()]
    tables={op:torch._C._dispatch_dump_table('aten::'+op) for op in ('tanh','tanh_backward')}
    scope=ElementwiseAdapter({'tanh':policy},env,runtime,Provider())
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    stepped=False; raised=False
    try:
        with scope:
            loss=model(torch.randn(32,16,device='npu')).square().mean(); loss.backward()
            optimizer.step(); stepped=True
    except Injected: raised=True
    assert raised and scope.failed and not stepped
    for p,q in zip(model.parameters(),before): torch.testing.assert_close(p.detach().cpu(),q,atol=0,rtol=0)
    assert not any(key[1]=='native' for key in scope.operator_counts)
    assert len(calls)==(2 if a.failure=='backward' else 1),calls
    assert tables=={op:torch._C._dispatch_dump_table('aten::'+op) for op in tables}
    return dict(status='pass',failure=a.failure,environment=env.__dict__,injected=True,
        optimizer_steps=0,parameters_unchanged=True,native_retries=0,scope_failed=scope.failed,
        registrations=len(calls),events=list(scope.events),dispatcher_restored=True,
        limitation='synthetic context failure, not a driver fault; this worker must exit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runtime-root',required=True)
    p.add_argument('--records',required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--failure',choices=('forward','backward','restore')); a=p.parse_args()
    if a.out.exists(): p.error('Output exists')
    report={'status':'fail','scope':'NPU model fault-injection controls, not native driver failure'}
    try:
        if a.failure: report.update(worker(a))
        else:
            report['workers']=[]
            for failure in ('forward','backward','restore'):
                out=a.out.with_name('failure-'+failure+'.json')
                cmd=[sys.executable,__file__,'--runtime-root',a.runtime_root,'--records',a.records,'--out',str(out),'--failure',failure]
                with out.with_suffix('.log').open('x') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                if result.returncode: raise RuntimeError(f'{failure}: exit {result.returncode}, inspect worker log')
                item=json.loads(out.read_text()); assert item['status']=='pass'; report['workers'].append(item)
            report['status']='pass'
    except Exception: report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        with a.out.open('x') as f: json.dump(report,f,indent=2)
    print('overall: '+report['status'].upper()); raise SystemExit(0 if report['status']=='pass' else 1)
