"""Cold-cache raw SiLU qualification, followed separately by adapter controls."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import importlib.metadata as md
import json
import logging
import os
from pathlib import Path
import sys
import traceback


def main():
    p=argparse.ArgumentParser(); p.add_argument('--controls',action='store_true'); a=p.parse_args()
    stage='operator-controls' if a.controls else 'operator-qualification'
    out=Path('/results')/stage; out.mkdir()
    os.environ['TRITON_CACHE_DIR']=str(out/'cold-cache')
    import torch
    import torch.nn.functional as F
    import triton
    import flag_gems
    sys.path.insert(0,'/runtime'); import runtime
    runtime.use('ascend'); runtime.set_device(0)
    from flagrt_adapter import Environment, OperatorPolicy, load_records
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.startup import package_identity
    from flagrt_adapter.environment import measure_compiler_build
    revision,identity=package_identity(flag_gems.__file__)
    build=measure_compiler_build(os.environ['FLAGRT_CANN_SHMEM_WHEEL'])
    env=Environment(torch.npu.get_device_name(0),torch.__version__,md.version('torch-npu'),revision,
                    os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,build)
    hits=Counter()
    class Handler(logging.Handler):
        def emit(self,r):
            if r.getMessage() in ('GEMS SILU FORWARD','GEMS SILU BACKWARD'): hits[r.getMessage()]+=1
    logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); handler=Handler(); logger.addHandler(handler)
    ops=['silu','silu_backward']; tables=lambda:{op:torch._C._dispatch_dump_table('aten::'+op) for op in ops}
    report={'status':'FAIL','environment':asdict(env),'provider_identity':identity,'cases':[]}
    try:
        records=load_records('/reviews/operators.json') if a.controls else ()
        for mode in (('off','on','unqualified') if a.controls else ('qual-0','qual-1','qual-2')):
            torch.manual_seed(1234)
            x=torch.randn(1,32,3072,device='npu',requires_grad=True); dy=torch.randn_like(x)
            ref=F.silu(x); refgrad,=torch.autograd.grad(ref,x,dy)
            before=tables(); counts=hits.copy()
            if a.controls:
                policy=OperatorPolicy(env,((1,32,3072),),experimental=mode!='off',
                                      capabilities=records if mode!='unqualified' else ())
                with ElementwiseAdapter({'silu':policy},env,runtime,flag_gems) as scope:
                    got=F.silu(x); grad,=torch.autograd.grad(got,x,dy)
                    runtime.synchronize(0)
                events=list(scope.events)
            else:
                with torch.no_grad(),flag_gems.use_gems(include=ops):
                    got=torch.ops.aten.silu.default(x)
                    grad=torch.ops.aten.silu_backward.default(dy,x)
                    runtime.synchronize(0)
                events=[]
            delta=hits-counts; target=not a.controls or mode=='on'
            assert delta==({'GEMS SILU FORWARD':1,'GEMS SILU BACKWARD':1} if target else {}),delta
            torch.testing.assert_close(got,ref,atol=1e-5 if target else 0,rtol=1e-4 if target else 0)
            torch.testing.assert_close(grad,refgrad,atol=1e-5 if target else 0,rtol=1e-4 if target else 0)
            assert before==tables()
            if a.controls and not target:
                assert events[0]['reason']==('experimental_disabled' if mode=='off' else 'capability_unverified')
            report['cases'].append(dict(mode=mode,shape=list(x.shape),dtype=str(x.dtype),
                forward_max_abs=float((got-ref).abs().max()),backward_max_abs=float((grad-refgrad).abs().max()),
                calls=dict(delta),events=events,restored=True,status='PASS'))
        report['status']='PASS'
    except BaseException:
        report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        logger.removeHandler(handler)
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
