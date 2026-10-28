"""Two synthetic model regressions on NPU. No business-model acceptance claim."""
import argparse
from contextlib import nullcontext
import copy
from dataclasses import replace
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback
import _adapter_package
from flagrt_adapter import Environment, OperatorPolicy, load_records
from flagrt_adapter.bridge import ElementwiseAdapter
from flagrt_adapter.startup import package_identity
from qualify_unary_npu import Hits


def run(args,report):
    import torch
    import flag_gems
    import triton
    sys.path.insert(0,args.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count()==1
    torch.use_deterministic_algorithms(True)
    revision,identity=package_identity(flag_gems.__file__)
    builds=[]
    for name in ('flagtree','triton-ascend'):
        try: builds.append(name+'=='+metadata.version(name))
        except metadata.PackageNotFoundError: pass
    env=Environment(torch.npu.get_device_name(0),torch.__version__,metadata.version('torch-npu'),
        revision,os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,';'.join(builds))
    records=load_records(args.records)
    base={op:OperatorPolicy(env,((32,64),),('torch.float32',),True,records,operator=op)
          for op in ('silu','sigmoid','tanh')}
    class Gated(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a=torch.nn.Linear(16,64); self.b=torch.nn.Linear(16,64); self.c=torch.nn.Linear(64,2)
        def forward(self,x): return self.c(torch.sigmoid(self.a(x))*torch.tanh(self.b(x)))
    class Mixed(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a=torch.nn.Linear(16,64); self.b=torch.nn.Linear(64,64); self.c=torch.nn.Linear(64,2)
        def forward(self,x): return self.c(torch.tanh(self.b(torch.nn.functional.silu(self.a(x)))))
    torch.manual_seed(20261023)
    models=(('sigmoid_tanh_gate',Gated(),('sigmoid','tanh')),('silu_tanh_mlp',Mixed(),('silu','tanh')))
    batches=[(torch.randn(32,16).to('npu'),torch.randn(32,2).to('npu')) for _ in range(3)]
    hits=Hits(); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    ops=tuple(op+s for op in base for s in ('','_backward'))
    before={op:torch._C._dispatch_dump_table('aten::'+op) for op in ops}
    report.update(environment=env.__dict__,identity=identity,models=[],data='seed 20261023 synthetic regression, 3 batches 32x16',tolerance={'atol':1e-5,'rtol':1e-4})
    for name,initial,used in models:
        item=dict(model=name,parameters=sum(p.numel() for p in initial.parameters()),trajectories=[])
        reference=[]
        for mode in ('native','target','disabled','unqualified'):
            model=copy.deepcopy(initial).to('npu'); optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            policies={op:p if mode=='target' else replace(p,experimental=mode!='disabled',capabilities=()) for op,p in base.items() if op in used}
            rows=[]
            for step,(x,y) in enumerate(batches):
                optimizer.zero_grad(set_to_none=True)
                scope=ElementwiseAdapter(policies,env,runtime,flag_gems) if mode!='native' else nullcontext()
                counts=hits.counts.copy()
                with scope:
                    pred=model(x); loss=(pred-y).square().mean(); loss.backward()
                runtime.synchronize(0)
                grads=[p.grad.detach().cpu().clone() for p in model.parameters()]
                optimizer.step()
                weights=[p.detach().cpu().clone() for p in model.parameters()]
                values=[pred.detach().cpu(),loss.detach().cpu()]+grads+weights
                if mode=='native': reference.append(values)
                atol,rtol=(1e-5,1e-4) if mode=='target' else (0,0)
                for a,b in zip(values,reference[step]): torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
                delta=hits.counts-counts
                if mode=='target':
                    assert sum(delta.values())==2*len(used),delta
                    for op in used:
                        for stage in ('FORWARD','BACKWARD'): assert delta[f'GEMS {op.upper()} {stage}']==1,delta
                else: assert not delta,delta
                if mode in ('disabled','unqualified'):
                    reason='experimental_disabled' if mode=='disabled' else 'capability_unverified'
                    for op in used: assert scope.operator_counts[(op,'native','forward',reason)]==1
                assert before=={op:torch._C._dispatch_dump_table('aten::'+op) for op in ops}
                rows.append(dict(step=step+1,loss=float(loss),hits=dict(delta),max_abs=max(float((a-b).abs().max()) for a,b in zip(values,reference[step]))))
            item['trajectories'].append(dict(mode=mode,steps=rows,status='pass'))
        report['models'].append(item)
        print(json.dumps(item),flush=True)
    report['status']='pass'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runtime-root',required=True)
    p.add_argument('--records',required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): p.error('Output exists')
    report={'status':'fail','scope':'NPU synthetic model mechanism regression'}
    try: run(a,report)
    except Exception: report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        with a.out.open('x') as f: json.dump(report,f,indent=2)
    print('overall: '+report['status'].upper()); raise SystemExit(0 if report['status']=='pass' else 1)
