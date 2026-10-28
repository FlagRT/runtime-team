"""Unmodified public DeepFM training regression, not CTR quality evaluation."""
import argparse
from contextlib import nullcontext
import copy
from dataclasses import replace
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import traceback

MODEL_REVISION = '1d39898a172ea80ecbe1268cd50ecac6670b4907'
MODEL_HASHES = {
    'build.py': '4651418ef428ee5e6b6be68e9119da4166ebff78547b2df8877f7bc65cb16964',
    'config.py': '2c5355658a94d5aee632299d5e3612685b09ceb77de27898feecabfac9fc2ded',
    'model.py': '319ca8968ee79f233a70338991cdb0f2cdd91dcb1d61329a14d3018a14434eef',
}


def load_public_model(root):
    root = Path(root).resolve()
    actual = {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in MODEL_HASHES}
    if actual != MODEL_HASHES:
        raise ValueError('Public DeepFM source identity mismatch')
    # Import only after identity check; build.py imports sibling config/model.
    for name in ('config', 'model'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent != root:
            raise RuntimeError('Conflicting model module: '+name)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('public_deepfm_build', root/'build.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_model(), module.DEFAULT_CONFIG, actual


def seed(torch, value, device):
    torch.manual_seed(value)
    if device == 'npu':
        torch.npu.manual_seed_all(value)


def trajectory(torch, initial, batches, scope_factory, device, synchronize,
               reference=None, hits=None, mode='native', failure_root=None):
    model = copy.deepcopy(initial).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    snapshots, rows = [], []
    for step, (x, y) in enumerate(batches):
        seed(torch, 23000+step, device)  # Same dropout mask for all paths.
        optimizer.zero_grad(set_to_none=True)
        before = hits.counts.copy() if hits else None
        scope = scope_factory()
        with scope:
            prediction = model(x)
            loss = torch.nn.functional.binary_cross_entropy(prediction, y)
            loss.backward()
        synchronize()
        grads = {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        assert all(torch.isfinite(v).all() for v in grads.values())
        optimizer.step()
        synchronize()
        tensors = {'prediction':prediction.detach().cpu(), 'loss':loss.detach().cpu()}
        tensors.update({'grad/'+n:v for n,v in grads.items()})
        tensors.update({'state/'+n:v.detach().cpu().clone() for n,v in model.state_dict().items()})
        # Include Adam moments rather than comparing only the visible parameters.
        for i, state in enumerate(optimizer.state.values()):
            for k,v in state.items():
                if isinstance(v,torch.Tensor): tensors[f'optimizer/{i}/{k}'] = v.detach().cpu().clone()
        for v in tensors.values(): assert torch.isfinite(v).all()
        delta = dict(hits.counts-before) if hits else {}
        if mode == 'target':
            assert delta == {'GEMS SIGMOID FORWARD':1,'GEMS SIGMOID BACKWARD':1}, delta
        else: assert not delta, delta
        if mode in ('disabled','unqualified'):
            reason = 'experimental_disabled' if mode=='disabled' else 'capability_unverified'
            assert scope.operator_counts[('sigmoid','native','forward',reason)] == 1
        maximum = 0.
        if reference is not None:
            assert tensors.keys() == reference[step].keys()
            for name, actual in tensors.items():
                expected = reference[step][name]
                difference = float((actual-expected).abs().max())
                maximum = max(maximum,difference)
                tolerance = dict(atol=1e-5,rtol=1e-4) if mode=='target' else dict(atol=0,rtol=0)
                try:
                    torch.testing.assert_close(actual,expected,**tolerance,msg=lambda m: name+': '+m)
                except AssertionError:
                    if failure_root is not None:
                        # Capture the failed comparison, without relaxing it or
                        # retrying a failed target call through the native path.
                        torch.save({'first_failed_tensor':name,'step':step+1,
                            'native':reference[step],'target':tensors,
                            'initial':{n:v.detach().cpu().clone() for n,v in initial.state_dict().items()},
                            'hits':delta},Path(failure_root)/'failed-comparison.pt')
                    raise
        rows.append(dict(step=step+1,loss=float(loss.detach()),hits=delta,gradient_tensors=len(grads),
                         state_tensors=len(model.state_dict()),compared_tensors=len(tensors),max_abs=maximum))
        snapshots.append(tensors)
    return snapshots, rows


def run(args, report):
    import torch
    import _adapter_package
    from flagrt_adapter import Environment, OperatorPolicy, load_records
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.startup import package_identity
    from qualify_unary_npu import Hits
    import flag_gems
    import triton
    sys.path.insert(0,args.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count()==1
    torch.use_deterministic_algorithms(True)
    revision, identity = package_identity(flag_gems.__file__)
    builds=[]
    for name in ('flagtree','triton-ascend'):
        try: builds.append(name+'=='+metadata.version(name))
        except metadata.PackageNotFoundError: pass
    env=Environment(torch.npu.get_device_name(0),torch.__version__,metadata.version('torch-npu'),
        revision,os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,';'.join(builds))
    policy=OperatorPolicy(env,((32,),(16,)),('torch.float32',),True,load_records(args.records),operator='sigmoid')
    seed(torch,20261023,'npu')
    initial, config, hashes = load_public_model(args.model_root)
    count=sum(p.numel() for p in initial.parameters())
    assert count==421452, count
    hits=Hits(); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    ops=('sigmoid','sigmoid_backward')
    dispatch={op:torch._C._dispatch_dump_table('aten::'+op) for op in ops}
    report.update(environment=env.__dict__,gems_identity=identity,model_revision=MODEL_REVISION,
        model_hashes=hashes,parameters=count,config=config.__dict__,batch_sizes=args.batch_sizes,
        data='random categorical fields and binary labels, seed 20261023; not CTR quality',
        train_mode=True,dropout=config.dropout,optimizer='Adam lr=1e-3',trajectories=[],
        tolerance={'target':{'atol':1e-5,'rtol':1e-4},'native_controls':{'atol':0,'rtol':0}})
    for batch_size in args.batch_sizes:
        batches=[(torch.stack([torch.randint(0,d,(batch_size,)) for d in config.field_dims],dim=1).to('npu'),
                  torch.randint(0,2,(batch_size,)).float().to('npu')) for _ in range(3)]
        reference=None
        for mode in ('native','native_repeat','disabled','unqualified','target'):
            report['active_case']={'batch_size':batch_size,'mode':mode}
            p = policy if mode=='target' else replace(policy,experimental=mode!='disabled',capabilities=())
            factory = (lambda: nullcontext()) if mode in ('native','native_repeat') else (
                lambda: ElementwiseAdapter({'sigmoid':p},env,runtime,flag_gems))
            snap, rows = trajectory(torch,initial,batches,factory,'npu',lambda:runtime.synchronize(0),reference,hits,mode,args.out.parent)
            if mode=='native': reference=snap
            assert dispatch=={op:torch._C._dispatch_dump_table('aten::'+op) for op in ops}
            item=dict(batch_size=batch_size,mode=mode,steps=rows,status='pass',dispatcher_restored=True)
            report['trajectories'].append(item)
            print(json.dumps(item),flush=True)
    report['status']='pass'
    report.pop('active_case',None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('runtime-root','model-root','records'): p.add_argument('--'+n,required=True)
    p.add_argument('--batch-size',dest='batch_sizes',type=int,choices=(16,32),action='append')
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    a.batch_sizes=a.batch_sizes or [32,16]
    if a.out.exists(): p.error('Output exists')
    report={'status':'fail','scope':'public DeepFM sigmoid adapter training regression'}
    try: run(a,report)
    except Exception: report['error']=traceback.format_exc(); print(report['error'],flush=True)
    finally:
        with a.out.open('x') as f: json.dump(report,f,indent=2)
    print('overall: '+report['status'].upper()); raise SystemExit(0 if report['status']=='pass' else 1)
