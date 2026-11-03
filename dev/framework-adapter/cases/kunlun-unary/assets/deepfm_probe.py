"""Unmodified public DeepFM on P800: bounded training and sigmoid routing.

Random initial weights/categorical data, not pretrained CTR quality validation.
Native controls must be exact; target tolerance is unchanged from the 910C probe.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

from model_probe import compare_tensor, mismatch_examples, validate_reference, MODEL_TOLERANCE, file_hash
from probe import Hits, run as inspect_environment
from model_admission import prepare_model, expected_native_reason, record_expected_block

MODEL_HASHES = {
    'build.py': '4651418ef428ee5e6b6be68e9119da4166ebff78547b2df8877f7bc65cb16964',
    'config.py': '2c5355658a94d5aee632299d5e3612685b09ceb77de27898feecabfac9fc2ded',
    'model.py': '319ca8968ee79f233a70338991cdb0f2cdd91dcb1d61329a14d3018a14434eef',
}


def load_model(root):
    root=Path(root).resolve()
    if {n:file_hash(root/n) for n in MODEL_HASHES}!=MODEL_HASHES:
        raise ValueError('Public DeepFM source identity mismatch')
    for name in ('model','config'):
        if name in sys.modules and Path(sys.modules[name].__file__).resolve().parent!=root:
            raise ValueError('Conflicting model module: '+name)
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('public_deepfm_build',root/'build.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_model(), module.DEFAULT_CONFIG


def make_batch(config, batch_size, step):
    import torch
    rng=torch.Generator().manual_seed(29000+step)
    x=torch.stack([torch.randint(0,d,(batch_size,),generator=rng) for d in config.field_dims],dim=1)
    y=torch.randint(0,2,(batch_size,),generator=rng).float()
    return x,y


def state_hash(model):
    return hashlib.sha256(b''.join(n.encode()+v.detach().cpu().contiguous().numpy().tobytes()
        for n,v in model.state_dict().items())).hexdigest()


def run(a,r):
    import torch
    inspect_environment(SimpleNamespace(runtime_root=a.runtime_root,stage='inspect'),r)
    r['status']='running'
    import runtime
    from flagrt_adapter.bridge import ElementwiseAdapter
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(20261023); torch.cuda.manual_seed_all(20261023)
    model,config=load_model(a.model)
    count=sum(p.numel() for p in model.parameters())
    if count!=421452: raise ValueError('Unexpected public DeepFM parameter count')
    initial_hash=state_hash(model)
    r.update(mode=a.mode,model_loaded=True,model_files=MODEL_HASHES,initial_state_sha256=initial_hash,
        parameter_count=count,public_config=config.__dict__,
        workload=dict(model='public DeepFM v1',dtype='torch.float32',batch=a.batch_size,steps=a.steps,
            initialization='random',initial_seed=20261023,data_seed=29000,dropout_seed=23000,
            data='synthetic categorical fields and binary labels',optimizer='Adam',lr=1e-3,
            eps=1e-8,weight_decay=0,foreach=False,deterministic=True,dropout=config.dropout),
        runtime_sources={'runtime/backends/kunlun/backend.py':file_hash(
            Path(a.runtime_root)/'runtime/backends/kunlun/backend.py')},
        tolerance=MODEL_TOLERANCE if a.mode=='on' else {'atol':0,'rtol':0},
        steps=[],comparisons=[],optimizer_steps=0)
    save=lambda:a.out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    actual,policies=prepare_model(a,r,'sigmoid',[a.batch_size])
    r['model_loaded']=True
    reference=None
    if a.reference:
        reference=json.loads((a.reference/'result.json').read_text())
        validate_reference(reference,r)
        if reference.get('initial_state_sha256')!=initial_hash:
            raise ValueError('Initial model states differ')
    if a.mode=='native': (a.out.parent/'tensors').mkdir()
    def check(name,value):
        value=value.detach().cpu().contiguous()
        if not bool(torch.isfinite(value).all()): raise ValueError('Non-finite '+name)
        filename=hashlib.sha256(name.encode()).hexdigest()+'.pt'
        item=dict(name=name,file=filename,shape=list(value.shape),dtype=str(value.dtype))
        if a.mode=='native': torch.save(value,a.out.parent/'tensors'/filename)
        else:
            expected=torch.load(a.reference/'tensors'/filename,map_location='cpu',weights_only=True)
            item.update(compare_tensor(value,expected,r['tolerance']))
        r['comparisons'].append(item)
        if item.get('mismatches',0):
            item['mismatch_examples']=mismatch_examples(value,expected,r['tolerance'])
            torch.save(dict(name=name,actual=value,reference=expected,tolerance=r['tolerance']),
                       a.out.parent/'failed-comparison.pt')
            save(); raise AssertionError(f'{name}: {item["mismatches"]} mismatches; max_abs={item["max_abs"]}')
    model=model.to('cuda:0').train()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,eps=1e-8,weight_decay=0,foreach=False)
    r['optimizer_created']=True
    gems=None
    if a.mode=='on': import flag_gems as gems
    hits=Hits('sigmoid'); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    tables=lambda:{n:torch._C._dispatch_dump_table('aten::'+n) for n in ('sigmoid','sigmoid_backward')}
    before=tables()
    sampled=model.linear.bias.detach().cpu().clone()
    try:
        for step in range(a.steps):
            r['active_step']=step+1
            x,y=make_batch(config,a.batch_size,step); x=x.to('cuda:0'); y=y.to('cuda:0')
            # Keep the original public dropout and BatchNorm; seed each path identically.
            torch.manual_seed(23000+step); torch.cuda.manual_seed_all(23000+step)
            optimizer.zero_grad(set_to_none=True)
            adapter=None if a.mode in ('native','native_repeat') else ElementwiseAdapter(policies,actual,runtime,gems)
            previous=hits.counts.copy()
            row=dict(step=step+1,status='running'); r['steps'].append(row); save()
            with adapter if adapter is not None else nullcontext():
                prediction=model(x)
                loss=torch.nn.functional.binary_cross_entropy(prediction,y)
                if not bool(torch.isfinite(loss)): raise ValueError('Non-finite loss')
                loss.backward(); runtime.synchronize(0)
            delta=dict(hits.counts-previous)
            row.update(loss=float(loss.detach()),provider_calls=delta,dispatcher_restored=tables()==before,
                routes=[] if adapter is None else [list(k)+[v] for k,v in adapter.operator_counts.items()])
            save()
            if not row['dispatcher_restored']: raise RuntimeError('Dispatcher not restored')
            if a.mode=='on':
                if delta!={'GEMS SIGMOID FORWARD':1,'GEMS SIGMOID BACKWARD':1}:
                    raise AssertionError('Expected one target forward/backward per step: '+str(delta))
                if any(k[1]!='flaggems' for k in adapter.operator_counts):
                    raise AssertionError('Strict target unexpectedly used native')
            else:
                if delta: raise AssertionError('Native path called FlagGems')
                if adapter is not None:
                    reason=expected_native_reason(a.mode)
                    if adapter.operator_counts[('sigmoid','native','forward',reason)]!=1:
                        raise AssertionError('Wrong native route/reason')
            prefix=f'step{step+1}/'
            check(prefix+'prediction',prediction); check(prefix+'loss',loss)
            for name,p in model.named_parameters():
                if p.grad is None: raise AssertionError('Missing gradient: '+name)
                check(prefix+'gradient/'+name,p.grad)
            # Failed forward/gradient check prevents this update. A later parameter
            # mismatch is recorded AFTER one update; it is not silently rolled back.
            optimizer.step(); runtime.synchronize(0); r['optimizer_steps']+=1
            for name,value in model.state_dict().items(): check(prefix+'state/'+name,value)
            for i,state in enumerate(optimizer.state.values()):
                for name,value in state.items():
                    if isinstance(value,torch.Tensor): check(prefix+f'optimizer/{i}/{name}',value)
            row.update(status='pass',gradient_tensors=len(list(model.parameters())),
                state_tensors=len(model.state_dict()),optimizer_steps=r['optimizer_steps'])
            save(); print(json.dumps(row),flush=True)
        if reference and [v['name'] for v in r['comparisons']]!=[v['name'] for v in reference['comparisons']]:
            raise AssertionError('Incomplete reference coverage')
        r['sampled_parameter_update']=float((model.linear.bias.detach().cpu()-sampled).abs().max())
        if not r['sampled_parameter_update']>0: raise AssertionError('No parameter update observed')
        if {n:file_hash(Path(a.model)/n) for n in MODEL_HASHES}!=MODEL_HASHES:
            raise RuntimeError('Public source changed during run')
        r['status']='pass'
    finally:
        r['provider_calls']=dict(hits.counts); r['dispatcher_restored']=tables()==before
        logger.removeHandler(hits)
        if not r['dispatcher_restored']: raise RuntimeError('Dispatcher restoration failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',required=True); p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--reference',type=Path)
    p.add_argument('--adapter-config',type=Path); p.add_argument('--batch-size',type=int,choices=(16,32),default=32)
    p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    p.add_argument('--mode',choices=('native','native_repeat','off','on','unqualified','denied','unknown','strict'),required=True)
    a=p.parse_args()
    if a.out.exists() or (a.mode not in ('native','strict') and not a.reference): p.error('New output/native reference required')
    if (a.mode not in ('native','native_repeat'))!=bool(a.adapter_config): p.error('Model adapter modes require schema-3 configuration')
    r=dict(status='error',stage='deepfm',mode=a.mode,
        scope='P800 public DeepFM sigmoid short training; synthetic data, not CTR quality acceptance')
    from flagrt_adapter.startup import TargetBlocked
    try: run(a,r)
    except TargetBlocked as exc:
        try: record_expected_block(a,r,exc)
        except Exception: r.update(status='error',error=traceback.format_exc())
    except Exception:
        r['error']=traceback.format_exc(); r['status']='fail' if r.get('steps') else 'error'
        if r.get('steps') and r['steps'][-1]['status']=='running': r['steps'][-1]['status']='fail'
        print(r['error'],flush=True)
    finally: a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('overall:',r['status'].upper(),flush=True)
    return {'pass':0,'fail':1,'error':2,'blocked':3}[r['status']]


if __name__=='__main__': raise SystemExit(main())
