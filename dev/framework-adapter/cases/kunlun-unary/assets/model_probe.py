"""P800 Qwen3-Embedding-0.6B: bounded FP32 causal-LM adapter regression.

Reuse the public device backend/model resolver and training-leg formulation.
This is a single-card integration probe, not the public two-rank acceptance.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
from model_admission import prepare_model, expected_native_reason, record_expected_block

WEIGHT_SHA = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'
MODEL_TOLERANCE = {'atol': 1e-5, 'rtol': 1e-4}


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def compare_tensor(value, expected, tolerance):
    import torch
    if value.shape!=expected.shape or value.dtype!=expected.dtype:
        raise ValueError('Tensor shape/dtype differs from the native reference')
    # Avoid repeated float expansion/isclose for identical full-model tensors.
    # Equal non-finite values must still fail, and other dtypes retain the old path.
    if (value.dtype in (torch.float32,torch.float16,torch.bfloat16)
            and tolerance['atol']>=0 and tolerance['rtol']>=0
            and torch.equal(value,expected)):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Non-finite comparison input')
        return {'max_abs':0.,'mismatches':0}
    x,y=value.reshape(-1),expected.reshape(-1)
    maximum=0.; mismatches=0
    for start in range(0,x.numel(),1048576):
        xx,yy=x[start:start+1048576].float(),y[start:start+1048576].float()
        if not torch.isfinite(xx).all() or not torch.isfinite(yy).all():
            raise ValueError('Non-finite comparison input')
        maximum=max(maximum,float((xx-yy).abs().max()))
        mismatches+=int((~torch.isclose(xx,yy,**tolerance)).sum())
    return {'max_abs':maximum,'mismatches':mismatches}


def mismatch_examples(value, expected, tolerance, limit=8):
    """Keep a small CPU excerpt for handoff; never change the acceptance rule."""
    import torch
    x,y=value.reshape(-1),expected.reshape(-1)
    examples=[]
    for start in range(0,x.numel(),1048576):
        xx,yy=x[start:start+1048576].float(),y[start:start+1048576].float()
        indices=(~torch.isclose(xx,yy,**tolerance)).nonzero().flatten()
        for index in indices[:limit-len(examples)].tolist():
            actual,reference=float(xx[index]),float(yy[index])
            examples.append(dict(flat_index=start+index,actual=actual,reference=reference,
                abs_error=abs(actual-reference),
                allowed_error=tolerance['atol']+tolerance['rtol']*abs(reference)))
        if len(examples)>=limit: break
    return examples


def validate_reference(reference, current):
    if reference.get('status')!='pass' or reference.get('mode')!='native':
        raise ValueError('Reference must be a completed native baseline')
    for key in ('environment','adapter_version','model_files','workload','runtime_sources'):
        if reference.get(key)!=current.get(key):
            raise ValueError('Reference identity mismatch: '+key)


def run(a,r):
    import torch
    from probe import run as inspect_environment, Hits
    # Same audited identity/native gate as the successful single-op case.
    inspect_environment(SimpleNamespace(runtime_root=a.runtime_root,stage='inspect'),r)
    r['status']='running'
    import runtime
    from flagrt_adapter.bridge import ElementwiseAdapter
    sys.path.insert(0,str(Path(a.runtime_root)/'runtime/proto'))
    from model_path import require_model
    model_path=Path(require_model(a.model,'explicit adapter model probe'))
    hashes={name:file_hash(model_path/name) for name in ('config.json','model.safetensors')}
    if hashes['model.safetensors']!=WEIGHT_SHA: raise ValueError('Unexpected 0.6B weight identity')
    config=json.loads((model_path/'config.json').read_text())
    if (config.get('num_hidden_layers'),config.get('hidden_size'),config.get('intermediate_size'),
        config.get('hidden_act'))!=(28,1024,3072,'silu'):
        raise ValueError('Not the expected full Qwen3-Embedding-0.6B architecture')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    r.update(model_path=str(model_path),model_files=hashes,mode=a.mode,
             workload=dict(model='Qwen3-Embedding-0.6B',formulation='synthetic causal LM',
                 dtype='torch.float32',batch=1,sequence_length=32,steps=a.steps,seed=1234,
                 data_seed=999,optimizer='AdamW',lr=1e-5,eps=1e-8,weight_decay=.01,
                 foreach=False,gradient_clip=1.0,attention='eager',deterministic=True),
             runtime_sources={name:file_hash(Path(a.runtime_root)/name) for name in (
                 'runtime/proto/proto_train_leg.py','runtime/proto/model_path.py',
                 'runtime/backends/kunlun/backend.py')},
             tolerance=MODEL_TOLERANCE if a.mode=='on' else {'atol':0,'rtol':0},
             steps=[],comparisons=[],optimizer_steps=0)
    save=lambda:a.out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    actual,policies=prepare_model(a,r,'silu',[1,32,3072])
    reference=None
    if a.reference:
        reference=json.loads((a.reference/'result.json').read_text())
        validate_reference(reference,r)
    if a.mode=='native': (a.out.parent/'tensors').mkdir()
    def check(name,value):
        value=value.detach().cpu().contiguous()
        if not torch.isfinite(value).all(): raise ValueError('Non-finite '+name)
        filename=hashlib.sha256(name.encode()).hexdigest()+'.pt'
        row=dict(name=name,file=filename,shape=list(value.shape),dtype=str(value.dtype))
        if a.mode=='native':
            torch.save(value,a.out.parent/'tensors'/filename)
        else:
            expected=torch.load(a.reference/'tensors'/filename,map_location='cpu',weights_only=True)
            row.update(compare_tensor(value,expected,r['tolerance']))
        r['comparisons'].append(row)
        if row.get('mismatches',0):
            row['mismatch_examples']=mismatch_examples(value,expected,r['tolerance'])
            save(); raise AssertionError(f'{name}: {row["mismatches"]} mismatches, max_abs={row["max_abs"]}')
    from transformers import AutoModelForCausalLM
    model,loading=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    r['model_loaded']=True
    r['loading_info']=loading
    r['parameter_count']=sum(p.numel() for p in model.parameters())
    r['parameter_tensors']=len(list(model.parameters()))
    if not all(p.requires_grad and bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Non-finite or frozen initial model weights')
    r['initial_parameter_sha256']=hashlib.sha256(b''.join(
        hashlib.sha256(p.detach().contiguous().numpy().tobytes()).digest() for p in model.parameters())).hexdigest()
    if reference and r['initial_parameter_sha256']!=reference['initial_parameter_sha256']:
        raise ValueError('Initial model states differ')
    model=model.to('cuda:0').train()
    model.config.use_cache=False
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,eps=1e-8,weight_decay=.01,foreach=False)
    r['optimizer_created']=True
    gems=None
    if a.mode=='on': import flag_gems as gems
    hits=Hits(); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    tables=lambda:{n:torch._C._dispatch_dump_table('aten::'+n) for n in ('silu','silu_backward')}
    before=tables()
    sample=model.get_input_embeddings().weight[100].detach().cpu().clone()
    try:
        for step in range(a.steps):
            r['active_step']=step+1
            ids=torch.randint(100,30000,(1,32),generator=torch.Generator().manual_seed(999+step)).to('cuda:0')
            # Force the monitored embedding row to occur, without real user data.
            ids[0,0]=100
            optimizer.zero_grad(set_to_none=True)
            adapter=None if a.mode in ('native','native_repeat') else ElementwiseAdapter(policies,actual,runtime,gems)
            previous=hits.counts.copy()
            row={'step':step+1,'status':'running'}; r['steps'].append(row); save()
            with adapter if adapter is not None else nullcontext():
                loss=model(input_ids=ids,labels=ids,use_cache=False).loss
                if not bool(torch.isfinite(loss)): raise ValueError('Non-finite loss')
                loss.backward()
                runtime.synchronize(0)
            delta=dict(hits.counts-previous)
            row.update(loss=float(loss.detach()),provider_calls=delta,dispatcher_restored=tables()==before,
                routes=[] if adapter is None else [list(k)+[v] for k,v in adapter.operator_counts.items()])
            save()
            if not row['dispatcher_restored']: raise RuntimeError('Dispatcher not restored')
            if a.mode=='on':
                if delta!={'GEMS SILU FORWARD':28,'GEMS SILU BACKWARD':28}:
                    raise AssertionError('Expected exactly 28 forward/backward calls: '+str(delta))
                if any(k[1]!='flaggems' for k in adapter.operator_counts):
                    raise AssertionError('Strict model target unexpectedly used native')
            else:
                if delta: raise AssertionError('Native path called FlagGems')
                if adapter is not None:
                    reason=expected_native_reason(a.mode)
                    if adapter.operator_counts[('silu','native','forward',reason)]!=28:
                        raise AssertionError('Wrong native route/reason')
            prefix=f'step{step+1}/'
            check(prefix+'loss',loss)
            gradients=0
            for name,p in model.named_parameters():
                if p.grad is None: raise AssertionError('Missing gradient: '+name)
                check(prefix+'gradient/'+name,p.grad); gradients+=1
            # Do not update after a failed loss/gradient comparison.
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step(); runtime.synchronize(0); r['optimizer_steps']+=1
            parameters=0
            for name,p in model.named_parameters():
                check(prefix+'parameter/'+name,p); parameters+=1
            row.update(status='pass',gradient_tensors=gradients,parameter_tensors=parameters,grad_norm=float(norm))
            save(); print(json.dumps(row),flush=True)
        if reference and [v['name'] for v in r['comparisons']]!=[v['name'] for v in reference['comparisons']]:
            raise AssertionError('Incomplete reference coverage')
        r['sampled_parameter_update']=float((model.get_input_embeddings().weight[100].detach().cpu()-sample).abs().max())
        if not r['sampled_parameter_update']>0: raise AssertionError('No observed parameter update')
        if hashes!={name:file_hash(model_path/name) for name in hashes}: raise RuntimeError('Model files changed during run')
        r['memory']=runtime.memory_stats(0)
        r['status']='pass'
    finally:
        r['provider_calls']=dict(hits.counts)
        r['dispatcher_restored']=tables()==before
        logger.removeHandler(hits)
        if not r['dispatcher_restored']: raise RuntimeError('Dispatcher restoration failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',required=True); p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--reference',type=Path)
    p.add_argument('--adapter-config',type=Path)
    p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    p.add_argument('--mode',choices=('native','native_repeat','off','on','unqualified','denied','unknown','strict'),required=True)
    a=p.parse_args()
    if a.out.exists() or (a.mode not in ('native','strict') and not a.reference): p.error('New output and native reference required')
    if (a.mode not in ('native','native_repeat'))!=bool(a.adapter_config): p.error('Model adapter modes require schema-3 configuration')
    r=dict(status='error',stage='model',mode=a.mode,scope='P800 full 0.6B single-card short training; not quality or distributed acceptance')
    from flagrt_adapter.startup import TargetBlocked
    try: run(a,r)
    except TargetBlocked as exc:
        try: record_expected_block(a,r,exc)
        except Exception: r.update(status='error',error=traceback.format_exc())
    except Exception:
        r['error']=traceback.format_exc(); r['status']='fail' if r.get('steps') else 'error'
        if r.get('steps') and r['steps'][-1]['status']=='running':
            r['steps'][-1]['status']='fail'
        print(r['error'],flush=True)
    finally: a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('overall:',r['status'].upper(),flush=True)
    return {'pass':0,'fail':1,'error':2,'blocked':3}[r['status']]


if __name__=='__main__': raise SystemExit(main())
