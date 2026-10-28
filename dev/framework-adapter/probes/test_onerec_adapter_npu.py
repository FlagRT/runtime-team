"""Full OneRec-8B bounded SGD adapter regression; synthetic tokens, not quality."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import sys
import traceback

WEIGHTS = {
    'model-00000-of-00003.safetensors': '54144c8a0fccb57ace38e288dd507401d6e6c89fabcc616748cb09db059964ca',
    'model-00001-of-00003.safetensors': '9c6bcb75cc225008d937c3ce875f4b5c90db45b0cdf0db9b2ef2bf80a301ed6c',
    'model-00002-of-00003.safetensors': 'dec154ac50baceb71093e6cb1e60cfcb3218d91e4eafc5d7fe371fac4564af0b',
}

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()

def run(a, r):
    import torch
    import _adapter_package
    from flagrt_adapter import Environment, OperatorPolicy, load_records
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.startup import package_identity
    from qualify_unary_npu import Hits
    import flag_gems, triton
    sys.path.insert(0, a.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count() == 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1014); torch.npu.manual_seed_all(1014)
    actual = {n:file_hash(Path(a.model_root)/n) for n in WEIGHTS}
    assert actual == WEIGHTS, 'Weight identity mismatch'
    revision, identity = package_identity(flag_gems.__file__)
    builds=[]
    for name in ('flagtree','triton-ascend'):
        try: builds.append(name+'=='+metadata.version(name))
        except metadata.PackageNotFoundError: pass
    env = Environment(torch.npu.get_device_name(0),torch.__version__,metadata.version('torch-npu'),
        revision,os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,';'.join(builds))
    policy = OperatorPolicy(env,((1,a.seq_len,12288),),('torch.bfloat16',),a.mode!='disabled',
        load_records(a.records) if a.mode=='target' else (),operator='silu')
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(a.model_root,local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager').to('npu').train()
    assert model.config.num_hidden_layers==36 and model.config.hidden_size==4096
    count=sum(p.numel() for p in model.parameters())
    assert count==8391013376 and all(p.requires_grad for p in model.parameters())
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    r.update(environment=env.__dict__,gems_identity=identity,weights=actual,parameter_count=count,
        mode=a.mode,optimizer='SGD lr=0.01, no momentum',batch=1,seq_len=a.seq_len,steps=a.steps,
        tolerance={'atol':1e-3,'rtol':1e-2} if a.mode=='target' else {'atol':0,'rtol':0},
        comparisons=[],runtime=runtime.__file__)
    def save(): a.out.write_text(json.dumps(r,indent=2))
    save()
    reference = None
    if a.mode!='native':
        reference=json.loads((Path(a.reference)/'result.json').read_text())
        assert reference['status']=='pass' and reference['mode']=='native'
        assert reference['weights']==actual and reference['environment']==env.__dict__
        assert reference['seq_len']==a.seq_len and reference['steps']==a.steps
    snapshots=a.out.parent/'tensors'
    if a.mode=='native': snapshots.mkdir()
    def compare(name, value):
        value=value.detach().cpu().contiguous()
        assert torch.isfinite(value).all(), name
        filename=hashlib.sha256(name.encode()).hexdigest()+'.pt'
        row={'name':name,'shape':list(value.shape),'file':filename}
        if a.mode=='native': torch.save(value,snapshots/filename)
        else:
            expected=torch.load(Path(a.reference)/'tensors'/filename,map_location='cpu',weights_only=True)
            assert value.shape==expected.shape and value.dtype==expected.dtype
            x,y=value.reshape(-1),expected.reshape(-1)
            maximum=0.; mismatches=0
            for start in range(0,x.numel(),1048576):
                xx=x[start:start+1048576].float(); yy=y[start:start+1048576].float()
                maximum=max(maximum,float((xx-yy).abs().max()))
                mismatches+=int((~torch.isclose(xx,yy,**r['tolerance'])).sum())
            row.update(max_abs=maximum,mismatches=mismatches)
            r['comparisons'].append(row)
            if mismatches:
                save(); raise AssertionError(f'{name}: {mismatches} mismatches, max_abs={maximum}')
            return
        r['comparisons'].append(row)
    hits=Hits(); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    names=('silu','silu_backward')
    before={n:torch._C._dispatch_dump_table('aten::'+n) for n in names}
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,foreach=False)
    r['step_results']=[]; r['optimizer_steps']=0
    for step in range(a.steps):
        r['active_step']=step+1
        scope=nullcontext() if a.mode in ('native','native_repeat') else ElementwiseAdapter({'silu':policy},env,runtime,flag_gems)
        tokens=torch.randint(0,model.config.vocab_size,(1,a.seq_len),generator=torch.Generator().manual_seed(1014+step)).to('npu')
        optimizer.zero_grad(set_to_none=True)
        hit_before=hits.counts.copy()
        with scope:
            loss=model(input_ids=tokens,labels=tokens).loss
            loss.backward()
        runtime.synchronize(0)
        delta=dict(hits.counts-hit_before)
        routes=[] if a.mode in ('native','native_repeat') else [list(k)+[v] for k,v in scope.operator_counts.items()]
        restored=before=={n:torch._C._dispatch_dump_table('aten::'+n) for n in names}
        row=dict(step=step+1,hits=delta,routes=routes,dispatcher_restored=restored,loss=float(loss.detach()))
        r['step_results'].append(row); save()
        assert restored
        if a.mode=='target':
            assert delta.get('GEMS SILU FORWARD',0)>=36 and delta.get('GEMS SILU BACKWARD',0)==36, delta
            assert not any(k[1]=='native' for k in scope.operator_counts), routes
        else:
            assert not delta, delta
            if a.mode in ('disabled','unqualified'):
                reason='experimental_disabled' if a.mode=='disabled' else 'capability_unverified'
                assert scope.operator_counts[('silu','native','forward',reason)]>=36, routes
        prefix=f'step{step+1}/'
        compare(prefix+'loss',loss)
        for name,p in model.named_parameters():
            assert p.grad is not None, name
            compare(prefix+'gradient/'+name,p.grad)
        optimizer.step(); runtime.synchronize(0)
        r['optimizer_steps']+=1
        for name,p in model.named_parameters(): compare(prefix+'parameter/'+name,p)
        row['compared_tensors']=799
        save(); print(json.dumps(row),flush=True)
    r['hits']=dict(hits.counts)
    r['dispatcher_restored']=all(v['dispatcher_restored'] for v in r['step_results'])
    if reference: assert [v['name'] for v in r['comparisons']]==[v['name'] for v in reference['comparisons']]
    r['memory']=runtime.memory_stats(0)
    r['status']='pass'
    save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','model-root','records'): p.add_argument('--'+name,required=True)
    p.add_argument('--reference'); p.add_argument('--mode',choices=('native','native_repeat','disabled','unqualified','target'),required=True)
    p.add_argument('--steps',type=int,choices=(1,2,3),default=1)
    p.add_argument('--seq-len',type=int,choices=(16,32),default=16)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists() or (a.mode!='native' and not a.reference): p.error('New output and native reference required')
    r={'status':'fail','scope':'full OneRec-8B, single-rank bounded adapter regression, not communication/quality'}
    try: run(a,r)
    except Exception: r['error']=traceback.format_exc(); print(r['error'],flush=True)
    finally: a.out.write_text(json.dumps(r,indent=2))
    print('overall: '+r['status'].upper(),flush=True)
    raise SystemExit(0 if r['status']=='pass' else 1)
