"""P800 full OneRec-8B BF16 short training, with a separate qualification mode.

Candidate uses reviewed single-op records in an explicit, oracle-checked
experiment. It neither creates nor bypasses a normal schema-3 model approval.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import traceback

from model_admission import (prepare_model, workload_from_report,
                             expected_native_reason, record_expected_block)
from model_probe import file_hash, compare_tensor, mismatch_examples, validate_reference

WEIGHTS = {
    'model-00000-of-00003.safetensors': '54144c8a0fccb57ace38e288dd507401d6e6c89fabcc616748cb09db059964ca',
    'model-00001-of-00003.safetensors': '9c6bcb75cc225008d937c3ce875f4b5c90db45b0cdf0db9b2ef2bf80a301ed6c',
    'model-00002-of-00003.safetensors': 'dec154ac50baceb71093e6cb1e60cfcb3218d91e4eafc5d7fe371fac4564af0b',
}
OPERATORS = {'silu': dict(shapes=[[1,16,12288]], dtypes=['torch.bfloat16'])}
TOLERANCE = dict(atol=1e-3, rtol=1e-2)  # Existing OneRec criterion, not tuned for P800.


def verify_architecture(config):
    expected = dict(model_type='qwen3', num_hidden_layers=36, hidden_size=4096,
                    intermediate_size=12288, hidden_act='silu')
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Not the pinned full OneRec-8B architecture')


def prepare_onerec(a, report):
    if a.mode != 'candidate':
        return prepare_model(a, report, 'silu', [1,16,12288], 'torch.bfloat16')
    # The normal on/off/unknown entry never reaches this experimental path.
    if a.adapter_config or not a.records or not a.reference:
        raise ValueError('Explicit candidate requires single-op evidence and a native oracle only')
    import flagrt_adapter
    from flagrt_adapter import load_records
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.startup import prepare_policies
    bridge = Path(flagrt_adapter.__file__).parent/'bridge.py'
    report['adapter_bridge_sha256'] = file_hash(bridge)
    report['workload_identity'] = asdict(workload_from_report(report, OPERATORS, file_hash(bridge)))
    actual = environment_from_record(report['environment'])
    config = dict(schema=2, enabled=True, require_target=True, records=None, operators=OPERATORS)
    policies, report['startup'] = prepare_policies(config, load_records(a.records), actual)
    report.update(qualification_experiment=True, model_approved=False,
                  acceptance='explicit candidate versus native; approval requires separate review')
    return actual, policies


def run(a, r):
    import torch
    from probe import run as inspect_environment, Hits
    inspect_environment(SimpleNamespace(runtime_root=a.runtime_root,stage='inspect'),r)
    r['status']='running'
    import runtime
    from flagrt_adapter.bridge import ElementwiseAdapter
    model_path=Path(a.model)
    hashes={name:file_hash(model_path/name) for name in (*WEIGHTS,'config.json','model.safetensors.index.json')}
    if any(hashes[name]!=sha for name,sha in WEIGHTS.items()):
        raise ValueError('OneRec shard identity differs from pinned public checkpoint')
    verify_architecture(json.loads((model_path/'config.json').read_text()))
    index=json.loads((model_path/'model.safetensors.index.json').read_text())
    if set(index['weight_map'].values())!=set(WEIGHTS): raise ValueError('Unexpected shard index')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(1014); torch.cuda.manual_seed_all(1014)
    r.update(mode=a.mode,model_files=hashes,
        workload=dict(model='OpenOneRec/OneRec-8B',revision='29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff',
            formulation='synthetic causal LM',dtype='torch.bfloat16',batch=1,sequence_length=16,
            steps=a.steps,seed=1014,data_seed=1014,optimizer='SGD',lr=.01,momentum=0.,
            foreach=False,gradient_checkpointing='non-reentrant',attention='eager',
            use_cache=False,deterministic=True),
        runtime_sources={name:file_hash(Path(a.runtime_root)/name) for name in (
            'runtime/backends/kunlun/backend.py','runtime/proto/model_path.py')},
        tolerance=TOLERANCE if a.mode in ('on','candidate') else dict(atol=0,rtol=0),
        steps=[],comparisons=[],optimizer_steps=0,model_loaded=False,optimizer_created=False,
        provider_calls={})
    save=lambda:a.out.write_text(json.dumps(r,indent=2)+'\n')
    save()
    actual,policies=prepare_onerec(a,r)
    reference=None
    if a.reference:
        reference=json.loads((a.reference/'result.json').read_text())
        validate_reference(reference,r)
        if reference['workload_identity']!=r['workload_identity']:
            raise ValueError('Measured model workload differs from native reference')
    if a.mode=='native': (a.out.parent/'tensors').mkdir()
    def check(name,value):
        value=value.detach().cpu().contiguous()
        if not torch.isfinite(value).all(): raise ValueError('Non-finite '+name)
        filename=hashlib.sha256(name.encode()).hexdigest()+'.pt'
        row=dict(name=name,file=filename,shape=list(value.shape),dtype=str(value.dtype))
        if a.mode=='native': torch.save(value,a.out.parent/'tensors'/filename)
        else:
            expected=torch.load(a.reference/'tensors'/filename,map_location='cpu',weights_only=True)
            row.update(compare_tensor(value,expected,r['tolerance']))
        r['comparisons'].append(row)
        if row.get('mismatches',0):
            row['mismatch_examples']=mismatch_examples(value,expected,r['tolerance'])
            save(); raise AssertionError(f'{name}: {row["mismatches"]} mismatches, max_abs={row["max_abs"]}')
    from transformers import AutoModelForCausalLM
    model,loading=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='eager',output_loading_info=True)
    r.update(model_loaded=True,loading_info=loading,
             parameter_count=sum(p.numel() for p in model.parameters()),
             parameter_tensors=len(list(model.parameters())))
    if r['parameter_count']!=8391013376 or r['parameter_tensors']!=399:
        raise ValueError('Incomplete model parameter set')
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('Weights did not load exactly')
    if not all(p.requires_grad and p.dtype==torch.bfloat16 and bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Non-finite, frozen, or wrong-dtype parameters')
    model=model.to('cuda:0').train()
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,foreach=False)
    r['optimizer_created']=True
    gems=None
    if a.mode in ('candidate','on'): import flag_gems as gems
    hits=Hits(); logger=logging.getLogger('flag_gems'); logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    tables=lambda:{n:torch._C._dispatch_dump_table('aten::'+n) for n in ('silu','silu_backward')}
    before=tables()
    try:
        for step in range(a.steps):
            r['active_step']=step+1
            tokens=torch.randint(0,model.config.vocab_size,(1,16),generator=torch.Generator().manual_seed(1014+step)).to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            adapter=None if a.mode in ('native','native_repeat') else ElementwiseAdapter(policies,actual,runtime,gems)
            previous=hits.counts.copy()
            row=dict(step=step+1,status='running'); r['steps'].append(row); save()
            with adapter if adapter is not None else nullcontext():
                loss=model(input_ids=tokens,labels=tokens,use_cache=False).loss
                if not bool(torch.isfinite(loss)): raise ValueError('Non-finite loss')
                loss.backward(); runtime.synchronize(0)
            delta=dict(hits.counts-previous)
            row.update(loss=float(loss.detach()),provider_calls=delta,dispatcher_restored=tables()==before,
                routes=[] if adapter is None else [list(k)+[v] for k,v in adapter.operator_counts.items()])
            save()
            if not row['dispatcher_restored']: raise RuntimeError('Dispatcher not restored')
            if a.mode in ('candidate','on'):
                if delta.get('GEMS SILU FORWARD',0)<36 or delta.get('GEMS SILU BACKWARD',0)!=36:
                    raise AssertionError('Missing target forward/backward: '+str(delta))
                if any(k[1]!='flaggems' for k in adapter.operator_counts):
                    raise AssertionError('Strict target unexpectedly used native')
            else:
                if delta: raise AssertionError('Native path called FlagGems')
                if adapter is not None:
                    reason=expected_native_reason(a.mode)
                    if adapter.operator_counts[('silu','native','forward',reason)]<36:
                        raise AssertionError('Wrong native route/reason')
            prefix=f'step{step+1}/'
            check(prefix+'loss',loss)
            for name,p in model.named_parameters():
                if p.grad is None: raise AssertionError('Missing gradient: '+name)
                check(prefix+'gradient/'+name,p.grad)
            optimizer.step(); runtime.synchronize(0); r['optimizer_steps']+=1
            for name,p in model.named_parameters(): check(prefix+'parameter/'+name,p)
            row.update(status='pass',compared_tensors=799)
            save(); print(json.dumps(row),flush=True)
        if reference and [v['name'] for v in r['comparisons']]!=[v['name'] for v in reference['comparisons']]:
            raise AssertionError('Incomplete reference coverage')
        if hashes!={name:file_hash(model_path/name) for name in hashes}:
            raise RuntimeError('Model source changed during experiment')
        r.update(status='pass',memory=runtime.memory_stats(0)); save()
    finally:
        r['provider_calls']=dict(hits.counts)
        r['dispatcher_restored']=tables()==before
        logger.removeHandler(hits)
        if not r['dispatcher_restored']: raise RuntimeError('Dispatcher restoration failed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','model'): p.add_argument('--'+name,required=True)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--reference',type=Path)
    p.add_argument('--adapter-config',type=Path); p.add_argument('--records',type=Path)
    p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    p.add_argument('--mode',choices=('native','native_repeat','off','on','unqualified','denied','unknown','strict','candidate'),required=True)
    a=p.parse_args()
    if a.out.exists() or (a.mode not in ('native','strict') and not a.reference):
        p.error('New output and native reference required')
    if a.mode=='candidate':
        if a.adapter_config or not a.records: p.error('Candidate requires only reviewed single-op records')
    elif a.records or (a.mode not in ('native','native_repeat'))!=bool(a.adapter_config):
        p.error('Normal model adapter modes require schema-3 configuration; no record-only bypass')
    r=dict(status='error',stage='onerec',mode=a.mode,scope='full OneRec-8B BF16 single-card bounded training; not model quality')
    from flagrt_adapter.startup import TargetBlocked
    try: run(a,r)
    except TargetBlocked as exc:
        try: record_expected_block(a,r,exc)
        except Exception: r.update(status='error',error=traceback.format_exc())
    except Exception:
        r.update(error=traceback.format_exc(),status='fail' if r.get('steps') else 'error')
        if r.get('steps') and r['steps'][-1]['status']=='running': r['steps'][-1]['status']='fail'
        print(r['error'],flush=True)
    finally: a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('overall:',r['status'].upper(),flush=True)
    return {'pass':0,'fail':1,'error':2,'blocked':3}[r['status']]


if __name__=='__main__': raise SystemExit(main())
