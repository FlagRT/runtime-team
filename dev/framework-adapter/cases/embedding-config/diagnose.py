"""Bounded, frozen-input localization; no model approval or kernel changes.

Capture native DeepFM tensors, verify native training against the frozen reference,
then compare direct provider / standalone adapter / unified adapter on identical
values. A completed diagnosis is not a successful numerical qualification.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=a.out.parent
    report={'status':'error','purpose':'localization only','results':[]}
    save=lambda:a.out.write_text(json.dumps(report,indent=2)+'\n')
    try:
        import torch
        import torch.nn.functional as F
        import deepfm_probe as probe
        import flag_gems as gems
        import runtime
        from flagrt_adapter.embedding import EmbeddingAdapter
        from flagrt_adapter.adapter import FrameworkAdapter
        from flagrt_adapter.embedding_policy import EmbeddingPolicy, EmbeddingSignature, load_embedding_records
        from flagrt_adapter.records import environment_from_record
        from flagrt_adapter.startup import package_content_identity

        def cpu(x): return x.detach().cpu().contiguous().clone()
        def digest(x): return hashlib.sha256(cpu(x).numpy().tobytes()).hexdigest()
        def diff(x,y):
            x,y=x.double(),y.double()
            return {'max_abs':float((x-y).abs().max()),'unequal':int((x!=y).sum())}

        samples=[]
        original_load=probe.load_model
        def load(root):
            model,config=original_load(root)
            def hook(name):
                def capture(module,args,output):
                    sample=dict(name=name,weight=cpu(module.weight),indices=cpu(args[0]))
                    samples.append(sample)
                    def gradient(g):
                        sample.update(gradient=cpu(g),gradient_shape=list(g.shape),
                                      gradient_stride=list(g.stride()))
                    output.register_hook(gradient)
                return capture
            for name,module in model.named_modules():
                if isinstance(module,torch.nn.Embedding): module.register_forward_hook(hook(name))
            return model,config
        probe.load_model=load
        native_dir=root/'native'; native_dir.mkdir()
        native={}
        args=SimpleNamespace(runtime_root='/input/public',model='/input/model',mode='native',
            batch_size=32,steps=3,out=native_dir/'result.json',reference=None,adapter_config=None)
        try: probe.run(args,native)
        finally:
            probe.load_model=original_load
            args.out.write_text(json.dumps(native,indent=2)+'\n')
        reference=json.loads(Path('/input/reference/result.json').read_text())
        assert [c['name'] for c in native['comparisons']]==[c['name'] for c in reference['comparisons']]
        native_diffs=[]
        for item in native['comparisons']:
            x=torch.load(native_dir/'tensors'/item['file'],weights_only=True)
            y=torch.load(Path('/input/reference/tensors')/item['file'],weights_only=True)
            native_diffs.append(dict(name=item['name'],**diff(x,y)))
        report.update(native_comparisons=native_diffs,environment=native['environment'],
                      adapter_package_identity=package_content_identity(__import__('flagrt_adapter').__file__)[0])
        assert len(samples)==6 and all('gradient' in s for s in samples)
        torch.save(samples,root/'captured.pt')
        report['capture_sha256']=probe.file_hash(root/'captured.pt')
        assert all(d['unequal']==0 for d in native_diffs), 'Instrumented native differs from frozen control'
        actual=environment_from_record(native['environment'])
        records=load_embedding_records('/input/review/records.json')
        signatures=tuple(EmbeddingSignature((3850,w),(32,5),'torch.float32','torch.int64') for w in (16,1))
        policy=EmbeddingPolicy(actual,signatures,records,experimental=True)
        assert all(policy.reason(actual,(s,'cuda')) is None for s in signatures)
        results_dir=root/'tensors';results_dir.mkdir()
        report['samples']=[]
        # First two steps include the step at which the model regression failed.
        for sample_id,s in enumerate(samples[:4]):
            w=s['weight'].to('cuda:0'); indices=s['indices'].to('cuda:0')
            grad=s['gradient'].to('cuda:0')
            # Recreate the expanded grad from linear.fc, not only its dense copy.
            if 0 in s['gradient_stride']:
                grad=grad[:,0:1,:].contiguous().expand(s['gradient_shape'])
            assert torch.equal(cpu(grad),s['gradient'])
            native_w=w.detach().clone().requires_grad_()
            native_out=F.embedding(indices,native_w)
            native_out.backward(grad);runtime.synchronize(0)
            reference_out,reference_grad=cpu(native_out),cpu(native_w.grad)
            double_w=s['weight'].double().requires_grad_()
            F.embedding(s['indices'],double_w).backward(s['gradient'].double())
            report['samples'].append(dict(sample=sample_id,table=s['name'],step=sample_id//2+1,
                shape=list(w.shape),gradient_stride=list(grad.stride()),
                native_vs_cpu64=diff(reference_grad,double_w.grad),
                weight_sha=digest(w),indices_sha=digest(indices),gradient_sha=digest(grad)))
            outputs={}
            for repeat in range(5):
                paths=['direct','standalone','unified']
                paths=paths[repeat%3:]+paths[:repeat%3]
                for path in paths:
                    calls=[]
                    class Spy:
                        def embedding(self,value,ind,*options):
                            assert options==(-1,False,False)
                            assert torch.equal(cpu(value),s['weight']) and torch.equal(cpu(ind),s['indices'])
                            calls.append('forward')
                            return gems.embedding(value,ind,*options)
                        def embedding_backward(self,value,ind,*options):
                            assert options==(3850,-1,False,False)
                            assert value.is_contiguous() and torch.equal(cpu(value),s['gradient'])
                            assert torch.equal(cpu(ind),s['indices'])
                            calls.append('backward')
                            return gems.embedding_backward(value,ind,*options)
                    provider=Spy()
                    weight=w.detach().clone().requires_grad_()
                    if path=='direct':
                        with torch.no_grad():
                            runtime.synchronize(0)
                            out=provider.embedding(weight,indices,-1,False,False);runtime.synchronize(0)
                            dw=provider.embedding_backward(grad.contiguous(),indices,3850,-1,False,False)
                            runtime.synchronize(0)
                        routes=[];copies=int(not grad.is_contiguous())
                    else:
                        adapter=(EmbeddingAdapter(policy,actual,runtime,provider) if path=='standalone'
                                 else FrameworkAdapter({'embedding':policy},actual,runtime,provider))
                        with adapter:
                            out=F.embedding(indices,weight);out.backward(grad);runtime.synchronize(0)
                        dw=weight.grad;copies=adapter.gradient_copies
                        routes=[list(k)+[v] for k,v in adapter.operator_counts.items()]
                        assert sum(adapter.operator_counts.values())==2
                    assert calls==['forward','backward']
                    out_cpu,dw_cpu=cpu(out),cpu(dw)
                    assert torch.isfinite(out_cpu).all() and torch.isfinite(dw_cpu).all()
                    key=f'{sample_id}-{repeat}-{path}'
                    torch.save({'forward':out_cpu,'backward':dw_cpu},results_dir/(key+'.pt'))
                    outputs[(repeat,path)]=dw_cpu
                    report['results'].append(dict(sample=sample_id,repeat=repeat,path=path,
                        forward_vs_native=diff(out_cpu,reference_out),backward_vs_native=diff(dw_cpu,reference_grad),
                        backward_vs_cpu64=diff(dw_cpu,double_w.grad),gradient_copies=copies,routes=routes,
                        file='tensors/'+key+'.pt',sha256=probe.file_hash(results_dir/(key+'.pt'))))
                    save()
            report.setdefault('pairwise',[]).append(dict(sample=sample_id,
                comparisons=[dict(a=list(k1),b=list(k2),**diff(v1,v2))
                             for (k1,v1),(k2,v2) in itertools.combinations(outputs.items(),2)]))
            print('sample',sample_id,'completed',flush=True)
        report['status']='pass'
    except Exception:
        report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally: save()
    return 0 if report['status']=='pass' else 1


if __name__=='__main__': raise SystemExit(main())
