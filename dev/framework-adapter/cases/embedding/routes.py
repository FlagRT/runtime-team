"""Explicit embedding off/on/missing-qualification hardware controls."""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import traceback

p=argparse.ArgumentParser();p.add_argument('--stage',choices=['routes'],required=True);a=p.parse_args()
r=dict(status='ERROR',stage='routes',cases=[])
try:
    import torch
    import torch.nn.functional as F
    import flag_gems as gems
    import runtime
    from environment import inspect_environment
    from flagrt_adapter.records import environment_from_record
    from flagrt_adapter.embedding import EmbeddingAdapter, EmbeddingPolicy, EmbeddingSignature, load_embedding_records
    inspect_environment(a,r); actual=environment_from_record(r['environment'])
    records=load_embedding_records('/input/review/records.json')
    class Hits(logging.Handler):
        def __init__(self): super().__init__();self.counts=Counter()
        def emit(self,record):
            if record.getMessage().startswith('GEMS EMBEDDING '): self.counts[record.getMessage()]+=1
    hits=Hits();logger=logging.getLogger('flag_gems.ops.embedding');logger.setLevel(logging.DEBUG);logger.addHandler(hits)
    tables=lambda:{op:torch._C._dispatch_dump_table('aten::'+op) for op in ('embedding','embedding_backward','embedding_dense_backward')}
    for width in (16,1):
        generator=torch.Generator().manual_seed(20261110+width)
        raw=torch.randn(3850,width,generator=generator)
        dy=torch.randn(32,5,width,generator=generator).cuda()
        ids=(torch.arange(160)%7).reshape(32,5).cuda()
        x=raw.cuda().requires_grad_(); native=F.embedding(ids,x);native.backward(dy)
        native_fw,native_bw=native.detach().cpu(),x.grad.cpu()
        for mode in ('off','on','unqualified'):
            hits.counts.clear();before=tables()
            sig=EmbeddingSignature((3850,width),(32,5))
            policy=EmbeddingPolicy(actual,(sig,),() if mode=='unqualified' else records,mode!='off')
            x=raw.cuda().requires_grad_()
            adapter=EmbeddingAdapter(policy,actual,runtime,gems)
            with adapter: y=F.embedding(ids,x);y.backward(dy)
            runtime.synchronize(0)
            fw,bw=y.detach().cpu(),x.grad.cpu()
            tol=dict(atol=1e-5,rtol=1e-4) if mode=='on' else dict(atol=0,rtol=0)
            torch.testing.assert_close(fw,native_fw,**tol);torch.testing.assert_close(bw,native_bw,**tol)
            assert tables()==before
            expected={'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1} if mode=='on' else {}
            assert dict(hits.counts)==expected
            reason={'off':'experimental_disabled','unqualified':'capability_unverified'}.get(mode)
            expected_routes=Counter({('flaggems','forward','completed'):1,('flaggems','backward','completed'):1}) if mode=='on' else Counter({('native','forward',reason):1})
            assert adapter.counts==expected_routes
            F.embedding(ids,raw.cuda().requires_grad_()).backward(dy)
            runtime.synchronize(0);assert dict(hits.counts)==expected
            row=dict(status='pass',width=width,mode=mode,hits=expected,events=list(adapter.events),
                restored=True,post_scope_native=True,forward_max_abs=(fw-native_fw).abs().max().item(),
                backward_max_abs=(bw-native_bw).abs().max().item())
            r['cases'].append(row);print('CASE',json.dumps(row),flush=True)
    r['status']='PASS'
except Exception:
    r['error']=traceback.format_exc();print(r['error'],flush=True)
finally: Path('/results/result.json').write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(0 if r['status']=='PASS' else 1)
