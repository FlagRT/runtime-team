"""Minimal direct-provider reproduction: no runtime or adapter imports.

Run in the recorded PPU environment with one visible device:
GEMS_VENDOR=nvidia python direct_replay.py --samples captured.pt --out result.json
Uses 4 frozen native DeepFM samples and an explicit unique-index control. Does
not modify kernels, approve models, change tolerances, or run optimizer steps.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import traceback

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    report={'status':'error','results':[]}
    try:
        import torch
        import torch.nn.functional as F
        import flag_gems as gems
        import inspect
        import importlib.metadata as metadata
        import triton
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(2)
        assert torch.cuda.device_count()==1
        report.update(chip=torch.cuda.get_device_name(0),torch=torch.__version__,
                      samples_sha256=hashlib.sha256(a.samples.read_bytes()).hexdigest())
        source=Path(inspect.getsourcefile(inspect.unwrap(gems.embedding_backward)))
        report['provider']=dict(module=gems.embedding_backward.__module__,source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            versions={**{n:metadata.version(n) for n in ('flag-gems','flagtree')},
                      'triton-module':triton.__version__})
        samples=torch.load(a.samples,map_location='cpu',weights_only=True)[:4]
        for sample_id,s in enumerate(samples):
            for control in ('captured','unique_indices'):
                index=s['indices'] if control=='captured' else torch.arange(s['indices'].numel()).reshape(s['indices'].shape)
                indices=index.to('cuda:0');grad=s['gradient'].to('cuda:0').contiguous()
                outputs={'native':[],'direct':[]}
                for repeat in range(5):
                    w=s['weight'].to('cuda:0').requires_grad_()
                    out=F.embedding(indices,w);out.backward(grad);torch.cuda.synchronize()
                    outputs['native'].append(w.grad.detach().cpu())
                    with torch.no_grad():
                        y=gems.embedding(w,indices,-1,False,False);torch.cuda.synchronize()
                        assert torch.equal(y,out)
                        dw=gems.embedding_backward(grad,indices,w.shape[0],-1,False,False)
                        torch.cuda.synchronize()
                        outputs['direct'].append(dw.cpu())
                for path,values in outputs.items():
                    differences=[float((x.double()-y.double()).abs().max()) for x,y in itertools.combinations(values,2)]
                    row=dict(sample=sample_id,table=s['name'],control=control,path=path,
                        repeats=len(values),repeated_index_count=int(index.numel()-index.unique().numel()),
                        repeat_max_abs=max(differences),nonidentical_pairs=sum(d>0 for d in differences))
                    if path=='direct':row['native_max_abs']=max(float((x.double()-outputs['native'][0].double()).abs().max()) for x in values)
                    report['results'].append(row)
                torch.save(outputs,a.out.parent/f'direct-{sample_id}-{control}.pt')
                print(sample_id,control,report['results'][-2:],flush=True)
        report['status']='pass'
    except Exception: report['error']=traceback.format_exc();print(report['error'],flush=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['status']=='pass' else 1

if __name__=='__main__':raise SystemExit(main())
