"""Direct FlagGems SiLU forward/backward qualification after compiler repair.

No production admission records are created. New cache, fixed tolerances.
"""
import os
import tempfile
os.environ['TRITON_CACHE_DIR'] = tempfile.mkdtemp(prefix='shmem-silu-cold-')
import json
import traceback
from pathlib import Path
import torch
import torch_npu
import flag_gems


def main():
    report={'status':'ERROR','checks':[],'cache':os.environ['TRITON_CACHE_DIR']}
    try:
        for shape in [(128,), (2, 17), (4, 1024)]:
            torch.manual_seed(123)
            cpu=torch.randn(shape,dtype=torch.float32)
            grad=torch.randn(shape,dtype=torch.float32)
            ref=cpu.clone().requires_grad_(); yref=torch.nn.functional.silu(ref); yref.backward(grad)
            x=cpu.npu(); y=flag_gems.silu(x)
            dx=flag_gems.silu_backward(grad.npu(),x); torch_npu.npu.synchronize()
            for name,got,want in [('forward',y.detach().cpu(),yref.detach()),('backward',dx.cpu(),ref.grad)]:
                ok=torch.allclose(got,want,atol=1e-5,rtol=1e-4)
                report['checks'].append({'shape':list(shape),'stage':name,'pass':ok,'max_abs':(got-want).abs().max().item()})
                assert ok, str(report['checks'][-1])
        report['status']='PASS'
    except Exception:
        report['error']=traceback.format_exc()
        print(report['error'],flush=True)
    Path('/results/silu-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
