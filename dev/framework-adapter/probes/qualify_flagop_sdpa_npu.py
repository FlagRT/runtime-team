"""Isolated direct FlagOP SDPA candidate test, not global A1 integration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback


def run(a,r):
    import torch
    import flag_gems
    import triton
    import torch.nn.functional as F
    sys.path.insert(0,a.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count()==1
    root=Path(a.flagop_root)
    rev=subprocess.check_output(['git','-c','safe.directory='+str(root),'-C',str(root),'rev-parse','HEAD'],text=True).strip()
    assert rev=='79b1b1aa867bd4dedd90eff69e6e9d717104e1fb'
    sys.path.insert(0,str(root))
    file=root/'ops/sdpa/register.py'
    spec=importlib.util.spec_from_file_location('tested_sdpa_register',file)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    r.update(source_revision=rev,register_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        torch=torch.__version__,triton=triton.__version__,paths={'gems':flag_gems.__file__,'triton':triton.__file__},cases=[])
    before=torch._C._dispatch_dump_table('aten::scaled_dot_product_attention')
    for causal in (False,True):
        torch.manual_seed(20261023)
        q,k,v=[torch.randn(1,2,64,64,device='npu',dtype=torch.float16,requires_grad=True) for _ in range(3)]
        dy=torch.randn_like(q)
        expected=F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        eg=torch.autograd.grad(expected,(q,k,v),dy)
        actual=mod._SDPA_A1_Function.apply(q,k,v,None,0.,causal,None,False)
        ag=torch.autograd.grad(actual,(q,k,v),dy)
        runtime.synchronize(0)
        torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)
        for x,y in zip(ag,eg): torch.testing.assert_close(x,y,atol=.02,rtol=.02)
        assert before==torch._C._dispatch_dump_table('aten::scaled_dot_product_attention')
        r['cases'].append(dict(shape=[1,2,64,64],dtype='float16',causal=causal,
            forward_max_abs=float((actual-expected).abs().max()),
            gradient_max_abs=[float((x-y).abs().max()) for x,y in zip(ag,eg)],status='pass'))
    r['status']='pass'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root',required=True); p.add_argument('--flagop-root',required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): p.error('Output exists')
    r={'status':'fail','scope':'direct candidate forward/math backward; no global register or model routing'}
    try: run(a,r)
    except Exception: r['error']=traceback.format_exc(); print(r['error'],flush=True)
    finally:
        with a.out.open('x') as f: json.dump(r,f,indent=2)
    print('overall: '+r['status'].upper()); raise SystemExit(0 if r['status']=='pass' else 1)
