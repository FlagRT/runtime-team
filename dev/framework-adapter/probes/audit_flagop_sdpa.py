"""Audit actual FlagOP register/backward on CPU with an explicitly fake forward.

This does NOT run Triton or qualify an accelerator. Source must be the pinned,
clean FlagOS-OP checkout. The native forward isolates the real wrapper backward.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import traceback
import torch
import torch.nn.functional as F

REV='79b1b1aa867bd4dedd90eff69e6e9d717104e1fb'


def run(root):
    git=['git','-c','safe.directory='+str(root),'-C',str(root)]
    if subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()!=REV:
        raise ValueError('Unexpected FlagOP revision')
    if subprocess.check_output(git+['status','--porcelain'],text=True).strip():
        raise ValueError('Dirty FlagOP checkout')
    file=root/'ops/sdpa/register.py'
    spec=importlib.util.spec_from_file_location('audited_sdpa_registration',file)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    native=F.scaled_dot_product_attention
    def forward(q,k,v,mask=None,dropout=0.,causal=False,scale=None,gqa=False):
        return native(q,k,v,mask,dropout,causal,scale=scale,enable_gqa=gqa)
    mod._load_kernel_mod=lambda *_:forward
    rows=[]
    for name, sq, sk, hq, hk, causal, mask_kind in (
        ('square',4,4,2,2,False,None),('square_causal',4,4,2,2,True,None),
        ('gqa',4,4,4,2,False,None),('additive_mask',4,4,2,2,False,'float'),
        ('all_masked_row',4,4,2,2,False,'bool'),('rectangular_causal',2,4,2,2,True,None)):
        torch.manual_seed(20261023)
        q=torch.randn(1,hq,sq,8,requires_grad=True); k=torch.randn(1,hk,sk,8,requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        mask=None
        if mask_kind=='bool':
            mask=torch.ones(1,1,sq,sk,dtype=torch.bool); mask[:,:,0,:]=False
        if mask_kind=='float': mask=torch.randn(1,1,sq,sk,requires_grad=True)*.1
        inputs=(q,k,v) if mask_kind!='float' else (q,k,v,mask)
        dy=torch.randn_like(q)
        expected=forward(q,k,v,mask,0.,causal,None,hq!=hk)
        grads=torch.autograd.grad(expected,inputs,dy,retain_graph=True)
        actual=mod._SDPA_A1_Function.apply(q,k,v,mask,0.,causal,None,hq!=hk)
        actual_grads=torch.autograd.grad(actual,inputs,dy)
        errors=[float((a-b).abs().max()) if torch.isfinite(a-b).all() else None for a,b in zip(actual_grads,grads)]
        finite=all(bool(torch.isfinite(x).all()) for x in actual_grads)
        match=finite and all(torch.allclose(a,b,atol=1e-5,rtol=1e-4) for a,b in zip(actual_grads,grads))
        rows.append(dict(case=name,forward='native test seam, not FlagOP kernel',
            gradient_match=match,finite=finite,gradient_max_abs=errors))
    # Global registration check: use a non-SDPA CPU math function to avoid
    # recursively entering the overridden ATen operation. Run in this process only.
    def fake(*args): return args[0].clone()
    mod._load_kernel_mod_mod=lambda _:SimpleNamespace(PLATFORM='CPU-control',sdpa_triton=fake)
    mod._load_kernel_mod=lambda *_:fake
    before=torch._C._dispatch_dump_table('aten::scaled_dot_product_attention')
    lib=mod.register_a1(dispatch_key='AutogradCPU')
    try:
        try:
            native(torch.randn(1,1,2,4),torch.randn(1,1,2,4),torch.randn(1,1,2,4))
            default_call={'status':'PASS'}
        except Exception:
            default_call={'status':'FAIL','error':traceback.format_exc()}
        mod.unhook_a1(lib)
        restored=torch._C._dispatch_dump_table('aten::scaled_dot_product_attention')==before
    finally:
        # Test-only process cleanup, not an advertised FlagOP recovery interface.
        lib._destroy()
    return dict(scope='CPU wrapper controls; fake/native forward, no Triton kernel',
        source_revision=REV,register_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        torch=torch.__version__,cases=rows,registration_default_call=default_call,
        unhook_restored_while_caller_retains_handle=restored,
        expected_limits_observed=all(r['gradient_match'] for r in rows[:4]) and
            all(not r['gradient_match'] for r in rows[4:]) and not restored)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): p.error('Output exists')
    try:
        result=run(a.root)
        result['status']='FAIL' if result['expected_limits_observed'] else 'ERROR'
    except Exception:
        result={'status':'ERROR','error':traceback.format_exc()}
    with a.out.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
    print('overall: '+result['status'])
    raise SystemExit(1 if result['status']=='FAIL' else 2)
