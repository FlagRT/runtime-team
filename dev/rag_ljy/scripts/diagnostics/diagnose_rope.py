#!/usr/bin/env python3
"""Isolate vLLM-Ascend Triton RoPE input/output stream dependencies on logical NPU 0."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import hashlib
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import torch
import torch_npu
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

torch.npu.set_device(0)
init_device_properties_triton()
torch.manual_seed(42)
q0=torch.randn((23,16,128),device='npu',dtype=torch.bfloat16)
k0=torch.randn((23,8,128),device='npu',dtype=torch.bfloat16)
p=torch.arange(23,device='npu',dtype=torch.int64)
angles=torch.arange(23,device='npu').float()[:,None]*torch.linspace(.001,1,64,device='npu')[None,:]
cache=torch.cat([angles.cos(),angles.sin()],dim=1).to(torch.bfloat16)
torch.npu.synchronize()
summary={}
for mode in ['none','before','after','both']:
    refs=[]
    for i in range(30):
        q=q0.clone();k=k0.clone()
        if mode in ['before','both']:torch.npu.synchronize()
        a,b=rope_forward_triton(q,k,cos_sin_cache=cache,positions=p,rope_dim=128)
        if mode in ['after','both']:torch.npu.synchronize()
        refs.append(a.float().cpu())
    summary[mode]={'unique_outputs':len({x.numpy().tobytes() for x in refs}),
                   'max_abs_diff':max((x-refs[0]).abs().max().item() for x in refs),
                   'hashes':[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in refs]}
    print(mode,summary[mode]['unique_outputs'],summary[mode]['max_abs_diff'],flush=True)
out=Path(__file__).resolve().parents[2]/'artifacts/diagnostics/runs'/f'rope-kernel-{datetime.now():%Y-%m-%d_%H%M%S}-{uuid4().hex[:6]}.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary,indent=2)+'\n')
