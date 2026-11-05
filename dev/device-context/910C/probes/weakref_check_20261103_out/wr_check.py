import sys, weakref
mode = sys.argv[1]
N = 30
import torch
import torch_npu  # noqa: F401  # 触发 npu 命名空间注册（容器 AUTOLOAD=0）
import acl  # noqa: F401
rc = acl.init()
n = acl.rt.get_device_count()
cnt = n[1] if isinstance(n, tuple) else n
print(f"[env] acl.init rc={rc} device_count={cnt}", flush=True)
if cnt and cnt >= 1:
    acl.rt.set_device(0)
import os
print(f"[env] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES','<unset>')}", flush=True)
refs = []
made = 0
for _ in range(N):
    if mode == "wr_tensor":
        o = torch.zeros(8, device="npu:0"); made += 1
    else:
        o = torch.npu.Stream(); made += 1
    if mode in ("wr_stream", "wr_tensor"):
        refs.append(weakref.ref(o))
    if mode != "wr_hold":
        del o
print(f"[{mode}] runtime_imported=False N={N} made={made} refs={len(refs)} strong={'yes' if mode=='wr_hold' else 'no'}", flush=True)
print(f"[{mode}] reached end, normal exit ...", flush=True)
