"""定位梯度 all_reduce 挂死：把「一个大的聚合通信」与「多次小通信」分开对比。

A) 先对 concat 后的整块梯度做一次 all_reduce（单次大通信）
B) 再逐参数 all_reduce（多次小通信），逐个打印，钉住具体张量
"""
import os, sys, time
import torch
import torch.distributed as dist
from datetime import timedelta
import flagcx  # noqa: F401  注册 cuda:flagcx

MODEL = os.environ["DC_MODEL"]
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dev = f"cuda:{lr}"

dist.init_process_group("cpu:gloo,cuda:flagcx", timeout=timedelta(seconds=120))
rank = dist.get_rank()
print(f"[rank{rank}] PG ready", flush=True)

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True, dtype=torch.float32).to(dev)
model.train()
print(f"[rank{rank}] model loaded", flush=True)

g = torch.Generator().manual_seed(999)
ids = torch.randint(0, 30000, (2, 64), generator=g).to(dev)
out = model(input_ids=ids, labels=ids)
out.loss.backward()
torch.cuda.synchronize()
print(f"[rank{rank}] backward done, loss={float(out.loss):.4f}", flush=True)

params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
print(f"[rank{rank}] grad tensors = {len(params)}", flush=True)

# ── A. 单次大通信（concat 成连续大块）──
flat = torch.cat([p.grad.detach().reshape(-1) for _, p in params])
print(f"[rank{rank}] A: flat numel={flat.numel()} bytes={flat.numel()*4}", flush=True)
t0 = time.time()
dist.all_reduce(flat, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
print(f"[rank{rank}] A: FLAT all_reduce OK  {time.time()-t0:.3f}s", flush=True)

# ── B. 多次小通信（逐参数）──
for i, (n, p) in enumerate(params):
    print(f"[rank{rank}] B#{i} {n} shape={tuple(p.grad.shape)} "
          f"contig={p.grad.is_contiguous()} dtype={p.grad.dtype}", flush=True)
    t0 = time.time()
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    print(f"[rank{rank}] B#{i} done {time.time()-t0:.3f}s", flush=True)

print(f"[rank{rank}] ALL DONE", flush=True)
