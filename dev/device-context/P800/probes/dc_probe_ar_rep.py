"""判定集合通信挂死的性质：是「重复调用累积」还是「特定张量」？

C) 对同一个 1024x1024 张量重复 all_reduce 200 次，每 10 次打印（看是否在某个 N 挂住）
D) 再对 310 个真实梯度张量按序做 all_reduce，每 5 个打印（看是否回到 B#2）
"""
import os, time
import torch
import torch.distributed as dist
from datetime import timedelta
import flagcx  # noqa: F401

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
g = torch.Generator().manual_seed(999)
ids = torch.randint(0, 30000, (2, 64), generator=g).to(dev)
out = model(input_ids=ids, labels=ids)
out.loss.backward()
torch.cuda.synchronize()
params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
print(f"[rank{rank}] setup done, grads={len(params)}", flush=True)

# ── C. 同一张量重复 200 次 ──
t = torch.ones(1024, 1024, device=dev)
t0 = time.time()
for i in range(200):
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if (i + 1) % 10 == 0:
        torch.cuda.synchronize()
        print(f"[rank{rank}] C rep={i+1} elapsed={time.time()-t0:.2f}s val={float(t[0,0])}", flush=True)
print(f"[rank{rank}] C: 200 reps OK", flush=True)

# ── D. 真实梯度按序 ──
for i, (n, p) in enumerate(params):
    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    if (i + 1) % 5 == 0 or i < 5:
        if (i + 1) % 5 == 0:
            torch.cuda.synchronize()
        print(f"[rank{rank}] D#{i} {n} shape={tuple(p.grad.shape)}", flush=True)
print(f"[rank{rank}] D: ALL {len(params)} grads OK", flush=True)
print(f"[rank{rank}] ALL DONE", flush=True)
