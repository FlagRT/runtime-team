"""E5：完全复刻训练腿的循环（AutoModelForCausalLM + AdamW + clip_grad + 20 步梯度 all_reduce），
但**不 import 原型** —— 据此把「原型层」从训练腿其余部分里切出来。
"""
import json
import os
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

MODEL = "/workspace/models/Qwen3-Embedding-0.6B"
STEPS, BATCH, SEQ, LR = 20, 4, 128, 1e-5

rank = int(os.environ["RANK"])
lr = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(lr)
dist.init_process_group("nccl")
dev = torch.cuda.current_device()
print(f"[e5] rank={rank} dev={dev} world={world}", flush=True)

model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).to(dev)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
print(f"[e5] model loaded", flush=True)

# 复刻训练腿的集合通信对照
t = torch.tensor([float(rank + 1)], device=dev)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
g_in = torch.tensor([float(rank)], device=dev)
g_out = [torch.zeros(1, device=dev) for _ in range(world)]
dist.all_gather(g_out, g_in)
if world == 2:
    if rank == 0:
        s = torch.arange(8, dtype=torch.float32, device=dev)
        r = torch.empty(8, dtype=torch.float32, device=dev)
        dist.send(s, dst=1)
        dist.recv(r, src=1)
    else:
        r = torch.empty(8, dtype=torch.float32, device=dev)
        dist.recv(r, src=0)
        s = torch.arange(8, dtype=torch.float32, device=dev) * 2
        dist.send(s, dst=0)
print(f"[e5] comm pass allreduce={t.item()} gather={sorted(float(x.item()) for x in g_out)}", flush=True)

torch.manual_seed(1234 + rank)
losses = []
t0 = time.time()
for step in range(STEPS):
    g = torch.Generator().manual_seed(999 + step)
    ids = torch.randint(0, 30000, (BATCH, SEQ), generator=g).to(dev)
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(round(float(loss.item()), 4))
dt = time.time() - t0
print(f"[e5] loss {losses[0]} -> {losses[-1]} in {dt:.2f}s", flush=True)

dist.barrier()
dist.destroy_process_group()
print(f"[e5 done] rank={rank}", flush=True)
