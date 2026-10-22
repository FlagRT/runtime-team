"""E10：E8（克隆）+ 训练腿独有的两处运行时调用，二分定位：
  E10_MODE=nodev     —— 不加
  E10_MODE=setdev    —— + runtime.set_device(lr)
  E10_MODE=stream    —— + runtime.set_device(lr) + st = runtime.create_stream()（一直持有到退出）

假设：`create_stream()` 造出的统一流在解释器退出期被 GC 释放 ⇒ 触发段错误。
"""
import json
import os
import sys
import time

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import runtime  # noqa: E402

MODE = os.environ.get("E10_MODE", "nodev")
MODEL = "/workspace/models/Qwen3-Embedding-0.6B"
OUT_DIR = os.environ.get("E10_OUT", "/workspace/probes_ppu/segv_isolate/e10/out")
STEPS, BATCH, SEQ, LR = 20, 4, 128, 1e-5

rank = int(os.environ["RANK"])
lr = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])

runtime.use("ppu")
runtime.current().device_count()
if MODE in ("setdev", "stream"):
    runtime.set_device(lr)
    print(f"[e10] mode={MODE} set_device({lr}) ok", flush=True)

# 一直持有到退出（与 proto_train_leg 同形）
STREAM = None
if MODE == "stream":
    STREAM = runtime.create_stream()
    print(f"[e10] create_stream -> {STREAM!r}", flush=True)

torch.cuda.set_device(lr)
dist.init_process_group("nccl")
dev_id = torch.cuda.current_device()

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).to(dev_id)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
print(f"[e10] mode={MODE} rank={rank} model loaded on {dev_id}", flush=True)

t = torch.tensor([float(rank + 1)], device=dev_id)
dist.all_reduce(t, op=dist.ReduceOp.SUM)
g_in = torch.tensor([float(rank)], device=dev_id)
g_out = [torch.zeros(1, device=dev_id) for _ in range(world)]
dist.all_gather(g_out, g_in)
if world == 2:
    if rank == 0:
        s = torch.arange(8, dtype=torch.float32, device=dev_id)
        r = torch.empty(8, dtype=torch.float32, device=dev_id)
        dist.send(s, dst=1)
        dist.recv(r, src=1)
    else:
        r = torch.empty(8, dtype=torch.float32, device=dev_id)
        dist.recv(r, src=0)
        s = torch.arange(8, dtype=torch.float32, device=dev_id) * 2
        dist.send(s, dst=0)

torch.manual_seed(1234 + rank)
losses = []
t0 = time.time()
for step in range(STEPS):
    g = torch.Generator().manual_seed(999 + step)
    ids = torch.randint(0, 30000, (BATCH, SEQ), generator=g).to(dev_id)
    out = model(input_ids=ids, labels=ids)
    opt.zero_grad(set_to_none=True)
    out.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(round(float(out.loss.item()), 4))
runtime.synchronize(lr)
dt = time.time() - t0

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"e10_rank{rank}.json"), "w") as f:
    json.dump({"mode": MODE, "rank": rank, "loss": losses[:1] + losses[-1:], "dt": round(dt, 2)}, f)
print(f"[e10] mode={MODE} rank{rank} loss {losses[0]}->{losses[-1]}", flush=True)
dist.destroy_process_group()
