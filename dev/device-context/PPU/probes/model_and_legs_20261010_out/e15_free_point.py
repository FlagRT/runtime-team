"""E15：机制验证 —— 崩溃是否发生在「释放我方 Stream 包装对象」这一刻。

E15_MODE:
  base    —— 与 E14 runtimestream 相同（退出期崩溃，作为对照）
  del     —— 在 destroy_process_group 之前显式 `del st; gc.collect()`（把释放提前到可观测处）
"""
import gc
import os
import sys

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

MODE = os.environ.get("E15_MODE", "base")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import runtime  # noqa: E402


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    runtime.use("ppu")
    runtime.set_device(local_rank)
    st = runtime.create_stream()
    print(f"[e15] mode={MODE} rank={rank} stream={st!r}", flush=True)

    dev_id = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/models/Qwen3-Embedding-0.6B", trust_remote_code=True, dtype=torch.float32).to(dev_id)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

    t = torch.tensor([float(rank + 1)], device=dev_id)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.manual_seed(1234 + rank)
    losses = []
    for step in range(20):
        g = torch.Generator().manual_seed(999 + step)
        ids = torch.randint(0, 30000, (4, 128), generator=g).to(dev_id)
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
    print(f"[e15] loss {losses[0]} -> {losses[-1]}", flush=True)

    if MODE == "del":
        print("[e15] >>> del st; gc.collect() （释放提前到此处）", flush=True)
        del st
        n = gc.collect()
        print(f"[e15] <<< gc.collect() 回收 {n} 个对象，仍未崩溃", flush=True)

    dist.destroy_process_group()
    print(f"[e15] rank={rank} destroy_process_group done", flush=True)


if __name__ == "__main__":
    main()
    print("[e15 exited main]", flush=True)
