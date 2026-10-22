"""E16：顺序判别 —— 同样调用，只换「原型触流」与「建进程组」的先后。

E16_MODE=after  —— init_process_group → runtime.use/set_device/create_stream（= E15 base）
E16_MODE=before —— runtime.use/set_device/create_stream → init_process_group（= E10 顺序）
"""
import os
import sys

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

MODE = os.environ.get("E16_MODE", "after")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import runtime  # noqa: E402


def bind_runtime(local_rank):
    runtime.use("ppu")
    runtime.set_device(local_rank)
    return runtime.create_stream()


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])

    if MODE == "before":
        st = bind_runtime(local_rank)
        print(f"[e16] mode={MODE} stream created BEFORE init_process_group", flush=True)
        dist.init_process_group("nccl")
    else:
        dist.init_process_group("nccl")
        st = bind_runtime(local_rank)
        print(f"[e16] mode={MODE} stream created AFTER init_process_group", flush=True)

    rank = dist.get_rank()
    world = dist.get_world_size()
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
    print(f"[e16] mode={MODE} loss {losses[0]} -> {losses[-1]}", flush=True)

    dist.destroy_process_group()
    print(f"[e16] mode={MODE} rank={rank} destroy done", flush=True)


if __name__ == "__main__":
    main()
    print("[e16 exited main]", flush=True)
