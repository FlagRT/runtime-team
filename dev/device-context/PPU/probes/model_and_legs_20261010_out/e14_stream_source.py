"""E14：把「流的创建位置与生命周期」对齐训练腿，比较两种来源：
  E14_MODE=torchstream   —— st = torch.cuda.Stream()（**不经我方包装**）
  E14_MODE=runtimestream —— st = runtime.create_stream()（我方统一流）

两者都在 init_process_group + set_device 之后、且在 main() 内创建，
在 main() 返回时释放（与 proto_train_leg 同形）。
"""
import os
import sys

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

MODE = os.environ.get("E14_MODE", "torchstream")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

import runtime  # noqa: E402


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    if MODE == "runtimestream":
        runtime.use("ppu")
        runtime.set_device(local_rank)
        st = runtime.create_stream()
    else:
        # 与 runtimestream 分支**同一位置、同一生命周期**，唯一差别是不经我方包装
        torch.cuda.set_device(local_rank)
        st = torch.cuda.Stream()
    print(f"[e14] mode={MODE} rank={rank} stream={st!r}", flush=True)

    dev_id = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(
        "/workspace/models/Qwen3-Embedding-0.6B", trust_remote_code=True, dtype=torch.float32).to(dev_id)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8)

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

    print(f"[e14] loss {losses[0]} -> {losses[-1]}", flush=True)
    dist.destroy_process_group()
    # st 在 main 返回时被释放（与 proto_train_leg 同形）


if __name__ == "__main__":
    main()
    print("[e14 exited main]", flush=True)
