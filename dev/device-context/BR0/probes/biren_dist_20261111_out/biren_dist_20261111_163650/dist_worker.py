import os, sys
import torch
import torch_br  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

BACKEND = os.environ.get("DIST_BACKEND", "sccl")
MODE = os.environ.get("DIST_MODE", "allreduce")

def worker(rank, world):
    os.environ["BIREN_VISIBLE_DEVICES"] = "0,1"
    torch_br.supa.set_device(rank)
    init = dist.init_process_group(
        backend=BACKEND,
        init_method=f"tcp://127.0.0.1:295{abs(hash(BACKEND+MODE)) % 90 + 10}",
        world_size=world, rank=rank,
    )
    t = torch.arange(8, dtype=torch.float64) + rank * 8
    t = t.to(f"supa:{rank}")
    dist.all_reduce(t)
    t2 = t.cpu()
    exp = sum(float(i % world) * 8 + i for i in range(8)) if MODE == "allreduce" else None
    if rank == 0:
        ok = exp is not None and abs(float(t2.sum()) - sum(float(v) for v in [sum(range(8)) + r * 64 for r in range(world)])) < 1e-6
        print("ALLREDUCE_VALUE=", [round(float(v), 2) for v in t2.tolist()], flush=True)
        print("ALLREDUCE_SUM=", round(float(t2.sum()), 2), "EXPECT=", round(sum(sum(range(8)) + r * 64 for r in range(world)), 2), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    if MODE == "allreduce":
        print("BIREN_DIST_ALLREDUCE_PASS", flush=True)
