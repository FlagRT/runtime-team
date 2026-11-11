import os
import torch
import torch_br  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank):
    dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29561",
                            world_size=2, rank=rank)
    x = torch.arange(4, dtype=torch.float64) + rank
    dist.all_reduce(x)
    if rank == 0:
        exp = torch.tensor([1.0, 3.0, 5.0, 7.0])
        print("CPU_GOT=", x.tolist(), "EXPECT=", exp.tolist(), "OK=", bool(torch.allclose(x, exp)), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("BIREN_DIST_CPU_ALLREDUCE_PASS", flush=True)
