import os
import torch
import torch_br  # noqa: F401  触发 sccl 注册（devices=["supa","cpu"]）
import torch.distributed as dist
import torch.multiprocessing as mp

PORT = os.environ.get("DIST_PORT", "29551")

def worker(rank):
    # ⚠️ 不在 worker 内改 BIREN_VISIBLE_DEVICES（容器级注入，进程内改会挂住设备上下文）
    torch_br.supa.set_device(rank)
    dist.init_process_group(backend="sccl", init_method=f"tcp://127.0.0.1:{PORT}",
                            world_size=2, rank=rank)
    x = (torch.arange(8, dtype=torch.float64) + rank * 8).to(f"supa:{rank}")
    dist.all_reduce(x)
    got = x.cpu()
    # 期望：各 rank i+8r 逐元素求和 = i*2 + 8*(0+1) = 2i+8
    exp = torch.arange(8, dtype=torch.float64) * 2 + 8
    if rank == 0:
        ok = bool(torch.allclose(got, exp))
        print("GOT   =", [round(v, 2) for v in got.tolist()], flush=True)
        print("EXPECT=", [round(v, 2) for v in exp.tolist()], flush=True)
        print("ALLREDUCE_OK=", ok, flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    print("BACKEND=sccl devices=[supa,cpu]（torch_br/__init__.py:179 注册）", flush=True)
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("BIREN_DIST_ALLREDUCE_PASS", flush=True)
