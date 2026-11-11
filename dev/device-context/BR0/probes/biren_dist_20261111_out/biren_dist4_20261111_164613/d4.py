import os, sys
import torch
import torch_br  # noqa: F401
import torch.distributed as dist

mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29571",
                        world_size=1, rank=0)
if mode == "supa":
    x = (torch.arange(8, dtype=torch.float64)).to("supa:0")
    dist.all_reduce(x)
    print("SUPA_W1_OK got=", [round(v, 2) for v in x.cpu().tolist()], flush=True)
elif mode == "cpu":
    try:
        x = torch.arange(8, dtype=torch.float64)
        dist.all_reduce(x)
        print("CPU_OK got=", [round(v, 2) for v in x.tolist()], flush=True)
    except Exception as e:
        print("CPU_FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:100]), flush=True)
dist.destroy_process_group()
print("DONE", flush=True)
