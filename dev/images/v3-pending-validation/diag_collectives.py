import os, json
import torch, torch_npu
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)
dist.init_process_group(backend="flagcx")
rank = dist.get_rank()
ws = dist.get_world_size()
dev = torch.device("npu", local_rank)
result = {"rank": rank}

# 1. all_gather (tensor list)
try:
    t = torch.full((4,), float(rank + 1), device=dev)
    out = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(out, t)
    result["all_gather"] = [o.cpu().tolist() for o in out]
except Exception as e:
    result["all_gather_error"] = repr(e)

# 2. broadcast
try:
    t2 = torch.full((4,), float(rank + 1), device=dev)
    dist.broadcast(t2, src=0)
    result["broadcast_from_rank0"] = t2.cpu().tolist()
except Exception as e:
    result["broadcast_error"] = repr(e)

# 3. all_gather_object (what DDP's _verify_param_shape likely uses internally -- uses c10d allgather via pickled tensor sizes)
try:
    objs = [None] * ws
    dist.all_gather_object(objs, {"rank": rank, "n": 338})
    result["all_gather_object"] = objs
except Exception as e:
    result["all_gather_object_error"] = repr(e)

print(json.dumps(result), flush=True)
dist.barrier()
dist.destroy_process_group()
