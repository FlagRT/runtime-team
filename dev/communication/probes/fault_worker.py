"""Deliberately exit rank 0 after a verified real device collective.

Use only with the process supervisor; rank 1 may block and must be reaped.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import timedelta

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--iterations", type=int, default=1)  # supervisor compatibility
p.add_argument("--out-dir", type=Path, required=True)
args = p.parse_args()
rank = int(os.environ["LOCAL_RANK"])
torch.flagos.set_device(rank)
dist.init_process_group("flagos", timeout=timedelta(seconds=20))
assert dist.get_world_size() == 2
value = torch.full((8,), rank + 1, dtype=torch.float32, device=f"flagos:{rank}")
torch.flagos.synchronize()
dist.all_reduce(value)
torch.flagos.synchronize()
assert bool(torch.all(value.cpu() == 3).item())
args.out_dir.mkdir(parents=True, exist_ok=True)
(args.out_dir / f"fault_marker_rank{rank}.json").write_text(json.dumps({
    "rank": rank, "run_id": os.environ.get("COMM_RUN_ID"),
    "collective_verified": True, "fault": "rank0_exit_42"}))
if rank == 0:
    os._exit(42)
dist.all_reduce(value)
torch.flagos.synchronize()
raise RuntimeError("peer exit was not detected")
