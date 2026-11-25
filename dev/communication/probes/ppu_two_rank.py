"""PPU identity, device arithmetic and exact NCCL tensor checks in two ranks."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


def normalize_uuid(uuid):
    return str(uuid).strip().lower().removeprefix('gpu-')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True, type=Path)
    p.add_argument('--expected-uuid', action='append', required=True)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 2 or rank != local or len(set(a.expected_uuid)) != 2:
        raise RuntimeError('Require the two expected local PPU devices')
    sys.path[:0] = [str(a.runtime_root.resolve()), str(Path(__file__).resolve().parents[1]/'training')]
    import runtime
    from distributed import TrainingGroup

    runtime.use('ppu')
    backend = runtime.current()
    if (backend.name, backend.device_type, runtime.device_count()) != ('ppu', 'cuda', 2):
        raise RuntimeError('Expected exactly two PPU devices, no backend fallback')
    visible = [torch.cuda.get_device_properties(i) for i in range(2)]
    visible_uuids = [normalize_uuid(prop.uuid) for prop in visible]
    if (set(visible_uuids) != {normalize_uuid(u) for u in a.expected_uuid}
            or not all('PPU' in prop.name.upper() for prop in visible)):
        raise RuntimeError('PPU UUID/vendor identity does not match the host reservation')
    runtime.set_device(local)
    device = torch.device('cuda', local)
    sync = lambda: runtime.synchronize(local, timeout_ms=15000)
    x = torch.ones((32, 32), dtype=torch.float32, device=device)
    y = x @ x
    event = torch.cuda.Event()
    event.record()
    sync()
    if not event.query() or not torch.equal(y.cpu(), torch.full((32, 32), 32.)):
        raise RuntimeError('PPU device arithmetic/event probe failed')
    print(f'rank {rank}: device probe passed', flush=True)
    group = TrainingGroup('nccl', rank, world, sync, timeout_seconds=30)
    counts = {'all_reduce': 0, 'all_gather': 0}
    try:
        for elements in (8, 2048):
            for i in range(5):
                data = torch.full((elements,), float(rank + i + 1), device=device)
                group.collective(dist.all_reduce, data)
                if not torch.equal(data.cpu(), torch.full((elements,), float(2*i + 3))):
                    raise RuntimeError('PPU AllReduce result mismatch')
                counts['all_reduce'] += 1
                data = torch.full((elements,), float(rank + i), device=device)
                peers = [torch.empty_like(data) for _ in range(world)]
                group.collective(lambda t: dist.all_gather(peers, t), data)
                if not all(torch.equal(peer.cpu(), torch.full((elements,), float(i+j)))
                           for j, peer in enumerate(peers)):
                    raise RuntimeError('PPU AllGather result mismatch')
                counts['all_gather'] += 1
    finally:
        group.close()
    result = dict(rank=rank, world_size=world, device=str(device), vendor='ppu', backend='nccl',
                  torch=torch.__version__, visible_uuid_order=visible_uuids,
                  local_uuid=visible_uuids[local], device_probe='PASS',
                  checks=counts, group_state=group.state, verdict='PASS')
    with (a.output/f'rank{rank}.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(f'rank {rank}: {counts}; group {group.state}', flush=True)


if __name__ == '__main__':
    main()
