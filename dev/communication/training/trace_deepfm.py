"""Diagnostic snapshots, not a performance benchmark; no collective-based oracle."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from deepfm_train import batch
from distributed import TrainingGroup, ManualDataParallel, manual_mean_gradients


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--collective', choices=['hccl', 'flagcx'], required=True)
    p.add_argument('--mode', choices=['manual', 'fused'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.steps <= 10:
        p.error('Diagnostic scope is 1..10 steps')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 2:
        p.error('This diagnostic requires exactly two ranks')
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use('ascend')
    runtime.set_device(local)
    device = torch.device(runtime.current().device_type, local)
    if device.type != 'npu':
        raise RuntimeError('No CPU fallback')
    sync = lambda: runtime.synchronize(local, timeout_ms=30000)
    sys.path.insert(0, args.model_root)
    from build import build_model, DEFAULT_CONFIG
    out = args.output / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    group = TrainingGroup(args.collective, rank, world, sync, 60)
    meta = {'rank': rank, 'backend': args.collective, 'mode': args.mode,
            'steps': args.steps, 'scope': 'diagnostic; synthetic inputs; no timing',
            'teardown': 'pending', 'snapshots': []}

    def save(label, tensors):
        sync()
        arrays = {name: value.detach().float().cpu().numpy()
                  for name, value in tensors}
        with (out / (label + '.npz')).open('xb') as f:
            np.savez(f, **arrays)
        meta['snapshots'].append(label)
        (out / 'trace.json').write_text(json.dumps(meta, indent=2) + '\n')

    try:
        torch.manual_seed(923)
        torch.set_num_threads(1)
        model = build_model().to(device).train()
        wrapped = ManualDataParallel(model, group)
        save('initial', model.named_parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(args.steps):
            x, y = batch(DEFAULT_CONFIG.field_dims, 64, 1023 + step)
            sl = slice(rank * 32, (rank + 1) * 32)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy(wrapped(x[sl].to(device)), y[sl].to(device))
            loss.backward()
            save(f'{step:02d}-local-grad', ((n, v.grad) for n, v in model.named_parameters()))
            manual_mean_gradients(model, group, args.mode == 'fused')
            save(f'{step:02d}-reduced-grad', ((n, v.grad) for n, v in model.named_parameters()))
            optimizer.step()
            save(f'{step:02d}-parameter', model.named_parameters())
        meta['body'] = 'completed; host comparison required'
    finally:
        (out / 'trace.json').write_text(json.dumps(meta, indent=2) + '\n')
        group.close()
        meta['teardown'] = group.state
        (out / 'trace.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
