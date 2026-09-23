"""DeepFM integration on c10d/DDP; no device backend or collective reimplementation.

The observed hook is deliberately synchronous, for correctness and measurement.
It does not claim compute/communication overlap or bounded in-process recovery.
"""
from datetime import timedelta
import time

import torch
import torch.distributed as dist


def shard_bounds(global_batch, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid rank/world_size')
    if global_batch % world_size or global_batch // world_size < 2:
        raise ValueError('Require equal local batches >=2 for training BatchNorm')
    local_batch = global_batch // world_size
    return rank * local_batch, (rank + 1) * local_batch


class TrainingGroup:
    def __init__(self, backend, rank, world_size, synchronize, timeout_seconds=60):
        if backend not in ('hccl', 'gloo'):
            raise ValueError('This increment supports hccl or explicit CPU/gloo only')
        if not 0 <= rank < world_size or timeout_seconds <= 0:
            raise ValueError('Invalid rank, world size or timeout')
        if dist.is_initialized():
            raise RuntimeError('Refusing to take ownership of an existing default group')
        self.backend, self.rank, self.world_size = backend, rank, world_size
        self.synchronize = synchronize
        self.state = 'INITIALIZING'
        self.reset_stats()
        dist.init_process_group(backend, rank=rank, world_size=world_size,
                                timeout=timedelta(seconds=timeout_seconds))
        self.state = 'OPEN'
        if str(dist.get_backend()) != backend:
            self.close()
            raise RuntimeError('Actual collective backend differs from requested backend')

    def require_open(self):
        if self.state != 'OPEN':
            raise RuntimeError(f'Communication group is {self.state}')

    def reset_stats(self):
        self.calls = 0
        self.payload_bytes = 0
        self.seconds = 0.0

    def snapshot(self):
        return {'gradient_allreduce_calls': self.calls,
                'gradient_payload_bytes': self.payload_bytes,
                'synchronous_hook_seconds': self.seconds,
                'payload_definition': 'input tensor bytes per rank; NOT link traffic',
                'timing_definition': 'all_reduce + mean + device sync wall time; no overlap'}

    def close(self):
        if self.state == 'CLOSED':
            return
        if dist.is_initialized():
            dist.destroy_process_group()
        self.state = 'CLOSED'


def observed_mean_hook(group, bucket):
    """DDP hook replaces its reduction, so explicitly divide SUM by world size."""
    group.require_open()
    tensor = bucket.buffer()
    start = time.perf_counter()
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(group.world_size)
        group.synchronize()
    except BaseException:
        group.state = 'FAILED'
        raise
    group.calls += 1
    group.payload_bytes += tensor.numel() * tensor.element_size()
    group.seconds += time.perf_counter() - start
    future = torch.futures.Future()
    future.set_result(tensor)
    return future
