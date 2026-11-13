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


class ObservedBucketTrace:
    """Record real GradBucket callbacks, not version-dependent DDP log guesses."""
    def __init__(self):
        self.steps = []
        self.active = None

    def begin(self, step):
        if self.active is not None or type(step) is not int or step != len(self.steps):
            raise ValueError('Measured backward steps must be sequential and non-overlapping')
        self.active = {'step': step, 'buckets': []}

    def record(self, bucket, tensor):
        if self.active is None:
            return  # Oracle and warmup callbacks are not measured samples.
        index = bucket.index()
        size = tensor.numel() * tensor.element_size()
        if (type(index) is not int or index < 0 or size <= 0
                or any(item['index'] == index for item in self.active['buckets'])):
            raise ValueError('Invalid or duplicate GradBucket in one backward')
        self.active['buckets'].append({'index': index, 'bytes': size})

    def end(self):
        if self.active is None or not self.active['buckets']:
            raise ValueError('Backward has no completed bucket observations')
        self.active['buckets'].sort(key=lambda item: item['index'])
        self.steps.append(self.active)
        self.active = None

    def sizes(self):
        if self.active is not None or not self.steps:
            raise ValueError('Incomplete bucket trace')
        first = self.steps[0]['buckets']
        if any(step['buckets'] != first for step in self.steps):
            raise ValueError('Measured fixed-shape bucket plan changed')
        return [item['bytes'] for item in first]


class TrainingGroup:
    def __init__(self, backend, rank, world_size, synchronize, timeout_seconds=60):
        if backend not in ('hccl', 'gloo', 'flagcx'):
            raise ValueError('Supported backends: hccl, gloo, flagcx')
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

    def collective(self, function, tensor, **kwargs):
        """Complete device work before consuming/reusing collective tensors.

        v3 FlagCX may return before writes complete even with async_op=False.
        A device recovery is NOT a process-group recovery; failure is terminal.
        """
        self.require_open()
        try:
            # Producer kernels (backward/clone/packing) may still be queued.
            # This conservative baseline does not assume FlagCX stream ordering.
            self.synchronize()
            function(tensor, **kwargs)
            self.synchronize()
        except BaseException:
            self.state = 'FAILED'
            raise

    def reset_stats(self):
        self.calls = 0
        self.payload_bytes = 0
        self.seconds = 0.0
        self.bucket_trace = ObservedBucketTrace()

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
        # Complete bucket production before the vendor collective consumes it.
        group.synchronize()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(group.world_size)
        group.synchronize()
        group.bucket_trace.record(bucket, tensor)
    except BaseException:
        group.state = 'FAILED'
        raise
    group.calls += 1
    group.payload_bytes += tensor.numel() * tensor.element_size()
    group.seconds += time.perf_counter() - start
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def manual_mean_gradients(model, group, fused=False, bucket_bytes=None):
    """Dense, same-dtype DeepFM gradients only; no DDP, list gather or overlap.

    All ranks must execute the same model/parameter order. Not a general unused
    parameter or uneven-input implementation. Packing/copying are timed.
    """
    group.require_open()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not gradients or any(g is None or g.is_sparse for g in gradients):
        raise ValueError('Manual reducer requires all dense gradients on all ranks')
    if len({(g.dtype, g.device) for g in gradients}) != 1:
        raise ValueError('Manual reducer requires one dtype/device')
    if bucket_bytes is not None:
        if fused:
            raise ValueError('Choose full fusion or bounded buckets, not both')
        if (isinstance(bucket_bytes, bool) or not isinstance(bucket_bytes, int)
                or bucket_bytes < gradients[0].element_size()):
            raise ValueError('Bucket must hold at least one element')
        if any(not g.is_contiguous() for g in gradients):
            raise ValueError('Bounded reducer requires contiguous gradients')
        return _bounded_mean_gradients(gradients, group, bucket_bytes)
    start = time.perf_counter()
    tensors = [torch.cat([g.reshape(-1) for g in gradients])] if fused else gradients
    for tensor in tensors:
        group.collective(dist.all_reduce, tensor, op=dist.ReduceOp.SUM)
        tensor.div_(group.world_size)
        group.calls += 1
        group.payload_bytes += tensor.numel() * tensor.element_size()
    if fused:
        offset = 0
        for gradient in gradients:
            gradient.copy_(tensors[0][offset:offset + gradient.numel()].view_as(gradient))
            offset += gradient.numel()
    try:
        group.synchronize()
    except BaseException:
        group.state = 'FAILED'
        raise
    group.seconds += time.perf_counter() - start


def _bounded_mean_gradients(gradients, group, bucket_bytes):
    """Stream through one reusable buffer, splitting even oversized parameters.

    Metadata holds views only. Completion before buffer reuse is mandatory;
    no communication/computation overlap is claimed. Cap excludes model grads.
    """
    start = time.perf_counter()
    capacity = min(bucket_bytes // gradients[0].element_size(),
                   sum(g.numel() for g in gradients))
    if capacity == 0:
        raise ValueError('Empty gradients')
    buffer = torch.empty(capacity, dtype=gradients[0].dtype, device=gradients[0].device)
    pieces, used = [], 0

    def flush(count):
        tensor = buffer[:count]
        group.collective(dist.all_reduce, tensor, op=dist.ReduceOp.SUM)
        tensor.div_(group.world_size)
        offset = 0
        for piece in pieces:
            piece.copy_(tensor[offset:offset + piece.numel()])
            offset += piece.numel()
        # Wait for unpack before the same buffer is overwritten.
        group.synchronize()
        group.calls += 1
        group.payload_bytes += count * tensor.element_size()
        pieces.clear()

    try:
        for gradient in gradients:
            flat, offset = gradient.view(-1), 0
            while offset < flat.numel():
                count = min(capacity - used, flat.numel() - offset)
                piece = flat[offset:offset + count]
                buffer[used:used + count].copy_(piece)
                pieces.append(piece)
                used += count
                offset += count
                if used == capacity:
                    flush(used)
                    used = 0
        if used:
            flush(used)
    except BaseException:
        group.state = 'FAILED'
        raise
    group.seconds += time.perf_counter() - start


class ManualDataParallel(torch.nn.Module):
    """Explicit initialization/buffer broadcasts; gradients reduced by caller."""
    def __init__(self, model, group):
        super().__init__()
        self.module, self.group = model, group
        for tensor in model.state_dict().values():
            group.collective(dist.broadcast, tensor, src=0)

    def forward(self, *args, **kwargs):
        # Match DDP's rank-zero buffer broadcast, not synchronized BatchNorm.
        for tensor in self.module.buffers():
            self.group.collective(dist.broadcast, tensor, src=0)
        return self.module(*args, **kwargs)
