"""Two-rank TP/PP execution examples, not a new partitioning framework.

Consumes the public device prototype and existing TrainingGroup. The supplied
plan fixes layer placement: TP splits hidden channels, PP places one layer on
each rank. PP is a serialized microbatch correctness baseline, not an overlap
or throughput claim. All data are synthetic; CPU autograd is the reference.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

from distributed import TrainingGroup
from deepfm_train import bind_runtime_device


@dataclass(frozen=True)
class MinimalPlan:
    mode: str
    input_dim: int = 16
    hidden_dim: int = 32
    output_dim: int = 8
    batch_size: int = 16
    microbatches: int = 2

    def validate(self, world_size=2):
        if self.mode not in ('tp', 'pp') or world_size != 2:
            raise ValueError('Minimal examples require TP/PP and exactly two ranks')
        dimensions = (self.input_dim, self.hidden_dim, self.output_dim,
                      self.batch_size, self.microbatches)
        if any(type(v) is not int or v <= 0 for v in dimensions):
            raise ValueError('Dimensions and microbatch count must be positive integers')
        if self.hidden_dim % world_size or self.batch_size % self.microbatches:
            raise ValueError('Require divisible hidden channels and equal microbatches')
        return self

    def hidden_slice(self, rank):
        self.validate()
        if rank not in (0, 1):
            raise ValueError('Invalid rank')
        width = self.hidden_dim // 2
        return slice(rank * width, (rank + 1) * width)


class Transport:
    """Completion before consuming/reusing buffers; no async claims."""
    def __init__(self, group):
        self.group = group
        self.calls = {'all_reduce': 0, 'send': 0, 'recv': 0}
        self.input_bytes = {key: 0 for key in self.calls}

    def call(self, operation, tensor, **kwargs):
        if operation not in self.calls:
            raise ValueError('Unknown transport operation')
        self.group.collective(getattr(dist, operation), tensor, **kwargs)
        self.calls[operation] += 1
        self.input_bytes[operation] += tensor.numel() * tensor.element_size()


def initial_weights(plan, seed):
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(plan.hidden_dim, plan.input_dim, generator=generator) / 8
    second = torch.randn(plan.output_dim, plan.hidden_dim, generator=generator) / 8
    return first, second


def step_data(plan, seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(plan.batch_size, plan.input_dim, generator=generator),
            torch.randn(plan.batch_size, plan.output_dim, generator=generator) / 4)


def reference(first, second, x, target):
    a = first.detach().clone().requires_grad_(True)
    b = second.detach().clone().requires_grad_(True)
    output = F.linear(F.relu(F.linear(x, a)), b)
    loss = F.mse_loss(output, target)
    loss.backward()
    return output.detach(), a.grad, b.grad, float(loss.detach())


def max_abs(actual, expected):
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError('Mismatched shape or nonfinite tensor')
    return float((actual.detach().cpu() - expected.detach().cpu()).abs().max())


def tp_step(plan, rank, first, second, x, target, transport):
    preactivation = F.linear(x, first)
    hidden = F.relu(preactivation)
    output = F.linear(hidden, second)
    transport.call('all_reduce', output, op=dist.ReduceOp.SUM)
    # Each rank owns different parameter shards. Do not divide this sum by 2.
    delta = 2 * (output - target) / target.numel()
    grad_second = delta.T @ hidden
    grad_first = ((delta @ second) * (preactivation > 0)).T @ x
    return output, grad_first, grad_second


def pp_step(plan, rank, weight, x, target, transport):
    count = plan.batch_size // plan.microbatches
    gradient = torch.zeros_like(weight)
    outputs = []
    for microbatch in range(plan.microbatches):
        section = slice(microbatch * count, (microbatch + 1) * count)
        if rank == 0:
            preactivation = F.linear(x[section], weight)
            activation = F.relu(preactivation).contiguous()
            transport.call('send', activation, dst=1)
            backward = torch.empty_like(activation)
            transport.call('recv', backward, src=1)
            gradient.add_((backward * (preactivation > 0)).T @ x[section])
        else:
            activation = torch.empty(count, plan.hidden_dim, device=weight.device)
            transport.call('recv', activation, src=0)
            output = F.linear(activation, weight)
            # Normalize by the full batch, not each microbatch independently.
            delta = 2 * (output - target[section]) / target.numel()
            gradient.add_(delta.T @ activation)
            transport.call('send', (delta @ weight).contiguous(), dst=0)
            outputs.append(output)
    return (torch.cat(outputs) if outputs else None), gradient


def run(args):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    plan = MinimalPlan(args.mode, hidden_dim=args.hidden_dim,
                       microbatches=args.microbatches).validate(world)
    if args.steps < 1 or args.steps > 1000:
        raise ValueError('Bounded step count required')
    torch.set_num_threads(1)
    runtime = None
    if args.backend == 'cpu':
        device, backend, sync = torch.device('cpu'), 'gloo', lambda: None
    else:
        if args.runtime_root is None or not args.prototype_revision:
            raise ValueError('Accelerators require explicit prototype path and revision')
        sys.path.insert(0, str(args.runtime_root.resolve()))
        import runtime
        device = bind_runtime_device(runtime, args.backend, local_rank)
        backend = args.collective
        if args.backend == 'kunlun' and backend != 'flagcx':
            raise ValueError('Kunlun requires FlagCX')
        sync = lambda: runtime.synchronize(local_rank, timeout_ms=30000)
    args.output.mkdir(parents=True, exist_ok=True)
    target_path = args.output / f'rank{rank}.json'
    if target_path.exists():
        raise FileExistsError(target_path)
    result = {
        'schema': 'communication.minimal_parallel/v1', 'rank': rank,
        'world_size': world, 'device_vendor': args.backend,
        'backend': backend, 'torch': torch.__version__, 'plan': asdict(plan),
        'prototype_revision': args.prototype_revision,
        'steps': args.steps, 'seed': args.seed,
        'scope': 'host-only' if args.backend == 'cpu' else 'accelerator integration',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'verdict': 'FAIL', 'checks': [],
        'limitations': 'Synthetic two-layer FP32 example. PP microbatches serialized; no overlap, model-scale throughput, multi-node, checkpoint or pool integration claim.',
    }
    group = None
    code = 1
    try:
        group = TrainingGroup(backend, rank, world, sync, timeout_seconds=60)
        transport = Transport(group)
        if runtime:
            result['runtime_info'] = runtime.current().info()
        first_cpu, second_cpu = initial_weights(plan, args.seed)
        section = plan.hidden_slice(rank)
        if plan.mode == 'tp':
            first = first_cpu[section].to(device).clone()
            second = second_cpu[:, section].to(device).contiguous().clone()
        else:
            weight = (first_cpu if rank == 0 else second_cpu).to(device).clone()
        # Force collective initialization on both ranks before point-to-point.
        ready = torch.ones(1, device=device)
        transport.call('all_reduce', ready, op=dist.ReduceOp.SUM)
        if float(ready.item()) != world:
            raise ValueError('Communication group warm-up mismatch')
        for step in range(args.steps):
            x_cpu, y_cpu = step_data(plan, args.seed + step + 1)
            expected, grad_a, grad_b, reference_loss = reference(first_cpu, second_cpu, x_cpu, y_cpu)
            x, y = x_cpu.to(device), y_cpu.to(device)
            forward_error = 0.0
            if plan.mode == 'tp':
                output, da, db = tp_step(plan, rank, first, second, x, y, transport)
                comparisons = [(output, expected), (da, grad_a[section]), (db, grad_b[:, section])]
                first.sub_(0.01 * da)
                second.sub_(0.01 * db)
            else:
                output, gradient = pp_step(plan, rank, weight, x, y, transport)
                comparisons = [(gradient, grad_a if rank == 0 else grad_b)]
                if output is not None:
                    comparisons.append((output, expected))
                    forward_error = max_abs(output, expected)
                weight.sub_(0.01 * gradient)
            for actual, wanted in comparisons:
                torch.testing.assert_close(actual.detach().cpu(), wanted,
                                           atol=1e-5, rtol=1e-4)
            first_cpu.sub_(0.01 * grad_a)
            second_cpu.sub_(0.01 * grad_b)
            parameter_pairs = ([(first, first_cpu[section]), (second, second_cpu[:, section])]
                               if plan.mode == 'tp' else [(weight, first_cpu if rank == 0 else second_cpu)])
            for actual, wanted in parameter_pairs:
                torch.testing.assert_close(actual.detach().cpu(), wanted, atol=1e-5, rtol=1e-4)
            if output is not None:
                forward_error = max_abs(output, expected)
            gradient_error = max(max_abs(actual, wanted) for actual, wanted in comparisons)
            parameter_error = max(max_abs(actual, wanted) for actual, wanted in parameter_pairs)
            result['checks'].append({'step': step, 'reference_loss': reference_loss,
                                     'forward_max_abs': forward_error,
                                     'tensor_check_max_abs': gradient_error,
                                     'parameter_max_abs': parameter_error})
        result['transport_calls'] = transport.calls
        result['transport_input_bytes'] = transport.input_bytes
        result['verdict'] = 'PASS'
        code = 0
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if runtime:
            result['translated_error'] = str(runtime.translate_error(exc, location='communication/parallel_minimal'))
        raise
    finally:
        if group is not None:
            try:
                group.close()
            except BaseException as exc:
                result['verdict'] = 'FAIL'
                result['cleanup_error'] = repr(exc)
                code = 1
            result['group_state'] = group.state
        target_path.write_text(json.dumps(result, indent=2) + '\n')
    return code


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['tp', 'pp'], required=True)
    parser.add_argument('--backend', choices=['cpu', 'ascend', 'kunlun'], required=True)
    parser.add_argument('--collective', choices=['hccl', 'flagcx'], default='flagcx')
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--prototype-revision', default='')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--microbatches', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    raise SystemExit(run(parse_args()))
