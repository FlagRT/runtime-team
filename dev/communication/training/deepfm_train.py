"""DeepFM data-parallel training, with a controlled single-global-batch oracle.

CPU/gloo results are explicitly host-only. Ascend requires the device team's
prototype and uses its public API, never importing vendor extensions here.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from distributed import (TrainingGroup, observed_mean_hook, shard_bounds,
                         ManualDataParallel, manual_mean_gradients)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'ascend'], required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--prototype-revision', default='unrecorded')
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--global-batch', type=int, default=64)
    p.add_argument('--seed', type=int, default=923)
    p.add_argument('--comm-mode', choices=['observed', 'native', 'manual', 'fused'], default='observed')
    p.add_argument('--collective', choices=['hccl', 'flagcx'], default='hccl')
    p.add_argument('--timeout-seconds', type=int, default=60)
    p.add_argument('--save-final-parameters', action='store_true', help='Diagnostic host-side comparison independent of collective backend')
    return p.parse_args()


def batch(field_dims, size, seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.stack([torch.randint(d, (size,), generator=generator)
                     for d in field_dims], dim=1)
    # Deterministic synthetic labels; this is not a CTR quality benchmark.
    y = ((x[:, 0] % 7 + x[:, 1] % 5) > 5).float()
    return x, y


def flat_parameters(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def rank_spread(model, group):
    value = flat_parameters(model)
    reference = value.clone()
    group.collective(dist.broadcast, reference, src=0)
    difference = (value - reference).abs().max()
    group.collective(dist.all_reduce, difference, op=dist.ReduceOp.MAX)
    return float(difference.item())


def wrap(model, group, device, mode):
    if mode in ('manual', 'fused'):
        return ManualDataParallel(model, group)
    kwargs = {'device_ids': [device.index], 'output_device': device.index} if device.type != 'cpu' else {}
    wrapped = DDP(model, broadcast_buffers=True, **kwargs)
    if mode == 'observed':
        wrapped.register_comm_hook(group, observed_mean_hook)
    return wrapped


def run(args):
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if args.steps < 1 or args.warmup < 0:
        raise ValueError('Require positive steps, nonnegative warmup and equal local batches >=2')
    if args.collective == 'flagcx' and (args.backend != 'ascend' or args.comm_mode not in ('manual', 'fused')):
        raise ValueError('v3 FlagCX requires Ascend + manual/fused mode; native DDP is unsupported')
    shard_start, shard_stop = shard_bounds(args.global_batch, rank, world)
    torch.set_num_threads(1)
    runtime = None
    if args.backend == 'ascend':
        if args.runtime_root is None:
            raise ValueError('Ascend requires --runtime-root')
        if args.prototype_revision == 'unrecorded':
            raise ValueError('Record the prototype commit with --prototype-revision')
        sys.path.insert(0, str(args.runtime_root.resolve()))
        import runtime
        runtime.use('ascend')
        if runtime.device_count() <= local_rank:
            raise RuntimeError('Insufficient visible NPU devices')
        runtime.set_device(local_rank)  # Triggers vendor backend registration before c10d.
        device = torch.device(runtime.current().device_type, local_rank)
        if device.type != 'npu':
            raise RuntimeError('Ascend must not silently fall back to CPU')
        sync = lambda: runtime.synchronize(local_rank, timeout_ms=30000)
        backend = args.collective
    else:
        device, backend = torch.device('cpu'), 'gloo'
        sync = lambda: None
    sys.path.insert(0, str(args.model_root.resolve()))
    from build import build_model, DEFAULT_CONFIG
    model_hashes = {f: hashlib.sha256((args.model_root / f).read_bytes()).hexdigest()
                    for f in ['model.py', 'config.py', 'build.py']}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f'rank{rank}.json'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    group = TrainingGroup(backend, rank, world, sync, args.timeout_seconds)
    local_batch = args.global_batch // world
    sl = slice(shard_start, shard_stop)
    result = {'backend': backend, 'device': str(device), 'rank': rank, 'world_size': world,
              'torch': torch.__version__, 'prototype_revision': args.prototype_revision,
              'model_sha256': model_hashes, 'comm_mode': args.comm_mode,
              'global_batch': args.global_batch, 'local_batch': local_batch,
              'dtype': 'float32', 'dataset': 'deterministic synthetic; not recommendation accuracy',
              'seed': args.seed, 'warmup_steps': args.warmup, 'measured_steps': args.steps,
              'evidence_scope': 'NPU integration' if runtime else 'CPU/gloo host-only'}
    try:
        if runtime:
            result['runtime_info'] = runtime.current().info()
        # Oracle: disable BN updates/dropout only in this controlled test.
        torch.manual_seed(args.seed)
        cpu = build_model()
        model = deepcopy(cpu).to(device).eval()
        cpu.eval()
        wrapped = wrap(model, group, device, args.comm_mode)
        x, y = batch(DEFAULT_CONFIG.field_dims, args.global_batch, args.seed + 1)
        reference_loss = torch.nn.functional.binary_cross_entropy(cpu(x), y)
        reference_loss.backward()
        loss = torch.nn.functional.binary_cross_entropy(wrapped(x[sl].to(device)), y[sl].to(device))
        loss.backward()
        if args.comm_mode in ('manual', 'fused'):
            manual_mean_gradients(model, group, args.comm_mode == 'fused')
        sync()
        comparisons = []
        for (name, p), (_, q) in zip(model.named_parameters(), cpu.named_parameters()):
            observed = p.grad.detach().cpu()
            comparisons.append({'name': name, 'max_abs': float((observed-q.grad).abs().max()),
                                'pass': bool(torch.allclose(observed,q.grad,rtol=2e-4,atol=2e-5))})
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        torch.optim.SGD(cpu.parameters(), lr=0.01).step()
        parameters = flat_parameters(model).cpu()
        expected = flat_parameters(cpu)
        result['oracle'] = {'mode': 'eval with autograd: BN frozen, dropout disabled',
                            'gradient_checks': comparisons,
                            'update_max_abs': float((parameters-expected).abs().max()),
                            'update_pass': bool(torch.allclose(parameters,expected,rtol=2e-4,atol=2e-5)),
                            'rank_parameter_max_abs': rank_spread(model, group)}
        if not all(c['pass'] for c in comparisons) or not result['oracle']['update_pass']:
            raise AssertionError('Global-batch gradient/update oracle failed')
        del wrapped, model, cpu
        # Fresh initialization: real train mode includes local BN and dropout.
        torch.manual_seed(args.seed)
        model = build_model().to(device).train()
        wrapped = wrap(model, group, device, args.comm_mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses, elapsed, sample_ids = [], [], []
        for step in range(args.warmup + args.steps):
            x, y = batch(DEFAULT_CONFIG.field_dims, args.global_batch, args.seed + 100 + step)
            if step == args.warmup:
                group.reset_stats()
            sync()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = wrapped(x[sl].to(device))
            loss = torch.nn.functional.binary_cross_entropy(prediction, y[sl].to(device))
            loss.backward()
            if args.comm_mode in ('manual', 'fused'):
                manual_mean_gradients(model, group, args.comm_mode == 'fused')
            optimizer.step()
            sync()
            seconds = time.perf_counter() - start
            finite = torch.tensor([int(torch.isfinite(loss).item())],device=device,dtype=torch.int32)
            group.collective(dist.all_reduce, finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise AssertionError('Non-finite training loss')
            if step >= args.warmup:
                losses.append(float(loss.item()))
                elapsed.append(seconds)
                sample_ids.append([step * args.global_batch + sl.start, step * args.global_batch + sl.stop])
        # Wall-time maximum across ranks, not the sum of per-rank throughput.
        duration = torch.tensor([sum(elapsed)],device=device,dtype=torch.float32)
        group.collective(dist.all_reduce, duration, op=dist.ReduceOp.MAX)
        spread = rank_spread(model, group)
        host_parameters = flat_parameters(model).cpu().numpy()
        result['final_parameter_sha256'] = hashlib.sha256(host_parameters.tobytes()).hexdigest()
        if args.save_final_parameters:
            import numpy as np
            with (args.output / f'rank{rank}-parameters.npy').open('xb') as output:
                np.save(output, host_parameters, allow_pickle=False)
        result['training'] = {'mode': 'train: local BatchNorm + dropout=0.2',
                              'losses': losses, 'step_seconds': elapsed,
                              'sample_id_ranges': sample_ids,
                              'parameter_count': sum(p.numel() for p in model.parameters()),
                              'rank_parameter_max_abs': spread,
                              'max_rank_measured_seconds': float(duration.item()),
                              'global_samples_per_second': args.global_batch*args.steps/float(duration.item()),
                              'gradient_communication': group.snapshot() if args.comm_mode != 'native' else
                                  {'observed': False, 'note': 'Native DDP communication is not instrumented'},
                              'bn_note': 'local minibatch statistics; buffers need not match at exit; '
                                         'not equivalent to single-device train-mode BatchNorm'}
        result['training']['parameters_finite'] = bool(torch.isfinite(flat_parameters(model)).all().item())
        if not result['training']['parameters_finite'] or spread > 1e-6:
            raise AssertionError('Parameters non-finite or different across ranks')
        if runtime:
            result['memory_stats_at_end'] = runtime.memory_stats(local_rank)
        result['verdict'] = 'PASS'
    except BaseException as exc:
        result['verdict'] = 'FAIL'
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if runtime:
            fe = runtime.translate_error(exc, location='communication/deepfm_train')
            result['error']['classification'] = {'category': str(fe.category), 'disposition': fe.disposition,
                                                 'mapped': fe.mapped, 'graded_by': fe.graded_by}
        raise
    finally:
        result['group_state'] = group.state
        result['teardown'] = 'pending'
        # Known FlagCX native cleanup abort cannot be caught by Python. Persist
        # functional evidence BEFORE teardown; it is not an overall PASS until
        # launcher exit and CLOSED state have independently been checked.
        with target.open('x') as f:
            json.dump(result,f,indent=2,ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps({'rank':rank,'functional_verdict':result.get('verdict'),
                          'error':result.get('error'), 'teardown':'pending'}),flush=True)
        try:
            group.close()
            result['group_state'] = group.state
            result['teardown'] = 'closed'
        except BaseException as cleanup_error:
            result['teardown'] = 'failed'
            result['teardown_error'] = str(cleanup_error)
            raise
        finally:
            temporary = target.with_suffix('.tmp')
            with temporary.open('x') as f:
                json.dump(result,f,indent=2,ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            temporary.replace(target)
    print(json.dumps({'rank':rank,'verdict':result['verdict'],'output':str(target)}),flush=True)


if __name__ == '__main__':
    run(parse_args())
