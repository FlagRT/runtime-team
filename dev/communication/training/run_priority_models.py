"""Run a priority model on two ranks with a CPU global-batch reference."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from distributed import TrainingGroup, manual_mean_gradients, shard_bounds
from priority_models import MODEL_NAMES, make_batch, make_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODEL_NAMES, required=True)
    p.add_argument('--backend', choices=['cpu', 'ascend', 'kunlun', 'ppu'], required=True)
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--revision', required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rank, size, local = (int(os.environ[x]) for x in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if size != 2 or rank != local or not 1 <= a.steps <= 100:
        raise ValueError('Requires two local ranks and 1–100 steps')
    torch.set_num_threads(1)
    backend, dtype = {'cpu': ('gloo', 'cpu'), 'ascend': ('hccl', 'npu'),
                      'kunlun': ('flagcx', 'cuda'), 'ppu': ('nccl', 'cuda')}[a.backend]
    if a.backend == 'cpu':
        device, sync = torch.device('cpu'), lambda: None
    else:
        if a.runtime_root is None or not (a.runtime_root/'runtime/__init__.py').is_file():
            raise ValueError('Explicit device prototype source required')
        sys.path.insert(0, str(a.runtime_root.resolve()))
        import runtime
        runtime.use(a.backend)
        if (runtime.current().name, runtime.current().device_type) != (a.backend, dtype):
            raise RuntimeError('Unexpected device backend; no fallback')
        if runtime.device_count() != 2:
            raise RuntimeError('Expose exactly two devices before launching')
        runtime.set_device(local)
        device, sync = torch.device(dtype, local), lambda: runtime.synchronize(local, timeout_ms=15000)
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/f'rank{rank}.json').open('x') as output:
        result = {'model': a.model, 'device_backend': a.backend, 'backend': backend,
                  'rank': rank, 'device': str(device), 'torch': torch.__version__,
                  'revision': a.revision, 'data': 'synthetic fixed seed; no model quality claim',
                  'steps': [], 'verdict': 'FAIL'}
        group = None
        try:
            group = TrainingGroup(backend, rank, size, sync, timeout_seconds=40)
            torch.manual_seed(7435)
            ref, objective = make_case(a.model)
            target = copy.deepcopy(ref).to(device)
            ref_opt = torch.optim.SGD(ref.parameters(), lr=0.02)
            opt = torch.optim.SGD(target.parameters(), lr=0.02)
            start, stop = shard_bounds(16, rank, size)
            result['sample_range'] = [start, stop]
            result['parameters'] = sum(param.numel() for param in target.parameters())
            for step in range(a.steps):
                inputs, labels = make_batch(a.model, 16, 32100+step)
                opt.zero_grad(set_to_none=True)
                ref_opt.zero_grad(set_to_none=True)
                full_logits = ref(*inputs)
                ref_loss = objective(full_logits, labels)
                ref_loss.backward()
                logits = target(*(x[start:stop].to(device) for x in inputs))
                loss = objective(logits, labels[start:stop].to(device))
                loss.backward()
                manual_mean_gradients(target, group, fused=True)
                got, expected = logits.detach().cpu(), full_logits.detach()[start:stop]
                torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)
                grad_error = 0.
                for param, ref_param in zip(target.parameters(), ref.parameters(), strict=True):
                    grad = param.grad.detach().cpu()
                    torch.testing.assert_close(grad, ref_param.grad, atol=2e-5, rtol=2e-4)
                    grad_error = max(grad_error, (grad - ref_param.grad).abs().max().item())
                opt.step()
                ref_opt.step()
                sync()
                param_error = 0.
                for param, ref_param in zip(target.parameters(), ref.parameters(), strict=True):
                    value = param.detach().cpu()
                    torch.testing.assert_close(value, ref_param, atol=2e-5, rtol=2e-4)
                    param_error = max(param_error, (value - ref_param).abs().max().item())
                result['steps'].append({'step': step, 'ref_loss': ref_loss.item(),
                  'logit_max_abs': (got-expected).abs().max().item(),
                  'grad_max_abs': grad_error, 'param_max_abs': param_error})
            flat = torch.cat([param.detach().reshape(-1).cpu() for param in target.parameters()])
            result['parameter_sha256'] = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
            result['communication'] = group.snapshot()
            if group.calls != a.steps or group.payload_bytes != a.steps*4*result['parameters']:
                raise RuntimeError('Incorrect gradient communication coverage')
            group.close()
            result['group_state'] = group.state
            result['verdict'] = 'PASS'
        except BaseException as exc:
            result['error'] = repr(exc)
            raise
        finally:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write('\n')


if __name__ == '__main__':
    main()
