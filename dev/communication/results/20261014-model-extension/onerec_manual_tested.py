"""Real OneRec-8B full-parameter SGD smoke, synthetic tokens, NOT quality training.

Two-rank explicit gradient averaging. No DDP, PEFT, model shrinking or CPU fallback.
SGD without momentum bounds optimizer memory; this is not an AdamW benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from distributed import TrainingGroup, manual_mean_gradients


def fingerprint(model):
    result = {}
    for name, p in model.named_parameters():
        raw = p.detach().cpu().contiguous().view(torch.uint8).numpy()
        result[name] = hashlib.sha256(raw.tobytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--collective', choices=['hccl', 'flagcx'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.steps <= 10:
        parser.error('Smoke test requires 1..10 steps')
    rank, local, world = [int(os.environ[k]) for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']]
    if world != 2:
        parser.error('Exactly two ranks required')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f'rank{rank}.json'
    if target.exists():
        raise FileExistsError(target)
    result = {'rank': rank, 'backend': args.collective, 'model': 'OpenOneRec/OneRec-8B',
              'revision': '29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff',
              'scope': 'real checkpoint, full-parameter SGD smoke, synthetic tokens; not recommendation quality',
              'dtype': 'bfloat16', 'local_batch': 1, 'global_batch': 2, 'seq_len': 16,
              'lr': 0.01, 'optimizer': 'SGD without momentum', 'steps': [], 'teardown': 'pending'}
    group = None
    try:
        sys.path.insert(0, args.runtime_root)
        import runtime
        runtime.use('ascend')
        runtime.set_device(local)
        device = torch.device(runtime.current().device_type, local)
        if device.type != 'npu':
            raise RuntimeError('No CPU fallback')
        sync = lambda: runtime.synchronize(local, timeout_ms=30000)
        torch.set_num_threads(2)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_root, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='eager').to(device).train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if model.config.num_hidden_layers != 36 or model.config.hidden_size != 4096:
            raise ValueError('Unexpected OneRec architecture')
        result['parameter_count'] = sum(p.numel() for p in model.parameters())
        result['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result['initial_sha256'] = fingerprint(model)
        target.write_text(json.dumps(result, indent=2) + '\n')
        group = TrainingGroup(args.collective, rank, world, sync, 120)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
        for step in range(args.steps):
            tokens = torch.randint(0, model.config.vocab_size, (1, 16),
                                 generator=torch.Generator().manual_seed(1014 + 2*step + rank)).to(device)
            optimizer.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            manual_mean_gradients(model, group, fused=False)
            finite = all(bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.requires_grad)
            if not finite or not torch.isfinite(loss).item():
                raise AssertionError('Non-finite loss/gradient')
            optimizer.step()
            sync()
            result['steps'].append({'step': step, 'loss': float(loss.item()),
                                    'seconds_with_finite_checks': time.perf_counter()-start,
                                    'gradients_finite': finite})
            target.write_text(json.dumps(result, indent=2) + '\n')
        result['final_sha256'] = fingerprint(model)
        result['changed_parameter_tensors'] = sum(result['initial_sha256'][k] != v
                                                 for k,v in result['final_sha256'].items())
        if result['changed_parameter_tensors'] == 0:
            raise AssertionError('No parameter changed')
        result['communication'] = group.snapshot()
        result['memory'] = runtime.memory_stats(local)
        result['body'] = 'completed; independent cross-rank hashes must match before acceptance'
    except BaseException as e:
        result['error'] = repr(e)
        raise
    finally:
        target.write_text(json.dumps(result, indent=2) + '\n')
        if group is not None:
            group.close()
            result['teardown'] = group.state
            target.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
