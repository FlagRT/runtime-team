"""Validate saved DeepFM NPU records; never turn missing launch evidence into PASS."""
import argparse
import json
from pathlib import Path


def summarize(root):
    manifest = json.loads((root / 'execution.json').read_text())
    if manifest['launcher_exit_codes'] != {'observed': 0, 'native': 0}:
        raise ValueError('Both launchers must exit successfully')
    runs = {}
    for mode in ('observed', 'native'):
        records = [json.loads((root / mode / f'rank{rank}.json').read_text())
                   for rank in range(2)]
        for rank, record in enumerate(records):
            if not (record['rank'] == rank and record['world_size'] == 2
                    and record['backend'] == 'hccl'
                    and record['evidence_scope'] == 'NPU integration'
                    and record['verdict'] == 'PASS' and record['group_state'] == 'CLOSED'
                    and record['comm_mode'] == mode
                    and record['global_batch'] == 64 and record['local_batch'] == 32
                    and record['warmup_steps'] == 5 and record['measured_steps'] == 50):
                raise ValueError(f'{mode}/rank{rank}: unexpected run metadata')
            oracle, training = record['oracle'], record['training']
            if not (len(oracle['gradient_checks']) == 17
                    and all(c['pass'] for c in oracle['gradient_checks'])
                    and oracle['update_pass'] and oracle['rank_parameter_max_abs'] == 0
                    and training['rank_parameter_max_abs'] == 0
                    and len(training['losses']) == len(training['step_seconds']) == 50
                    and len(training['sample_id_ranges']) == 50):
                raise ValueError(f'{mode}/rank{rank}: incomplete correctness evidence')
            for step, (start, stop) in enumerate(training['sample_id_ranges']):
                if [start, stop] != [(step + 5) * 64 + rank * 32,
                                    (step + 5) * 64 + (rank + 1) * 32]:
                    raise ValueError('Shards must cover each global batch exactly once')
        runs[mode] = records
    reference = runs['observed'][0]
    for records in runs.values():
        for record in records:
            for key in ('model_sha256', 'prototype_revision', 'torch', 'seed', 'dtype'):
                if record[key] != reference[key]:
                    raise ValueError(f'Comparison configuration mismatch: {key}')
    deltas = [max(abs(a - b) for a, b in zip(
        runs['observed'][rank]['training']['losses'],
        runs['native'][rank]['training']['losses'])) for rank in range(2)]
    if not all(delta < 1e-6 for delta in deltas):
        raise ValueError('Observed/native loss comparison failed (<1e-6)')
    return {
        'evidence_scope': '910C/HCCL DeepFM integration; synthetic data; single run per mode',
        'verdict': 'PASS', 'launcher_exit_codes': manifest['launcher_exit_codes'],
        'world_size': 2, 'warmup_steps': 5, 'measured_steps': 50,
        'gradient_tensors_checked_per_rank': 17,
        'gradient_reference_max_abs': max(c['max_abs'] for records in runs.values()
            for record in records for c in record['oracle']['gradient_checks']),
        'update_reference_max_abs': max(record['oracle']['update_max_abs']
            for records in runs.values() for record in records),
        'final_rank_parameter_max_abs': 0.0,
        'native_vs_observed_loss_curve_max_abs_by_rank': deltas,
        'global_samples_per_second': {mode: records[0]['training']['global_samples_per_second']
            for mode, records in runs.items()},
        'observed_gradient_communication_by_rank': [r['training']['gradient_communication']
            for r in runs['observed']],
        'performance_caveat': 'Single diagnostic runs, not end-to-end throughput or speedup; '
            'exclude data generation and additional post-step validation collectives',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    with (args.root / 'summary.json').open('x') as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
