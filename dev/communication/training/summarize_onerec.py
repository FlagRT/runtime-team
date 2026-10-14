"""Fail-closed acceptance for the exact two-rank OneRec smoke configuration."""
import argparse
import json
import math
from pathlib import Path


def summarize(root, steps):
    launch = json.loads((root/'execution.json').read_text())
    ranks = [json.loads((root/f'rank{r}.json').read_text()) for r in range(2)]
    checks = {'launcher_zero': launch['exit_code'] == 0}
    for r, d in enumerate(ranks):
        checks[f'rank{r}_identity'] = d['rank'] == r and d['model'] == 'OpenOneRec/OneRec-8B'
        checks[f'rank{r}_full_parameters'] = d.get('parameter_count') == d.get('trainable_parameter_count') == 8391013376
        checks[f'rank{r}_steps'] = len(d['steps']) == steps and [s['step'] for s in d['steps']] == list(range(steps))
        checks[f'rank{r}_finite'] = all(s['gradients_finite'] and math.isfinite(s['loss']) for s in d['steps'])
        checks[f'rank{r}_closed'] = d['teardown'] == 'CLOSED' and 'error' not in d
        checks[f'rank{r}_updated'] = d.get('changed_parameter_tensors',0) > 0
        c = d.get('communication', {})
        checks[f'rank{r}_calls'] = c.get('gradient_allreduce_calls') == (1 if d.get('comm_mode','manual') == 'fused' else 399)*steps
        checks[f'rank{r}_payload'] = c.get('gradient_payload_bytes') == 8391013376*2*steps
    for key in ('backend','revision','dtype','optimizer','lr','seq_len','global_batch','local_batch'):
        checks[f'matching_{key}'] = ranks[0][key] == ranks[1][key]
    checks['matching_mode'] = ranks[0].get('comm_mode','manual') == ranks[1].get('comm_mode','manual')
    for key in ('initial_sha256','final_sha256'):
        a,b = [d.get(key,{}) for d in ranks]
        checks[f'matching_{key}'] = len(a) == 399 and a == b
    return {'verdict': 'PASS' if all(checks.values()) else 'FAIL',
            'scope': 'synthetic-token full-parameter SGD smoke; not quality/convergence or default-backend acceptance',
            'backend': ranks[0]['backend'], 'steps_per_rank': steps, 'checks': checks,
            'losses_per_rank': [[s['loss'] for s in d['steps']] for d in ranks]}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--steps',type=int,required=True)
    a=p.parse_args()
    if a.steps < 1: p.error('steps must be positive')
    result=summarize(a.root,a.steps)
    with (a.root/'summary.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(result['verdict'])
    raise SystemExit(0 if result['verdict']=='PASS' else 1)
