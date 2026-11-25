"""Aggregate priority-model worker evidence; fail closed on incomplete runs."""
import argparse
import json
import math
from pathlib import Path


def inspect(path, name, device_backend, steps):
    if (path/'exit_code').read_text().strip() != '0':
        raise ValueError(f'{name}: launcher did not exit cleanly')
    ranks = [json.loads((path/f'rank{i}.json').read_text()) for i in (0, 1)]
    collective = {'cpu': 'gloo', 'ascend': 'hccl', 'kunlun': 'flagcx', 'ppu': 'nccl'}[device_backend]
    for i, r in enumerate(ranks):
        if (r['rank'], r['model'], r['device_backend'], r['backend'], r['verdict'],
                r['group_state'], r['sample_range']) != (i, name, device_backend, collective,
                                                          'PASS', 'CLOSED', [8*i, 8*(i+1)]):
            raise ValueError(f'{name}: rank identity or outcome mismatch')
        if not r['parameters'] > 0 or [s['step'] for s in r['steps']] != list(range(steps)):
            raise ValueError(f'{name}: incomplete coverage')
        for step in r['steps']:
            if not all(math.isfinite(step[k]) and step[k] >= 0 for k in
                       ('ref_loss', 'logit_max_abs', 'grad_max_abs', 'param_max_abs')):
                raise ValueError(f'{name}: bad numerical evidence')
        c = r['communication']
        if (c['gradient_allreduce_calls'], c['gradient_payload_bytes']) != (steps, steps*4*r['parameters']):
            raise ValueError(f'{name}: incorrect communication accounting')
    for key in ('parameter_sha256', 'revision', 'parameters', 'torch'):
        if ranks[0][key] != ranks[1][key]:
            raise ValueError(f'{name}: ranks disagree on {key}')
    return {
        'model': name, 'device_backend': device_backend, 'collective': collective,
        'steps': steps, 'ranks': len(ranks), 'verdict': 'PASS', 'torch': ranks[0]['torch'],
        'parameters': ranks[0]['parameters'], 'revision': ranks[0]['revision'],
        'final_parameters_identical': True,
        'max_abs': {key: max(s[key] for r in ranks for s in r['steps'])
                    for key in ('logit_max_abs', 'grad_max_abs', 'param_max_abs')},
        'communication': {k: ranks[0]['communication'][k] for k in
                          ('gradient_allreduce_calls', 'gradient_payload_bytes')},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('--backend', choices=('cpu', 'ascend', 'kunlun', 'ppu'), required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--models', nargs='+', choices=('din', 'dssm', 'dcn-mix', 'sasrec'),
                   default=('din', 'dssm'))
    a = p.parse_args()
    if a.steps <= 0 or len(set(a.models)) != len(a.models):
        p.error('steps must be positive and models must be unique')
    print(json.dumps([inspect(a.evidence/name, name, a.backend, a.steps)
                      for name in a.models], indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
