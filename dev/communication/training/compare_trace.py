"""Independent host-only comparison of diagnostic gradient/parameter snapshots."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(root):
    traces = [json.loads((root / f'rank{r}/trace.json').read_text()) for r in range(2)]
    if traces[0]['snapshots'] != traces[1]['snapshots']:
        raise ValueError('Incomplete or mismatched rank snapshots')
    result = {'scope': 'instrumented diagnostic; host copies alter timing',
              'ranks': traces, 'snapshots': {}, 'source_sha256': {}}
    for label in traces[0]['snapshots']:
        paths = [root / f'rank{r}' / f'{label}.npz' for r in range(2)]
        for path in paths:
            result['source_sha256'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
            if a.files != b.files:
                raise ValueError('Mismatched parameter names')
            differences = {k: float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
            result['snapshots'][label] = {'rank_max_abs': max(differences.values()),
                'per_parameter_max_abs': differences,
                'finite': all(bool(np.isfinite(a[k]).all() and np.isfinite(b[k]).all()) for k in a.files)}
            if label.endswith('-reduced-grad'):
                prefix = label.split('-')[0]
                with np.load(root/'rank0'/f'{prefix}-local-grad.npz', allow_pickle=False) as x, \
                     np.load(root/'rank1'/f'{prefix}-local-grad.npz', allow_pickle=False) as y:
                    result['snapshots'][label]['host_mean_max_abs'] = max(
                        float(np.max(np.abs((x[k]+y[k])/2-a[k]))) for k in a.files)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    with a.output.open('x') as f:
        json.dump(compare(a.root), f, indent=2)
        f.write('\n')
