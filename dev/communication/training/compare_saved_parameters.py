"""Compare two rank snapshots without invoking the collective under test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare(root):
    arrays=[np.load(root/f'rank{rank}-parameters.npy',allow_pickle=False) for rank in range(2)]
    if arrays[0].shape != arrays[1].shape or arrays[0].dtype != arrays[1].dtype:
        raise ValueError('Snapshot shape/dtype mismatch')
    finite=all(np.isfinite(a).all() for a in arrays)
    maximum=float(np.max(np.abs(arrays[0]-arrays[1]))) if finite else None
    return {'method':'Independent host NumPy comparison; no c10d calls',
            'parameter_count':int(arrays[0].size),'dtype':str(arrays[0].dtype),
            'finite':bool(finite),'max_abs':maximum,
            'raw_array_sha256':[hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
            'verdict':'PASS' if finite and maximum <= 1e-6 else 'FAIL'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();result=compare(args.root)
    with (args.root/'independent_comparison.json').open('x') as output: json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['verdict']=='PASS' else 1)
