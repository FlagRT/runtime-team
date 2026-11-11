"""Verify the complete visible UUID set before running P800 model tests."""
import argparse
import json
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-uuid', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if 'XPU_EVENT_KL3_ENABLE' in os.environ:
        raise ValueError('K-3: XPU_EVENT_KL3_ENABLE must be unset')
    actual = [str(torch.cuda.get_device_properties(i).uuid)
              for i in range(torch.cuda.device_count())]
    normalize = lambda values: sorted(v.removeprefix('GPU-') for v in values)
    passed = normalize(actual) == normalize(args.expected_uuid)
    result = {'visible_uuid_order': actual, 'expected_uuid_set': args.expected_uuid,
              'torch': torch.__version__, 'verdict': 'PASS' if passed else 'FAIL'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
