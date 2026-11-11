"""One bounded two-rank case with timeout and launcher-level acceptance."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from summarize_onerec import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collective', choices=['flagcx', 'hccl'], required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--bucket-mib', type=int, default=64)
    parser.add_argument('--mode', choices=['manual', 'fused', 'bucketed'], default='bucketed')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error('timeout must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(Path(__file__).with_name('onerec_smoke.py')),
               '--runtime-root', args.runtime_root, '--model-root', args.model_root,
               '--collective', args.collective, '--mode', args.mode,
               '--bucket-mib', str(args.bucket_mib), '--seq-len', str(args.seq_len),
               '--steps', str(args.steps), '--output', str(args.output)]
    start = time.time()
    timed_out = False
    with (args.output/'launcher.log').open('x') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    execution = dict(argv=command, exit_code=code, timed_out=timed_out,
                     elapsed_seconds=time.time()-start)
    (args.output/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    try:
        result = summarize(args.output, args.steps)
    except (KeyError, FileNotFoundError, ValueError, TypeError) as error:
        result = dict(verdict='ERROR', reason=str(error), execution=execution)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('overall: '+result['verdict'], flush=True)
    return {'PASS': 0, 'FAIL': 1, 'ERROR': 2}[result['verdict']]


if __name__ == '__main__':
    raise SystemExit(main())
