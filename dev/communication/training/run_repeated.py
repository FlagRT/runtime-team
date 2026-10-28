"""Run alternating manual/fused DeepFM trials; retain every launcher exit code.

CPU evidence remains host-only. No bypass of FlagCX teardown, no auto recovery.
The caller is responsible for verifying the locked image and idle device lease.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'ascend', 'kunlun'], required=True)
    p.add_argument('--collective', choices=['hccl', 'flagcx'], default='flagcx')
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--prototype-revision', required=True)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--launch-timeout', type=int, default=300)
    p.add_argument('--first-mode', choices=['manual', 'fused'], default='manual')
    p.add_argument('--save-final-parameters', action='store_true')
    args = p.parse_args()
    if args.repeats < 3 or args.launch_timeout < 1: p.error('Require >=3 repeats and positive timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'scope': {'cpu':'CPU/Gloo host-only','ascend':'Ascend integration',
                          'kunlun':'Kunlun integration'}[args.backend],
                'prototype_revision': args.prototype_revision, 'runs': []}
    try:
        for repeat in range(args.repeats):
            modes = (args.first_mode, 'fused' if args.first_mode == 'manual' else 'manual')
            for mode in (modes if repeat % 2 == 0 else modes[::-1]):
                name = f'{repeat+1:02d}-{mode}'
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
                cmd = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
                       '--nproc_per_node=2', '--rdzv_backend=static', '--master_addr=127.0.0.1',
                       f'--master_port={port}', str(Path(__file__).with_name('deepfm_train.py')),
                       '--backend', args.backend, '--comm-mode', mode, '--global-batch', '64',
                       '--steps', '50', '--warmup', '5', '--seed', '923',
                       '--prototype-revision', args.prototype_revision,
                       '--model-root', str(args.model_root.resolve()), '--output', str((args.output/name).resolve())]
                if args.backend in ('ascend','kunlun'):
                    if args.runtime_root is None: raise ValueError('Accelerator requires runtime root')
                    cmd += ['--runtime-root', str(args.runtime_root.resolve()), '--collective', args.collective]
                if args.save_final_parameters:
                    cmd += ['--save-final-parameters']
                start = time.time(); timed_out = False
                with (args.output/f'{name}.log').open('x') as log:
                    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
                    try: code = process.wait(timeout=args.launch_timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try: code = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); code = process.wait()
                manifest['runs'].append({'name': name, 'mode': mode, 'repeat': repeat+1,
                    'argv': cmd, 'exit_code': code, 'timed_out': timed_out,
                    'launcher_seconds': time.time()-start})
                print(f'{name}: launcher={code} timeout={timed_out}', flush=True)
                if code != 0: raise RuntimeError(f'{name} failed; no silent retry')
    finally:
        (args.output/'execution.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__': main()
