"""Bounded, repeated TP/PP execution with fail-closed evidence aggregation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


CASES = [('tp-h32', 'tp', 32, 2), ('tp-h64', 'tp', 64, 2),
         ('pp-m2', 'pp', 32, 2), ('pp-m4', 'pp', 32, 4)]


def summarize(root):
    manifest = json.loads((root / 'execution.json').read_text())
    runs = manifest['runs']
    if manifest['steps'] != 50 or manifest['repeats'] < 3:
        raise ValueError('Acceptance requires >=3 launches per case and 50 steps')
    expected = {(name, repeat) for name, *_ in CASES for repeat in range(1, manifest['repeats'] + 1)}
    seen, records = set(), []
    for run in runs:
        key = (run['case'], run['repeat'])
        if key not in expected or key in seen or run['exit_code'] != 0 or run['timed_out']:
            raise ValueError('Invalid, duplicate, timed-out or failed launcher')
        seen.add(key)
        case = next(c for c in CASES if c[0] == run['case'])
        for rank in range(2):
            record = json.loads((root / run['name'] / f'rank{rank}.json').read_text())
            if (record['rank'] != rank or record['world_size'] != 2 or record['steps'] != 50
                    or record['verdict'] != 'PASS' or record['group_state'] != 'CLOSED'
                    or record['device_vendor'] != manifest['device_vendor']
                    or record['source_sha256'] != manifest['source_sha256']['parallel_minimal.py']
                    or record['prototype_revision'] != manifest['prototype_revision']):
                raise ValueError('Incomplete rank/environment metadata')
            backend = manifest['collective'] if manifest['device_vendor'] != 'cpu' else 'gloo'
            if record['backend'] != backend:
                raise ValueError('Wrong collective backend')
            plan = record['plan']
            if (plan['mode'], plan['hidden_dim'], plan['microbatches']) != case[1:]:
                raise ValueError('Wrong parallel plan')
            if (plan['input_dim'], plan['output_dim'], plan['batch_size']) != (16, 8, 16):
                raise ValueError('Unexpected example dimensions')
            if record['tolerance'] != {'atol': 1e-5, 'rtol': 1e-4}:
                raise ValueError('Acceptance tolerance changed')
            checks = record['checks']
            if len(checks) != 50 or [v['step'] for v in checks] != list(range(50)):
                raise ValueError('Incomplete or repeated step coverage')
            for check in checks:
                if not all(math.isfinite(v) and v >= 0 for k, v in check.items() if k != 'step'):
                    raise ValueError('Invalid numeric check')
                if max(check[k] for k in ('forward_max_abs', 'tensor_check_max_abs', 'parameter_max_abs')) > 1e-5:
                    raise ValueError('Absolute reference error exceeds acceptance')
            count = 50 * plan['microbatches'] if plan['mode'] == 'pp' else 0
            wanted_calls = {'all_reduce': 51 if plan['mode'] == 'tp' else 1, 'send': count, 'recv': count}
            if record['transport_calls'] != wanted_calls:
                raise ValueError('Transport call coverage mismatch')
            wanted_bytes = ({'all_reduce': 4 + 50 * 16 * 8 * 4, 'send': 0, 'recv': 0}
                            if plan['mode'] == 'tp' else
                            {'all_reduce': 4, 'send': 50 * 16 * plan['hidden_dim'] * 4,
                             'recv': 50 * 16 * plan['hidden_dim'] * 4})
            if record['transport_input_bytes'] != wanted_bytes:
                raise ValueError('Transport byte coverage mismatch')
            records.append(record)
    if seen != expected:
        raise ValueError('Missing planned launches')
    return {
        'verdict': 'PASS', 'scope': manifest['scope'], 'device_vendor': manifest['device_vendor'],
        'collective': manifest['collective'], 'cases': len(CASES), 'launches': len(runs),
        'rank_records': len(records), 'steps_per_launch': 50,
        'forward_max_abs': max(v['forward_max_abs'] for r in records for v in r['checks']),
        'tensor_check_max_abs': max(v['tensor_check_max_abs'] for r in records for v in r['checks']),
        'parameter_max_abs': max(v['parameter_max_abs'] for r in records for v in r['checks']),
        'prototype_revision': manifest['prototype_revision'], 'source_sha256': manifest['source_sha256'],
        'limits': 'Two-layer synthetic FP32 TP/serialized PP examples, not DeepFM/OneRec model partitioning, multi-node, pipeline overlap, performance gains or checkpoint recovery.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cpu', 'ascend', 'kunlun'], required=True)
    p.add_argument('--collective', choices=['flagcx', 'hccl'], default='flagcx')
    p.add_argument('--runtime-root', type=Path)
    p.add_argument('--prototype-revision', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launch-timeout', type=int, default=120)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 3 or not 1 <= args.launch_timeout <= 600:
        p.error('Require >=3 repeats and bounded timeout')
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['parallel_minimal.py', 'distributed.py', 'deepfm_train.py', 'run_parallel_matrix.py']
    manifest = {'scope': 'CPU/Gloo host-only' if args.backend == 'cpu' else args.backend + ' integration',
                'device_vendor': args.backend, 'collective': 'gloo' if args.backend == 'cpu' else args.collective,
                'prototype_revision': args.prototype_revision, 'steps': 50, 'repeats': args.repeats,
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources},
                'runs': [], 'verdict': 'RUNNING'}
    try:
        for repeat in range(1, args.repeats + 1):
            for name, mode, hidden, microbatches in (CASES if repeat % 2 else CASES[::-1]):
                run_name = f'{repeat:02d}-{name}'
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                command = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
                           '--rdzv_backend=static', '--master_addr=127.0.0.1', f'--master_port={port}',
                           str(Path(__file__).with_name('parallel_minimal.py')),
                           '--mode', mode, '--backend', args.backend, '--hidden-dim', str(hidden),
                           '--microbatches', str(microbatches), '--steps', '50',
                           '--prototype-revision', args.prototype_revision,
                           '--output', str((args.output / run_name).resolve())]
                if args.backend != 'cpu':
                    if args.runtime_root is None:
                        raise ValueError('Accelerator requires runtime root')
                    command += ['--runtime-root', str(args.runtime_root.resolve()), '--collective', args.collective]
                start, timed_out = time.monotonic(), False
                with (args.output / f'{run_name}.log').open('x') as log:
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '1'})
                    try:
                        code = proc.wait(timeout=args.launch_timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            code = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            code = proc.wait()
                manifest['runs'].append({'case': name, 'repeat': repeat, 'name': run_name,
                    'argv': command, 'exit_code': code, 'timed_out': timed_out,
                    'launcher_seconds': time.monotonic() - start})
                (args.output / 'execution.json').write_text(json.dumps(manifest, indent=2) + '\n')
                print(f'{run_name}: exit={code} timeout={timed_out}', flush=True)
                if code != 0 or timed_out:
                    raise RuntimeError('Parallel launcher failed; no silent retry')
        result = summarize(args.output)
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        manifest['verdict'] = 'PASS'
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        manifest.update(verdict='FAIL', error=repr(exc))
        raise
    finally:
        (args.output / 'execution.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
