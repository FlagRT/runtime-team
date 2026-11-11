#!/usr/bin/env python3
"""Real NPU watermarks under burst, staggered and continuous arrivals.

Default is read-only preflight. --run explicitly enables extra NPU allocations.
No API restarts, auxiliary servers, retries, or background suite scheduling.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]


def http(url, body=None, timeout=30):
    request = Request(url, data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=timeout)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, {k.lower(): v for k, v in response.headers.items()}, json.load(response)


def ratios(health):
    stats = health['device_memory']['stats']
    measured = max(stats['used_mb'], stats['total_mb'] - stats['free_mb'])
    return measured / stats['total_mb'], (measured + health['memory_admission']['reserved_mib']) / stats['total_mb']


def phase_ok(health, phase):
    memory = health['device_memory']
    admission = health['memory_admission']
    if memory['status'] != 'ok' or memory.get('stale') or admission['reason'] == 'stale_telemetry':
        return False
    actual, effective = ratios(health)
    if phase == 'high':
        return actual > .85 and admission['reason'] == 'memory_pressure'
    if phase == 'middle':
        return .75 < actual < .85 and .75 < effective <= .85 and admission['reason'] == 'memory_pressure'
    return effective <= .75 and admission['status'] == 'ready'


def response_ok(row, phase):
    if phase in ('high', 'middle'):
        if row.get('status') != 503 or not isinstance(row.get('body'), dict):
            return False
        detail = row['body'].get('detail', {})
        return (row.get('status') == 503 and isinstance(detail, dict)
                and detail.get('reason') == 'memory_pressure' and bool(row.get('headers', {}).get('retry-after')))
    return row.get('status') == 200


def conflicting_processes(proc_root=Path('/proc'), own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    # A timeout launcher repeats our script path in argv. Exclude only our
    # ancestor launchers, not another run's timeout or an ancestor test suite.
    ancestors = set()
    pid = own_pid
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        try:
            status = (proc_root/str(pid)/'status').read_text()
            pid = int(next(line.split()[1] for line in status.splitlines()
                           if line.startswith('PPid:')))
        except (OSError, ValueError, StopIteration):
            break
    found = []
    names = ('check_online_pressure.py', 'run_pressure_suite.py',
             'check_memory_admission_live.py', 'check_watermarks_modes.py',
             'check_request_memory_modes.py', 'check_measured_watermarks.py', 'check_dynamic_admission_live.py')
    for entry in proc_root.glob('[0-9]*/cmdline'):
        try:
            pid = int(entry.parent.name)
            argv = entry.read_bytes().decode().split('\0')
            launcher = argv and Path(argv[0]).name in ('timeout', 'env', 'bash', 'sh')
            if pid == own_pid or (pid in ancestors and launcher):
                continue
            if any(Path(arg).name in names for arg in argv):
                found.append({'pid': pid, 'argv': argv})
        except (OSError, UnicodeError):
            pass
    return found


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8088')
    parser.add_argument('--modes', nargs='+', choices=['burst', 'staggered', 'continuous'], default=['burst', 'staggered', 'continuous'])
    parser.add_argument('--queries', type=Path, default=ROOT/'data/queries/nfcorpus-50.jsonl')
    parser.add_argument('--model-root', type=Path, default=Path('/mnt/raid/jliu171/models'))
    parser.add_argument('--hold-seconds', type=float, default=15)
    parser.add_argument('--stagger-seconds', type=float, default=2)
    parser.add_argument('--max-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.run:
        parser.error('External-allocation experiment retired. Use check_request_memory_modes.py --run for real RAG requests only.')
    if min(args.hold_seconds, args.stagger_seconds, args.max_seconds) <= 0:
        parser.error('Durations must be positive')
    lock = open('/tmp/rag-watermarks-modes.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError('Another test is running: ' + json.dumps(conflicts))
    def health():
        code, _, data = http(args.url+'/health')
        if code != 200:
            raise RuntimeError(f'health HTTP {code}')
        return data
    initial = health()
    config = initial['memory_admission']['config']
    if not (initial['status'] == 'ready' and initial['inflight'] == 0
            and initial['memory_admission']['reserved_mib'] == 0
            and phase_ok(initial, 'baseline') and ratios(initial)[0] < .70
            and config['enabled'] and config['high_watermark'] == .85
            and config['low_watermark'] == .75 and config['request_budget_mib'] == 2048
            and config['safety_margin_mib'] == 1024 and initial['max_inflight'] >= 8
            and initial['device_memory']['device'] == 'npu:0'
            and initial['device_memory']['source'] == 'device-context.runtime.memory_stats'
            and initial['device_memory']['unit'] == 'MiB'):
        raise RuntimeError('Preflight failed: '+json.dumps(initial))
    queries = [json.loads(line) for line in args.queries.read_text().splitlines() if line.strip()]
    if not queries or any(not isinstance(q.get('query'), str) or not q['query'] for q in queries):
        raise ValueError('Input must contain nonempty query strings')
    print(json.dumps({'preflight': 'passed', 'health': initial, 'queries': len(queries),
                      'run_requested': args.run}, indent=2), flush=True)
    if not args.run:
        return 0

    out = ROOT/'experiment_outputs/checks/watermarks_modes'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out), flush=True)
    write_lock = threading.Lock()
    stream = (out/'events.jsonl').open('w')
    last_progress = 0.0
    def record(kind, **data):
        nonlocal last_progress
        with write_lock:
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 't': time.monotonic(), 'kind': kind, **data})+'\n')
            stream.flush()
            if kind == 'health' and time.monotonic()-last_progress >= 5:
                h = data['health']
                actual, effective = ratios(h)
                print(f"[{data['mode']}/{data['phase']}] memory={actual:.2%} "
                      f"effective={effective:.2%} inflight={h['inflight']} "
                      f"admission={h['memory_admission']['reason']}", flush=True)
                last_progress = time.monotonic()
            elif kind in ('phase_start', 'calibration', 'monitor_error', 'generator_error'):
                print(kind+': '+json.dumps(data), flush=True)
    summary = {'status': 'failed', 'watermarks': {}, 'drain_overlap': {}, 'errors': [],
               'pressure_source': 'extra allocations in this checker, not natural RAG query usage',
               'not_covered': ['budget calibration', 'request_budget branch', 'stale telemetry branch', 'NPU compute utilization', 'universal OOM prevention']}
    stop = threading.Event()
    abort = threading.Event()
    buffers = []
    torch = device = runtime = module = None
    monitor_thread = generator_thread = None
    pool = ThreadPoolExecutor(max_workers=4)
    futures = []
    rows = []
    state = {'mode': 'setup', 'phase': 'setup'}
    slots = threading.BoundedSemaphore(4)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupt)
    signal.alarm(args.max_seconds)
    def check():
        if abort.is_set():
            raise RuntimeError('Monitor or request generator aborted; inspect events.jsonl')
    def pause(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            check()
            time.sleep(min(.1, max(0, end-time.monotonic())))
    def wait_for(predicate, timeout=60):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            check()
            h = health()
            record('wait_health', mode=state['mode'], phase=state['phase'], health=h)
            if predicate(h):
                return h
            pause(.25)
        raise RuntimeError('Condition timeout: '+json.dumps(h))
    def release():
        buffers.clear()
        if module is not None:
            gc.collect()
            module.empty_cache()
            runtime.synchronize()
    def read():
        runtime.synchronize()
        return runtime.memory_stats(0)
    def request(index, mode, phase, scheduled):
        row = {'index': index, 'mode': mode, 'phase': phase, 'scheduled': scheduled,
               'sent': time.monotonic(), 'query_id': queries[index % len(queries)].get('query_id')}
        try:
            row['before'] = health()
            row['status'], row['headers'], row['body'] = http(args.url+'/query', {'query': queries[index % len(queries)]['query']})
            row['after'] = health()
        except Exception as exc:
            row['error'] = repr(exc)
        finally:
            row['completed'] = time.monotonic()
            rows.append(row)
            record('request', **row)
            slots.release()
    counter = 0
    def submit():
        nonlocal counter
        if not slots.acquire(blocking=False):
            record('client_capacity_exceeded', **state)
            abort.set()
            return
        mode, phase = state['mode'], state['phase']
        futures.append(pool.submit(request, counter, mode, phase, time.monotonic()))
        counter += 1
    def monitor():
        while not stop.is_set():
            try:
                h = health()
                record('health', **state, health=h)
                if h['device_memory']['status'] != 'ok' or h['device_memory'].get('stale') or h['memory_admission']['reason'] == 'stale_telemetry':
                    raise RuntimeError('Unavailable or stale memory telemetry')
                if h['device_memory']['stats']['free_mb'] < 8192:
                    raise RuntimeError('8GiB free guard breached')
                if h['memory_admission']['reason'] == 'memory_pressure' and h['inflight'] > 0:
                    summary['drain_overlap'][state['mode']] = True
            except Exception as exc:
                record('monitor_error', error=repr(exc))
                abort.set()
                return
            stop.wait(.25)
    generation_stop = threading.Event()
    def generate(interval):
        deadline = time.monotonic()
        try:
            while not generation_stop.is_set() and not abort.is_set():
                if generation_stop.wait(max(0, deadline-time.monotonic())):
                    return
                submit()
                deadline += interval
        except Exception as exc:
            record('generator_error', error=repr(exc))
            abort.set()
    def drain():
        for future in futures:
            future.result(timeout=35)
        wait_for(lambda h: h['inflight'] == 0 and h['memory_admission']['reserved_mib'] == 0)
    def target(ratio, rising=False):
        # Bounded 128MiB steps; re-read device-wide telemetry after every allocation.
        for _ in range(512):
            check()
            stats = read()
            actual = max(stats['used_mb'], stats['total_mb']-stats['free_mb'])/stats['total_mb']
            record('allocator', **state, stats=stats, extra_mib=len(buffers)*128)
            if stats['free_mb'] < 8192:
                raise RuntimeError('Allocator 8GiB guard breached')
            if rising and actual >= ratio:
                return
            if not rising and actual <= ratio:
                return
            if rising:
                # Small extra reserve above hard guard for allocator overhead / concurrent activity.
                if stats['free_mb'] < 8192+128+512:
                    raise RuntimeError('Guard prevents reaching target; do not lower guard')
                buffers.append(torch.empty(128*1024*1024, dtype=torch.uint8, device=device))
            else:
                if not buffers:
                    raise RuntimeError('Cannot lower external memory usage')
                buffers.pop()
                gc.collect()
                module.empty_cache()
            pause(.05)
        raise RuntimeError('Allocation step cap reached')
    try:
        print('Preparing source snapshots and package versions...', flush=True)
        shutil.copyfile(args.queries, out/'queries.jsonl')
        for relative in ('src', 'scripts', 'tests'):
            shutil.copytree(ROOT/relative, out/'source'/relative, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(ROOT.parent/'device-context/prototype', out/'source/device-context', ignore=shutil.ignore_patterns('__pycache__'))
        for name, command in {'git_head': ['git', 'rev-parse', 'HEAD'], 'git_diff': ['git', 'diff', '--', 'dev/rag_ljy', 'dev/device-context'], 'packages': [sys.executable, '-m', 'pip', 'freeze']}.items():
            result = subprocess.run(command, cwd=ROOT.parent.parent, capture_output=True, text=True, timeout=60)
            (out/(name+'.txt')).write_text(result.stdout+'\n'+result.stderr)
        model_manifest = {}
        for name in ('Qwen3-Embedding-0.6B', 'Qwen3-Reranker-0.6B'):
            print('Hashing model files: '+name, flush=True)
            directory = args.model_root/'Qwen'/name
            files = sorted(p for p in directory.rglob('*') if p.is_file() and '.cache' not in p.parts)
            if not files:
                raise RuntimeError('Cannot fingerprint model: '+str(directory))
            model_manifest[name] = {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files}
        (out/'manifest.json').write_text(json.dumps({'args': vars(args), 'health_before': initial,
            'query_sha256': sha256(args.queries), 'models': model_manifest,
            'command': shlex.join([sys.executable, *sys.argv]), 'minimum_free_mib': 8192,
            'sampling': 'API device-context samples; health polled every 0.25s (not independent device reads)'}, default=str, indent=2))
        sys.path.insert(0, str(ROOT/'scripts'))
        from _bootstrap import load_project_env
        load_project_env()
        from rag_engine.device import prepare_torch_device
        torch, device = prepare_torch_device('npu:0')
        print('Device ready; starting calibration and experiment monitoring.', flush=True)
        import runtime as runtime_module
        runtime = runtime_module
        module = getattr(torch, device.type)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        # Serial calibration, no intentional pressure. Conservative continuous rate <= 1 QPS.
        for _ in range(6):
            submit()
            drain()
        calibration = list(rows)
        if any(row.get('status') != 200 for row in calibration):
            raise RuntimeError('Baseline calibration failed')
        interval = max(1.0, 2*max(row['completed']-row['sent'] for row in calibration))
        record('calibration', continuous_interval_seconds=interval, stagger_interval_seconds=args.stagger_seconds)
        for mode in args.modes:
            state.update(mode=mode, phase='transition')
            summary['drain_overlap'][mode] = False
            generation_stop.clear()
            if mode != 'burst':
                generator_thread = threading.Thread(target=generate, args=(args.stagger_seconds if mode == 'staggered' else interval,), daemon=True)
                generator_thread.start()
            for phase in ('baseline', 'high', 'middle', 'recovery'):
                state['phase'] = 'transition'
                if phase == 'high':
                    target(.84, rising=True)
                    if mode == 'burst':
                        # A request accepted just below high water can retain its budget
                        # across the pause. Evidence is required, never assumed.
                        for _ in range(4):
                            submit()
                        pause(.3)
                    target(.855, rising=True)
                elif phase == 'middle':
                    target(.80)
                elif phase == 'recovery':
                    release()
                wait_for(lambda h: phase_ok(h, phase))
                state['phase'] = phase
                record('phase_start', **state)
                if mode == 'burst':
                    for _ in range(4):
                        submit()
                    pause(args.hold_seconds)
                else:
                    pause(max(args.hold_seconds, 5*(args.stagger_seconds if mode == 'staggered' else interval)))
                state['phase'] = 'transition'
                # Finish all requests labeled with this plateau before changing allocations.
                for future in list(futures):
                    future.result(timeout=35)
                phase_rows = [r for r in rows if r['mode'] == mode and r['phase'] == phase]
                valid = len(phase_rows) >= 3 and all(response_ok(r, phase) and phase_ok(r['before'], phase) and phase_ok(r['after'], phase) for r in phase_rows)
                summary['watermarks'][mode+'/'+phase] = {'passed': valid, 'requests': len(phase_rows)}
                if not valid:
                    raise RuntimeError('Plateau verification failed: '+mode+'/'+phase)
            generation_stop.set()
            if generator_thread:
                generator_thread.join(35)
                if generator_thread.is_alive():
                    raise RuntimeError('Generator failed to stop')
            drain()
            # Transitional requests may see 200 or explicit memory_pressure only.
            if any(not (response_ok(r, 'baseline') or response_ok(r, 'high')) for r in rows if r['mode'] == mode):
                raise RuntimeError('Unexpected response (including capacity rejection): '+mode)
        summary['status'] = 'passed' if all(summary['drain_overlap'].values()) else 'incomplete'
        if summary['status'] == 'incomplete':
            summary['errors'].append('Watermark plateaus passed, but pressure with inflight work was not observed in every mode; drain-under-pressure remains unverified there.')
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['errors'].append(repr(exc))
    finally:
        signal.alarm(0)
        print('Cleaning up allocations and checking service recovery...', flush=True)
        # Prevent a second Ctrl-C from interrupting normal cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        generation_stop.set()
        stop.set()
        for thread in (generator_thread, monitor_thread):
            if thread:
                thread.join(35)
        try:
            release()
        except BaseException as exc:
            summary['errors'].append('release: '+repr(exc))
            summary['status'] = 'failed'
        pool.shutdown(wait=True)
        try:
            end = time.monotonic()+60
            while True:
                final = health()
                record('cleanup_health', health=final)
                if final['inflight'] == 0 and final['memory_admission']['reserved_mib'] == 0 and phase_ok(final, 'recovery'):
                    summary['service_recovered'] = True
                    break
                if time.monotonic() > end:
                    raise RuntimeError('Service did not recover within 60s')
                time.sleep(.5)
        except BaseException as exc:
            summary['errors'].append('recovery: '+repr(exc))
            summary['status'] = 'failed'
        summary['http_counts'] = {str(code): sum(r.get('status') == code for r in rows) for code in (200, 503, None)}
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        stream.close()
    print(json.dumps(summary, indent=2))
    return {'passed': 0, 'failed': 1, 'incomplete': 2}[summary['status']]


if __name__ == '__main__':
    sys.exit(main())
