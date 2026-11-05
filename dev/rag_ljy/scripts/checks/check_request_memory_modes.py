#!/usr/bin/env python3
"""Request-only NPU pressure. Never allocates device memory or restarts services.

--run sends real /query calls. Without it, only inspect health and input.
A reservation-only high watermark NEVER counts as actual 85% device usage.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

# These utilities are stdlib HTTP/process/file helpers; main() is never invoked.
from check_watermarks_modes import ROOT, http, ratios, phase_ok, response_ok, conflicting_processes, sha256


def measured_phase_ok(health, phase):
    if health['memory_admission'].get('watermark_basis') != 'device_measured_usage':
        return phase_ok(health, phase)  # historical evidence keeps its original policy
    memory, admission = health['device_memory'], health['memory_admission']
    if memory['status'] != 'ok' or memory.get('stale') or admission['reason'] == 'stale_telemetry':
        return False
    actual, _ = ratios(health)
    if phase == 'high':
        return actual >= .85 and admission['reason'] == 'memory_pressure'
    if phase == 'middle':
        return .75 < actual < .85 and admission['reason'] == 'memory_pressure'
    return actual <= .75 and admission['status'] == 'ready'


def assess(samples, requests):
    samples = list(samples)
    for request in requests:
        body = request.get('body')
        detail = body.get('detail', {}) if isinstance(body, dict) else {}
        if isinstance(detail, dict) and 'admission_snapshot' in detail:
            samples.append({'t':detail['decision_monotonic'], 'health':detail['admission_snapshot'], 'source':'server_admission_decision'})
    samples.sort(key=lambda s:s['t'])
    highs = [s for s in samples if measured_phase_ok(s['health'], 'high')]
    peak = max((ratios(s['health'])[0] for s in samples), default=0)
    result = {'peak_actual_percent': round(peak*100, 3),
              'http_counts': dict(Counter(str(r.get('status')) for r in requests)),
              'rejection_reasons': dict(Counter(r.get('reason') for r in requests if r.get('status') == 503))}
    if not highs:
        result['status'] = 'threshold_not_reached' if peak < .85 else 'incomplete'
        return result
    first = highs[0]
    result['first_actual_high'] = first
    middle = [s for s in samples if s['t'] > first['t'] and measured_phase_ok(s['health'], 'middle')]
    low = [s for s in samples if s['t'] > first['t'] and measured_phase_ok(s['health'], 'recovery')]
    def observed_response(phase, earliest):
        for r in requests:
            body = r.get('body')
            detail = body.get('detail', {}) if isinstance(body, dict) else {}
            if not isinstance(detail, dict): detail = {}
            decision = detail.get('admission_snapshot')
            if decision and detail['decision_monotonic'] >= earliest and response_ok(r, phase) and measured_phase_ok(decision, phase):
                return True
            if r.get('sent', 0) >= earliest and response_ok(r, phase) and r.get('before') and r.get('after') and measured_phase_ok(r['before'], phase) and measured_phase_ok(r['after'], phase):
                return True
        return False
    checks = {
        'actual_above_85': True,
        'high_memory_rejection': observed_response('high', first['t']),
        'middle_band_hysteresis': bool(middle) and observed_response('middle', first['t']),
        'recovery_after_middle': bool(middle) and any(s['t'] > middle[0]['t'] for s in low)
                                and observed_response('recovery', middle[0]['t']),
        'accepted_work_completed_across_pressure': any(r.get('status') == 200
                and r['sent'] < first['t'] < r.get('response_received', r['completed']) for r in requests),
    }
    result.update(checks=checks, status='passed' if all(checks.values()) else 'incomplete')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--url', default='http://127.0.0.1:8088')
    p.add_argument('--queries', type=Path, default=ROOT/'experiment_outputs/inputs/request_pressure/queries.jsonl')
    p.add_argument('--model-root', type=Path, default=Path('/mnt/raid/jliu171/models'))
    p.add_argument('--modes', nargs='+', choices=['burst', 'staggered', 'continuous'], default=['burst', 'staggered', 'continuous'])
    p.add_argument('--levels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument('--batches', type=int, nargs='+', default=list(range(1, 33)))
    p.add_argument('--tiers', nargs='+', default=['original', '1024', '2048', '4096', '6144'])
    p.add_argument('--calibrate', action='store_true', help='with --run, serial real-request search for a threshold-reaching profile before arrival-mode tests')
    p.add_argument('--resume-calibration', type=Path, help='previous events.jsonl on this unchanged service; reuse measured real-forward growth for safe next step')
    p.add_argument('--stagger-seconds', type=float, default=.5)
    p.add_argument('--continuous-seconds', type=float, default=15)
    p.add_argument('--max-seconds', type=int, default=1200)
    a = p.parse_args()
    hold = ROOT/'experiment_outputs/checks/request_memory_modes/safety_hold.json'
    if a.run and hold.exists():
        p.error('A real-forward calibration breached the free-memory guard. Automatic escalation is suspended until the per-forward memory protection is corrected. Evidence: '+str(hold))
    if a.resume_calibration and not a.calibrate:
        p.error('--resume-calibration requires --calibrate')
    if min(a.levels+[a.stagger_seconds, a.continuous_seconds, a.max_seconds]) <= 0 or max(a.levels) > 128:
        p.error('Positive limits required; levels must be <=128')
    lock = open('/tmp/rag-watermarks-modes.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    conflicts = conflicting_processes()
    if conflicts:
        raise RuntimeError('Other test processes: '+json.dumps(conflicts))
    def health():
        code, _, h = http(a.url+'/health')
        if code != 200:
            raise RuntimeError('health failed')
        return h
    initial = health()
    cfg = initial['memory_admission']['config']
    if not (initial['max_inflight'] == 0 and initial['inflight'] == 0
            and cfg['enabled'] and cfg['high_watermark'] == .85 and cfg['low_watermark'] == .75
            and cfg.get('estimate_requests') and cfg['request_budget_mib'] is None and cfg['safety_margin_mib'] == 1024
            and initial['memory_admission']['reserved_mib'] == 0
            and (phase_ok(initial, 'baseline') or (a.calibrate and ratios(initial)[0] < .85
                 and initial['memory_admission']['status'] == 'ready'))
            and initial['device_memory']['source'] == 'device-context.runtime.memory_stats'):
        raise RuntimeError('Need idle API, --max-inflight 0 and unchanged memory guards: '+json.dumps(initial))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.model_root/'Qwen/Qwen3-Embedding-0.6B/tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    inputs = [json.loads(line) for line in a.queries.read_text().splitlines() if line.strip()]
    if not inputs or any(not isinstance(r.get('query'), str) or not 0 < len(r['query'].strip()) <= 32768 for r in inputs):
        raise ValueError('Each input needs a nonempty query of at most 32768 characters')
    # Preserve input text and provenance; constructed context is explicitly labeled.
    inputs.sort(key=lambda r: len(r['query']))
    groups = {tier: [r for r in inputs if r.get('tier', 'original') == tier] for tier in a.tiers}
    if any(not group for group in groups.values()):
        raise ValueError('Missing requested input tier; run prepare_request_pressure_inputs.py or select --tiers original for custom inputs')
    if min(a.batches) < 1 or max(a.batches) > initial.get('reranker_batch_limit', 4):
        raise ValueError('Requested batch exceeds deployed reranker_batch_limit; deploy the reviewed larger-batch configuration first')
    print(json.dumps({'preflight': 'passed', 'max_inflight': 0, 'query_count': len(inputs),
          'query_chars_min_max': [len(inputs[0]['query']), len(inputs[-1]['query'])],
          'tier_counts': {t: len(g) for t, g in groups.items()}, 'batches': a.batches,
          'run_requested': a.run, 'pressure_source': 'real RAG HTTP requests only'}, indent=2), flush=True)
    if not a.run:
        return 0
    out = ROOT/'experiment_outputs/checks/request_memory_modes'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out), flush=True)
    shutil.copyfile(a.queries, out/'queries_original.jsonl')
    (out/'queries_used.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in inputs))
    for part in ('src', 'scripts'):
        shutil.copytree(ROOT/part, out/'source'/part, ignore=shutil.ignore_patterns('__pycache__'))
    print('Saving model hashes and software versions (CPU/file I/O only)...', flush=True)
    models = {}
    for name in ('Qwen3-Embedding-0.6B', 'Qwen3-Reranker-0.6B'):
        root = a.model_root/'Qwen'/name
        files = sorted(f for f in root.rglob('*') if f.is_file() and '.cache' not in f.parts)
        if not files:
            raise RuntimeError('Missing model files: '+str(root))
        models[name] = {str(f.relative_to(root)): sha256(f) for f in files}
    for name, command in {'git_head': ['git', 'rev-parse', 'HEAD'],
                          'packages': [sys.executable, '-m', 'pip', 'freeze']}.items():
        r = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
        (out/(name+'.txt')).write_text(r.stdout+'\n'+r.stderr)
    (out/'manifest.json').write_text(json.dumps({'args': vars(a), 'initial_health': initial,
        'model_hashes': models,
        'query_sha256': sha256(a.queries), 'command': shlex.join([sys.executable, *sys.argv]),
        'input_policy': 'source text and provenance preserved; full original queries plus explicitly labeled corpus-context stress requests',
        'limit': 'health observes cached device-context samples; short peaks may be missed; no compute-utilization measurement'}, default=str, indent=2))
    log = (out/'events.jsonl').open('w')
    mutex = threading.Lock()
    rows, samples, futures = [], [], []
    stop, unsafe, reached = threading.Event(), threading.Event(), threading.Event()
    state = {'mode': 'setup', 'profile': None, 'tier': a.tiers[0]}
    count = 0
    pool = ThreadPoolExecutor(max_workers=128)
    slots = threading.BoundedSemaphore(128)
    summary = {'status': 'incomplete', 'modes': {}, 'errors': [], 'pressure_source': 'real RAG HTTP requests only'}
    prior_points = []
    if a.resume_calibration:
        previous = [json.loads(line) for line in a.resume_calibration.read_text().splitlines()]
        for profile in {json.dumps(r.get('profile'), sort_keys=True) for r in previous if r['kind'] == 'health' and r.get('profile')}:
            profile = json.loads(profile)
            tier = profile.get('tier')
            if tier not in groups:
                continue
            work = profile['reranker_batch_size'] * min(8192, (profile.get('token_cap') or max(r['tokens'] for r in groups[tier])) + 512)
            peak = max(r['health']['device_memory']['stats']['used_mb'] for r in previous
                       if r['kind'] == 'health' and r.get('profile') == profile)
            prior_points.append((work, peak))
        prior_points.sort()
        shutil.copyfile(a.resume_calibration, out/'prior_calibration_events.jsonl')
    def record(kind, **data):
        with mutex:
            log.write(json.dumps({'kind': kind, 'utc': datetime.now(timezone.utc).isoformat(), **data})+'\n')
            log.flush()
    def sample():
        h = health()
        with mutex:
            row = {'t': time.monotonic(), 'mode': state['mode'], 'profile': state['profile'],
                   'sent_so_far': count, 'completed_so_far': len(rows), 'health': h}
            samples.append(row)
        record('health', **row)
        memory = h['device_memory']
        if (memory['status'] != 'ok' or memory.get('stale')
                or h['memory_admission']['reason'] == 'stale_telemetry'
                or memory['stats']['free_mb'] < 8192):
            unsafe.set()
        if ratios(h)[0] > .85:
            reached.set()
        return h
    def monitor():
        last = 0
        while not stop.is_set():
            try:
                h = sample()
                if time.monotonic()-last > 5:
                    actual, effective = ratios(h)
                    print(f"[{state['mode']}] actual={actual:.2%} effective={effective:.2%} "
                          f"inflight={h['inflight']} admission={h['memory_admission']['reason']}", flush=True)
                    last = time.monotonic()
            except Exception as exc:
                record('monitor_error', t=time.monotonic(), error=repr(exc))
                unsafe.set()
                return
            stop.wait(.25)
    def check():
        if unsafe.is_set():
            raise RuntimeError('Telemetry/error/free-memory guard stopped new requests; inspect events')
    def send(index, mode, profile, payload, input_record):
        row = {'index': index, 'mode': mode, 'profile': profile, 'input_record': {k: v for k, v in input_record.items() if k != 'query'},
               'payload': payload, 'sent': time.monotonic()}
        try:
            row['before'] = health()
            row['status'], row['headers'], row['body'] = http(a.url+'/query', payload, timeout=120)
            row['after'] = health()
            detail = row['body'].get('detail') if isinstance(row['body'], dict) else None
            row['reason'] = detail.get('reason') if isinstance(detail, dict) else detail
            if row['status'] != 200 and not (row['status'] == 503 and row['reason'] in ('memory_pressure', 'request_budget') and row['headers'].get('retry-after')):
                unsafe.set()
        except Exception as exc:
            row['error'] = repr(exc)
            unsafe.set()
        finally:
            row['completed'] = time.monotonic()
            with mutex:
                rows.append(row)
            record('request', **row)
            slots.release()
    def submit(batch=4, probe=False):
        nonlocal count
        check()
        if reached.is_set() and not probe:
            return False
        if not slots.acquire(blocking=False):
            record('client_saturated', t=time.monotonic(), mode=state['mode'])
            raise RuntimeError('128 client requests outstanding; no hidden internal request backlog allowed')
        with mutex:
            index = count
            count += 1
        group = groups[a.tiers[0]] if probe else groups[state['tier']]
        input_record = group[index % len(group)]
        query = input_record['query']
        cap = (state['profile'] or {}).get('token_cap')
        if cap and not probe:
            query = tokenizer.decode(tokenizer.encode(query, add_special_tokens=False).ids[:cap], skip_special_tokens=False)
        payload = {'query': query, 'reranker_batch_size': batch,
                   'retriever_top_k': 100, 'coarse_top_k': 50, 'fine_top_k': 5}
        futures.append(pool.submit(send, index, state['mode'], state['profile'], payload, input_record))
        return True
    def wait_seconds(seconds):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            check()
            if reached.is_set():
                return
            time.sleep(min(.1, max(0, end-time.monotonic())))
    def settle(timeout=120):
        for f in list(futures):
            f.result(timeout=125)
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            h = sample()
            if h['inflight'] == 0 and h['memory_admission']['reserved_mib'] == 0:
                return h
            time.sleep(.25)
        raise RuntimeError('Accepted work did not drain; do not restart automatically')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.alarm(a.max_seconds)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        for mode in (['calibration'] if a.calibrate else a.modes):
            pilot_history = list(prior_points)
            measured_slope = 0.0
            for (w1, p1), (w2, p2) in zip(pilot_history, pilot_history[1:]):
                if w2 > w1 and p2 > p1:
                    measured_slope = max(measured_slope, (p2-p1)/(w2-w1))
            state.update(mode=mode, profile=None)
            reached.clear()
            starting = sample()
            if not (phase_ok(starting, 'baseline') or (a.calibrate and ratios(starting)[0] < .85
                    and starting['memory_admission']['status'] == 'ready')):
                raise RuntimeError('Next mode has no recovered low baseline; no manual cache clearing allowed')
            for tier, batch in [(t, b) for t in a.tiers for b in a.batches]:
                state['tier'] = tier
                # Serial pilot before each larger real model forward. Do not infer
                # safety from a larger number of waiting HTTP requests.
                h = sample()
                if h['memory_admission']['status'] != 'ready':
                    raise RuntimeError('Admission is paused; cannot escalate the real workload until it recovers')
                stats = h['device_memory']['stats']
                tokens = max(r.get('tokens', len(r['query'])) for r in groups[tier])
                work = batch * min(8192, tokens + 512)
                token_cap = None
                if pilot_history:
                    prior_work, prior_peak = max(pilot_history, key=lambda item: item[0])
                    # Conservative extrapolation of measured increases; never
                    # jump straight to an unmeasured large forward near the guard.
                    extra = max(0, work-prior_work)
                    predicted = prior_peak + extra * max(.6, measured_slope)
                    if predicted > stats['total_mb']-8192:
                        available_growth = (stats['total_mb']-8192-prior_peak) / max(.6, measured_slope)
                        token_cap = min(tokens, int((prior_work + .8*available_growth)/batch)-512)
                        if token_cap < 128 or available_growth <= 0:
                            record('unsafe_next_profile', t=time.monotonic(), tier=tier, batch=batch, predicted_peak_mib=predicted)
                            raise RuntimeError('No safe next measured profile within 8GiB guard')
                        work = batch * min(8192, token_cap + 512)
                        record('refined_profile', t=time.monotonic(), tier=tier, batch=batch, token_cap=token_cap)
                state['profile'] = {'tier': tier, 'reranker_batch_size': batch, 'arrival_level': 'serial_pilot', 'token_cap': token_cap}
                begin = time.monotonic()
                submit(batch)
                settle()
                check()
                peak = max((s['health']['device_memory']['stats']['used_mb'] for s in samples if s['t'] >= begin), default=stats['used_mb'])
                if pilot_history:
                    pw, pp = max(pilot_history, key=lambda item: item[0])
                    if work > pw and peak > pp:
                        measured_slope = max(measured_slope, (peak-pp)/(work-pw))
                pilot_history.append((work, peak))
                if reached.is_set():break
                if a.calibrate:continue
                for level in a.levels:
                    state['profile'] = {'tier': tier, 'reranker_batch_size': batch, 'arrival_level': level, 'token_cap': token_cap}
                    print(f'{mode}: {state["profile"]}', flush=True)
                    if mode == 'continuous':
                        # level / 4 QPS, open loop; no retries and no wait-for-response pacing.
                        end, next_send = time.monotonic()+a.continuous_seconds, time.monotonic()
                        while time.monotonic() < end and not reached.is_set():
                            submit(batch)
                            next_send += 4/level
                            wait_seconds(max(0, next_send-time.monotonic()))
                    else:
                        for _ in range(level):
                            if not submit(batch):break
                            if mode == 'staggered':wait_seconds(a.stagger_seconds)
                    if reached.is_set():break
                    settle()
                    check()
                if reached.is_set():break
            if reached.is_set():
                print('Actual memory exceeded 85%; stopping load growth and observing rejection/recovery.', flush=True)
                end = time.monotonic()+120
                while time.monotonic() < end:
                    check()
                    h = sample()
                    submit(1, probe=True)  # real small query, still subject to the same admission
                    time.sleep(1)
                    if phase_ok(h, 'recovery') and h['inflight'] == 0:break
            settle()
            check()
            selected_samples = [s for s in samples if s['mode'] == mode]
            selected_rows = [r for r in rows if r['mode'] == mode]
            summary['modes'][mode] = assess(selected_samples, selected_rows)
            (out/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps({'mode': mode, 'result': summary['modes'][mode]['status']}), flush=True)
        summary['status'] = 'passed' if all(r['status'] == 'passed' for r in summary['modes'].values()) else 'incomplete'
    except BaseException as exc:
        summary['status'] = 'failed'
        summary['errors'].append(repr(exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('Stopping submissions; allowing accepted real requests to drain. No cache clearing or restart.', flush=True)
        pool.shutdown(wait=True)
        try:
            final = settle()
            summary['health_after'] = final
            summary['service_recovered'] = phase_ok(final, 'recovery')
            if not summary['service_recovered']:
                summary['status'] = 'incomplete' if summary['status'] != 'failed' else 'failed'
        except Exception as exc:
            summary['errors'].append('drain: '+repr(exc))
            summary['status'] = 'failed'
        stop.set()
        watcher.join(35)
        for mode in (['calibration'] if a.calibrate else a.modes):
            if mode not in summary['modes']:
                ss, rr = [s for s in samples if s['mode'] == mode], [r for r in rows if r['mode'] == mode]
                summary['modes'][mode] = assess(ss, rr) if ss else {'status': 'not_run'}
        summary['total_sent'] = count
        summary['total_completed_http'] = len(rows)
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        min_free = min((s['health']['device_memory']['stats']['free_mb'] for s in samples), default=float('inf'))
        summary['minimum_observed_free_mib'] = min_free if samples else None
        if min_free < 8192:
            hold.write_text(json.dumps({'reason': 'real inference breached 8GiB free guard; empirical growth extrapolation is not a safe bound',
                                        'run': str(out), 'minimum_free_mib': min_free}, indent=2))
        (out/'summary.json').write_text(json.dumps(summary, indent=2))
        log.close()
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'passed' else 1 if summary['status'] == 'failed' else 2


if __name__ == '__main__':
    sys.exit(main())
