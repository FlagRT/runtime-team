#!/usr/bin/env python3
"""Check a running RAG API with serial, simultaneous, and staggered requests.

Run from dev/rag_ljy:
  python scripts/check_multi_request/check_requests.py --backend milvus

Start serve_queries.py separately against the named backend. --backend labels the
report; it does not change or verify the server's configured database. Output is
saved under artifacts/checks/multi_request/<Asia-Shanghai-date-time>-<unique-id>/.
Uses only the Python standard library. No automatic retries hide rejected work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4
CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_QUERIES = PROJECT / 'data/queries/nfcorpus-50.jsonl'


def http(url, timeout, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    # Local container endpoints must not go through the host's HTTP proxy.
    try:
        response = build_opener(ProxyHandler({})).open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode('utf-8', errors='replace')
        try:
            content = json.loads(raw)
        except ValueError:
            content = {'invalid_json': raw}
        return response.status, response.headers.get('X-Request-ID'), content


def send(base_url, query, timeout, origin):
    start = time.perf_counter()
    row = {'query_id': query['query_id'], 'query': query['query'],
           'sent_ms': (start - origin) * 1000}
    try:
        status, request_id, results = http(base_url + '/query', timeout, {'query': query['query']})
        row.update(status=status, request_id=request_id, results=results)
    except Exception as error:
        row.update(status=None, request_id=None, results=None, error_type=type(error).__name__)
    end = time.perf_counter()
    row.update(completed_ms=(end - origin) * 1000, latency_ms=(end - start) * 1000)
    return row


def run_scenario(name, queries, url, timeout, stagger_ms):
    origin = time.perf_counter()
    if name == 'serial':
        return [send(url, q, timeout, origin) for q in queries]
    barrier = threading.Barrier(len(queries) + 1)
    schedule = {}

    def worker(index, query):
        barrier.wait()
        target = schedule['start'] + (index * stagger_ms / 1000 if name == 'staggered' else 0)
        time.sleep(max(0, target - time.perf_counter()))
        return send(url, query, timeout, origin)

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(worker, i, q) for i, q in enumerate(queries)]
        schedule['start'] = time.perf_counter()
        barrier.wait()
        return [future.result() for future in futures]


def same_results(reference, actual):
    """Strict ordered identity/content comparison; tolerate small score roundoff."""
    if not isinstance(reference, list) or not isinstance(actual, list) or not reference or len(reference) != len(actual):
        return False
    for expected, observed in zip(reference, actual):
        if not isinstance(expected, dict) or not isinstance(observed, dict):
            return False
        for key in ('rank', 'document_id', 'chunk_id', 'title', 'text', 'source_uri', 'retrieval_ranks'):
            if key not in expected or key not in observed or expected[key] != observed[key]:
                return False
        for key in ('rerank_score', 'rrf_score'):
            try:
                if not math.isclose(float(expected[key]), float(observed[key]), rel_tol=1e-4, abs_tol=1e-6):
                    return False
            except (KeyError, TypeError, ValueError):
                return False
    return True


def validate(all_rows):
    baseline = {r['query_id']: r for r in all_rows['serial']}
    seen = set()
    checks = {}
    for scenario, rows in all_rows.items():
        failures = []
        for row in rows:
            identifier = row['request_id']
            if row['status'] != 200:
                failures.append(f"{row['query_id']}: HTTP {row['status']} / {row.get('error_type', '')}")
            if not identifier or identifier in seen:
                failures.append(f"{row['query_id']}: missing or duplicate X-Request-ID")
            seen.add(identifier)
            ref = baseline[row['query_id']]
            if not same_results(ref['results'], row['results']):
                failures.append(f"{row['query_id']}: invalid/empty results or mismatch against serial reference")
        overlapping_pairs = sum(
            max(a['sent_ms'], b['sent_ms']) < min(a['completed_ms'], b['completed_ms'])
            for i, a in enumerate(rows) for b in rows[i + 1:])
        if scenario != 'serial' and not overlapping_pairs:
            failures.append('No overlapping HTTP requests observed; reduce --stagger-ms and rerun')
        checks[scenario] = {'passed': not failures, 'requests': len(rows),
                            'overlapping_http_pairs': overlapping_pairs, 'failures': failures}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default='http://127.0.0.1:8088')
    parser.add_argument('--backend', required=True, choices=['milvus', 'elasticsearch'], help='report label; configure server separately')
    parser.add_argument('--queries', type=Path, default=DEFAULT_QUERIES)
    parser.add_argument('--count', type=int, default=2, help='same N queries in each of three scenarios')
    parser.add_argument('--stagger-ms', type=float, default=20)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--output-dir', type=Path, default=PROJECT / 'artifacts/checks/multi_request')
    args = parser.parse_args()
    if args.count < 2 or not math.isfinite(args.stagger_ms) or args.stagger_ms <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('count must be >=2; stagger-ms and timeout must be finite and positive')
    queries = []
    for line in args.queries.read_text(encoding='utf-8').splitlines():
        if line.strip():
            q = json.loads(line)
            queries.append({'query_id': str(q['query_id']), 'query': q['query']})
    queries = queries[:args.count]
    if len(queries) != args.count or len({q['query_id'] for q in queries}) != args.count or any(not isinstance(q['query'], str) or not q['query'].strip() for q in queries):
        parser.error('Input must contain enough valid queries with unique IDs')
    now = datetime.now(CHINA_TIME)
    output = args.output_dir / f'{now:%Y-%m-%d_%H%M%S}-{args.backend}-{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save('queries.json', queries)
    summary = {'started_at': now.isoformat(), 'backend_label': args.backend, 'url': args.url,
               'count_per_scenario': args.count, 'stagger_ms': args.stagger_ms,
               'status': 'running', 'note': 'Client HTTP overlap does not prove internal stage overlap; inspect server stage logs using X-Request-ID.'}
    save('summary.json', summary)
    print(f'Results: {output}', flush=True)
    try:
        status, _, health = http(args.url.rstrip('/') + '/health', args.timeout)
        summary['health_before'] = health
        if status != 200 or not isinstance(health, dict) or health.get('status') != 'ready':
            raise RuntimeError('Server is not ready')
        if args.count > health.get('max_inflight', 0):
            raise RuntimeError('Requested count exceeds reported capacity, or server lacks multi-request support')
        all_rows = {}
        for scenario in ('serial', 'simultaneous', 'staggered'):
            rows = run_scenario(scenario, queries, args.url.rstrip('/'), args.timeout, args.stagger_ms)
            all_rows[scenario] = rows
            save(scenario + '.json', rows)
            print(f'{scenario}: {sum(r["status"] == 200 for r in rows)}/{len(rows)} HTTP successes', flush=True)
            if any(r['status'] is None for r in rows):
                raise RuntimeError('Transport failure/timeout; stopping to avoid adding work while previous requests may still run')
        summary['checks'] = validate(all_rows)
        summary['status'] = 'passed' if all(c['passed'] for c in summary['checks'].values()) else 'failed'
    except Exception as error:
        summary.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        summary['finished_at'] = datetime.now(CHINA_TIME).isoformat()
        save('summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
