#!/usr/bin/env python3
"""Compare sequential/concurrent retrieval separately on both databases."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from benchmark_sequential import (
    Settings, create_embedder, create_retrieval_store, load_queries,
    hybrid_search, measure_stage, package_versions, run_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backends', nargs='+', choices=['milvus', 'elasticsearch'],
                        default=['milvus', 'elasticsearch'])
    parser.add_argument('--modes', nargs=2, choices=['sequential', 'concurrent'],
                        default=['sequential', 'concurrent'])
    parser.add_argument('--es-index', default='nfcorpus-chunks-v1')
    parser.add_argument('--milvus-collection', default='nfcorpus_chunks_v1')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'artifacts/benchmarks')
    args = parser.parse_args()
    if args.limit < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error('limit/repeats must be positive; warmup must be nonnegative')
    if len(set(args.modes)) != 2 or len(set(args.backends)) != len(args.backends):
        parser.error('modes and backends must not contain duplicates')
    source = sorted(load_queries(args.queries), key=lambda q: q.query_id)
    if len(source) < args.limit:
        parser.error(f'Need {args.limit} queries; input contains {len(source)}')
    queries = sorted(random.Random(args.seed).sample(source, args.limit), key=lambda q: q.query_id)
    started = datetime.now(timezone.utc)
    output = args.output_dir / f'comparison-{started:%Y%m%dT%H%M%SZ}-{uuid4().hex[:6]}'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'queries.jsonl').write_text(''.join(
        json.dumps(asdict(q), ensure_ascii=False) + '\n' for q in queries), encoding='utf-8')
    print(f'Output: {output}', flush=True)
    settings = replace(Settings.from_env(), elasticsearch_index_name=args.es_index,
                       milvus_collection_name=args.milvus_collection)
    stores = {}
    for backend in args.backends:
        store = create_retrieval_store(replace(settings, retrieval_backend=backend))
        store.require_connection()
        if not store.has_index():
            raise RuntimeError(f'Missing resource: {store.resource_name}')
        stores[backend] = store
    print(f'Precomputing {len(queries)} vectors once; excluded from retrieval timing...', flush=True)
    embedder = create_embedder(settings.embedding_model_path, device=args.device,
                            output_dims=settings.embedding_dims,
                            instruction=settings.retrieval_instruction)
    vectors = embedder.encode_queries([q.query for q in queries], batch_size=8)
    if len(vectors) != len(queries):
        raise RuntimeError('Unexpected vector count')
    vector_text = json.dumps(dict(zip([q.query_id for q in queries], vectors)))
    (output / 'query_vectors.json').write_text(vector_text + '\n', encoding='utf-8')
    query_vectors = dict(zip([q.query_id for q in queries], vectors))
    metadata = {
        'started_utc': started.isoformat(),
        'scope': 'retrieval-only', 'query_source': str(args.queries),
        'selection': 'seeded sample from query IDs sorted lexicographically',
        'query_vectors_sha256': hashlib.sha256(vector_text.encode()).hexdigest(),
        'resolved_device': str(embedder.device), 'dtype': str(getattr(embedder, 'dtype', None) or embedder.model.dtype),
        'embedding_backend': getattr(embedder, 'backend', 'transformers'),
        'embedding_model': str(settings.embedding_model_path),
        'embedding_dims': settings.embedding_dims,
        'retrieval_instruction': settings.retrieval_instruction,
        'retriever_top_k': 50, 'coarse_top_k': 30,
        'num_candidates': 100, 'rrf_rank_constant': 60,
        'packages': package_versions(), 'backend_order': args.backends,
        'mode_order': args.modes,
    }
    summaries = {}
    fingerprints = {}
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='rag-compare') as pool:
        for backend, store in stores.items():
            summaries[backend] = {}
            fingerprints[backend] = {}
            for mode in args.modes:
                print(f'Running {backend}: {mode}', flush=True)
                observed = {}

                def measure(query, timings):
                    with measure_stage(timings, 'total_ms'):
                        results = hybrid_search(
                            store, query.query, query_vectors[query.query_id],
                            execution_mode=mode, executor=pool, timings=timings)
                    # Validation work is outside the retrieval timer.
                    observed[query.query_id] = [hit['_id'] for hit in results]
                    return results

                summaries[backend][mode] = run_benchmark(
                    queries, measure, output / backend / mode,
                    warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                    metadata={**metadata, 'backend': backend, 'resource': store.resource_name},
                    execution_mode=mode)
                fingerprints[backend][mode] = observed
    (output / 'result_ids.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    lines = [
        '# Retrieval comparison', '',
        f'Run started (UTC): {started.isoformat()}', '',
        f'{len(queries)} identical queries; {args.repeats} measured pass(es) per mode; '
        f'{args.warmup} excluded warm-ups per mode. Embeddings computed once; no reranking.', '',
        'Each database runs separately. Requests are serial; concurrent means BM25 and dense '
        'overlap within one request. All times are client-observed milliseconds.', '',
        '| Database | Mode | Success/fail | BM25 mean | Dense mean | Search mean | Search P50 | Search P95 | RRF mean | Total mean |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    notes = []
    for backend, modes in summaries.items():
        for mode, summary in modes.items():
            stats = summary['latency_ms']
            cells = [f'{stats[field][stat]:.3f}' if field in stats else 'N/A'
                     for field, stat in [('bm25_ms', 'mean'), ('dense_ms', 'mean'),
                                         ('search_ms', 'mean'), ('search_ms', 'p50'),
                                         ('search_ms', 'p95'), ('rrf_ms', 'mean'), ('total_ms', 'mean')]]
            lines.append(f"| {backend} | {mode} | {summary['successful']}/{summary['failed']} | " + ' | '.join(cells) + ' |')
        seq, con = modes['sequential'], modes['concurrent']
        complete = all(s['failed'] == 0 and s['successful'] == len(queries) * args.repeats
                       for s in (seq, con))
        if complete:
            speedup = seq['latency_ms']['search_ms']['mean'] / con['latency_ms']['search_ms']['mean']
            notes.extend(['', f'{backend}: mean search speedup **{speedup:.2f}x**.'])
        else:
            notes.extend(['', f'{backend}: incomplete measurements; no speedup claimed.'])
        equal = sum(fingerprints[backend]['sequential'].get(q.query_id) ==
                    fingerprints[backend]['concurrent'].get(q.query_id)
                    for q in queries if q.query_id in fingerprints[backend]['sequential']
                    and q.query_id in fingerprints[backend]['concurrent'])
        notes.append(f'Identical final fused result ID order: {equal}/{len(queries)} queries (last observation).')
    lines.extend(notes)
    lines.extend(['', 'Quick baseline on a shared host, not a statistically robust capacity test. '
                  'Modes run in the recorded order, so caching and background load may affect the difference. '
                  'Use --modes concurrent sequential for an order-reversed follow-up.', ''])
    report = '\n'.join(lines)
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print(f'Saved report and raw artifacts: {output}', flush=True)
    if any(s['failed'] for modes in summaries.values() for s in modes.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
