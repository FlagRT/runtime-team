#!/usr/bin/env python3
"""Replay the saved 50-query/200-case baseline and test fresh vLLM embeddings.

Requires the local fixed vLLM embedding server, Milvus and Elasticsearch.
Preserves the original retrieval-only scope (no reranker or generation).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from uuid import uuid4

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'scripts'))
from _bootstrap import load_project_env
from rag_engine.config import Settings
from rag_engine.retrieval import hybrid_search
from rag_engine.stores import create_retrieval_store
from rag_engine.vllm_embedding import VllmQwen3Embedder


def ids(hits):
    return [h['_id'] for h in hits]


def cross_engine_metrics(rows, old_vectors):
    """Compare unique queries per backend; avoids double-counting both modes."""
    result = {}
    for backend in ('milvus', 'elasticsearch'):
        selected = [r for r in rows if r['backend'] == backend and r['mode'] == 'sequential']
        cosines = []
        overlap = []
        top1_changes = []
        for r in selected:
            a, b = r['vector'], old_vectors[r['query_id']]
            cosines.append(sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(x*x for x in b)))
            old, new = ids(r['replay_hits']), ids(r['vllm_hits'])
            overlap.append(len(set(old) & set(new)))
            if old[:1] != new[:1]:
                top1_changes.append({'query_id': r['query_id'], 'old': old[0], 'new': new[0]})
        result[backend] = {
            'vector_cosine_min': min(cosines), 'vector_cosine_mean': sum(cosines)/len(cosines),
            'top30_overlap_min': min(overlap), 'top30_overlap_mean': sum(overlap)/len(overlap),
            'same_top30_set': sum(set(ids(r['replay_hits'])) == set(ids(r['vllm_hits'])) for r in selected),
            'same_top5_order': sum(ids(r['replay_hits'])[:5] == ids(r['vllm_hits'])[:5] for r in selected),
            'top1_changes': top1_changes,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=PROJECT / 'artifacts/benchmarks/concurrent_vs_sequential_2026-09-23/comparison-20260923T111642Z-231050')
    args = parser.parse_args()
    load_project_env()
    baseline = args.baseline
    queries = [json.loads(line) for line in (baseline / 'queries.jsonl').read_text().splitlines() if line]
    assert len(queries) == 50 and len({q['query_id'] for q in queries}) == 50
    old_vectors = json.loads((baseline / 'query_vectors.json').read_text())
    old_ids = json.loads((baseline / 'result_ids.json').read_text())
    stamp = datetime.now(timezone.utc)
    out = PROJECT / 'artifacts/benchmarks' / f'vllm-baseline-200-{stamp:%Y-%m-%d_%H%M%SZ}-{uuid4().hex[:6]}'
    out.mkdir()
    shutil.copyfile(baseline / 'queries.jsonl', out / 'queries.jsonl')
    summary = {
        'started_utc': stamp.isoformat(), 'baseline': str(baseline),
        'queries_sha256': hashlib.sha256((baseline / 'queries.jsonl').read_bytes()).hexdigest(),
        'scope': '50 identical queries x 2 databases x 2 retrieval modes; no reranking',
        'note': 'Original baseline saves ordered IDs only, not scores. Fresh vLLM embedding per case. Requests serial; concurrent mode overlaps dense/BM25 within a request.',
        'status': 'running', 'checks': {},
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(f'Results: {out}', flush=True)
    settings = replace(Settings.from_env(), elasticsearch_index_name='nfcorpus-chunks-v1', milvus_collection_name='nfcorpus_chunks_v1')
    embedder = VllmQwen3Embedder(settings.embedding_model_path, output_dims=settings.embedding_dims, instruction=settings.retrieval_instruction)
    embedder.require_connection()
    rows = {}
    try:
        with (out / 'cases.jsonl').open('w') as raw, ThreadPoolExecutor(max_workers=2) as pool:
            for backend in ('milvus', 'elasticsearch'):
                store = create_retrieval_store(replace(settings, retrieval_backend=backend))
                store.require_connection()
                assert store.has_index(), f'Missing {store.resource_name}'
                for mode in ('sequential', 'concurrent'):
                    for q in queries:
                        qid = q['query_id']
                        replay = hybrid_search(store, q['query'], old_vectors[qid], execution_mode=mode, executor=pool)
                        vector = embedder.encode_queries([q['query']], batch_size=1)[0]
                        timings = {}
                        fresh = hybrid_search(store, q['query'], vector, execution_mode=mode, executor=pool, timings=timings)
                        row = {'backend': backend, 'mode': mode, 'query_id': qid,
                               'vector': vector, 'replay_hits': replay, 'vllm_hits': fresh,
                               'historical_ids': old_ids[backend][mode][qid], 'timings_ms': timings}
                        raw.write(json.dumps(row) + '\n'); raw.flush()
                        rows[backend, mode, qid] = row
                    print(f'{backend}/{mode}: 50 baseline replays + 50 fresh vLLM cases complete', flush=True)
        checks = summary['checks']
        allrows = list(rows.values())
        summary['cross_engine_detail'] = cross_engine_metrics(allrows, old_vectors)
        checks['baseline_replay_vs_historical_ids'] = {'matched': sum(ids(r['replay_hits']) == r['historical_ids'] for r in allrows), 'total': 200}
        checks['vllm_vs_historical_ids'] = {'matched': sum(ids(r['vllm_hits']) == r['historical_ids'] for r in allrows), 'total': 200}
        for backend in ('milvus', 'elasticsearch'):
            pairs = [(rows[backend, 'sequential', q['query_id']], rows[backend, 'concurrent', q['query_id']]) for q in queries]
            checks[backend] = {
                'queries': 50,
                'exact_vectors_between_modes': sum(a['vector'] == b['vector'] for a,b in pairs),
                'exact_vllm_order_between_modes': sum(ids(a['vllm_hits']) == ids(b['vllm_hits']) for a,b in pairs),
                'exact_vllm_results_including_scores_between_modes': sum(a['vllm_hits'] == b['vllm_hits'] for a,b in pairs),
                'vllm_vs_replayed_pytorch_order': sum(ids(a['vllm_hits']) == ids(a['replay_hits']) for a,b in pairs),
                'vllm_vs_replayed_pytorch_top1': sum(ids(a['vllm_hits'])[:1] == ids(a['replay_hits'])[:1] for a,b in pairs),
                'cross_engine_different_query_ids': [a['query_id'] for a,b in pairs if ids(a['vllm_hits']) != ids(a['replay_hits'])],
            }
        checks['identical_vectors_all_four_passes'] = sum(len({json.dumps(rows[b,m,q['query_id']]['vector']) for b in ('milvus','elasticsearch') for m in ('sequential','concurrent')}) == 1 for q in queries)
        summary['status'] = 'passed' if checks['identical_vectors_all_four_passes'] == 50 and all(checks[b]['exact_vllm_results_including_scores_between_modes'] == 50 for b in ('milvus','elasticsearch')) else 'mismatch'
        summary['status_definition'] = 'Exact vLLM repeatability; historical/cross-engine comparisons reported separately.'
    except Exception as exc:
        summary.update(status='error', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        summary['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
    if summary['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
