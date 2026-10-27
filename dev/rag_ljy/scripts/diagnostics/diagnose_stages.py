#!/usr/bin/env python3
"""Capture each stage while reproducing serial and concurrent RAG scheduling.

Requires a running local vLLM embedding endpoint. This uses an isolated pipeline,
not the production HTTP API. Full vectors and candidates are retained for diagnosis.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import load_project_env
load_project_env()
from rag_engine.query_runtime import create_pipeline, format_hits


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


async def main():
    project = Path(__file__).resolve().parents[2]
    source = project/'data/queries/nfcorpus-50.jsonl'
    queries = [json.loads(l) for l in source.read_text().splitlines()][:2]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H%M%S')
    out = Path(__file__).resolve().parents[2]/'artifacts/diagnostics/runs'/f'diagnosis-{stamp}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    print(out, flush=True)
    records = []
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(1) as model, ThreadPoolExecutor(4) as search:
        pipeline = await loop.run_in_executor(model, create_pipeline, 'npu:0')
        async def one(q, label, delay=0):
            await asyncio.sleep(delay)
            row = {'query_id':q['query_id'], 'scenario':label}
            # Preserve the same separate model/retrieval scheduling used by server.py.
            v = await loop.run_in_executor(model, pipeline.embed, q['query'])
            row['vector'] = v
            row['vector_hash'] = fingerprint(v)
            hits = await loop.run_in_executor(search, pipeline.search, q['query'], v)
            row['candidates'] = hits
            row['candidate_hash'] = fingerprint(hits)
            ranked = await loop.run_in_executor(model, pipeline.rank, q['query'], hits)
            row['results'] = format_hits(ranked)
            records.append(row)
            print(label, q['query_id'], row['vector_hash'][:12], row['candidate_hash'][:12],flush=True)
        try:
            for repeat in range(3):
                for q in queries:await one(q, f'{repeat}-serial')
                await asyncio.gather(*(one(q,f'{repeat}-simultaneous') for q in queries))
                await asyncio.gather(*(one(q,f'{repeat}-staggered',i*.02) for i,q in enumerate(queries)))
        finally:
            await loop.run_in_executor(model,pipeline.close)
            (out/'stages.json').write_text(json.dumps(records,indent=2)+'\n')
    summary = {}
    for q in queries:
        rows=[r for r in records if r['query_id']==q['query_id']]
        summary[q['query_id']] = {'unique_vectors':len({r['vector_hash'] for r in rows}),
            'unique_candidates':len({r['candidate_hash'] for r in rows}),
            'unique_final_orders':len({tuple(h['chunk_id'] for h in r['results']) for r in rows})}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    asyncio.run(main())
