#!/usr/bin/env python3
"""Compare vLLM query vectors with the saved Transformers benchmark vectors."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
from uuid import uuid4

from _bootstrap import load_project_env
load_project_env()
from rag_engine.config import Settings
from rag_engine.vllm_embedding import VllmQwen3Embedder


def main():
    project = Path(__file__).resolve().parents[2]
    reference = project / 'artifacts/benchmarks/concurrent_vs_sequential_2026-09-23/comparison-20260923T111642Z-231050'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--reference', type=Path, default=reference)
    parser.add_argument('--url', default='http://127.0.0.1:8089/v1')
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    settings = Settings.from_env()
    queries = [json.loads(line) for line in (args.reference/'queries.jsonl').read_text().splitlines()][:args.count]
    saved = json.loads((args.reference/'query_vectors.json').read_text())
    model = VllmQwen3Embedder(settings.embedding_model_path,
                            output_dims=settings.embedding_dims,
                            instruction=settings.retrieval_instruction, url=args.url)
    vectors = model.encode_queries([q['query'] for q in queries])
    rows = []
    for query, vector in zip(queries, vectors):
        ref = saved[query['query_id']]
        cosine = sum(a*b for a,b in zip(ref, vector)) / math.sqrt(sum(a*a for a in ref)*sum(b*b for b in vector))
        rows.append({'query_id':query['query_id'], 'cosine_similarity':cosine,
                     'dimensions':len(vector), 'passed':len(ref)==len(vector)==1024 and cosine>=0.999})
    now = datetime.now(timezone(timedelta(hours=8)))
    output = project/'artifacts/checks/embeddings'/f'vllm-embedding-{now:%Y-%m-%d_%H%M%S}-{uuid4().hex[:6]}'
    output.mkdir(parents=True)
    summary = {'started_at':now.isoformat(), 'reference':str(args.reference),
               'cosine_threshold':0.999, 'checks':rows, 'passed':all(r['passed'] for r in rows),
               'note':'Vector similarity is a compatibility smoke test, not a retrieval quality evaluation.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'vectors.json').write_text(json.dumps(dict(zip([q['query_id'] for q in queries],vectors)))+'\n')
    print(output)
    print(json.dumps(summary,indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
