#!/usr/bin/env python3
"""Verify rebuilt corpus IDs and stored dense vectors across both databases."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from _bootstrap import load_project_env
from rag_engine.config import Settings
from rag_engine.stores import create_retrieval_store
from elasticsearch.helpers import scan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    load_project_env()
    settings = replace(Settings.from_env(), elasticsearch_index_name='nfcorpus-chunks-v1', milvus_collection_name='nfcorpus_chunks_v1')
    es = create_retrieval_store(replace(settings, retrieval_backend='elasticsearch'))
    mv = create_retrieval_store(replace(settings, retrieval_backend='milvus'))
    # Request vectors explicitly: recent Elasticsearch can omit them from _source.
    e = {}
    for hit in scan(es.client, index=es.resource_name, query={'query': {'match_all': {}}, '_source': False, 'fields': ['chunk_id', 'embedding']}, size=256):
        fields = hit['fields']; v = fields['embedding']
        if len(v) == 1 and isinstance(v[0], list):
            v = v[0]
        e[hit['_id']] = v
    m = {}
    it = mv.client.query_iterator(collection_name=mv.resource_name, batch_size=256, output_fields=['chunk_id','embedding'], consistency_level='Strong')
    try:
        while True:
            batch = it.next()
            if not batch:
                break
            m.update((r['chunk_id'],r['embedding']) for r in batch)
    finally:
        it.close()
    same_ids = set(e) == set(m)
    max_diff = max((abs(float(a)-float(b)) for k in e.keys() & m.keys() for a,b in zip(e[k],m[k])), default=0)
    dimensions_ok = all(len(v)==1024 for v in list(e.values())+list(m.values()))
    result = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'elasticsearch_count':len(e), 'milvus_count':len(m), 'identical_chunk_ids':same_ids, 'dimensions_1024':dimensions_ok, 'max_stored_vector_absolute_difference':max_diff, 'passed':len(e)==len(m)==6543 and same_ids and dimensions_ok and max_diff<1e-6}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
