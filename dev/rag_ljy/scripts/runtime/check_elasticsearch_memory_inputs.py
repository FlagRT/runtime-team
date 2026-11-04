#!/usr/bin/env python3
"""Read-only CPU validation of the Elasticsearch corpus used by memory trials."""
import hashlib
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from _bootstrap import load_project_env


def check():
    load_project_env()
    os.environ['RETRIEVAL_BACKEND'] = 'elasticsearch'
    from rag_engine.config import Settings
    from elasticsearch import Elasticsearch, helpers
    from tokenizers import Tokenizer
    settings = Settings.from_env()
    tokenizer_path = settings.reranker_model_path / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    limit = int(os.environ.get('RAG_RERANK_DOCUMENT_TOKENS', '640'))
    client = Elasticsearch(settings.elasticsearch_url, basic_auth=settings.basic_auth, request_timeout=30)
    try:
        index = settings.elasticsearch_index_name
        mapping = client.indices.get_mapping(index=index)
        if len(mapping) != 1:
            raise RuntimeError('Expected one concrete Elasticsearch index')
        concrete = next(iter(mapping))
        props = mapping[concrete]['mappings']['properties']
        if props['embedding']['type'] != 'dense_vector' or props['embedding']['dims'] != settings.embedding_dims:
            raise RuntimeError('Elasticsearch embedding mapping does not match model configuration')
        count = client.count(index=index)['count']
        embedded = client.count(index=index, query={'exists': {'field': 'embedding'}})['count']
        rows, documents = [], set()
        for hit in helpers.scan(client, index=index, query={'query': {'match_all': {}}, '_source': ['text', 'document_id', 'chunk_id']}):
            item = hit['_source']
            tokens = len(tokenizer.encode(item['text'], add_special_tokens=False).ids)
            documents.add(item['document_id'])
            rows.append((hit['_id'], hashlib.sha256(item['text'].encode()).hexdigest(), tokens))
        maximum = max((r[2] for r in rows), default=0)
        if count != 6543 or embedded != count or len(rows) != count or len(documents) != 3633:
            raise RuntimeError(f'Expected 3633 documents / 6543 embedded chunks; got documents={len(documents)}, chunks={count}, embedded={embedded}, scanned={len(rows)}')
        if maximum > limit:
            raise RuntimeError(f'Corpus maximum {maximum} tokens exceeds configured document bound {limit}; review before running')
        # Exercise real CPU lexical retrieval too: valid vectors/counts alone
        # do not prove long requests fit Elasticsearch's Boolean-clause limit.
        from rag_engine.stores.elasticsearch import ElasticsearchStore
        inputs = Path(__file__).resolve().parents[2] / 'experiment_outputs/inputs/request_pressure/queries.jsonl'
        records = [json.loads(line) for line in inputs.read_text().splitlines()]
        pressure = [r for r in records if r.get('tier') in ('1024', '2048', '4096', '6144')]
        if {r['tier'] for r in pressure} != {'1024','2048','4096','6144'}:
            raise RuntimeError('Missing pressure input tiers')
        store = ElasticsearchStore(settings)
        try:
            for record in pressure:
                if not store.bm25_search(record['query'], size=100):
                    raise RuntimeError('Pressure input has no lexical retrieval results')
        finally:
            store.client.close()
        return {'lexical_pressure_queries_checked': len(pressure),
                'lexical_policy': 'original multi_match; on clause overflow, first 256 unique analyzed terms per field',
                'pressure_inputs_sha256': hashlib.sha256(inputs.read_bytes()).hexdigest(),
                'status': 'passed', 'backend': 'elasticsearch', 'index': concrete,
                'documents': len(documents), 'chunks': count, 'embedded_chunks': embedded,
                'maximum_document_tokens': maximum, 'document_token_limit': limit,
                'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                'corpus_sha256': hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()}
    finally:
        client.close()


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True))
