#!/usr/bin/env python3
"""CPU-only: export ALL NFCorpus queries and provenance-labeled long RAG inputs.

Long inputs ask a corpus question with source abstracts as context; they are
constructed stress requests, not original human queries. No NPU allocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'experiment_outputs/inputs/request_pressure')
    p.add_argument('--per-length', type=int, default=16)
    a = p.parse_args()
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(a.tokenizer/'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    queries = []
    sources = []
    for f in sorted((a.dataset/'queries').glob('*.parquet')):
        sources.append(f)
        queries.extend(pq.read_table(f).to_pylist())
    if not queries:
        raise ValueError('No corpus queries found')
    docs_file = a.dataset/'documents.jsonl'
    docs = [json.loads(line) for line in docs_file.read_text().splitlines() if line.strip()]
    rows = [{'query_id': str(q['_id']), 'query': q['text'], 'tier': 'original',
             'input_kind': 'original_nfcorpus_query',
             'tokens': len(tokenizer.encode(q['text'], add_special_tokens=False).ids)} for q in queries]
    docs_terms = [set(re.findall(r'[a-z]{4,}', d['title'].lower()+' '+d['text'].lower())) for d in docs]
    for target in (1024, 2048, 4096, 6144):
        for i in range(a.per_length):
            q = queries[(i*193) % len(queries)]
            terms = set(re.findall(r'[a-z]{4,}', q['text'].lower()))
            selected = sorted(range(len(docs)), key=lambda j: (-len(terms & docs_terms[j]), j))
            text = f"Find research relevant to this question: {q['text']}\nUse these source abstracts as context:\n"
            ids = []
            for j in selected:
                d = docs[j]
                text += f"\nSource {d['document_id']}: {d['title']}\n{d['text']}\n"
                ids.append(d['document_id'])
                tokens = tokenizer.encode(text, add_special_tokens=False).ids
                if len(tokens) >= target:
                    text = tokenizer.decode(tokens[:target], skip_special_tokens=False)
                    break
            # Respect the API character ceiling, preserving a literal prefix.
            text = text[:32768]
            rows.append({'query_id': f"context-{target}-{q['_id']}", 'query': text,
                         'tier': str(target), 'input_kind': 'constructed_context_request',
                         'original_query_id': str(q['_id']), 'source_document_ids': ids,
                         'target_tokens': target, 'tokens': len(tokenizer.encode(text, add_special_tokens=False).ids),
                         'note': 'Original corpus abstracts, selected by lexical overlap; final abstract may be truncated. Executes real RAG inference; not an original dataset query.'})
    a.output.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)
    (a.output/'queries.jsonl').write_text(content)
    manifest = {'original_queries': len(queries), 'constructed_requests': len(rows)-len(queries),
                'sha256': hashlib.sha256(content.encode()).hexdigest(),
                'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources+[docs_file]},
                'tokenizer': str(a.tokenizer),
                'tiers': {t: {'count': sum(r['tier']==t for r in rows), 'tokens_min_max': [min(r['tokens'] for r in rows if r['tier']==t), max(r['tokens'] for r in rows if r['tier']==t)]} for t in sorted({r['tier'] for r in rows})}}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
