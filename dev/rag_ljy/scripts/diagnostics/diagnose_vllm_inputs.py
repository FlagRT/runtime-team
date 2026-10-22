#!/usr/bin/env python3
"""Replay byte-identical token IDs against vLLM to isolate engine repeatability.

Run from the project directory after starting serve_embeddings.py.
RAG_DIAG_LABEL labels the experiment; each run saves a unique result file.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json,sys,time,hashlib,math
from pathlib import Path
from urllib.request import Request,build_opener,ProxyHandler
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from rag_engine.vllm_embedding import VllmQwen3Embedder
p=PROJECT/Path('data/queries/nfcorpus-50.jsonl')
qs=[json.loads(l) for l in p.read_text().splitlines()][:2]
m=VllmQwen3Embedder(Path('/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B'))
tokens=m.tokenizer([f'Instruct: {m.instruction}\nQuery:{q["query"]}' for q in qs],padding=False,truncation=True,max_length=8192)['input_ids']
payloads=[json.dumps({'model':'rag-embedding','input':[t],'encoding_format':'float'}).encode() for t in tokens]
print('token_lengths',list(map(len,tokens)),flush=True)
rows=[]
for cycle in range(15):
 for i,body in enumerate(payloads):
  with build_opener(ProxyHandler({})).open(Request('http://127.0.0.1:8089/v1/embeddings',data=body,headers={'Content-Type':'application/json'}),timeout=120) as r:
   data=json.load(r)
  v=data['data'][0]['embedding'];n=math.sqrt(sum(x*x for x in v));v=[x/n for x in v]
  rows.append({'cycle':cycle,'query_id':qs[i]['query_id'],'input_sha256':hashlib.sha256(body).hexdigest(),'vector':v,'vector_hash':hashlib.sha256(json.dumps(v).encode()).hexdigest()})
 time.sleep([0,.02,.2][cycle%3])
import os
from datetime import datetime,timezone,timedelta
from uuid import uuid4
label=os.getenv('RAG_DIAG_LABEL','repeat')
stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d_%H%M%S')
out=Path(__file__).resolve().parents[2]/'artifacts/diagnostics/runs'/f'fixed-input-{stamp}-{label}-{uuid4().hex[:6]}.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'label':label,'tokens':tokens,'rows':rows},indent=2))
print(out,flush=True)
for q in qs:
 r=[r for r in rows if r['query_id']==q['query_id']]
 print(q['query_id'],'unique inputs',len({x['input_sha256'] for x in r}),'unique vectors',len({x['vector_hash'] for x in r}),flush=True)
