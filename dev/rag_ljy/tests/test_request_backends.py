"""Database selection belongs to each request; model instances remain shared."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest
from fastapi.testclient import TestClient

from rag_engine.pipeline import RetrievalPipeline
from rag_engine.server import create_app
from rag_engine.config import Settings


class Store:
    def __init__(self,name,barrier=None): self.backend_name=name; self.barrier=barrier
    def results(self):
        if self.barrier: self.barrier.wait(5)
        return [{'_id':self.backend_name,'_score':1,'_source':{
            'document_id':self.backend_name,'chunk_id':self.backend_name,'title':self.backend_name,
            'text':self.backend_name,'source_uri':'test://'+self.backend_name}}]
    def bm25_search(self,*a,**kw): return self.results()
    def dense_search(self,*a,**kw): return self.results()


def pipeline(barrier=None):
    stores={b:Store(b,barrier) for b in ('milvus','elasticsearch')}
    embedder=SimpleNamespace(encode_queries=lambda *a,**kw:[[1.0]])
    reranker=SimpleNamespace(rerank=lambda q,hits,**kw:[{**h,'_rerank_score':1} for h in hits])
    return RetrievalPipeline(stores['milvus'],embedder,reranker,stores=stores)


def test_two_http_backends_overlap_without_changing_default_or_crossing_results():
    p=pipeline(threading.Barrier(4))
    with TestClient(create_app(pipeline_factory=lambda _:p,memory_probe_enabled=False)) as client:
        assert client.get('/health').json()['retrieval_backends']==['elasticsearch','milvus']
        with ThreadPoolExecutor(2) as pool:
            futures={b:pool.submit(client.post,'/query',json={'query':'same question','retrieval_backend':b})
                     for b in ('milvus','elasticsearch')}
            for backend,future in futures.items():
                result=future.result(10)
                assert result.status_code==200
                assert result.headers['x-retrieval-backend']==backend
                assert result.json()[0]['document_id']==backend
        assert p.store is p.stores['milvus']


def test_omitted_backend_uses_default_and_invalid_backend_is_rejected():
    p=pipeline()
    with TestClient(create_app(pipeline_factory=lambda _:p,memory_probe_enabled=False)) as client:
        result=client.post('/query',json={'query':'q'})
        assert result.status_code==200 and result.headers['x-retrieval-backend']=='milvus'
        assert client.post('/query',json={'query':'q','retrieval_backend':'both'}).status_code==422
        del p.stores['elasticsearch']
        assert client.post('/query',json={'query':'q','retrieval_backend':'elasticsearch'}).status_code==422


def test_pipeline_initializes_two_clients_but_only_one_pair_of_models(monkeypatch):
    from rag_engine.query_runtime import create_pipeline
    monkeypatch.setenv('RETRIEVAL_BACKEND','milvus')
    monkeypatch.setenv('RAG_RETRIEVAL_BACKENDS','milvus,elasticsearch')
    monkeypatch.setenv('ELASTICSEARCH_PASSWORD','test-only')
    monkeypatch.setenv('RAG_GENERATION_ENABLED','false')
    settings=Settings.from_env(); calls=[]
    def store(settings):
        calls.append(settings.retrieval_backend)
        return SimpleNamespace(backend_name=settings.retrieval_backend,
            require_connection=lambda:None,has_index=lambda:True)
    monkeypatch.setattr('rag_engine.create_retrieval_store',store)
    monkeypatch.setattr('rag_engine.vllm_embedding.create_embedder',lambda *a,**kw: calls.append('embedding'))
    monkeypatch.setattr('rag_engine.reranker.Qwen3Reranker',lambda *a,**kw: calls.append('reranker'))
    p=create_pipeline('cpu',settings=settings)
    try:
        assert calls==['milvus','elasticsearch','embedding','reranker']
        assert set(p.stores)=={'milvus','elasticsearch'}
    finally:p.close()


def test_enabled_backends_must_contain_default(monkeypatch):
    monkeypatch.setenv('RETRIEVAL_BACKEND','milvus')
    monkeypatch.setenv('RAG_RETRIEVAL_BACKENDS','elasticsearch')
    with pytest.raises(ValueError,match='Default'):Settings.from_env()
