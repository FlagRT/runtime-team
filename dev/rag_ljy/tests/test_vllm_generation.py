from types import SimpleNamespace
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

from rag_engine.generation import GenerationLimits, PromptBuilder
from rag_engine.vllm_generation import VLLMOneRecGenerator, GenerationServiceError
from rag_engine.admission import MemoryAdmissionConfig
from rag_engine.server import create_app
from test_generation import CharacterTokenizer
from test_generation_server import GenerationPipeline
from test_server import ControlledMemoryMonitor


def test_exact_prompt_tokens_and_original_answer(monkeypatch):
    g=VLLMOneRecGenerator.__new__(VLLMOneRecGenerator)
    g._tokenizer_lock=threading.Lock()
    g.limits=GenerationLimits(1000,64)
    g.builder=PromptBuilder(CharacterTokenizer(),g.limits)
    g.model_path=Path('/model'); g.url='http://127.0.0.1:8090'
    hits=[{'text':'The original source.', 'document_id':'doc'}]
    expected=g.builder.build('Question',hits)
    def remote(url,body):
        assert url.endswith('/v1/completions')
        assert body['prompt']==expected['input_ids']
        assert body['max_tokens']==32 and body['temperature']==0
        return {'choices':[{'text':'Answer [1]', 'finish_reason':'stop'}],
                'usage':{'prompt_tokens':len(body['prompt']), 'completion_tokens':4}}
    monkeypatch.setattr('rag_engine.vllm_generation.call_json',remote)
    result=g.generate('Question',hits,32)
    assert result['answer']=='Answer [1]' and result['sources']==expected['sources']
    assert result['execution_memory_estimate']['reserved_mib']==0
    assert result['generation_backend']=='vllm'


def dual_app(monkeypatch, *, safety=150):
    primary,remote=ControlledMemoryMonitor(100),ControlledMemoryMonitor(100)
    sentinel=object()
    monkeypatch.setattr('rag_engine.server.DeviceMemoryMonitor',
        lambda *a,**kw: remote if kw.get('probe') is sentinel else primary)
    pipeline=GenerationPipeline(); pipeline.finish_generation.set()
    pipeline.generator.backend='vllm'; pipeline.generator.physical_die=6
    pipeline.generator.execution_profile['backend']='vllm'
    estimator=SimpleNamespace(fault=None,estimate=lambda _: {'estimated_incremental_mib':100})
    app=create_app(pipeline_factory=lambda _:pipeline,max_inflight=0,
        generation_memory_probe_factory=lambda die: sentinel,
        memory_estimator_factory=lambda _:estimator,
        memory_admission=MemoryAdmissionConfig(enabled=True,estimate_requests=True,safety_margin_mib=safety))
    return app,pipeline,primary,remote


def test_remote_high_middle_low_and_no_primary_reservation_on_rejection(monkeypatch):
    app,pipeline,primary,remote=dual_app(monkeypatch)
    with TestClient(app) as client:
        for used in (850,800,751):
            remote.refresh(used)
            response=client.post('/query',json={'query':'q','generate':True,'max_new_tokens':16,'priority':'high'})
            assert response.status_code==503 and response.headers['retry-after']=='1'
            assert response.json()['detail']['reason']=='memory_pressure'
            assert response.json()['detail']['physical_die']==6
            assert client.get('/health').json()['memory_admission']['reserved_mib']==0
        assert pipeline.calls==[]
        remote.refresh(750)
        response=client.post('/query',json={'query':'q','generate':True,'max_new_tokens':16})
        assert response.status_code==200
        health=client.get('/health').json()
        assert health['stage_queues']['generation']['workers']==4
        assert health['memory_admission_by_stage']['generation']['reserved_mib']==0


def test_remote_free_margin_and_stale_telemetry(monkeypatch):
    app,pipeline,primary,remote=dual_app(monkeypatch,safety=200)
    with TestClient(app) as client:
        remote.refresh(810)
        result=client.post('/query',json={'query':'q','generate':True,'max_new_tokens':16})
        assert result.status_code==503 and result.json()['detail']['reason']=='request_budget'
        remote.refresh(100,stale=True)
        result=client.post('/query',json={'query':'q','generate':True,'max_new_tokens':16})
        assert result.status_code==503 and result.json()['detail']['reason']=='stale_telemetry'
        assert client.post('/query',json={'query':'retrieval-only'}).status_code==200
        primary.refresh(900); remote.refresh(100)
        assert client.post('/query',json={'query':'q','generate':True,'max_new_tokens':16}).status_code==503


def test_accepted_remote_generation_completes_after_high_watermark(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    app,pipeline,primary,remote=dual_app(monkeypatch)
    pipeline.finish_generation.clear()
    with TestClient(app) as client, ThreadPoolExecutor(1) as caller:
        result=caller.submit(client.post,'/query',json={'query':'accepted','generate':True,'max_new_tokens':16})
        try:
            assert pipeline.generating.wait(5)
            remote.refresh(900)
            rejected=client.post('/query',json={'query':'reject','generate':True,'max_new_tokens':16})
            assert rejected.status_code==503
        finally: pipeline.finish_generation.set()
        assert result.result(5).status_code==200
        assert pipeline.calls==['accepted']


def test_verify_local_vllm_rejects_wrong_physical_die(tmp_path):
    from rag_engine.vllm_generation import verify_engine
    process=tmp_path/'123'; process.mkdir()
    args=['python','-m','vllm.entrypoints.openai.api_server','--port','8090','--model','/model',
          '--dtype','bfloat16','--served-model-name','onerec-8b','--max-model-len','5120','--max-num-seqs','4']
    (process/'cmdline').write_bytes(('\0'.join(args)+'\0').encode())
    (process/'environ').write_bytes(b'ASCEND_RT_VISIBLE_DEVICES=6\0RAG_ASCEND_PINNED_DIE=6\0')
    assert verify_engine('/model','http://127.0.0.1:8090',6,proc_root=tmp_path)['physical_die']==6
    with pytest.raises(ValueError,match='device'):
        verify_engine('/model','http://127.0.0.1:8090',7,proc_root=tmp_path)
