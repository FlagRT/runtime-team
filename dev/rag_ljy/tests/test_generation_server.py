import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_engine.admission import MemoryAdmissionConfig
from rag_engine.server import create_app
from test_server import ControlledMemoryMonitor, FakePipeline


class GenerationPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.generating, self.finish_generation = threading.Event(), threading.Event()
        self.generator = SimpleNamespace(
            execution_profile={'model_path': 'test-only'},
            configure_memory_guard=lambda *a: None,
            validate_request=self.validate_generation)

    def validate_generation(self, query, tokens):
        if tokens > 64:
            raise ValueError('configured generation limit')

    def generate(self, query, hits, tokens, timings):
        self.threads.append(threading.get_ident())
        self.generating.set()
        assert self.finish_generation.wait(5)
        if query == 'generation-fails':
            raise RuntimeError('private model failure')
        return {'answer': query + ' [1]', 'sources': [hits[0]['_source']],
                'usage': {'input_tokens': 32, 'output_tokens': 8}}


def generation_app(monkeypatch, pipeline, monitor):
    monkeypatch.setattr('rag_engine.server.DeviceMemoryMonitor', lambda *a, **kw: monitor)
    estimator = SimpleNamespace(fault=None, version='test-only',
        estimate=lambda request: {'estimated_incremental_mib': 300 if request.generate else 100})
    return create_app(pipeline_factory=lambda _: pipeline, max_inflight=0,
        stage_queue_capacity=2, memory_estimator_factory=lambda _: estimator,
        memory_admission=MemoryAdmissionConfig(enabled=True, estimate_requests=True, safety_margin_mib=150))


def test_generation_keeps_budget_until_output_ready_and_accepted_request_finishes_under_pressure(monkeypatch):
    pipeline, monitor = GenerationPipeline(), ControlledMemoryMonitor(100)
    app = generation_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client, ThreadPoolExecutor(1) as callers:
        pending = callers.submit(client.post, '/query', json={'query': 'answer', 'generate': True, 'max_new_tokens': 32})
        try:
            assert pipeline.generating.wait(5)
            health = client.get('/health').json()
            assert health['memory_admission']['active_reserved_mib'] == 300
            assert health['stage_queues']['generation']['capacity'] == 2
            assert health['stage_queues']['generation']['running'] == 1
            monitor.refresh(900)
            rejected = client.post('/query', json={'query': 'reject', 'generate': True, 'max_new_tokens': 32})
            assert rejected.status_code == 503 and rejected.headers['retry-after'] == '1'
            assert rejected.json()['detail']['reason'] == 'memory_pressure'
        finally:
            pipeline.finish_generation.set()
        response = pending.result(5)
        assert response.status_code == 200 and response.json()['answer'] == 'answer [1]'
        assert response.json()['hits'][0]['text'] == 'answer'
        assert response.json()['timings_ms']['total_ms'] > 0
        # Embedding, reranking and generation all use the initialization thread.
        assert len(set(pipeline.threads)) == 1
        assert client.get('/health').json()['memory_admission']['pending_release_mib'] == 300
        monitor.refresh(800)
        assert client.post('/query', json={'query': 'middle'}).status_code == 503
        monitor.refresh(700)
        old = client.post('/query', json={'query': 'retrieval-only'})
        assert old.status_code == 200 and isinstance(old.json(), list)


def test_generation_validates_before_running_and_failure_retains_pending_budget(monkeypatch):
    pipeline, monitor = GenerationPipeline(), ControlledMemoryMonitor(100)
    pipeline.finish_generation.set()
    with TestClient(generation_app(monkeypatch, pipeline, monitor)) as client:
        invalid = client.post('/query', json={'query': 'q', 'generate': True, 'max_new_tokens': 65})
        assert invalid.status_code == 422 and not pipeline.calls
        failed = client.post('/query', json={'query': 'generation-fails', 'generate': True, 'max_new_tokens': 16})
        assert failed.status_code == 500 and 'private' not in failed.text
        assert client.get('/health').json()['memory_admission']['pending_release_mib'] == 300
        monitor.refresh(100)
        assert client.post('/query', json={'query': 'next'}).status_code == 200


def test_generation_cannot_start_with_disabled_or_fixed_admission():
    pipeline = GenerationPipeline()
    with pytest.raises(ValueError, match='requires dynamic'):
        with TestClient(create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=False)):
            pass


def test_http_waiter_cancellation_keeps_generation_task_and_reservation(monkeypatch):
    pipeline, monitor = GenerationPipeline(), ControlledMemoryMonitor(100)
    app = generation_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client:
        endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/query')
        from fastapi import Response
        from rag_engine.server import QueryRequest

        async def exercise():
            waiter = asyncio.create_task(endpoint(QueryRequest(query='cancelled', generate=True, max_new_tokens=16), Response()))
            for _ in range(200):
                if pipeline.generating.is_set():
                    break
                await asyncio.sleep(.005)
            assert pipeline.generating.is_set()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert app.state.memory_admission.snapshot(monitor.snapshot())['active_reserved_mib'] == 300
            assert len(app.state.tasks) == 1
            pipeline.finish_generation.set()
            await asyncio.gather(*list(app.state.tasks))
            await asyncio.sleep(0)
            assert app.state.memory_admission.snapshot(monitor.snapshot())['pending_release_mib'] == 300
        try:
            client.portal.call(exercise)
        finally:
            pipeline.finish_generation.set()
