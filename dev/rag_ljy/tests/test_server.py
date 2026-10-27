"""Exercise persistent model ownership and HTTP behavior without NPU hardware."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from fastapi import Response

from rag_engine.server import QueryRequest, create_app as _create_app


def create_app(*args, **kwargs):
    # Most tests exercise fake pipeline scheduling without initializing hardware.
    kwargs.setdefault("memory_probe_enabled", False)
    return _create_app(*args, **kwargs)


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.threads = []
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, query, batch_size, timings):
        self.calls.append(query)
        self.threads.append(threading.get_ident())
        return [query]

    def search(self, query, vector, retriever_top_k, coarse_top_k, timings):
        assert vector == [query]
        if query == "block":
            self.started.set()
            assert self.release.wait(timeout=5)
        if query == "fail":
            raise RuntimeError("private database connection details")
        return [query]

    def rank(self, query, candidates, top_k, batch_size, timings):
        assert candidates == [query]
        self.threads.append(threading.get_ident())
        return [{
            "_rerank_score": 0.9,
            "_rrf_score": 0.03,
            "_retrieval_ranks": {"dense": 1},
            "_source": {
                "document_id": "doc", "chunk_id": "chunk", "title": "Title",
                "text": query, "source_uri": "sample://doc",
            },
        }]


def test_reuses_models_on_initialization_thread_and_retains_output_contract():
    pipeline = FakePipeline()
    loads = []

    def factory(device):
        loads.append((device, threading.get_ident()))
        return pipeline

    app = create_app("npu:3", factory)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health.pop("device_memory") == {"enabled": False, "status": "disabled"}
        assert health == {
            "status": "ready", "device": "npu:3", "busy": False,
            "inflight": 0, "max_inflight": 8, "embedding_backend": "transformers",
            "retrieval_backend": "unknown", "execution_mode": "concurrent",
            "stage_queues": {
                "embedding": {"capacity": 8, "queued": 0, "running": 0, "workers": 1},
                "retrieval": {"capacity": 8, "queued": 0, "running": 0, "workers": 4},
                "rerank": {"capacity": 8, "queued": 0, "running": 0, "workers": 1},
            },
        }
        for question in ("first", "second"):
            response = client.post("/query", json={"query": question})
            assert response.status_code == 200
            assert response.json() == [{
                "rank": 1, "rerank_score": 0.9, "rrf_score": 0.03,
                "retrieval_ranks": {"dense": 1}, "document_id": "doc",
                "chunk_id": "chunk", "title": "Title", "text": question,
                "source_uri": "sample://doc",
            }]
        assert len(loads) == 1
        assert loads[0][0] == "npu:3"
        assert pipeline.threads == [loads[0][1]] * 4
        assert app.state.pipeline is pipeline
    assert app.state.pipeline is None
    assert not app.state.ready


@pytest.mark.parametrize("body", [
    {}, {"query": "   "}, {"query": "x", "fine_top_k": 100},
    {"query": "x", "reranker_batch_size": 100},
    {"query": "x", "device": "npu:9"},
])
def test_invalid_requests_do_not_run_inference(body):
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline)) as client:
        assert client.post("/query", json=body).status_code == 422
        assert pipeline.calls == []


def test_failed_request_does_not_reload_models_or_stop_server():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline)) as client:
        response = client.post("/query", json={"query": "fail"})
        assert response.status_code == 500
        assert "private" not in response.text
        assert client.post("/query", json={"query": "next"}).status_code == 200
        assert len(pipeline.calls) == 2


def test_capacity_full_rejected_and_health_responsive():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline, max_inflight=1)) as client:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(client.post, "/query", json={"query": "block"})
            try:
                assert pipeline.started.wait(timeout=5)
                assert client.get("/health").json()["busy"] is True
                response = client.post("/query", json={"query": "another"})
                assert response.status_code == 503
                assert response.headers["retry-after"] == "1"
                assert len(pipeline.calls) == 1
            finally:
                pipeline.release.set()
            assert pending.result(timeout=5).status_code == 200
        assert client.post("/query", json={"query": "later"}).status_code == 200


def test_model_loading_failure_prevents_startup():
    def factory(_):
        raise RuntimeError("Model is missing")

    with pytest.raises(RuntimeError, match="Model is missing"):
        with TestClient(create_app(pipeline_factory=factory)):
            pytest.fail("Startup must fail")


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_server_selects_mode_and_closes_pipeline_on_initialization_thread(monkeypatch, mode):
    import rag_engine.server as server

    pipeline = FakePipeline()
    loads = []
    closed = []
    pipeline.close = lambda: closed.append(threading.get_ident())

    def factory(device, *, execution_mode):
        loads.append((device, execution_mode, threading.get_ident()))
        return pipeline

    monkeypatch.setattr(server, "create_pipeline", factory)
    app = create_app("npu:3", execution_mode=mode)
    with TestClient(app) as client:
        assert client.post("/query", json={"query": "first"}).status_code == 200
        assert client.post("/query", json={"query": "second"}).status_code == 200
        assert not closed
    assert loads == [("npu:3", mode, pipeline.threads[0])]
    assert closed == [loads[0][2]]
    assert app.state.pipeline is None


def test_cancelled_request_keeps_busy_until_inference_finishes():
    pipeline = FakePipeline()
    app = create_app(pipeline_factory=lambda _: pipeline)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/query")

    async def scenario():
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(endpoint(QueryRequest(query="block"), Response()))
            try:
                assert await asyncio.to_thread(pipeline.started.wait, 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert len(app.state.tasks) == 1
            finally:
                pipeline.release.set()
            await asyncio.gather(*list(app.state.tasks))
            await asyncio.sleep(0)
            assert not app.state.tasks
            result = await endpoint(QueryRequest(query="next"), Response())
            assert result[0]["text"] == "next"

    asyncio.run(scenario())


def test_second_request_finishes_while_first_retrieval_is_blocked(caplog):
    """No batch barrier: fast request embeds, searches and reranks before slow one."""
    import logging
    caplog.set_level(logging.INFO, logger="rag_engine.server")
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline, max_inflight=2)) as client:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(client.post, "/query", json={"query": "block"})
            try:
                assert pipeline.started.wait(5)
                fast = client.post("/query", json={"query": "fast"})
                assert fast.status_code == 200
                assert fast.json()[0]["text"] == "fast"
                assert not pending.done()
                assert client.get("/health").json()["inflight"] == 1
                failed = client.post("/query", json={"query": "fail"})
                assert failed.status_code == 500
                assert not pending.done()
            finally:
                pipeline.release.set()
            slow = pending.result(5)
        assert slow.status_code == 200
        assert slow.json()[0]["text"] == "block"
        assert len({r.headers['x-request-id'] for r in (fast, slow, failed)}) == 3
    messages = [record.getMessage() for record in caplog.records]
    fast_id, slow_id = fast.headers['x-request-id'], slow.headers['x-request-id']
    def event(identifier, stage, kind):
        return next(i for i, m in enumerate(messages)
                    if f'request_id={identifier} stage={stage} event={kind}' in m)
    assert event(slow_id, 'retrieval', 'start') < event(fast_id, 'embedding', 'start')
    assert event(fast_id, 'rerank', 'end') < event(slow_id, 'retrieval', 'end')
    assert len(set(pipeline.threads)) == 1


def test_shutdown_drains_disconnected_requests_before_pipeline_close():
    pipeline = FakePipeline()
    closed = []
    pipeline.close = lambda: closed.append(len(pipeline.threads))
    app = create_app(pipeline_factory=lambda _: pipeline)
    endpoint = next(route.endpoint for route in app.routes if route.path == '/query')

    async def scenario():
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        task = asyncio.create_task(endpoint(QueryRequest(query='block'), Response()))
        assert await asyncio.to_thread(pipeline.started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        shutdown = asyncio.create_task(context.__aexit__(None, None, None))
        try:
            await asyncio.sleep(0)
            assert not closed and not shutdown.done()
        finally:
            pipeline.release.set()
        await asyncio.wait_for(shutdown, 5)
        assert closed == [2]  # embedding AND reranking drained before close
    asyncio.run(scenario())


@pytest.mark.parametrize('options', [{'max_inflight': 0}, {'retrieval_workers': 0}, {'embedding_workers': 0}, {'stage_queue_capacity': 0}])
def test_invalid_scheduler_limits(options):
    with pytest.raises(ValueError):
        create_app(**options)


def test_two_admitted_requests_bound_model_queue_and_reject_third():
    pipeline = FakePipeline()
    entered = threading.Event()
    release = threading.Event()
    original = pipeline.embed

    def blocked_embed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    pipeline.embed = blocked_embed
    app = create_app(pipeline_factory=lambda _: pipeline, max_inflight=2)
    endpoint = next(route.endpoint for route in app.routes if route.path == '/query')

    async def scenario():
        from fastapi import HTTPException
        async with app.router.lifespan_context(app):
            first = asyncio.create_task(endpoint(QueryRequest(query='first'), Response()))
            assert await asyncio.to_thread(entered.wait, 5)
            second = asyncio.create_task(endpoint(QueryRequest(query='second'), Response()))
            try:
                await asyncio.sleep(0)  # second endpoint registers its request
                assert len(app.state.tasks) == 2
                with pytest.raises(HTTPException) as rejected:
                    await endpoint(QueryRequest(query='third'), Response())
                assert rejected.value.status_code == 503
                assert rejected.value.headers['Retry-After'] == '1'
                assert rejected.value.headers['X-Request-ID']
            finally:
                release.set()
            results = await asyncio.gather(first, second)
            assert [r[0]['text'] for r in results] == ['first', 'second']
    asyncio.run(scenario())


def test_remote_embedding_overlaps_without_blocking_reranker():
    """A slow HTTP embedding must not hold the local rerank initialization thread."""
    from types import SimpleNamespace
    pipeline = FakePipeline()
    pipeline.embedder = SimpleNamespace(backend='vllm', device='remote-vllm')
    entered, release = threading.Event(), threading.Event()
    loads, embedding_threads, rank_threads = [], [], []
    original_embed, original_rank = pipeline.embed, pipeline.rank

    def embed(query, *args):
        embedding_threads.append(threading.get_ident())
        if query == 'slow-embedding':
            entered.set()
            assert release.wait(5)
        return original_embed(query, *args)

    def rank(*args):
        rank_threads.append(threading.get_ident())
        return original_rank(*args)

    def factory(_):
        loads.append(threading.get_ident())
        return pipeline

    pipeline.embed, pipeline.rank = embed, rank
    app = create_app(pipeline_factory=factory, embedding_workers=2, stage_queue_capacity=1)
    with TestClient(app) as client, ThreadPoolExecutor(1) as callers:
        slow = callers.submit(client.post, '/query', json={'query': 'slow-embedding'})
        try:
            assert entered.wait(5)
            fast = client.post('/query', json={'query': 'fast'})
            assert fast.status_code == 200 and fast.json()[0]['text'] == 'fast'
            assert not slow.done()
            health = client.get('/health').json()
            assert health['stage_queues']['embedding']['workers'] == 2
            assert health['stage_queues']['embedding']['running'] == 1
            assert health['stage_queues']['rerank']['workers'] == 1
            assert rank_threads == loads
            assert loads[0] not in embedding_threads
        finally:
            release.set()
        assert slow.result(5).status_code == 200
    assert rank_threads == loads * 2


def test_memory_sampling_does_not_block_health_or_inference():
    from types import SimpleNamespace
    import time
    entered, release = threading.Event(), threading.Event()
    sampling_threads = []
    pipeline = FakePipeline()
    pipeline.reranker = SimpleNamespace(device="npu:0")
    selected = []
    def sample():
        sampling_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return {"enabled": True, "status": "ok", "stats": {
            "total_mb": 100, "used_mb": 20, "free_mb": 80}}
    def factory(device):
        selected.append(device)
        return SimpleNamespace(sample=sample)
    app = create_app("auto", lambda _: pipeline, memory_probe_enabled=True,
                     memory_probe_factory=factory, memory_probe_interval_s=.01)
    with TestClient(app) as client:
        try:
            assert entered.wait(5)
            assert client.get("/health").json()["device_memory"]["status"] == "pending"
            assert client.post("/query", json={"query": "while-probe-blocked"}).status_code == 200
            assert sampling_threads[0] not in pipeline.threads
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while client.get("/health").json()["device_memory"]["status"] == "pending":
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert client.get("/health").json()["device_memory"]["stats"]["free_mb"] == 80
    assert selected == ["npu:0"]
