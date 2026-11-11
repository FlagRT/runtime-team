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
        assert health.pop("memory_admission") == {"enabled": False, "status": "disabled"}
        assert health == {
            "status": "ready", "device": "npu:3", "busy": False,
            "inflight": 0, "max_inflight": 8, "embedding_backend": "transformers",
            "reranker_batch_limit": 4,
            "memory_estimation": {"enabled": False, "version": None,
                                  "status": "fixed_or_disabled", "fault": None},
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


def test_generation_request_on_retrieval_only_api_is_rejected_before_inference():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline)) as client:
        assert client.post('/query', json={'query': 'q', 'generate': True}).status_code == 422
        assert not pipeline.calls


def test_failed_request_does_not_reload_models_or_stop_server():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline)) as client:
        response = client.post("/query", json={"query": "fail"})
        assert response.status_code == 500
        assert "private" not in response.text
        assert client.post("/query", json={"query": "next"}).status_code == 200
        assert len(pipeline.calls) == 2


def test_reranker_batch_limit_is_configurable_and_checked_before_admission():
    pipeline = FakePipeline()
    batches = []
    original = pipeline.rank
    def rank(query, candidates, top_k, batch_size, timings):
        batches.append(batch_size)
        return original(query, candidates, top_k, batch_size, timings)
    pipeline.rank = rank
    with TestClient(create_app(pipeline_factory=lambda _: pipeline,
                              max_reranker_batch_size=16)) as client:
        assert client.get('/health').json()['reranker_batch_limit'] == 16
        assert client.post('/query', json={'query': 'large real forward', 'reranker_batch_size': 16}).status_code == 200
        assert batches == [16]
        assert client.post('/query', json={'query': 'too large', 'reranker_batch_size': 17}).status_code == 422
        assert batches == [16]


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


class ControlledMemoryMonitor:
    """Use controlled device observations while exercising the real HTTP API."""

    def __init__(self, used=300):
        self.refresh(used)

    def refresh(self, used, **updates):
        import time
        self.sample = {"status": "ok", "unit": "MiB", "scope": "device", "age_seconds": 0,
                       "sample_started_monotonic": time.monotonic(), "stale": False,
                       "stats": {"total_mb": 1000, "used_mb": used, "free_mb": 1000 - used},
                       **updates}

    def start(self):
        pass

    def snapshot(self):
        return dict(self.sample)

    async def close(self):
        pass


def memory_admission_app(monkeypatch, pipeline, monitor, *, request_budget_mib=300, **options):
    from rag_engine.admission import MemoryAdmissionConfig
    monkeypatch.setattr("rag_engine.server.DeviceMemoryMonitor", lambda *a, **kw: monitor)
    return create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=True,
        memory_admission=MemoryAdmissionConfig(enabled=True, request_budget_mib=request_budget_mib,
            safety_margin_mib=150, high_watermark=.85, low_watermark=.75), **options)


def test_http_memory_pressure_rejects_then_recovers_below_low_watermark(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="rag_engine.server")
    monitor = ControlledMemoryMonitor(900)
    pipeline = FakePipeline()
    app = memory_admission_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client:
        rejected = client.post("/query", json={"query": "too-full"})
        assert rejected.status_code == 503 and rejected.headers["retry-after"] == "1"
        assert rejected.headers["x-request-id"]
        assert rejected.json()["detail"]["reason"] == "memory_pressure"
        assert not pipeline.calls
        monitor.refresh(800)
        assert client.post("/query", json={"query": "still-paused"}).status_code == 503
        monitor.refresh(300)
        response = client.post("/query", json={"query": "recovered"})
        assert response.status_code == 200 and response.json()[0]["text"] == "recovered"
        assert client.get("/health").json()["memory_admission"]["accepted"] == 1
    assert any('"reason": "memory_pressure"' in r.getMessage() for r in caplog.records)
    assert any('"accepted": true' in r.getMessage() for r in caplog.records)


def test_disabled_count_limit_accepts_more_than_eight_but_keeps_memory_guard(monkeypatch):
    import time
    monitor, pipeline = ControlledMemoryMonitor(100), FakePipeline()
    entered, release = threading.Event(), threading.Event()
    original = pipeline.embed

    def blocked_embed(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    pipeline.embed = blocked_embed
    app = memory_admission_app(monkeypatch, pipeline, monitor,
                               max_inflight=0, request_budget_mib=50)
    with TestClient(app) as client, ThreadPoolExecutor(10) as callers:
        pending = []
        try:
            pending = [callers.submit(client.post, '/query', json={'query': f'q{i}'})
                       for i in range(10)]
            assert entered.wait(5)
            end = time.monotonic() + 5
            while time.monotonic() < end:
                state = client.get('/health').json()
                if state['inflight'] == 10:
                    break
                time.sleep(.01)
            assert state['max_inflight'] == 0 and state['inflight'] == 10
            assert state['stage_queues']['embedding']['capacity'] == 8
            assert state['memory_admission']['active_reserved_mib'] == 500
            monitor.refresh(900)
            response = client.post('/query', json={'query': 'pressure'})
            assert response.status_code == 503
            assert response.json()['detail']['reason'] == 'memory_pressure'
            assert response.headers['retry-after'] == '1'
        finally:
            release.set()
        assert all(f.result(5).status_code == 200 for f in pending)
        monitor.refresh(100)
        assert client.post('/query', json={'query': 'recovered'}).status_code == 200


def test_http_downstream_budget_prevents_second_request_while_first_searches(monkeypatch):
    monitor, pipeline = ControlledMemoryMonitor(), FakePipeline()
    app = memory_admission_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client, ThreadPoolExecutor(1) as caller:
        first = caller.submit(client.post, "/query", json={"query": "block"})
        try:
            assert pipeline.started.wait(5)
            state = client.get("/health").json()
            assert state["inflight"] == 1 and state["max_inflight"] == 8
            assert state["memory_admission"]["active_reserved_mib"] == 300
            rejected = client.post("/query", json={"query": "cannot-spend-same-free-memory"})
            assert rejected.status_code == 503
            assert rejected.json()["detail"]["reason"] == "request_budget"
            assert len(pipeline.calls) == 1
        finally:
            pipeline.release.set()
        assert first.result(5).status_code == 200  # Accepted requests can still drain.
        state = client.get("/health").json()["memory_admission"]
        assert state["reserved_requests"] == 0 and state["pending_release_mib"] == 300
        assert client.post("/query", json={"query": "old-sample"}).status_code == 503
        monitor.refresh(300)
        assert client.post("/query", json={"query": "new-sample"}).status_code == 200


def test_http_failed_request_releases_budget_after_fresh_observation(monkeypatch):
    monitor, pipeline = ControlledMemoryMonitor(), FakePipeline()
    app = memory_admission_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client:
        assert client.post("/query", json={"query": "fail"}).status_code == 500
        state = client.get("/health").json()["memory_admission"]
        assert state["reserved_requests"] == 0 and state["pending_release_mib"] == 300
        monitor.refresh(300)
        assert client.post("/query", json={"query": "next"}).status_code == 200


@pytest.mark.parametrize("bad_sample", [
    {"status": "pending"}, {"status": "unavailable"}, {"stale": True},
    {"scope": "backend_reported_unverified"},
])
def test_http_bad_telemetry_blocks_new_requests_until_fresh_read(monkeypatch, bad_sample):
    monitor, pipeline = ControlledMemoryMonitor(), FakePipeline()
    monitor.refresh(300, **bad_sample)
    app = memory_admission_app(monkeypatch, pipeline, monitor)
    with TestClient(app) as client:
        assert client.post("/query", json={"query": "untrusted"}).status_code == 503
        assert not pipeline.calls
        monitor.refresh(300)
        assert client.post("/query", json={"query": "fresh"}).status_code == 200


def test_disconnected_http_client_keeps_memory_budget_until_actual_completion(monkeypatch):
    monitor, pipeline = ControlledMemoryMonitor(), FakePipeline()
    app = memory_admission_app(monkeypatch, pipeline, monitor)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/query")

    async def scenario():
        from fastapi import HTTPException
        async with app.router.lifespan_context(app):
            caller = asyncio.create_task(endpoint(QueryRequest(query="block"), Response()))
            try:
                assert await asyncio.to_thread(pipeline.started.wait, 5)
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
                assert app.state.memory_admission.snapshot(monitor.snapshot())["active_reserved_mib"] == 300
                monitor.refresh(300)  # A new read cannot free a LIVE request's reservation.
                with pytest.raises(HTTPException) as rejected:
                    await endpoint(QueryRequest(query="too-soon"), Response())
                assert rejected.value.status_code == 503
            finally:
                pipeline.release.set()
            await asyncio.gather(*list(app.state.tasks))
            await asyncio.sleep(0)
            state = app.state.memory_admission.snapshot(monitor.snapshot())
            assert state["active_reserved_mib"] == 0 and state["pending_release_mib"] == 300
            monitor.refresh(300)
            assert app.state.memory_admission.snapshot(monitor.snapshot())["reserved_mib"] == 0

    asyncio.run(scenario())


def test_memory_admission_cannot_be_enabled_without_probing():
    from rag_engine.admission import MemoryAdmissionConfig
    with pytest.raises(ValueError, match="requires memory probing"):
        create_app(memory_admission=MemoryAdmissionConfig(enabled=True, request_budget_mib=300))


def test_simultaneous_burst_admits_only_requests_whose_budgets_fit(monkeypatch):
    from fastapi import HTTPException
    monitor, pipeline = ControlledMemoryMonitor(), FakePipeline()
    app = memory_admission_app(monkeypatch, pipeline, monitor, request_budget_mib=200)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/query")

    async def scenario():
        async with app.router.lifespan_context(app):
            callers = [asyncio.create_task(endpoint(QueryRequest(query="block"), Response()))
                       for _ in range(8)]
            try:
                assert await asyncio.to_thread(pipeline.started.wait, 5)
                async with asyncio.timeout(3):
                    while sum(task.done() for task in callers) < 6:
                        await asyncio.sleep(.001)
                state = app.state.memory_admission.snapshot(monitor.snapshot())
                assert len(app.state.tasks) == state["reserved_requests"] == 2
                assert state["active_reserved_mib"] == 400
            finally:
                pipeline.release.set()
            results = await asyncio.gather(*callers, return_exceptions=True)
            rejected = [result for result in results if isinstance(result, HTTPException)]
            assert len(rejected) == 6 and all(result.status_code == 503 for result in rejected)
            assert all(result.detail["reason"] == "request_budget" for result in rejected)
            assert sum(isinstance(result, list) for result in results) == 2
            assert not app.state.tasks
    asyncio.run(scenario())


def test_real_background_monitor_supplies_admission_timing_and_recovers_from_pending():
    from types import SimpleNamespace
    import time
    from rag_engine.admission import MemoryAdmissionConfig
    entered, release = threading.Event(), threading.Event()
    def sample():
        entered.set()
        assert release.wait(5)
        return {"status": "ok", "unit": "MiB", "scope": "backend_reported_unverified",
                "stats": {"total_mb": 1000, "used_mb": 300, "free_mb": 700}}
    pipeline = FakePipeline()
    app = create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=True,
        memory_probe_factory=lambda _: SimpleNamespace(sample=sample), memory_probe_interval_s=.01,
        memory_admission=MemoryAdmissionConfig(enabled=True, request_budget_mib=300,
            safety_margin_mib=50, allow_unverified_scope=True))
    with TestClient(app) as client:
        try:
            assert entered.wait(5)
            assert client.post("/query", json={"query": "pending"}).status_code == 503
        finally:
            release.set()
        deadline = time.monotonic() + 3
        while client.get("/health").json()["memory_admission"]["status"] != "ready":
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert client.post("/query", json={"query": "ready"}).status_code == 200
        # No request or health call performs a synchronous driver read.
        assert client.get("/health").json()["device_memory"]["sample_started_monotonic"] > 0


def test_dynamic_estimate_runs_before_any_model_work_and_rejects_large_input(monkeypatch):
    from rag_engine.admission import MemoryAdmissionConfig
    monitor, pipeline = ControlledMemoryMonitor(100), FakePipeline()
    monkeypatch.setattr('rag_engine.server.DeviceMemoryMonitor', lambda *a, **kw: monitor)
    class Estimator:
        version = 'test'
        def estimate(self, request):
            return {'estimated_incremental_mib': len(request.query)*100}
    app = create_app(pipeline_factory=lambda _: pipeline, memory_probe_enabled=True,
                     memory_admission=MemoryAdmissionConfig(enabled=True, estimate_requests=True, safety_margin_mib=100),
                     memory_estimator_factory=lambda _: Estimator(), max_inflight=0)
    with TestClient(app) as client:
        rejected = client.post('/query', json={'query': 'large request'})
        assert rejected.status_code == 503
        assert rejected.json()['detail']['reason'] == 'request_budget'
        assert rejected.headers['retry-after'] == '1'
        assert not pipeline.calls
        response = client.post('/query', json={'query': 'ok'})
        assert response.status_code == 200 and response.headers['x-memory-estimate-mib'] == '200'
        h = client.get('/health').json()
        assert h['memory_admission']['pending_release_mib'] == 200
        assert h['memory_estimation']['enabled']


def test_sample_latches_pressure_without_http_polling(monkeypatch):
    monitor = ControlledMemoryMonitor(850)
    app = memory_admission_app(monkeypatch, FakePipeline(), monitor, request_budget_mib=50)
    with TestClient(app) as client:
        monitor.on_sample(monitor.snapshot())
        monitor.refresh(800)
        response = client.post('/query', json={'query': 'still-paused'})
        assert response.status_code == 503
        assert response.json()['detail']['reason'] == 'memory_pressure'
        monitor.refresh(750)
        assert client.post('/query', json={'query': 'low'}).status_code == 200


def test_idle_pressure_reclaims_unused_cache_in_model_executor(monkeypatch):
    from types import SimpleNamespace
    monitor = ControlledMemoryMonitor(850)
    pipeline = FakePipeline()
    reclaimed = threading.Event()
    pipeline.reranker = SimpleNamespace(device='cpu', reclaim_unused_memory=reclaimed.set)
    app = memory_admission_app(monkeypatch, pipeline, monitor, request_budget_mib=50)
    with TestClient(app) as client:
        client.portal.call(monitor.on_sample, monitor.snapshot())
        assert reclaimed.wait(5)
        assert client.get('/health').json()['memory_admission']['reason'] == 'memory_pressure'
        monitor.refresh(750)
        assert client.post('/query', json={'query':'after-real-low-sample'}).status_code == 200
