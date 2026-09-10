"""Exercise persistent model ownership and HTTP behavior without NPU hardware."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from rag_engine.server import QueryRequest, create_app


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.threads = []
        self.started = threading.Event()
        self.release = threading.Event()

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        self.threads.append(threading.get_ident())
        if kwargs["query"] == "block":
            self.started.set()
            assert self.release.wait(timeout=5)
        if kwargs["query"] == "fail":
            raise RuntimeError("private database connection details")
        return [{
            "_rerank_score": 0.9,
            "_rrf_score": 0.03,
            "_retrieval_ranks": {"dense": 1},
            "_source": {
                "document_id": "doc", "chunk_id": "chunk", "title": "Title",
                "text": kwargs["query"], "source_uri": "sample://doc",
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
        assert client.get("/health").json() == {
            "status": "ready", "device": "npu:3", "busy": False,
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
        assert pipeline.threads == [loads[0][1], loads[0][1]]
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


def test_busy_request_rejected_and_health_responsive():
    pipeline = FakePipeline()
    with TestClient(create_app(pipeline_factory=lambda _: pipeline)) as client:
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


def test_cancelled_request_keeps_busy_until_inference_finishes():
    pipeline = FakePipeline()
    app = create_app(pipeline_factory=lambda _: pipeline)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/query")

    async def scenario():
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(endpoint(QueryRequest(query="block")))
            try:
                assert await asyncio.to_thread(pipeline.started.wait, 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert app.state.busy
            finally:
                pipeline.release.set()
            # Enqueue a barrier on the same worker, then allow callbacks to run.
            await asyncio.get_running_loop().run_in_executor(app.state.executor, lambda: None)
            await asyncio.sleep(0)
            assert not app.state.busy
            result = await endpoint(QueryRequest(query="next"))
            assert result[0]["text"] == "next"

    asyncio.run(scenario())
