"""Multi-request API with bounded embedding, retrieval, and reranking queues."""

from __future__ import annotations

import asyncio
import logging
import json
from time import perf_counter_ns
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from .query_runtime import create_pipeline, format_hits
from .retrieval import validate_execution_mode
from .scheduler import StageQueue


logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=32768)
    retriever_top_k: int = Field(default=50, ge=1, le=100)
    coarse_top_k: int = Field(default=30, ge=20, le=50)
    fine_top_k: int = Field(default=5, ge=3, le=5)
    embedding_batch_size: int = Field(default=8, ge=1, le=8)
    reranker_batch_size: int = Field(default=4, ge=1, le=4)


def create_app(
    device: str = "auto", pipeline_factory=None, *, execution_mode: str = "concurrent",
    max_inflight: int = 8, retrieval_workers: int = 4,
    embedding_workers: int = 2, stage_queue_capacity: int | None = None,
) -> FastAPI:
    """Advance requests independently through bounded FIFO stage queues.

    Remote vLLM embedding calls use a separate pool. Local accelerator models
    remain on their initialization thread. Full stage queues apply backpressure
    within the overall admitted-request bound; new excess requests receive 503.
    """
    validate_execution_mode(execution_mode)
    if stage_queue_capacity is None:
        stage_queue_capacity = max_inflight
    if min(max_inflight, retrieval_workers, embedding_workers, stage_queue_capacity) < 1:
        raise ValueError("Worker counts and request/queue capacities must be positive")
    factory = pipeline_factory or partial(create_pipeline, execution_mode=execution_mode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-inference")
        retrieval_executor = ThreadPoolExecutor(
            max_workers=retrieval_workers, thread_name_prefix="rag-retrieval")
        embedding_executor = None
        app.state.stage_queues = {}
        app.state.ready = False
        app.state.tasks = set()
        app.state.executor = executor
        app.state.retrieval_executor = retrieval_executor
        try:
            app.state.pipeline = await asyncio.get_running_loop().run_in_executor(
                executor, factory, device)
            embedder = getattr(app.state.pipeline, "embedder", None)
            app.state.embedding_backend = getattr(embedder, "backend", "transformers")
            app.state.device = str(getattr(embedder, "device", device))
            remote_embedding = app.state.embedding_backend == "vllm"
            if remote_embedding:
                embedding_executor = ThreadPoolExecutor(
                    max_workers=embedding_workers, thread_name_prefix="rag-embedding-http")
            app.state.stage_queues = {
                "embedding": StageQueue("embedding", embedding_executor or executor,
                    capacity=stage_queue_capacity, workers=embedding_workers if remote_embedding else 1),
                "retrieval": StageQueue("retrieval", retrieval_executor,
                    capacity=stage_queue_capacity, workers=retrieval_workers),
                "rerank": StageQueue("rerank", executor,
                    capacity=stage_queue_capacity, workers=1),
            }
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            # Accepted requests keep ownership even if their HTTP waiter is cancelled.
            # Drain all stages before closing the pipeline or either executor.
            try:
                await asyncio.gather(*list(app.state.tasks), return_exceptions=True)
                for queue in app.state.stage_queues.values():
                    await queue.close()
                close = getattr(getattr(app.state, "pipeline", None), "close", None)
                if close is not None:
                    await asyncio.get_running_loop().run_in_executor(executor, close)
            finally:
                if embedding_executor is not None:
                    embedding_executor.shutdown(wait=True, cancel_futures=True)
                retrieval_executor.shutdown(wait=True, cancel_futures=True)
                executor.shutdown(wait=True, cancel_futures=True)
                app.state.pipeline = None

    app = FastAPI(title="RAG retrieval and reranking", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Models are not ready")
        active = len(app.state.tasks)
        return {"status": "ready", "device": app.state.device, "busy": bool(active),
                "inflight": active, "max_inflight": max_inflight,
                "embedding_backend": app.state.embedding_backend,
                "retrieval_backend": getattr(getattr(app.state.pipeline, "store", None), "backend_name", "unknown"),
                "execution_mode": execution_mode,
                "stage_queues": {name: queue.snapshot()
                                 for name, queue in app.state.stage_queues.items()}}

    async def process(request, request_id):
        pipeline = app.state.pipeline
        timings = {}
        started = perf_counter_ns()

        async def stage(name, operation):
            queued = perf_counter_ns()

            def work():
                begin = perf_counter_ns()
                timings[name + "_queue_ms"] = (begin - queued) / 1_000_000
                logger.info("request_id=%s stage=%s event=start monotonic_ns=%d queue_ms=%.3f",
                            request_id, name, begin, timings[name + "_queue_ms"])
                try:
                    return operation()
                finally:
                    logger.info("request_id=%s stage=%s event=end monotonic_ns=%d",
                                request_id, name, perf_counter_ns())

            return await app.state.stage_queues[name].run(work)

        status = "error"
        try:
            vector = await stage("embedding", partial(
                pipeline.embed, request.query, request.embedding_batch_size, timings))
            candidates = await stage("retrieval", partial(
                pipeline.search, request.query, vector, request.retriever_top_k,
                request.coarse_top_k, timings))
            hits = await stage("rerank", partial(
                pipeline.rank, request.query, candidates, request.fine_top_k,
                request.reranker_batch_size, timings))
            result = format_hits(hits)
            status = "ok"
            return result
        finally:
            timings["total_ms"] = (perf_counter_ns() - started) / 1_000_000
            logger.info("request_id=%s event=finished status=%s timings=%s",
                        request_id, status, json.dumps(timings, sort_keys=True))

    @app.post("/query")
    async def query(request: QueryRequest, response: Response):
        request_id = uuid4().hex
        headers = {"X-Request-ID": request_id}
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Service is not ready", headers=headers)
        # No await between capacity check and task registration: atomic on this loop.
        if len(app.state.tasks) >= max_inflight:
            raise HTTPException(status_code=503, detail="Request capacity is full; retry later",
                                headers={**headers, "Retry-After": "1"})
        response.headers["X-Request-ID"] = request_id
        task = asyncio.create_task(process(request, request_id), name="rag-" + request_id)
        app.state.tasks.add(task)
        logger.info("request_id=%s event=accepted inflight=%d", request_id, len(app.state.tasks))

        def finished(completed):
            app.state.tasks.discard(completed)
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error("request_id=%s query failed", request_id,
                                 exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except Exception as error:
            raise HTTPException(status_code=500, detail="Query failed; see server logs",
                                headers=headers) from error

    return app
