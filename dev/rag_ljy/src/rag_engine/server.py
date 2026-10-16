"""Persistent retrieval API with one model pair on one inference thread."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .query_runtime import create_pipeline, format_hits


logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=32768)
    retriever_top_k: int = Field(default=50, ge=1, le=100)
    coarse_top_k: int = Field(default=30, ge=20, le=50)
    fine_top_k: int = Field(default=5, ge=3, le=5)
    embedding_batch_size: int = Field(default=8, ge=1, le=8)
    reranker_batch_size: int = Field(default=4, ge=1, le=4)


def create_app(device: str = "npu:0", pipeline_factory=None) -> FastAPI:
    factory = pipeline_factory or create_pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # NPU initialization and all inference run on the same dedicated thread.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-npu")
        app.state.busy = False
        app.state.ready = False
        app.state.executor = executor
        try:
            app.state.pipeline = await asyncio.get_running_loop().run_in_executor(
                executor, factory, device
            )
            app.state.ready = True
            logger.info("RAG models loaded on %s; ready for repeated queries", device)
            yield
        finally:
            app.state.ready = False
            # Do not release the models while an in-flight inference is running.
            executor.shutdown(wait=True, cancel_futures=True)
            app.state.pipeline = None

    app = FastAPI(title="RAG retrieval and reranking", lifespan=lifespan)

    @app.get("/health")
    async def health():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return {"status": "ready", "device": device, "busy": app.state.busy}

    @app.post("/query")
    async def query(request: QueryRequest):
        if app.state.busy:
            raise HTTPException(
                status_code=503,
                detail="An inference is already running; retry after it completes",
                headers={"Retry-After": "1"},
            )
        # This check/set has no await between it and is atomic on the event loop.
        app.state.busy = True

        def infer():
            hits = app.state.pipeline.retrieve(**request.model_dump())
            return format_hits(hits)

        future = asyncio.get_running_loop().run_in_executor(
            app.state.executor, infer
        )

        def finished(completed):
            app.state.busy = False
            # Retrieve failures even when the HTTP client has disconnected.
            if not completed.cancelled():
                error = completed.exception()
                if error is not None:
                    logger.error("RAG query failed", exc_info=(type(error), error, error.__traceback__))

        future.add_done_callback(finished)
        try:
            # A disconnected client must not release the busy flag while the
            # underlying NPU operation continues to use the shared models.
            return await asyncio.shield(future)
        except Exception as error:
            raise HTTPException(
                status_code=500, detail="Query failed; see server logs"
            ) from error

    return app
