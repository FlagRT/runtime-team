"""Shared model initialization and output for CLI and persistent queries."""

from __future__ import annotations


def create_pipeline(device: str, *, settings=None, execution_mode: str = "concurrent", safety_mib=5120):
    from . import Settings, create_retrieval_store
    from .vllm_embedding import create_embedder
    from .pipeline import RetrievalPipeline
    from .reranker import Qwen3Reranker
    from .retrieval import validate_execution_mode

    validate_execution_mode(execution_mode)
    if settings is None:
        settings = Settings.from_env()
    from dataclasses import replace
    backends = settings.retrieval_backends or (settings.retrieval_backend,)
    if settings.retrieval_backend not in backends:
        raise ValueError('Default retrieval backend must be enabled')
    stores = {}
    for backend in backends:
        store = create_retrieval_store(replace(settings, retrieval_backend=backend))
        store.require_connection()
        if not store.has_index():
            raise RuntimeError(
                f"{store.backend_name} resource {store.resource_name} does not exist; "
                "run scripts/data/create_index.py and ingest documents first"
            )
        stores[backend] = store
    store = stores[settings.retrieval_backend]
    embedder = create_embedder(
        settings.embedding_model_path,
        device=device,
        output_dims=settings.embedding_dims,
        instruction=settings.retrieval_instruction,
    )
    reranker = Qwen3Reranker(
        settings.reranker_model_path,
        device=device,
        instruction=settings.retrieval_instruction,
    )
    generator = None
    if settings.generation_enabled:
        from .generation import GenerationLimits, OneRecGenerator
        limits = GenerationLimits(settings.generation_max_input_tokens, settings.generation_max_new_tokens)
        if settings.generation_backend == 'vllm':
            from .vllm_generation import VLLMOneRecGenerator
            generator = VLLMOneRecGenerator(settings.generation_model_path,
                url=settings.generation_url, physical_die=settings.generation_physical_die, limits=limits)
        else:
            generator = OneRecGenerator(settings.generation_model_path, device=device,
                limits=limits, safety_mib=safety_mib)
    return RetrievalPipeline(store, embedder, reranker, execution_mode=execution_mode, generator=generator, stores=stores)


def format_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "rank": rank,
            "rerank_score": hit["_rerank_score"],
            "rrf_score": hit["_rrf_score"],
            "retrieval_ranks": hit["_retrieval_ranks"],
            "document_id": hit["_source"]["document_id"],
            "chunk_id": hit["_source"]["chunk_id"],
            "title": hit["_source"]["title"],
            "text": hit["_source"]["text"],
            "source_uri": hit["_source"]["source_uri"],
        }
        for rank, hit in enumerate(hits, start=1)
    ]
