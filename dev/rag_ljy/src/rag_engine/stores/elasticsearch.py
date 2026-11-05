"""Elasticsearch sparse and dense retrieval backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import logging

from elasticsearch import Elasticsearch, BadRequestError, helpers

from ..config import Settings


def index_definition(embedding_dims: int) -> dict[str, Any]:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_version": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "title": {"type": "text"},
                "text": {"type": "text"},
                "source_uri": {"type": "keyword"},
                "metadata": {"type": "object", "dynamic": True},
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        },
    }


class ElasticsearchStore:
    backend_name = "elasticsearch"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = Elasticsearch(
            settings.elasticsearch_url,
            basic_auth=settings.basic_auth,
            request_timeout=60,
        )

    @property
    def resource_name(self) -> str:
        return self.settings.elasticsearch_index_name

    def require_connection(self) -> None:
        if not self.client.ping():
            raise RuntimeError(
                f"Cannot connect to Elasticsearch at {self.settings.elasticsearch_url}"
            )

    def has_index(self) -> bool:
        return bool(self.client.indices.exists(index=self.resource_name))

    def create_index(self, recreate: bool = False) -> bool:
        exists = self.has_index()
        if exists and not recreate:
            return False
        if exists:
            self.client.indices.delete(index=self.resource_name)
        self.client.indices.create(
            index=self.resource_name,
            **index_definition(self.settings.embedding_dims),
        )
        return True

    def bulk_index(self, chunks: Iterable[dict[str, Any]]) -> tuple[int, list[Any]]:
        actions = (
            {
                "_op_type": "index",
                "_index": self.resource_name,
                "_id": chunk["chunk_id"],
                "_source": chunk,
            }
            for chunk in chunks
        )
        success, errors = helpers.bulk(
            self.client,
            actions,
            raise_on_error=False,
            refresh="wait_for",
        )
        return success, errors

    def bm25_search(self, query: str, size: int = 50) -> list[dict[str, Any]]:
        try:
            response = self.client.search(
                index=self.resource_name, size=size,
                query={"multi_match": {"query": query, "fields": ["title^2", "text"], "type": "best_fields"}},
            )
        except BadRequestError as error:
            detail = str(error.body).lower()
            if not any(marker in detail for marker in
                       ('too_many_clauses', 'too_many_nested_clauses', 'too many nested clauses', 'maxclausecount')):
                raise
            # Bound only lexical retrieval. The caller's full query still goes
            # to embedding and reranking. Do not raise the cluster-wide limit.
            bounded = self._bounded_bm25_query(query)
            response = self.client.search(index=self.resource_name, size=size, query=bounded)
        return _normalize_hits(response, "bm25")

    def _bounded_bm25_query(self, query: str) -> dict[str, Any]:
        fields = []
        counts = {}
        for field, boost in (("title", 2.0), ("text", 1.0)):
            analysis = self.client.indices.analyze(index=self.resource_name, field=field, text=query)
            terms = list(dict.fromkeys(item["token"] for item in analysis["tokens"]))
            selected = terms[:256]
            counts[field] = {"unique_terms": len(terms), "selected_terms": len(selected)}
            if selected:
                # Already analyzed terms are not analyzed/expanded again.
                fields.append({"bool": {"should": [{"term": {field: term}} for term in selected],
                                        "minimum_should_match": 1, "boost": boost}})
        logging.getLogger(__name__).warning(
            "event=bm25_clause_limit_fallback policy=first_256_unique_analyzed_terms_per_field counts=%s", counts)
        return {"dis_max": {"queries": fields}} if fields else {"match_none": {}}

    def dense_search(
        self,
        query_vector: list[float],
        size: int = 50,
        num_candidates: int = 100,
    ) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.resource_name,
            size=size,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": size,
                "num_candidates": max(size, num_candidates),
            },
        )
        return _normalize_hits(response, "dense")


def _normalize_hits(response: Any, channel: str) -> list[dict[str, Any]]:
    hits = response["hits"]["hits"]
    return [
        {
            "_id": hit["_id"],
            "_source": hit["_source"],
            "_score": float(hit.get("_score") or 0.0),
            "_channel": channel,
        }
        for hit in hits
    ]
