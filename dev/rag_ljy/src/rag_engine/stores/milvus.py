"""Milvus sparse BM25 and dense vector retrieval backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from ..config import Settings


OUTPUT_FIELDS = [
    "chunk_id",
    "document_id",
    "document_version",
    "chunk_index",
    "title",
    "text",
    "source_uri",
    "metadata",
]


class MilvusStore:
    backend_name = "milvus"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        client_options: dict[str, Any] = {
            "uri": settings.milvus_uri,
            "db_name": settings.milvus_database,
        }
        if settings.milvus_token:
            client_options["token"] = settings.milvus_token
        self.client = MilvusClient(**client_options)

    @property
    def resource_name(self) -> str:
        return self.settings.milvus_collection_name

    def require_connection(self) -> None:
        try:
            self.client.list_collections()
        except Exception as error:
            raise RuntimeError(
                f"Cannot connect to Milvus at {self.settings.milvus_uri}"
            ) from error

    def has_index(self) -> bool:
        return bool(self.client.has_collection(collection_name=self.resource_name))

    def create_index(self, recreate: bool = False) -> bool:
        exists = self.has_index()
        if exists and not recreate:
            return False
        if exists:
            self.client.drop_collection(collection_name=self.resource_name)

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=64,
            is_primary=True,
        )
        schema.add_field(
            field_name="document_id", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(field_name="document_version", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=8192
        )
        schema.add_field(
            field_name="text", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="search_text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(
            field_name="source_uri", datatype=DataType.VARCHAR, max_length=8192
        )
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.settings.embedding_dims,
        )
        schema.add_field(
            field_name="sparse_embedding",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            Function(
                name="search_text_bm25",
                input_field_names=["search_text"],
                output_field_names=["sparse_embedding"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        self.client.create_collection(
            collection_name=self.resource_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        return True

    def bulk_index(self, chunks: Iterable[dict[str, Any]]) -> tuple[int, list[Any]]:
        entities = [self._to_entity(chunk) for chunk in chunks]
        if not entities:
            return 0, []
        self.client.upsert(collection_name=self.resource_name, data=entities)
        return len(entities), []

    def bm25_search(self, query: str, size: int = 50) -> list[dict[str, Any]]:
        response = self.client.search(
            collection_name=self.resource_name,
            data=[query],
            anns_field="sparse_embedding",
            limit=size,
            output_fields=OUTPUT_FIELDS,
        )
        return _normalize_hits(response, "bm25")

    def dense_search(
        self,
        query_vector: list[float],
        size: int = 50,
        num_candidates: int = 100,
    ) -> list[dict[str, Any]]:
        # Milvus AUTOINDEX chooses its own candidate-search parameters.
        _ = num_candidates
        response = self.client.search(
            collection_name=self.resource_name,
            data=[query_vector],
            anns_field="embedding",
            limit=size,
            output_fields=OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE"},
        )
        return _normalize_hits(response, "dense")

    def _to_entity(self, chunk: dict[str, Any]) -> dict[str, Any]:
        embedding = chunk.get("embedding")
        if embedding is None:
            # Milvus requires its dense-vector field on every row. A BM25-only
            # smoke test uses a placeholder that a later upsert replaces.
            embedding = [0.0] * self.settings.embedding_dims
            embedding[0] = 1.0
        if len(embedding) != self.settings.embedding_dims:
            raise ValueError(
                f"Chunk {chunk['chunk_id']} has embedding dimension {len(embedding)}; "
                f"expected {self.settings.embedding_dims}"
            )
        title = str(chunk["title"])
        text = str(chunk["text"])
        return {
            "chunk_id": str(chunk["chunk_id"]),
            "document_id": str(chunk["document_id"]),
            "document_version": int(chunk["document_version"]),
            "chunk_index": int(chunk["chunk_index"]),
            "title": title,
            "text": text,
            "search_text": f"{title}\n{text}",
            "source_uri": str(chunk["source_uri"]),
            "metadata": chunk.get("metadata", {}),
            "embedding": embedding,
        }


def _normalize_hits(response: Any, channel: str) -> list[dict[str, Any]]:
    hits = response[0] if response else []
    normalized = []
    for hit in hits:
        source = dict(hit.get("entity") or {})
        identifier = str(hit.get("id") or source.get("chunk_id"))
        source.setdefault("chunk_id", identifier)
        normalized.append(
            {
                "_id": identifier,
                "_source": source,
                "_score": float(hit.get("distance") or 0.0),
                "_channel": channel,
            }
        )
    return normalized
