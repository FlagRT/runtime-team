"""Environment-backed configuration for the RAG pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBEDDING_REPO = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_REPO = "Qwen/Qwen3-Reranker-0.6B"
SUPPORTED_RETRIEVAL_BACKENDS = frozenset({"elasticsearch", "milvus"})


@dataclass(frozen=True)
class Settings:
    retrieval_backend: str
    elasticsearch_url: str
    elasticsearch_username: str
    elasticsearch_password: str
    elasticsearch_index_name: str
    milvus_uri: str
    milvus_token: str
    milvus_database: str
    milvus_collection_name: str
    model_dir: Path
    embedding_model_path: Path
    reranker_model_path: Path
    embedding_dims: int
    retrieval_instruction: str

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(
            os.getenv("MODEL_DIR", "/mnt/raid/jliu171/models")
        ).expanduser()
        port = os.getenv("ELASTICSEARCH_PORT", "9200")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        retrieval_backend = os.getenv(
            "RETRIEVAL_BACKEND", "elasticsearch"
        ).strip().lower()
        if retrieval_backend not in SUPPORTED_RETRIEVAL_BACKENDS:
            choices = ", ".join(sorted(SUPPORTED_RETRIEVAL_BACKENDS))
            raise ValueError(
                f"RETRIEVAL_BACKEND must be one of: {choices}; "
                f"got {retrieval_backend!r}"
            )
        password = os.getenv("ELASTICSEARCH_PASSWORD") or os.getenv(
            "ELASTIC_PASSWORD", ""
        )
        if retrieval_backend == "elasticsearch" and not password:
            raise RuntimeError(
                "ELASTIC_PASSWORD (or ELASTICSEARCH_PASSWORD) is not set"
            )

        return cls(
            retrieval_backend=retrieval_backend,
            elasticsearch_url=os.getenv(
                "ELASTICSEARCH_URL", f"http://127.0.0.1:{port}"
            ),
            elasticsearch_username=os.getenv(
                "ELASTICSEARCH_USERNAME", "elastic"
            ),
            elasticsearch_password=password,
            elasticsearch_index_name=os.getenv(
                "ELASTICSEARCH_INDEX",
                os.getenv("RAG_INDEX_NAME", "rag-chunks-v1"),
            ),
            milvus_uri=os.getenv(
                "MILVUS_URI", f"http://127.0.0.1:{milvus_port}"
            ),
            milvus_token=os.getenv("MILVUS_TOKEN", ""),
            milvus_database=os.getenv("MILVUS_DATABASE", "default"),
            milvus_collection_name=os.getenv(
                "MILVUS_COLLECTION", "rag_chunks_v1"
            ),
            model_dir=model_dir,
            embedding_model_path=Path(
                os.getenv(
                    "EMBEDDING_MODEL_PATH",
                    str(model_dir / DEFAULT_EMBEDDING_REPO),
                )
            ),
            reranker_model_path=Path(
                os.getenv(
                    "RERANKER_MODEL_PATH",
                    str(model_dir / DEFAULT_RERANKER_REPO),
                )
            ),
            embedding_dims=int(os.getenv("EMBEDDING_DIMS", "1024")),
            retrieval_instruction=os.getenv(
                "RETRIEVAL_INSTRUCTION",
                "Given a web search query, retrieve relevant passages that answer the query",
            ),
        )

    @property
    def basic_auth(self) -> tuple[str, str]:
        return self.elasticsearch_username, self.elasticsearch_password

    @property
    def resource_name(self) -> str:
        if self.retrieval_backend == "elasticsearch":
            return self.elasticsearch_index_name
        return self.milvus_collection_name

    @property
    def index_name(self) -> str:
        """Backward-compatible Elasticsearch index name."""
        return self.elasticsearch_index_name
