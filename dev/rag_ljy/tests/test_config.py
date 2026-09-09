import pytest

from rag_engine.config import Settings


DATABASE_ENV_KEYS = (
    "RETRIEVAL_BACKEND",
    "ELASTIC_PASSWORD",
    "ELASTICSEARCH_PASSWORD",
    "ELASTICSEARCH_INDEX",
    "MILVUS_URI",
    "MILVUS_COLLECTION",
)


def clear_database_environment(monkeypatch) -> None:
    for key in DATABASE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_milvus_backend_does_not_require_elasticsearch_password(monkeypatch) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_BACKEND", "milvus")

    settings = Settings.from_env()

    assert settings.retrieval_backend == "milvus"
    assert settings.resource_name == "rag_chunks_v1"
    assert settings.elasticsearch_password == ""


def test_elasticsearch_backend_requires_password(monkeypatch) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_BACKEND", "elasticsearch")

    with pytest.raises(RuntimeError, match="ELASTIC_PASSWORD"):
        Settings.from_env()


def test_unknown_backend_is_rejected(monkeypatch) -> None:
    clear_database_environment(monkeypatch)
    monkeypatch.setenv("RETRIEVAL_BACKEND", "unknown")

    with pytest.raises(ValueError, match="RETRIEVAL_BACKEND"):
        Settings.from_env()
