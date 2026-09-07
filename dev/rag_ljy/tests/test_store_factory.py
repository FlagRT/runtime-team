import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_engine.stores.factory import create_retrieval_store


@pytest.mark.parametrize(
    ("backend", "module_name", "class_name"),
    [
        ("elasticsearch", "rag_engine.stores.elasticsearch", "ElasticsearchStore"),
        ("milvus", "rag_engine.stores.milvus", "MilvusStore"),
    ],
)
def test_factory_creates_only_the_selected_backend(
    monkeypatch, backend: str, module_name: str, class_name: str
) -> None:
    module = ModuleType(module_name)

    class FakeStore:
        def __init__(self, settings) -> None:
            self.settings = settings

    setattr(module, class_name, FakeStore)
    monkeypatch.setitem(sys.modules, module_name, module)
    settings = SimpleNamespace(retrieval_backend=backend)

    store = create_retrieval_store(settings)

    assert isinstance(store, FakeStore)
    assert store.settings is settings


def test_factory_rejects_unknown_backend() -> None:
    settings = SimpleNamespace(retrieval_backend="unknown")

    with pytest.raises(ValueError, match="Unsupported retrieval backend"):
        create_retrieval_store(settings)
