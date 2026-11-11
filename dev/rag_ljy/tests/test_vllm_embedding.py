"""Contract checks without loading an accelerator or a vLLM engine."""
import math
import sys
from types import SimpleNamespace

import pytest

from rag_engine.vllm_embedding import VllmQwen3Embedder, create_embedder


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    calls = []
    def tokenize(texts, **kwargs):
        calls.append((texts, kwargs))
        return {'input_ids': [[i + 1] for i in range(len(texts))]}
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenize)))
    return VllmQwen3Embedder(tmp_path), calls


def test_format_order_and_normalization(adapter):
    model, calls = adapter
    payloads = []
    def request(payload):
        payloads.append(payload)
        return {'data': [{'index': 1, 'embedding': [0, 2] + [0]*1022},
                         {'index': 0, 'embedding': [3, 0] + [0]*1022}]}
    model._request = request
    vectors = model.encode_queries(['first', 'second'])
    assert vectors[0][:2] == [1, 0] and vectors[1][:2] == [0, 1]
    assert calls[0][0][0] == 'Instruct: ' + model.instruction + '\nQuery:first'
    assert calls[0][1] == {'truncation': True, 'max_length': 8192, 'padding': False}
    assert payloads[0]['input'] == [[1], [2]]
    assert payloads[0]['encoding_format'] == 'float'
    model.encode_documents(['first', 'second'])
    assert calls[1][0] == ['first', 'second']


@pytest.mark.parametrize('data', [[], [{'index': 1, 'embedding': [1]*1024}],
    [{'index': 0, 'embedding': [1]*100}], [{'index': 0, 'embedding': [0]*1024}],
    [{'index': 0, 'embedding': [math.nan]*1024}]])
def test_invalid_vectors_fail_without_fallback(adapter, data):
    model, _ = adapter
    model._request = lambda p: {'data': data}
    with pytest.raises(ValueError):
        model.encode_queries(['query'])


def test_dimension_reduction_renormalizes(adapter):
    model, _ = adapter
    model.output_dims = 32
    model._request = lambda p: {'data': [{'index': 0, 'embedding': [1]*1024}]}
    vector = model.encode_queries(['query'])[0]
    assert len(vector) == 32
    assert sum(v*v for v in vector) == pytest.approx(1)


def test_backend_selection_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('RAG_EMBEDDING_BACKEND', 'unknown')
    with pytest.raises(ValueError, match='RAG_EMBEDDING_BACKEND'):
        create_embedder(tmp_path)
