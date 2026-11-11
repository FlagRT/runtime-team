"""Verify routing and placement without vendor libraries or accelerator hardware."""

import sys
from types import SimpleNamespace

import pytest

from rag_engine import device


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.delenv("DC_BACKEND", raising=False)
    monkeypatch.delenv("RAG_DTYPE", raising=False)
    monkeypatch.setattr(device, "_selected_backend", None)
    calls = []
    backends = {name: SimpleNamespace(device_type=prefix) for name, prefix in (
        ("ascend", "npu"), ("kunlun", "cuda"), ("flagos", "flagos"),
    )}
    runtime = SimpleNamespace(
        get=lambda name: backends[name],
        use=lambda name: calls.append(("use", name)),
        device_count=lambda: 4,
        set_device=lambda index: calls.append(("set_device", index)),
    )
    torch = SimpleNamespace(
        device=lambda name: name, bfloat16="bf16", float32="fp32", float16="fp16",
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 4,
            set_device=lambda index: calls.append(("native_cuda", index)),
        ),
    )
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls, runtime, torch


@pytest.mark.parametrize("backend,requested,expected", [
    (None, "npu:3", "npu:3"),
    (None, "auto", "npu:0"),
    ("kunlun", "auto", "cuda:0"),
    ("kunlun", "cuda:2", "cuda:2"),
    ("flagos", "auto", "flagos:0"),
])
def test_vendor_routing(stack, monkeypatch, backend, requested, expected):
    calls, _, torch = stack
    if backend:
        monkeypatch.setenv("DC_BACKEND", backend)
    assert device.prepare_torch_device(requested) == (torch, expected)
    assert calls == [("use", backend or "ascend"), ("set_device", int(expected.split(":")[1]))]


def test_conflicting_device_fails_before_binding(stack, monkeypatch):
    monkeypatch.setenv("DC_BACKEND", "kunlun")
    with pytest.raises(ValueError, match="uses cuda, not npu"):
        device.prepare_torch_device("npu:0")
    assert stack[0] == []


@pytest.mark.parametrize("requested", ["npu:-1", "npu:abc", "invalid", "npu:4"])
def test_invalid_or_invisible_device_is_not_bound(stack, requested):
    with pytest.raises(ValueError):
        device.prepare_torch_device(requested)
    assert not any(call[0] == "set_device" for call in stack[0])


def test_cpu_bypasses_runtime_even_with_vendor_env(stack, monkeypatch):
    monkeypatch.setenv("DC_BACKEND", "ascend")
    monkeypatch.setitem(sys.modules, "runtime", None)
    assert device.prepare_torch_device("cpu")[1] == "cpu:0"
    assert stack[0] == []


def test_native_cuda_is_preserved_without_backend_selection(stack, monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime", None)
    assert device.prepare_torch_device("cuda:2")[1] == "cuda:2"
    assert stack[0] == [("native_cuda", 2)]


def test_missing_runtime_has_actionable_error(stack, monkeypatch):
    monkeypatch.setitem(sys.modules, "runtime", None)
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        device.prepare_torch_device("npu:0")


def test_backend_cannot_change_after_initialization(stack, monkeypatch):
    device.prepare_torch_device("npu:0")
    monkeypatch.setenv("DC_BACKEND", "kunlun")
    with pytest.raises(RuntimeError, match="one device-context backend"):
        device.prepare_torch_device("auto")


def test_vendor_errors_are_not_hidden(stack):
    def fail(index):
        raise RuntimeError("vendor initialization failed")
    stack[1].set_device = fail
    with pytest.raises(RuntimeError, match="vendor initialization failed"):
        device.prepare_torch_device("auto")


def test_dtype_uses_resolved_device_and_explicit_override(stack, monkeypatch):
    torch = stack[2]
    assert device.inference_dtype(torch, "cpu:0") == "fp32"
    for resolved in ("npu:0", "cuda:0", "flagos:0"):
        assert device.inference_dtype(torch, resolved) == "bf16"
    monkeypatch.setenv("RAG_DTYPE", "float16")
    assert device.inference_dtype(torch, "cuda:0") == "fp16"
    monkeypatch.setenv("RAG_DTYPE", "bad")
    with pytest.raises(ValueError, match="RAG_DTYPE"):
        device.inference_dtype(torch, "npu:0")


def test_both_models_use_runtime_device_and_dtype(stack, monkeypatch, tmp_path):
    from rag_engine.embedding import Qwen3Embedder
    from rag_engine.reranker import Qwen3Reranker

    monkeypatch.setenv("DC_BACKEND", "kunlun")
    stack[2].nn = SimpleNamespace(functional=object())
    loads, placements = [], []

    class Model:
        def to(self, target):
            placements.append(target)
            return self

        def eval(self):
            return self

    def load_model(path, **kwargs):
        loads.append(kwargs)
        return Model()

    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: {"yes": 1, "no": 2}[token],
        encode=lambda *args, **kwargs: [1, 2],
    )
    model_factory = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=model_factory, AutoModelForCausalLM=model_factory,
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
    ))
    embedder = Qwen3Embedder(tmp_path)
    reranker = Qwen3Reranker(tmp_path)
    assert embedder.device == reranker.device == "cuda:0"
    assert placements == ["cuda:0", "cuda:0"]
    assert [load["torch_dtype"] for load in loads] == ["bf16", "bf16"]
