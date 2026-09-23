"""CPU control tests. Mocked target execution is NOT NPU kernel evidence."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from transformers.activations import SiLUActivation
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

import qwen_scoped_adapter as adapter
from qwen_scoped_adapter import QwenScopedGems


@pytest.fixture
def model():
    return torch.nn.Sequential(Qwen3RMSNorm(8, eps=1e-6), SiLUActivation()).eval()


@pytest.mark.parametrize("enabled,reason", [(True, "device"), (False, "disabled")])
def test_native_and_restore(model, enabled, reason):
    x = torch.randn(2, 8)
    with torch.inference_mode():
        expected = model(x)
        with QwenScopedGems(model, enabled=enabled, allowed_ops=("silu", "rms_norm")) as scope:
            torch.testing.assert_close(model(x), expected)
            assert {r["reason"] for r in scope.routes.values()} == {reason}
        assert all("forward" not in m.__dict__ for m in model)
        torch.testing.assert_close(model(x), expected)


def test_operator_allowlist(model):
    with torch.inference_mode(), QwenScopedGems(model, allowed_ops=("silu",)) as scope:
        model(torch.randn(2, 8))
    assert scope.counts[("rms_norm", "native", "operator_not_enabled", "selected")] == 1


def test_grad_uses_native(model):
    x = torch.randn(2, 8, requires_grad=True)
    with QwenScopedGems(model, allowed_ops=("silu", "rms_norm")) as scope:
        model(x).sum().backward()
    assert x.grad is not None
    assert {r["reason"] for r in scope.routes.values()} == {"inference_only"}


def test_default_does_not_enable_unvalidated_ops(model):
    with torch.inference_mode(), QwenScopedGems(model) as scope:
        model(torch.ones(2, 8))
    assert {r["reason"] for r in scope.routes.values()} == {"operator_not_enabled"}


def test_existing_override_preserved(model):
    custom = lambda x: x
    model[1].forward = custom
    with pytest.raises(RuntimeError, match="override"):
        with QwenScopedGems(model):
            pass
    assert "forward" not in model[0].__dict__
    assert model[1].forward is custom


def test_nested_rejected_outer_survives(model):
    with QwenScopedGems(model) as outer:
        installed = model[0].forward
        with pytest.raises(RuntimeError):
            with QwenScopedGems(model):
                pass
        with pytest.raises(RuntimeError):
            outer.__enter__()
        assert model[0].forward is installed
    assert "forward" not in model[0].__dict__


def test_exception_exit_restores(model):
    with pytest.raises(ValueError):
        with QwenScopedGems(model):
            raise ValueError("controlled body failure")
    assert all("forward" not in m.__dict__ for m in model)


def test_argument_error_not_swallowed(model):
    with QwenScopedGems(model), pytest.raises(TypeError):
        model[0](unknown=torch.ones(2, 8))


def test_foreign_thread_rejected(model):
    with QwenScopedGems(model), ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="single-thread"):
            pool.submit(model, torch.ones(2, 8)).result()


def test_unknown_class_rejected():
    class Lookalike(torch.nn.Module):
        def forward(self, x):
            return x
    with pytest.raises(ValueError):
        with QwenScopedGems(Lookalike()):
            pass


def test_bad_allowlist_rejected(model):
    with pytest.raises(ValueError):
        QwenScopedGems(model, allowed_ops=("unknown",))


def test_missing_gems_falls_back(model, monkeypatch):
    scope = QwenScopedGems(model)
    monkeypatch.setattr(scope, "_reason", lambda *args: None)
    def absent(name):
        raise ModuleNotFoundError("absent", name="flag_gems")
    monkeypatch.setattr(adapter.importlib, "import_module", absent)
    with torch.inference_mode(), scope:
        model(torch.ones(2, 8))
    assert {r["reason"] for r in scope.routes.values()} == {"missing_flag_gems"}


def test_dependency_error_not_hidden(model, monkeypatch):
    scope = QwenScopedGems(model)
    monkeypatch.setattr(scope, "_reason", lambda *args: None)
    def broken(name):
        raise ModuleNotFoundError("broken", name="sqlalchemy")
    monkeypatch.setattr(adapter.importlib, "import_module", broken)
    with torch.inference_mode(), pytest.raises(ModuleNotFoundError):
        with scope:
            model(torch.ones(2, 8))
    assert not scope.counts
    assert all("forward" not in m.__dict__ for m in model)


def test_partial_write_error_not_retried(model, monkeypatch):
    scope = QwenScopedGems(model)
    monkeypatch.setattr(scope, "_reason", lambda *args: None)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda *_: None), raising=False)
    calls = []
    def fail(x, *args):
        calls.append(True)
        x.add_(1)
        raise RuntimeError("partial write")
    monkeypatch.setattr(adapter.importlib, "import_module", lambda _: SimpleNamespace(rms_norm=fail))
    x = torch.ones(2, 8)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="partial write"):
        with scope:
            model(x)
    assert len(calls) == 1
    assert not any(k[1] == "native" for k in scope.counts)
    torch.testing.assert_close(x, torch.full_like(x, 2))
    assert all("forward" not in m.__dict__ for m in model)
