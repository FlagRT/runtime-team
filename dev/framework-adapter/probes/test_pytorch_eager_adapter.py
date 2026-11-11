"""Standalone tests: neither this suite nor the adapter imports vLLM."""
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn.functional as F

import pytorch_eager_adapter as adapter
from pytorch_eager_adapter import PreferGems


def test_no_vllm_import():
    assert not any(n == "vllm" or n.startswith(("vllm.", "vllm_fl")) for n in sys.modules)


@pytest.fixture(scope="session", autouse=True)
def stays_independent_of_vllm():
    yield
    test_no_vllm_import()


@pytest.mark.parametrize("module", [False, True])
def test_cpu_native_and_scope_restore(module):
    x = torch.randn(3, 16)
    fn = torch.nn.SiLU() if module else F.silu
    expected = fn(x)
    with PreferGems() as mode:
        torch.testing.assert_close(fn(x), expected)
        assert mode.routes[-1].reason == "device"
    count = len(mode.routes)
    fn(x)
    assert len(mode.routes) == count


def test_nested_scope_rejected_and_restored():
    with PreferGems():
        with pytest.raises(RuntimeError, match="Nested"):
            with PreferGems():
                pass
    with PreferGems():
        F.silu(torch.ones(2))


def test_scope_is_thread_local():
    with PreferGems() as mode, ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(F.silu, torch.ones(2)).result()
        assert not mode.routes


def test_execution_error_not_retried(monkeypatch):
    # Only routing control is mocked; real kernels are tested separately below.
    monkeypatch.setattr(adapter, "_common_reason", lambda *a: None)
    x = torch.ones(2)
    def fail(t):
        t.add_(1)
        raise RuntimeError("partial write")
    monkeypatch.setattr(adapter, "_load_gems", lambda: SimpleNamespace(silu=fail))
    with PreferGems() as mode:
        with pytest.raises(RuntimeError, match="partial write"):
            F.silu(x)
    assert [r.backend for r in mode.routes] == ["flaggems", "flaggems"]
    torch.testing.assert_close(x, torch.full((2,), 2.0))
    # Exception exit restores the original API as well.
    assert F.silu(x).shape == x.shape


def test_missing_optional_library(monkeypatch):
    monkeypatch.setattr(adapter, "_common_reason", lambda *a: None)
    def absent():
        raise ModuleNotFoundError("missing", name="flag_gems")
    monkeypatch.setattr(adapter, "_load_gems", absent)
    with PreferGems() as mode:
        F.silu(torch.ones(2))
    assert mode.routes[-1].reason == "missing_flag_gems"


def test_broken_dependency_not_silently_hidden(monkeypatch):
    monkeypatch.setattr(adapter, "_common_reason", lambda *a: None)
    def broken():
        raise ModuleNotFoundError("broken dependency", name="triton")
    monkeypatch.setattr(adapter, "_load_gems", broken)
    with PreferGems(), pytest.raises(ModuleNotFoundError, match="broken dependency"):
        F.silu(torch.ones(2))


@pytest.fixture
def npu():
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    torch.manual_seed(135)
    yield torch.device("npu:0")
    torch.npu.synchronize()


def close(a, b, dtype):
    tol = 0.03 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(a.cpu().float(), b.cpu().float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (7, 512)])
@pytest.mark.parametrize("module", [False, True])
def test_npu_silu(npu, dtype, shape, module):
    cpu = torch.randn(shape).to(dtype)
    x = cpu.to(npu)
    fn = torch.nn.SiLU() if module else F.silu
    native = fn(x)
    with torch.no_grad(), PreferGems() as mode:
        result = fn(x)
    assert mode.routes[-1].backend == "flaggems"
    assert result.device == x.device and result.dtype == x.dtype
    close(result, native, dtype)
    close(result, F.silu(cpu.float()), dtype)
    torch.testing.assert_close(x.cpu(), cpu, atol=0, rtol=0)


@pytest.mark.parametrize("seed", [0, 1, 7, 135])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_native_gems_without_test_barrier(npu, seed, dtype):
    """No test synchronization between native and adapted invocations."""
    torch.manual_seed(seed)
    cpu = torch.randn(1, 128).to(dtype)
    wc = torch.randn(128).to(dtype)
    x, w = cpu.to(npu), wc.to(npu)
    native = F.rms_norm(x, [128], w)
    with torch.no_grad(), PreferGems() as mode:
        result = F.rms_norm(x, [128], w)
    after = F.rms_norm(x, [128], w)
    eps = torch.finfo(dtype).eps
    ref = cpu.float() * torch.rsqrt(cpu.float().square().mean(-1, keepdim=True) + eps)
    ref *= wc.float()
    assert mode.routes[-1].backend == "flaggems"
    for out in (native, result, after):
        close(out, ref, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (7, 512)])
@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("eps", [None, 1e-6])
def test_npu_rms(npu, dtype, shape, module, eps):
    cpu = torch.randn(shape).to(dtype)
    x = cpu.to(npu)
    weight = torch.randn(shape[-1], dtype=dtype, device=npu)
    layer = torch.nn.RMSNorm((shape[-1],), eps=eps, device=npu, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(weight)
    fn = layer if module else lambda t: F.rms_norm(t, (shape[-1],), weight, eps)
    weight_snapshot = weight.cpu()
    epsilon = torch.finfo(dtype).eps if eps is None else eps
    reference = cpu.float() * torch.rsqrt(cpu.float().square().mean(-1, keepdim=True) + epsilon)
    reference *= weight_snapshot.float()
    with torch.no_grad():
        native = fn(x)
        native_snapshot = native.cpu()
        close(native_snapshot, reference, dtype)
        with PreferGems() as mode:
            result = fn(x)
    assert mode.routes[-1].backend == "flaggems"
    close(result, reference, dtype)
    torch.testing.assert_close(native.cpu(), native_snapshot, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), weight_snapshot, atol=0, rtol=0)
    close(native, reference, dtype)
    close(result, native, dtype)
    assert result.device == x.device and result.dtype == x.dtype


@pytest.mark.parametrize("case", ["dtype", "layout", "empty", "inplace", "disabled", "autograd"])
def test_npu_silu_preflight(npu, case):
    cpu = torch.randn(3, 128).half()
    x = cpu.to(npu)
    if case == "dtype": x = x.float()
    if case == "layout": x = x[:, ::2]
    if case == "empty": x = x[:0]
    if case == "autograd": x.requires_grad_()
    inplace = case == "inplace"
    ref = x.detach().clone().requires_grad_(x.requires_grad)
    expected = F.silu(ref, inplace=inplace)
    with PreferGems(enabled=case != "disabled") as mode:
        result = F.silu(x, inplace=inplace)
    assert mode.routes[-1].reason == case
    close(result, expected, x.dtype)
    if inplace: assert result.data_ptr() == x.data_ptr()
    if case == "autograd":
        result.sum().backward()
        expected.sum().backward()
        close(x.grad, ref.grad, x.dtype)


@pytest.mark.parametrize("case", ["missing_weight", "normalized_shape", "weight", "autograd"])
def test_npu_rms_preflight(npu, case):
    x = torch.randn(3, 128, device=npu).half()
    shape = (3, 128) if case == "normalized_shape" else (128,)
    weight = torch.ones(shape, device=npu, dtype=x.dtype)
    if case == "missing_weight": weight = None
    if case == "weight": weight = torch.ones(256, device=npu, dtype=x.dtype)[::2]
    if case == "autograd": weight.requires_grad_()
    native = F.rms_norm(x, shape, weight)
    with PreferGems() as mode:
        result = F.rms_norm(x, shape, weight)
    assert mode.routes[-1].reason == case
    close(result, native, x.dtype)
    if case == "autograd":
        actual_grad = torch.autograd.grad(result.sum(), weight)[0]
        native_grad = torch.autograd.grad(native.sum(), weight)[0]
        close(actual_grad, native_grad, weight.dtype)


def test_npu_input_changes_not_cached(npu):
    with torch.no_grad(), PreferGems() as mode:
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            F.silu(torch.ones(3, 128, device=npu, dtype=dtype)).cpu()
    assert [r.backend for r in mode.routes] == ["flaggems", "native", "flaggems"]


def test_npu_keyword_calls(npu):
    x = torch.randn(3, 128, device=npu).half()
    w = torch.ones(128, device=npu, dtype=x.dtype)
    with torch.no_grad(), PreferGems() as mode:
        a = F.silu(input=x, inplace=False)
        b = F.rms_norm(input=x, normalized_shape=[128], weight=w, eps=1e-6)
    assert [r.backend for r in mode.routes] == ["flaggems", "flaggems"]
    close(a, F.silu(x), x.dtype)
    close(b, F.rms_norm(x, [128], w, 1e-6), x.dtype)


@pytest.mark.parametrize("seed", [0, 1, 7, 135])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [None, 1e-6])
def test_npu_rms_seed_sweep(npu, seed, dtype, eps):
    """Retest the first-run discrepancy; preserve inputs and native output."""
    torch.manual_seed(seed)
    cpu = torch.randn(1, 128).to(dtype)
    x = cpu.to(npu)
    weight = torch.randn(128, device=npu, dtype=dtype)
    weight_before = weight.cpu()
    native = F.rms_norm(x, [128], weight, eps)
    native_before = native.cpu()
    epsilon = torch.finfo(dtype).eps if eps is None else eps
    ref = cpu.float() * torch.rsqrt(cpu.float().square().mean(-1, keepdim=True) + epsilon)
    ref *= weight_before.float()
    with torch.no_grad(), PreferGems() as mode:
        result = F.rms_norm(x, [128], weight, eps)
    assert mode.routes[-1].backend == "flaggems"
    close(result, ref, dtype)
    close(native_before, ref, dtype)
    torch.testing.assert_close(native.cpu(), native_before, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), weight_before, atol=0, rtol=0)
    torch.testing.assert_close(x.cpu(), cpu, atol=0, rtol=0)
