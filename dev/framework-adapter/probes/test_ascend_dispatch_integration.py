"""Real NPU kernels through dispatch; no model/platform-plugin claim.

Run inside the pinned Ascend container with the feature checkout on PYTHONPATH.
The small input matrix is a functional baseline, not a performance benchmark.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_fl.dispatch import CachedOp, SelectionPolicy, policy_context
from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
from vllm_fl.dispatch.backends.vendor.ascend.ascend import AscendBackend
from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend


BACKENDS = {"flagos": FlagGemsBackend, "vendor": AscendBackend,
            "reference": ReferenceBackend}


@pytest.fixture(autouse=True)
def device():
    torch.npu.set_device(0)
    torch.manual_seed(135)
    with torch.no_grad():
        yield
    torch.npu.synchronize()


def check(actual, expected, dtype):
    atol, rtol = (0.03, 0.03) if dtype == torch.bfloat16 else (0.005, 0.005)
    torch.testing.assert_close(actual.cpu().float(), expected.float(),
                               atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 128), (7, 512)])
def test_silu(backend, dtype, shape):
    cpu = torch.randn(*shape).to(dtype)
    gate, up = cpu.float().chunk(2, dim=-1)
    expected = torch.nn.functional.silu(gate) * up
    actual = BACKENDS[backend]().silu_and_mul(None, cpu.to("npu"))
    check(actual, expected, dtype)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("residual", [False, True])
def test_rms(backend, dtype, residual):
    cpu = torch.randn(7, 128).to(dtype)
    weight = torch.randn(128).to(dtype)
    res = torch.randn_like(cpu) if residual else None
    summed = cpu.float() + res.float() if residual else cpu.float()
    expected = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
    expected = expected * weight.float()
    obj = SimpleNamespace(weight=weight.to("npu"), variance_epsilon=1e-6)
    actual = BACKENDS[backend]().rms_norm(
        obj, cpu.to("npu"), res.to("npu") if residual else None)
    if residual:
        actual, actual_res = actual
        check(actual_res, summed, dtype)
    check(actual, expected, dtype)


def rope_reference(x, cos, sin, pos, interleaved):
    c, s = cos[pos].float().unsqueeze(1), sin[pos].float().unsqueeze(1)
    x = x.float()
    if interleaved:
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a*c-b*s, b*c+a*s), -1).flatten(-2)
    a, b = x.chunk(2, -1)
    return torch.cat((a*c-b*s, b*c+a*s), -1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_rope(backend, dtype, interleaved, inplace, strided):
    q = torch.randn(7, 4, 64).to(dtype)
    k = torch.randn(7, 2, 64).to(dtype)
    angles = torch.randn(32, 32)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    pos = torch.tensor([0, 2, 4, 7, 9, 15, 23], dtype=torch.int64)
    eq, ek = (rope_reference(t, cos, sin, pos, interleaved) for t in (q, k))
    nq, nk = q.to("npu"), k.to("npu")
    if strided:
        def make_strided(t):
            out = torch.empty(*t.shape[:-1], t.shape[-1] * 2,
                              dtype=t.dtype, device="npu")[..., ::2]
            return out.copy_(t)
        nq, nk = make_strided(nq), make_strided(nk)
    aq, ak = BACKENDS[backend]().rotary_embedding(
        None, nq, nk, cos.to("npu"), sin.to("npu"), pos.to("npu"),
        rotary_interleaved=interleaved, inplace=inplace)
    check(aq, eq, dtype)
    check(ak, ek, dtype)
    if inplace:
        assert aq.data_ptr() == nq.data_ptr()
        assert ak.data_ptr() == nk.data_ptr()
        check(nq, eq, dtype)
        check(nk, ek, dtype)
    else:
        torch.testing.assert_close(nq.cpu(), q, rtol=0, atol=0)
        torch.testing.assert_close(nk.cpu(), k, rtol=0, atol=0)


def test_real_cached_dispatch_preflight(monkeypatch):
    """Controlled capability rejection selects actual native kernels.

    The row-count predicate below is test injection, not an Ascend limitation.
    """
    import os
    import vllm_fl.dispatch as dispatch
    from vllm_fl.dispatch import OpImpl, BackendImplKind as Kind

    dispatch.reset_default_manager()
    mgr = dispatch.get_default_manager()
    mgr._state.initialized, mgr._state.init_pid = True, os.getpid()
    calls = []

    def tracked(name, backend):
        def invoke(obj, x):
            calls.append(name)
            return backend.silu_and_mul(obj, x)
        return invoke

    mgr.registry.register_many([
        OpImpl("silu_and_mul", "test.flagos", Kind.DEFAULT,
               tracked("flagos", FlagGemsBackend()),
               supports_inputs=lambda obj, x: x.shape[0] == 1,
               runtime_fallback_safe=False),
        OpImpl("silu_and_mul", "test.ascend", Kind.VENDOR,
               tracked("vendor", AscendBackend()), vendor="ascend",
               runtime_fallback_safe=False),
    ])
    monkeypatch.setattr(dispatch, "_OP_FAST_PATH_ENABLED", True)
    try:
        with policy_context(SelectionPolicy(strict=True)):
            op = CachedOp("silu_and_mul")
            for rows in (1, 7, 1):
                cpu = torch.randn(rows, 128).half()
                a, b = cpu.float().chunk(2, -1)
                check(op(None, cpu.to("npu")), torch.nn.functional.silu(a)*b, cpu.dtype)
        assert calls == ["flagos", "vendor", "flagos"]
        assert mgr.get_failed_impls() == {}
    finally:
        dispatch.reset_default_manager()


def test_builtin_rotary_dtype_routing():
    """Actual platform discovery + builtin predicate; no injected capability."""
    import vllm_fl.dispatch as dispatch

    dispatch.reset_default_manager()
    try:
        mgr = dispatch.get_default_manager()
        assert "vendor.ascend" in [i.impl_id for i in mgr.resolve_candidates("rotary_embedding")]
        policy = SelectionPolicy.from_dict(strict=True, per_op_order={
            "rotary_embedding": ["vendor", "reference"]})
        with policy_context(policy):
            op = CachedOp("rotary_embedding")
            for dtype in (torch.float16, torch.bfloat16, torch.float16,
                          torch.float32, torch.float16):
                q = torch.randn(7, 4, 64).to(dtype)
                k = torch.randn(7, 2, 64).to(dtype)
                angles = torch.randn(32, 32)
                cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
                pos = torch.arange(7)
                args = (None, q.npu(), k.npu(), cos.npu(), sin.npu(), pos.npu())
                candidates = mgr.resolve_candidates("rotary_embedding")
                supported = [i for i in candidates if i.supports_call(*args)]
                assert supported[0].impl_id == ("reference.torch" if dtype == torch.float32 else "vendor.ascend")
                aq, ak = op(*args, inplace=False)
                check(aq, rope_reference(q, cos, sin, pos, False), dtype)
                check(ak, rope_reference(k, cos, sin, pos, False), dtype)
        assert mgr.get_failed_impls() == {}
    finally:
        dispatch.reset_default_manager()
