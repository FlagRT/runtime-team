"""Real-XPU regressions; run through run_blas_acceptance.sh, not host unittest."""
import json
import os
from pathlib import Path

import pytest
import torch
import flag_gems
from flag_gems.testing import assert_close


@pytest.mark.parametrize("shape", [(1, 1, 32), (15, 160, 1024), (495, 5333, 71), (64, 64, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("matrix_bias", [False, True])
def test_direct_addmm_contract(shape, dtype, matrix_bias, monkeypatch):
    """Check four shapes, broadcast bias, dtype and actual fused-kernel dispatch."""
    assert torch.cuda.is_available(), "This test requires a real XPU device"
    # The backend loader may load this file under a private module alias.
    # Probe the globals of the function actually registered by FlagGems.
    globals_ = flag_gems.addmm.__globals__
    assert Path(globals_["__file__"]).name == "addmm.py"
    original_kernel = globals_["addmm_kernel"]
    launches = []

    class KernelProbe:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                # Bias and destination intentionally have different element types.
                assert args[2].dtype == torch.float32
                assert args[3].dtype == dtype
                launches.append(True)
                return original_kernel[grid](*args, **kwargs)
            return launch

    monkeypatch.setitem(globals_, "addmm_kernel", KernelProbe())
    m, n, k = shape
    generator = torch.Generator(device="cpu").manual_seed(20261027)
    a_cpu = torch.randn((m, k), generator=generator).to(dtype)
    b_cpu = torch.randn((k, n), generator=generator).to(dtype)
    bias_shape = (m, n) if matrix_bias else (n,)
    bias_cpu = torch.randn(bias_shape, generator=generator).to(dtype)
    reference = torch.addmm(bias_cpu.double(), a_cpu.double(), b_cpu.double(),
                            alpha=-0.999, beta=0.5).to(dtype)
    a, b, bias = (value.to(flag_gems.device) for value in (a_cpu, b_cpu, bias_cpu))
    with torch.inference_mode(), flag_gems.use_gems():
        result = torch.addmm(bias, a, b, alpha=-0.999, beta=0.5)
    torch.cuda.synchronize()
    assert launches == [True], "Expected exactly one direct ADDMM-kernel invocation"
    assert result.dtype == dtype
    assert tuple(result.shape) == (m, n)
    actual = result.cpu()
    assert torch.isfinite(actual).all()
    assert_close(actual, reference, dtype, reduce_dim=k)
    record = {
        "shape_mnk": shape, "dtype": str(dtype), "matrix_bias": matrix_bias,
        "seed": 20261027, "alpha": -0.999, "beta": 0.5,
        "reference": "CPU FP64 computed from exact quantized input tensors; cast to output dtype",
        "max_abs_error": (actual.double() - reference.double()).abs().max().item(),
        "relative_l2_error": ((actual.double() - reference.double()).norm()
                              / reference.double().norm().clamp_min(1e-30)).item(),
        "direct_addmm_kernel_calls": len(launches), "status": "passed",
    }
    directory = Path(os.environ["XPU_FIX_IR_DIR"]).parent / "contract-results"
    directory.mkdir(exist_ok=True)
    name = f"{'-'.join(map(str, shape))}-{dtype}-{matrix_bias}.json"
    (directory / name).write_text(json.dumps(record, indent=2) + "\n")
