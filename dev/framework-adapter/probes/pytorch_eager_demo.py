"""Ordinary nn layers on NPU; no vLLM or vllm_fl imports."""
import json
import sys
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from pytorch_eager_adapter import PreferGems


def main():
    torch.npu.set_device(0)
    x = torch.linspace(-2, 2, 3 * 128).reshape(3, 128).half().npu()
    model = torch.nn.Sequential(
        torch.nn.RMSNorm(128, eps=1e-6, device="npu", dtype=torch.float16),
        torch.nn.SiLU(),
    ).eval()
    with torch.inference_mode():
        native = model(x)
        with PreferGems() as mode:
            output = model(x)
            # Float32 is outside the validated accelerated dtype scope.
            fallback = F.silu(x.float())
        torch.testing.assert_close(output.cpu(), native.cpu(), atol=0.005, rtol=0.005)
        torch.testing.assert_close(fallback.cpu(), F.silu(x.float()).cpu())
    assert [r.backend for r in mode.routes] == ["flaggems", "flaggems", "native"]
    assert not any(n == "vllm" or n.startswith(("vllm.", "vllm_fl")) for n in sys.modules)
    print(json.dumps({"routes": [asdict(r) for r in mode.routes],
                      "shape": list(output.shape), "vllm_imported": False}))


if __name__ == "__main__":
    main()
