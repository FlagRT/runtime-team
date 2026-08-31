"""Small helpers for selecting PyTorch CPU or Ascend NPU devices."""

from __future__ import annotations


def prepare_torch_device(device: str):
    import torch

    if device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "torch_npu is required for an npu device; run inside the NPU container"
            ) from error
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("torch_npu is installed, but no Ascend NPU is available")
    elif device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch, torch.device(device)


def inference_dtype(torch, device: str):
    if device.startswith(("npu", "cuda")):
        return torch.bfloat16
    return torch.float32
