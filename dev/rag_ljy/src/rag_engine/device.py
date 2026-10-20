"""Device-context integration; vendor libraries stay in the container's PyTorch."""

from __future__ import annotations

import importlib
import os
import re


_selected_backend: str | None = None


def device_selection(device: str = "auto") -> tuple[str | None, str | None, int]:
    """Resolve configuration without importing PyTorch or initializing hardware.

    CPU always bypasses runtime. Legacy CUDA without DC_BACKEND stays native;
    Kunlun must be selected explicitly because it shares the CUDA namespace.
    """
    if device == "auto":
        prefix, ordinal = None, 0
    else:
        match = re.fullmatch(r"(cpu|npu|cuda|flagos|mlu)(?::([0-9]+))?", device)
        if match is None:
            raise ValueError("device must be auto or a device such as npu:0, cuda:0, cpu")
        prefix, index = match.groups()
        ordinal = int(index or 0)
    if prefix == "cpu":
        return None, prefix, ordinal
    backend = os.getenv("DC_BACKEND", "").strip()
    if not backend:
        if prefix == "cuda":
            return None, prefix, ordinal
        backend = {"npu": "ascend", "flagos": "flagos", "mlu": "cambricon"}.get(prefix, "ascend")
    return backend, prefix, ordinal


def prepare_torch_device(device: str = "auto"):
    """Bind the device on the calling inference thread and return torch/device."""
    global _selected_backend
    backend_name, prefix, ordinal = device_selection(device)
    if backend_name is None:
        import torch

        if prefix == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but no CUDA device is available")
            if ordinal >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {ordinal} is not visible")
            torch.cuda.set_device(ordinal)
        return torch, torch.device(f"{prefix}:{ordinal}")

    try:
        runtime = importlib.import_module("runtime")
    except ModuleNotFoundError as error:
        if error.name != "runtime":
            raise
        raise RuntimeError(
            "device-context runtime is not importable. Add the repository's "
            "dev/device-context/prototype to PYTHONPATH (in the standard container: "
            "/workspace/dev/device-context/prototype)."
        ) from error
    if not all(hasattr(runtime, name) for name in ("use", "get", "device_count", "set_device")):
        raise RuntimeError("Imported an unrelated 'runtime' module; check PYTHONPATH")
    if _selected_backend is not None and _selected_backend != backend_name:
        raise RuntimeError("RAG supports one device-context backend per process; restart to switch")
    backend = runtime.get(backend_name)
    if prefix is not None and prefix != backend.device_type:
        raise ValueError(
            f"DC_BACKEND={backend_name} uses {backend.device_type}, not {prefix}; "
            f"use --device {backend.device_type}:{ordinal} or --device auto"
        )
    runtime.use(backend_name)
    count = runtime.device_count()
    if ordinal >= count:
        raise ValueError(f"Device index {ordinal} is not visible for {backend_name} ({count} devices)")
    runtime.set_device(ordinal)
    _selected_backend = backend_name
    # set_device loads the selected vendor extension before torch.device is built.
    import torch

    return torch, torch.device(f"{backend.device_type}:{ordinal}")


def inference_dtype(torch, device):
    """Preserve BF16 accelerator / FP32 CPU defaults; allow an explicit override."""
    default = "float32" if str(device).split(":", 1)[0] == "cpu" else "bfloat16"
    name = os.getenv("RAG_DTYPE", default)
    if name not in {"float32", "float16", "bfloat16"}:
        raise ValueError("RAG_DTYPE must be float32, float16, or bfloat16")
    return getattr(torch, name)
