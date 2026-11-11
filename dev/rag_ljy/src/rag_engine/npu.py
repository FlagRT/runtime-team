"""Compatibility imports; new callers should use rag_engine.device."""

from .device import inference_dtype, prepare_torch_device

__all__ = ["inference_dtype", "prepare_torch_device"]
