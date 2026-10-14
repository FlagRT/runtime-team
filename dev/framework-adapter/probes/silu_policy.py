"""Dependency-free, fail-closed admission for the experimental SiLU adapter."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Environment:
    chip: str
    torch: str
    torch_npu: str
    gems_revision: str


@dataclass(frozen=True)
class InputSpec:
    device: str
    dtype: str
    shape: tuple
    contiguous: bool
    strided: bool = True
    exact_tensor: bool = True
    inplace: bool = False
    autocast: bool = False


@dataclass(frozen=True)
class SiluPolicy:
    # Caller must measure environment identity; values are not auto-detected.
    expected: Environment
    shapes: tuple
    dtypes: tuple = ("torch.float32",)
    experimental: bool = False

    def reason(self, actual, spec):
        if not self.experimental:
            return "experimental_disabled"
        if actual != self.expected:
            return "environment_mismatch"
        if spec.device != "npu":
            return "device"
        if not spec.exact_tensor:
            return "tensor_subclass"
        if spec.inplace:
            return "inplace"
        if spec.autocast:
            return "autocast"
        if spec.dtype not in self.dtypes:
            return "dtype"
        if not spec.strided or not spec.contiguous:
            return "layout"
        if not spec.shape or any(n <= 0 for n in spec.shape):
            return "empty_or_scalar"
        if spec.shape not in self.shapes:
            return "shape"
        return None
