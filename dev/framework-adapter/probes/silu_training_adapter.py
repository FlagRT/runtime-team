"""Alias for the canonical bridge, preserving historical probe imports."""
import sys
import _adapter_package  # noqa: F401
from flagrt_adapter import bridge

sys.modules[__name__] = bridge
