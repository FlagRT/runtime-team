#!/usr/bin/env python3
"""统一运行时 API 子包：对外语义对象与抽象（错误分级、设备、流）。"""

from .errors import DISPOSITION, ErrorCategory, FlagosError, translate_via_backend

__all__ = ["FlagosError", "ErrorCategory", "DISPOSITION", "translate_via_backend"]
