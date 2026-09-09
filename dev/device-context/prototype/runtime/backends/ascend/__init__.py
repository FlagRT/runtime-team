#!/usr/bin/env python3
"""昇腾后端（torch_npu + 已有 conformance 资产适配）。"""

from .backend import AscendBackend, build

__all__ = ["AscendBackend", "build"]
