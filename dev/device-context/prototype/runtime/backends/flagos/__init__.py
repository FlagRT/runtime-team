#!/usr/bin/env python3
"""FlagOS 后端（torch_fl / flagos 设备后端适配，用于锁定训练镜像）。"""

from .backend import FlagosBackend, build

__all__ = ["FlagosBackend", "build"]
