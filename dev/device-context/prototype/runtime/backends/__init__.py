#!/usr/bin/env python3
"""Backend 插件包：接口规范（base）+ 注册表（registry）+ 各厂商实现子目录。

新增厂商：新建子目录 + 实现 RuntimeBackend + 提供 build() 工厂，
然后在 registry._KNOWN_BACKENDS 登记即可被自动发现。
"""

from .base import RuntimeBackend
from . import registry

__all__ = ["RuntimeBackend", "registry"]
