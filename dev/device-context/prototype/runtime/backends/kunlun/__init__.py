#!/usr/bin/env python3
"""昆仑芯 P800 后端（XPytorch / torch.cuda 命名空间适配）。

定位：统一运行时原型的**第二个接入实例**（第一个是昇腾）。
      迁移的是《运行时层接口约定》的规范与方法；910C 的实现与结论不迁移。

命名空间：昆仑芯这栈上设备 API 走 `torch.cuda`（`USE_XPU=OFF`），
          故 `device_type = "cuda"` 而后端名 `name = "kunlun"` —— 见 backend.py 顶部说明。
"""

from .backend import KunlunBackend, KunlunEventAdapter, build

__all__ = ["KunlunBackend", "KunlunEventAdapter", "build"]
