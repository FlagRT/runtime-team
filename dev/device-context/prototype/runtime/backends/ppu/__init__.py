#!/usr/bin/env python3
"""平头哥 PPU（真武 ZW810E）后端（复用 `torch.cuda` 命名空间）。

定位：统一运行时原型的**第四个接入实例**（前三个：昇腾 910C / 昆仑芯 P800 / 寒武纪 MLU590）。
      迁移的是《运行时层接口约定》的规范与方法；**其余实例的实现与结论一律不迁移**
      —— 本后端的能力声明**逐项来自本机实测**（证据见 `PPU/probes/`）。

命名空间：平头哥这栈上设备 API 走 `torch.cuda`（而 `torch.xpu.device_count() == 0`），
          故 `device_type = "cuda"` 而后端名 `name = "ppu"` —— 见 backend.py 顶部说明。
"""

from .backend import PpuBackend, PpuEventAdapter, build

__all__ = ["PpuBackend", "PpuEventAdapter", "build"]
