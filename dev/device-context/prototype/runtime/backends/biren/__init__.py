#!/usr/bin/env python3
"""壁仞 BR_0（Biren166M，`torch.supa` 命名空间）后端。

定位：统一运行时原型的**第五个接入实例**（前四个：昇腾 910C / 昆仑芯 P800 / 寒武纪 MLU590 / 平头哥 PPU）。
      按《新芯片接入手册》铁律：**新建实例目录 `BR0/`，只迁规范/方法/框架代码**；
      本后端的能力声明**逐项来自本机实测**（证据见 `BR0/probes/`），不迁任何既有芯片的实现与结论。

命名空间：设备 API 走 `torch.supa`（pip 包 `torch-br`，PyTorch PrivateUse1），
          故 `device_type = "supa"`、后端名 `name = "biren"` —— 详见 backend.py 顶部说明。
"""

from .backend import BirenBackend, BirenEventAdapter, build

__all__ = ["BirenBackend", "BirenEventAdapter", "build"]
