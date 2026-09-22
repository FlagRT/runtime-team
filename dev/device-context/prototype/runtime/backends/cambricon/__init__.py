#!/usr/bin/env python3
"""寒武纪 MLU 后端（`torch_mlu` / PrivateUse1 命名空间适配）。

定位：统一运行时原型的**第三个接入实例**（前两家：昇腾 910C、昆仑芯 P800）。
      按《运行时层接口约定》§2 的 Backend 插件接入规范**新建**；
      **不迁移**前两家的实现与结论。

命名空间：本栈上 MLU 经 `torch_mlu` 注册为 PyTorch **PrivateUse1**，
          设备串前缀是 `mlu` ⇒ `device_type = "mlu"`，后端名 `name = "cambricon"`。
          ⚠️ PrivateUse1 为**进程级单例**，不得与 `torch_npu` / `torch_fl` 同进程混用。

⚠️ 证据等级：**尚未在寒武纪真机上跑过**（两台测试机缺 `docker` 组权限）。
   凡 API 形态与取值见 `backend.py` 顶部「未实测清单」，勿当作已验证结论。
"""

from .backend import CambriconBackend, CambriconEventAdapter, build

__all__ = ["CambriconBackend", "CambriconEventAdapter", "build"]
