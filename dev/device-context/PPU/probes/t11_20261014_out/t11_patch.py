#!/usr/bin/env python3
"""
t11_patch.py — T11 修法对照用的**注入补丁**（不改仓库代码；以 sitecustomize 形态加载）
═══════════════════════════════════════════════════════════════════════════════

用法（容器内）：
  PYTHONPATH=<本目录> T11_PATCH=<模式> python3 <载体>

模式：
  control       只 import 本层、不打补丁（与"未注入"作对照，隔离 import 顺序的影响）
  note_off      `note_stream_created` 变 no-op ⇒ 登记表 `_stream_ctx` **不再挂弱引用**
  strong        `_id_holder` 改为**强持有**（返回取回原对象的 lambda）⇒ 所有登记表都不再建弱引用
  owned_off     `_register_owned_stream` 变 no-op（优先级路径的归属登记整块去掉）

⚠️ 加载纪律：本文件必须是**合法 UTF-8 且 bytes 字面量全 ASCII** —— 实测踩过一次
   （bytes 里混中文 ⇒ sitecustomize 整体 SyntaxError，补丁**静默没生效**却看起来"跑过了"）。
   故本文件用 `T11_PATCH_LOADED` 明标是否真的加载成功。
"""
import os
import sys

_PATCH = os.environ.get("T11_PATCH", "control")

try:
    # ⚠️ sitecustomize 在解释器启动早期加载 ⇒ 此刻 `sys.path` 里**还没有**原型根
    #    （脚本目录才是 sys.path[0]）⇒ 必须显式补上，否则 `import runtime` 直接
    #    ModuleNotFoundError（实测踩过：表现为 T11_PATCH_FAILED，容易被误读成"补丁无效"）。
    _root = os.environ.get("DC_ROOT", "") or os.getcwd()
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)

    import runtime                                                       # noqa: F401
    from runtime.backends.base import RuntimeBackend

    if _PATCH == "note_off":
        RuntimeBackend.note_stream_created = lambda self, s: None
    elif _PATCH == "strong":
        RuntimeBackend._id_holder = staticmethod(lambda obj: (lambda: obj))
    elif _PATCH == "owned_off":
        RuntimeBackend._register_owned_stream = lambda self, s, h: None
    elif _PATCH == "control":
        pass
    else:
        sys.stderr.write(f"[t11-patch] unknown patch: {_PATCH}\n")

    sys.stderr.write(f"[t11-patch] T11_PATCH_LOADED patch={_PATCH}\n")
    sys.stderr.flush()
except Exception as _e:                                                  # noqa: BLE001
    sys.stderr.write(f"[t11-patch] T11_PATCH_FAILED patch={_PATCH} err={_e!r}\n")
    sys.stderr.flush()
