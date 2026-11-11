#!/usr/bin/env python3
"""壁仞 BR_0：流优先级事实探针（能设 / 能回读 / 有范围 API 三问分离）。

纪律：`Stream(priority=...)` 收下参数 ≠ 优先级生效；判据是**回读一致**。
本探针只采集事实，不改变后端能力声明。
"""
import sys
import torch
import torch_br  # noqa: F401

dev = torch.supa
print("=== 流优先级事实探针（biren / torch.supa）===")
print("torch:", torch.__version__, "| torch_br:", getattr(torch_br, "__version__", "?"))
print("devices:", dev.device_count())

# ── 问 1：命名空间/类上有没有范围查询入口 ──
for owner, name in [
    ("torch.supa", "stream_priority_range"),
    ("torch.supa.Stream", "priority_range"),
    ("torch.supa.Stream", "priority"),
]:
    obj = dev if owner == "torch.supa" else dev.Stream
    print("has %-28s: %s" % (owner + "." + name, hasattr(obj, name)))

# ── 问 2：构造能否收 priority，且回读是否一致 ──
for prio in (0, -5, 7):
    try:
        s = dev.Stream(priority=prio)
        rb = None
        for attr in ("priority", "priority_value"):
            if hasattr(s, attr):
                try:
                    rb = getattr(s, attr)
                    rb = rb() if callable(rb) else rb
                except Exception as exc:
                    rb = "ERR:%s" % type(exc).__name__
                break
        print("Stream(priority=%3d): 构造成功 | 回读 %s=%r | 一致=%s"
              % (prio, attr if rb is not None else "无属性", rb,
                 "是" if rb == prio else ("无法判定" if rb is None else "否")))
    except Exception as exc:
        print("Stream(priority=%3d): 构造失败 %s: %s" % (prio, type(exc).__name__, str(exc).splitlines()[0][:90]))

# ── 问 3：默认流回读 ──
try:
    cur = dev.current_stream()
    rb = getattr(cur, "priority", "无属性")
    print("current_stream().priority:", rb() if callable(rb) else rb)
except Exception as exc:
    print("current_stream 回读失败:", type(exc).__name__)

# ── 问 4：CUDA 式入口是否存在（对照用）──
print("dev._C 有 get_stream_priority_range:", hasattr(getattr(dev, "_C", object()), "get_stream_priority_range"))
print("=== 探针结束 ===")
