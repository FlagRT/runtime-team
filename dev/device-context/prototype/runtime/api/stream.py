#!/usr/bin/env python3
"""
多流 Stream 统一抽象（runtime/api/stream.py）

对应职责：统一 Stream 语义（本方向核心子层之一）。

设计来源：
  - 事件语义契约（E1-E4 + wait_host 有界等待）
  - 910C 阶段完成的 **16 项流语义子项核查**（本抽象的验收基线）

封装对象：
  - Stream：后端原生流的统一包装（wait_event / wait_stream / synchronize / context）
  - Event ：后端原生事件的统一包装（record / query / wait_host）

⚠️ 两条硬性纪律（实现多流代码时必须遵守）：
  1. **跨流传递内存必须 record_stream**
     PyTorch 缓存分配器按流跟踪内存；若 tensor 在流 A 分配、在流 B 使用后于流 A 释放，
     分配器不知道流 B 仍在使用，可能把该内存重分配给流 A 的新 tensor → 数据竞争。
     正确做法：`tensor.record_stream(using_stream)`。
     **2026-09-29（工作包 B-3）**：本纪律的实现路径加了契约约束 —— 能力位
     `supports("record_stream")`，上层可**提前预判**；后端未声明该能力时，
     本层走**保守同步路径**（同步当前设备后放行），而不是等到 `getattr` 失败才报错。
  2. **错误隔离是分层的**
     - API 调用级失败（如 107015）：只影响该次调用，其他流不受影响（已实测）
     - 芯片级故障（如 AICORE_TIMEOUT / AICORE_EXCEPTION）：影响该设备上**所有流**，
       恢复必须走设备级 `recover_device(mode="real")`，**流级重试无效**

⚠️ `.native` 是**显式逃生舱**（修订建议 §3）：
    一旦上层取用 `.native`，该处代码即视为**绑定该厂商**，不得再声称跨芯片可移植。
    为此本层对公开属性的取用**计数审计**（`backend.info()["native_accesses"]`）；
    层内部一律走私有 `_native_obj`，**不计数**，避免把"内部实现"记成"上层破坏可移植性"。
"""

import warnings
from typing import Optional


class Stream:
    """统一流抽象：包装后端原生流对象，屏蔽厂商差异。"""

    def __init__(self, backend, native):
        self._backend = backend
        self._native = native

    @property
    def backend(self):
        return self._backend

    @property
    def _native_obj(self):
        """**层内部**用的原生对象访问口（不计数）。"""
        return self._native

    @property
    def native(self):
        """后端原生流对象（**显式逃生舱**）。

        ⚠️ 使用即视为**绑定该厂商**：该处代码不得再声称跨芯片可移植。
        取用次数会记入后端审计（`info()["native_accesses"]`），便于事后定位破坏点。
        正常路径应优先使用本类的统一方法（wait_event / wait_stream / synchronize / context）。
        """
        self._backend.note_native_access("Stream")
        return self._native

    def wait_event(self, event) -> None:
        """等待事件：建立跨流依赖（流 A record → 本流 wait → 可见）。

        兼容传入统一 Event 或后端原生事件对象。
        """
        native = event._native_obj if isinstance(event, Event) else event
        self._native.wait_event(native)

    def wait_stream(self, other) -> None:
        """等待另一流此前所有任务完成（粒度比 event 粗，过度等待会更慢）。

        兼容传入统一 Stream 或后端原生流对象。
        """
        native = other._native_obj if isinstance(other, Stream) else other
        self._native.wait_stream(native)

    def synchronize(self, timeout_ms: Optional[int] = None) -> None:
        """等待本流任务完成；timeout_ms 非空时为有界等待（超时抛 TimeoutError）。"""
        self._backend.check_stream_usable(self._native)      # 工作包 C：绑定语义拦截
        if timeout_ms is None:
            self._native.synchronize()
            return
        self._backend.synchronize_stream(self._native, timeout_ms)

    def context(self):
        """返回切换到本流的上下文管理器（with 使用）。"""
        self._backend.check_stream_usable(self._native)      # 工作包 C：绑定语义拦截
        return self._backend.stream_context(self._native)

    def record_stream(self, tensor) -> None:
        """告知缓存分配器：tensor 在本流上仍会被使用。

        跨流传递内存时必须调用，否则内存可能被提前回收 → 数据竞争。

        2026-09-29（工作包 B-3）行为定义：
          · 后端**声明**能力 `record_stream` 且 tensor 原生支持 ⇒ 走原生路径（零额外开销）；
          · 否则 ⇒ **保守同步路径**：同步当前设备后再放行（**慢但正确**），
            并 `warnings.warn` + 计入后端退化审计（`info()["degradations"]`）。
            —— 这里刻意**不再抛 AttributeError**：跨流内存被提前回收是"偶发数据错乱"，
            比降级慢一点危险得多；且"是否支持"应可**预判**（`supports("record_stream")`）。
        """
        fn = getattr(tensor, "record_stream", None)
        if self._backend.supports("record_stream") and callable(fn):
            fn(self._native)
            return

        reason = ("后端未声明能力 record_stream"
                  if not self._backend.supports("record_stream")
                  else f"tensor 类型 {type(tensor).__name__} 无 record_stream()")
        self._backend.note_degradation("record_stream_conservative")
        warnings.warn(
            f"[runtime] {reason} ⇒ 走**保守同步路径**（同步当前设备后放行）。"
            "这会丢失跨流重叠带来的性能收益；如需消除，请改用声明了该能力的后端/张量类型。",
            stacklevel=2,
        )
        self._backend.conservative_stream_sync(self._native)

    def release(self) -> bool:
        """释放本流；**仅当它由本层拥有**时真的销毁，否则 no-op 返回 `False`。

        什么时候本层拥有：走 `create_stream(priority=…)`（厂商 C API 建流 + 包装）时
        —— 实测 torch **不拥有**这种流（丢弃包装对象 + gc 后句柄仍可用）。
        `create_stream()`（厂商原生路径）造的流由厂商拥有 ⇒ 本方法 no-op，
        **不会越权销毁**别人的流。

        ⚠️ 释放后**不得再使用**该流（`synchronize()` / `context()` / `record_stream()`
        会当场报错，而不是静默失败）。
        """
        return self._backend.release_stream(self._native)

    def __repr__(self) -> str:
        return f"<Stream backend={self._backend.name} native={type(self._native).__name__}>"


class Event:
    """统一事件抽象：包装后端原生事件对象。"""

    def __init__(self, backend, native):
        self._backend = backend
        self._native = native

    @property
    def backend(self):
        return self._backend

    @property
    def _native_obj(self):
        """**层内部**用的原生对象访问口（不计数）。"""
        return self._native

    @property
    def native(self):
        """后端原生事件对象（**显式逃生舱**，取用计入审计，语义同 `Stream.native`）。"""
        self._backend.note_native_access("Event")
        return self._native

    def record(self, stream: Optional[Stream] = None) -> None:
        """在指定流（或当前流）记录完成点。"""
        self._native.record(stream._native_obj if stream else None)

    def query(self) -> bool:
        """查询是否已完成。注意：未 record 的事件其 query 语义由后端契约定义。"""
        return self._native.query()

    def wait_host(self, timeout_ms: int) -> bool:
        """主机侧有界等待（避免无限阻塞）。返回 True 表示已完成。"""
        return self._backend.wait_event_host(self._native, timeout_ms)

    def wait(self, stream: Optional[Stream] = None) -> None:
        """让指定流（默认当前流）等待本事件完成。

        等价于 stream.wait_event(self) 的反向写法；stream 可为统一 Stream 或原生流。
        """
        if stream is None:
            fn = getattr(self._native, "wait", None)
            if fn is None:
                raise NotImplementedError("该后端原生事件不支持 wait()")
            fn()
            return
        native = stream._native_obj if isinstance(stream, Stream) else stream
        self._native.wait(native)

    def synchronize(self) -> None:
        self._native.synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        """与另一事件之间的耗时（毫秒）；后端不支持时抛 NotImplementedError。"""
        fn = getattr(self._native, "elapsed_time", None)
        if fn is None:
            raise NotImplementedError("该后端不支持 elapsed_time")
        end = end_event._native_obj if isinstance(end_event, Event) else end_event
        return fn(end)

    def __repr__(self) -> str:
        return f"<Event backend={self._backend.name} native={type(self._native).__name__}>"
