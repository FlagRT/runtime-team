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
  2. **错误隔离是分层的**
     - API 调用级失败（如 107015）：只影响该次调用，其他流不受影响（已实测）
     - 芯片级故障（如 AICORE_TIMEOUT / AICORE_EXCEPTION）：影响该设备上**所有流**，
       恢复必须走设备级 `recover_device(mode="real")`，**流级重试无效**
"""

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
    def native(self):
        """后端原生流对象（需要厂商特有操作时才用）。"""
        return self._native

    def wait_event(self, event) -> None:
        """等待事件：建立跨流依赖（流 A record → 本流 wait → 可见）。

        兼容传入统一 Event 或后端原生事件对象。
        """
        native = event.native if isinstance(event, Event) else event
        self._native.wait_event(native)

    def wait_stream(self, other) -> None:
        """等待另一流此前所有任务完成（粒度比 event 粗，过度等待会更慢）。

        兼容传入统一 Stream 或后端原生流对象。
        """
        native = other.native if isinstance(other, Stream) else other
        self._native.wait_stream(native)

    def synchronize(self, timeout_ms: Optional[int] = None) -> None:
        """等待本流任务完成；timeout_ms 非空时为有界等待（超时抛 TimeoutError）。"""
        if timeout_ms is None:
            self._native.synchronize()
            return
        self._backend.synchronize_stream(self._native, timeout_ms)

    def context(self):
        """返回切换到本流的上下文管理器（with 使用）。"""
        return self._backend.stream_context(self._native)

    def record_stream(self, tensor) -> None:
        """告知缓存分配器：tensor 在本流上仍会被使用。

        跨流传递内存时必须调用，否则内存可能被提前回收 → 数据竞争。
        """
        fn = getattr(tensor, "record_stream", None)
        if fn is None:
            raise AttributeError(
                "该 tensor 不支持 record_stream；跨流传递内存需要此能力，"
                "请确认使用的是支持该语义的框架对象"
            )
        fn(self._native)

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
    def native(self):
        return self._native

    def record(self, stream: Optional[Stream] = None) -> None:
        """在指定流（或当前流）记录完成点。"""
        self._native.record(stream.native if stream else None)

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
        native = stream.native if isinstance(stream, Stream) else stream
        self._native.wait(native)

    def synchronize(self) -> None:
        self._native.synchronize()

    def elapsed_time(self, end_event: "Event") -> float:
        """与另一事件之间的耗时（毫秒）；后端不支持时抛 NotImplementedError。"""
        fn = getattr(self._native, "elapsed_time", None)
        if fn is None:
            raise NotImplementedError("该后端不支持 elapsed_time")
        end = end_event.native if isinstance(end_event, Event) else end_event
        return fn(end)

    def __repr__(self) -> str:
        return f"<Event backend={self._backend.name} native={type(self._native).__name__}>"
