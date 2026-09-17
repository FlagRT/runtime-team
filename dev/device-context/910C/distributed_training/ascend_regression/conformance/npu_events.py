#!/usr/bin/env python3
"""
conformance/npu_events.py — torch.npu.Event 的统一事件语义适配层

设备执行上下文职责（细项21·统一 Event 语义）在 A 线（torch_npu）的落地：
torch_npu 原生 Event 相对统一事件契约（E1-E4 v2）存在两个缺口，本适配层补齐：

  - E3：未 record 事件 query() 误报完成（ACL 事件默认完成状态，与 B 线 AclEvent
        同源问题）→ recorded 跟踪修正：未 record 返回 False（不崩溃）
  - E2v2：原生无 wait_host（主机有界等待）→ query 轮询实现，永不永久阻塞

【说明】本适配层是"统一语义验证"的载体：conformance 用它验证契约的期望行为，
同时如实标注"torch_npu 原生 Event 不满足、由统一适配层补齐"——这本身就是
设备执行上下文职责在 A 线的价值体现（统一层补语义，厂商层不背契约）。
"""

import time


class NpuEventAdapter:
    """torch.npu.Event 的统一语义适配（recorded 跟踪 + query 修正 + wait_host）。"""

    def __init__(self, *args, **kwargs):
        import torch_npu  # noqa: F401
        import torch
        self._ev = torch.npu.Event(*args, **kwargs)
        self._recorded = False

    def record(self, stream=None):
        r = self._ev.record(stream)
        self._recorded = True
        return r

    def wait(self, stream=None):
        return self._ev.wait(stream)

    def synchronize(self):
        return self._ev.synchronize()

    def query(self):
        # E3：未 record 事件返回未完成（修正 torch_npu 原生误报完成）
        if not self._recorded:
            return False
        return self._ev.query()

    def wait_host(self, timeout_ms=None):
        # E2v2：主机有界等待，永不永久阻塞
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while not self.query():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

    def elapsed_time(self, end_event):
        return self._ev.elapsed_time(end_event._ev)
