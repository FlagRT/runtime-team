"""按执行兼容性隔离的优先级请求队列：入队、取队、队列状态查询。

- 队列按 compatibility_key（模型ID|模型版本|执行profile）分 lane，不同
  lane 的请求绝不混批（兼容性隔离在入队时即确定）；
- lane 内取队序：(-priority, arrival_mono_ns, 入队序号)，同档优先级严格
  FIFO，规则确定、可测试；
- 本队列不做组批决策（分桶/预算/封口由 Planner = dynamic_batching.
  BatchCoordinator 承担），也不依赖任何模型类型、推理框架或芯片 SDK。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from dynamic_batching.errors import (
    DuplicateRequestError,
    RequestValidationError,
)

from ..errors import SchedEngineError
from .request import SchedRequest


class UnknownLaneError(SchedEngineError):
    """取队/查询的兼容键不存在。"""


@dataclass(frozen=True)
class QueueLaneStatus:
    """单条 lane 的状态快照。"""

    compatibility_key: str
    depth: int
    by_priority: dict[int, int]
    head_request_id: str | None
    oldest_arrival_mono_ns: int | None


@dataclass(frozen=True)
class _QueuedItem:
    order_key: tuple[int, int, int]  # (-priority, arrival_mono_ns, seq)
    compat_key: str
    request: SchedRequest


class RequestQueue:
    """lane 内入队 O(1)，取队时排序（确定序），跨 lane 互不影响。"""

    def __init__(self) -> None:
        self._lanes: dict[str, list[_QueuedItem]] = {}
        self._seq = itertools.count()
        self._pending_ids: set[str] = set()

    @property
    def depth(self) -> int:
        return len(self._pending_ids)

    @property
    def lane_keys(self) -> list[str]:
        return sorted(self._lanes)

    def enqueue(self, request: SchedRequest, compatibility_key: str) -> None:
        rid = request.request_id
        if not isinstance(rid, str) or not rid:
            raise RequestValidationError("request_id 必须为非空字符串")
        if rid in self._pending_ids:
            raise DuplicateRequestError(f"request_id 重复: {rid}")
        if not isinstance(compatibility_key, str) or not compatibility_key:
            raise RequestValidationError("compatibility_key 必须为非空字符串")
        item = _QueuedItem(
            order_key=(-request.priority, request.arrival_mono_ns, next(self._seq)),
            compat_key=compatibility_key,
            request=request,
        )
        self._lanes.setdefault(compatibility_key, []).append(item)
        self._pending_ids.add(rid)

    def dequeue(
        self, compatibility_key: str, max_items: int | None = None
    ) -> list[SchedRequest]:
        """按取队规则弹出该 lane 的请求；不影响其他 lane。"""
        lane = self._lanes.get(compatibility_key)
        if lane is None:
            raise UnknownLaneError(f"未知兼容键 lane: {compatibility_key}")
        lane.sort(key=lambda item: item.order_key)
        n = len(lane) if max_items is None else min(len(lane), max(0, int(max_items)))
        taken = lane[:n]
        del lane[:n]
        if not lane:
            self._lanes.pop(compatibility_key, None)
        for item in taken:
            self._pending_ids.discard(item.request.request_id)
        return [item.request for item in taken]

    def dequeue_all(self) -> list[tuple[str, SchedRequest]]:
        """跨 lane 排空：全部条目合并按取队键全局有序，返回 (兼容键, 请求)。

        跨 lane 本无执行先后语义（兼容键隔离、各自独立成批）；此处的全局
        序仅用于确定性的取队观测（dequeue trace）与回放进 coordinator 的
        次序，同键决胜保证可重复运行。
        """
        items = [item for lane in self._lanes.values() for item in lane]
        items.sort(key=lambda item: item.order_key)
        self._lanes.clear()
        self._pending_ids.clear()
        return [(item.compat_key, item.request) for item in items]

    def lane_status(self, compatibility_key: str) -> QueueLaneStatus:
        lane = self._lanes.get(compatibility_key)
        if lane is None:
            raise UnknownLaneError(f"未知兼容键 lane: {compatibility_key}")
        by_priority: dict[int, int] = {}
        for item in lane:
            pr = item.request.priority
            by_priority[pr] = by_priority.get(pr, 0) + 1
        return QueueLaneStatus(
            compatibility_key=compatibility_key,
            depth=len(lane),
            by_priority=by_priority,
            head_request_id=lane[0].request.request_id if lane else None,
            oldest_arrival_mono_ns=(
                min((i.request.arrival_mono_ns for i in lane), default=None)
            ),
        )

    def status(self) -> dict[str, QueueLaneStatus]:
        """所有非空 lane 的状态查询。"""
        return {key: self.lane_status(key) for key in sorted(self._lanes)}

    def pending_ids(self) -> set[str]:
        return set(self._pending_ids)
