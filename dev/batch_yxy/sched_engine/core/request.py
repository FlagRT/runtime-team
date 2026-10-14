"""调度引擎统一请求类型：与模型专用输入类型、推理框架、芯片 SDK 解耦。

SchedRequest.payload 由模型画像层（解释调度特征）与推理框架适配层（解释
执行输入）共同消费，调度核心本身不解释 payload 内容，也不复制大张量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SchedRequest:
    """入队请求。

    - priority：取队优先级（int，越大越先出队，默认 0）；同优先级按
      (arrival_mono_ns, 提交顺序) FIFO。优先级不绕过任何硬约束。
    - features：模型画像提取的调度特征（如 token_length / sequence_length /
      nnz），由画像层或引擎在入队时填充，供观测与分桶决策使用。
    - deadline_mono_ns 仅记录，本引擎不做 SLA 控制。
    """

    request_id: str
    model_id: str
    payload: Any
    arrival_mono_ns: int
    priority: int = 0
    features: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    deadline_mono_ns: int | None = None

    @property
    def dequeue_key(self) -> tuple[int, int]:
        """取队排序键：优先级降序，再按到达时间升序（提交序号由队列决胜）。"""
        return (-self.priority, self.arrival_mono_ns)
