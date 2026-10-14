"""设备能力画像：隔离批量、Shape、资源预算与执行能力的芯片差异。

字段语义：max_batch_size / max_total_tokens 为芯片侧执行预算上限（引擎
在策略生效前取 min(策略, 设备) 钳制）；max_sequence_length 为单请求
Shape 上限；capabilities 描述已验证的执行能力。

状态语义（诚实标注）：
- verified：已在真实芯片上完成可运行链路验证（verified_with 注明模型+框架）；
- pending：仅登记画像契约，未在任何真实卡上验证 —— 引擎拒绝运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

STATUS_VERIFIED = "verified"
STATUS_PENDING = "pending"


@dataclass(frozen=True)
class DeviceProfile:
    device_id: str
    vendor: str
    status: str
    max_batch_size: int | None
    max_total_tokens: int | None
    max_sequence_length: int | None
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    verified_with: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""
