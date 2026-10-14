"""设备能力画像层：批量/Shape/资源预算的芯片差异隔离与 6 类芯片注册表。"""

from .base import (
    STATUS_PENDING,
    STATUS_VERIFIED,
    DeviceProfile,
)
from .registry import (
    ALL_DEVICES,
    all_devices,
    get_device,
)

__all__ = [
    "DeviceProfile",
    "ALL_DEVICES",
    "all_devices",
    "get_device",
    "STATUS_VERIFIED",
    "STATUS_PENDING",
]
