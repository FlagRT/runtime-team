"""6 类目标芯片的设备画像注册表（仅昇腾 910C 为 verified，其余 pending）。

矩阵（芯片 × 状态 | 已验证组合）：

| 芯片        | 厂商     | 状态       | verified_with                     |
|-------------|----------|------------|-----------------------------------|
| ascend-910c | 华为昇腾 | verified   | Qwen3-Embedding-0.6B@vllm-embed   |
| kunlun      | 昆仑芯   | pending    | —                                 |
| cambricon   | 寒武纪   | pending    | —                                 |
| thead       | 平头哥   | pending    | —                                 |
| biren       | 壁仞     | pending    | —                                 |
| enflame     | 燧原     | pending    | —                                 |

910C 的预算数值来源：2026-09-23 NPU bench 验证运行所采用的策略
v1-910c-single-card（max_batch_size=32 / max_total_tokens=32768 /
桶上界至 4096）。注意：这是"已验证运行采用的预算上限"，不是设备极限
探测值；设备极限需专门画像探针另行测量后更新。
"""

from __future__ import annotations

from .base import (
    STATUS_PENDING,
    STATUS_VERIFIED,
    DeviceProfile,
)

ASCEND_910C = DeviceProfile(
    device_id="ascend-910c",
    vendor="华为昇腾",
    status=STATUS_VERIFIED,
    max_batch_size=32,
    max_total_tokens=32768,
    max_sequence_length=4096,
    capabilities=("fixed-length-padding-batch", "single-card-davinci0"),
    verified_with=("Qwen3-Embedding-0.6B@vllm-embed",),
    notes=(
        "单卡 davinci0（容器内相对索引 0）；数值为已验证运行采用的策略预算，"
        "非设备极限探测值"
    ),
)

KUNLUN = DeviceProfile(
    device_id="kunlun",
    vendor="昆仑芯",
    status=STATUS_PENDING,
    max_batch_size=None,
    max_total_tokens=None,
    max_sequence_length=None,
    notes="待适配：未在真实卡上验证，画像契约占位，预算字段留空（不得标记支持）",
)

CAMBRICON = DeviceProfile(
    device_id="cambricon",
    vendor="寒武纪",
    status=STATUS_PENDING,
    max_batch_size=None,
    max_total_tokens=None,
    max_sequence_length=None,
    notes="待适配：未在真实卡上验证",
)

THEAD = DeviceProfile(
    device_id="thead",
    vendor="平头哥",
    status=STATUS_PENDING,
    max_batch_size=None,
    max_total_tokens=None,
    max_sequence_length=None,
    notes="待适配：未在真实卡上验证",
)

BIREN = DeviceProfile(
    device_id="biren",
    vendor="壁仞",
    status=STATUS_PENDING,
    max_batch_size=None,
    max_total_tokens=None,
    max_sequence_length=None,
    notes="待适配：未在真实卡上验证",
)

ENFLAME = DeviceProfile(
    device_id="enflame",
    vendor="燧原",
    status=STATUS_PENDING,
    max_batch_size=None,
    max_total_tokens=None,
    max_sequence_length=None,
    notes="待适配：未在真实卡上验证",
)

ALL_DEVICES: tuple[DeviceProfile, ...] = (
    ASCEND_910C,
    KUNLUN,
    CAMBRICON,
    THEAD,
    BIREN,
    ENFLAME,
)

_REGISTRY: dict[str, DeviceProfile] = {d.device_id: d for d in ALL_DEVICES}


def get_device(device_id: str) -> DeviceProfile | None:
    return _REGISTRY.get(device_id)


def all_devices() -> tuple[DeviceProfile, ...]:
    return ALL_DEVICES
