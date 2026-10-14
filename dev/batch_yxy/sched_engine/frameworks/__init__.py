"""推理框架适配层：统一契约（提交批次/获取结果/错误映射）与 vLLM Embedding 接入。"""

from .base import (
    DECLARED_FRAMEWORKS,
    STATUS_DECLARED,
    STATUS_VERIFIED,
    VERIFIED_FRAMEWORKS,
    BatchHandle,
    FrameworkAdapter,
)
from .vllm_embed import VllmEmbedFrameworkAdapter

__all__ = [
    "FrameworkAdapter",
    "BatchHandle",
    "VllmEmbedFrameworkAdapter",
    "VERIFIED_FRAMEWORKS",
    "DECLARED_FRAMEWORKS",
    "STATUS_VERIFIED",
    "STATUS_DECLARED",
]
