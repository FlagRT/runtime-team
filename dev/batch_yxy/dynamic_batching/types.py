"""组批核心数据类型：请求、策略、批次计划与兼容键。

接口语义对齐 docs/Qwen3-Embedding-0.6B-动态组批与长度感知分桶开发实现文档.md 第 3 节。
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PolicyValidationError


@dataclass(frozen=True)
class BatchRequest:
    """待组批请求。

    input_token_count 必须与实际执行输入一致（同一 tokenizer、含特殊 token、
    截断之后的值）；payload_ref 由执行适配层解释，调度核心不复制大张量；
    arrival_mono_ns 使用单调时钟；deadline_mono_ns 仅记录，本次不做 SLA 控制。
    """

    request_id: str
    model_id: str
    model_version: str
    input_token_count: int
    payload_ref: object
    arrival_mono_ns: int
    execution_profile: str | None = None
    deadline_mono_ns: int | None = None


def compatibility_key(request: BatchRequest) -> str:
    """兼容键（硬约束）：模型ID|模型版本|执行profile。兼容键不同的请求绝不进入同一批次。"""
    profile = request.execution_profile or "default"
    return f"{request.model_id}|{request.model_version}|{profile}"


@dataclass(frozen=True)
class BatchPolicy:
    """一个模型在某个策略版本下的组批规则。

    bucket_upper_bounds 为固定桶上界，必须严格递增且覆盖已验证长度范围；
    超出 max_length 的请求在 submit 时明确拒绝（RequestTooLongError），
    不静默放入最大桶。max_total_tokens 必须 >= 最大桶上界，否则单条超长请求
    永远无法成批。max_padding_ratio 为硬约束：加入某请求会使批次 Padding 比例
    超过该值时，该请求不进入本批（保留给后续批次）。
    """

    model_id: str
    policy_version: str
    bucket_upper_bounds: tuple[int, ...]
    max_batch_size: int
    max_total_tokens: int
    max_wait_ms: float
    allow_adjacent_bucket_merge: bool = False
    max_padding_ratio: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise PolicyValidationError("model_id 必须为非空字符串")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise PolicyValidationError("policy_version 必须为非空字符串")
        bounds = self.bucket_upper_bounds
        if not isinstance(bounds, tuple) or not bounds:
            raise PolicyValidationError("bucket_upper_bounds 必须为非空 tuple")
        for b in bounds:
            if not isinstance(b, int) or isinstance(b, bool) or b < 1:
                raise PolicyValidationError(f"桶上界必须为正整数: {bounds}")
        for i in range(len(bounds) - 1):
            if bounds[i] >= bounds[i + 1]:
                raise PolicyValidationError(f"桶上界必须严格递增: {bounds}")
        if not isinstance(self.max_batch_size, int) or self.max_batch_size < 1:
            raise PolicyValidationError("max_batch_size 必须 >= 1")
        if (
            not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < bounds[-1]
        ):
            raise PolicyValidationError(
                f"max_total_tokens({self.max_total_tokens}) 必须 >= 最大桶上界"
                f"({bounds[-1]})"
            )
        if not isinstance(self.max_wait_ms, (int, float)) or self.max_wait_ms < 0:
            raise PolicyValidationError("max_wait_ms 必须 >= 0")
        if self.max_padding_ratio is not None and not (
            0.0 <= self.max_padding_ratio < 1.0
        ):
            raise PolicyValidationError("max_padding_ratio 取值应为 [0, 1)")

    @property
    def max_length(self) -> int:
        """策略最大已验证长度 = 最大桶上界。"""
        return self.bucket_upper_bounds[-1]

    def bucket_index(self, token_count: int) -> int | None:
        """返回 token_count 所在桶下标；超出最大上界返回 None。"""
        for i, bound in enumerate(self.bucket_upper_bounds):
            if token_count <= bound:
                return i
        return None

    @property
    def wait_window_ns(self) -> int:
        return int(self.max_wait_ms * 1_000_000)


@dataclass(frozen=True)
class BatchPlan:
    """一次可执行的批次计划。

    request_ids[i] 与 payload_refs[i]、token_lengths[i] 及执行结果第 i 项一一对应；
    padded_length 为该批在定长 Padding 执行路径下所需的 pad 后长度估算
    （= max(token_lengths)），实际设备路径是否产生该 Padding 由执行端测量决定。
    """

    batch_id: str
    policy_version: str
    compatibility_key: str
    bucket_ids: tuple[int, ...]
    request_ids: tuple[str, ...]
    payload_refs: tuple[object, ...]
    token_lengths: tuple[int, ...]
    total_tokens: int
    padded_length: int | None
    dispatch_reason: str
    created_mono_ns: int
