"""动态组批错误类型。非法输入显式报错，不默默截断或吞并。"""

from __future__ import annotations


class BatchCoordinatorError(Exception):
    """组批模块错误基类。"""


class PolicyValidationError(BatchCoordinatorError):
    """BatchPolicy 配置非法。"""


class RequestValidationError(BatchCoordinatorError):
    """BatchRequest 提交参数非法。"""


class DuplicateRequestError(RequestValidationError):
    """request_id 重复。"""


class UnknownPolicyError(RequestValidationError):
    """model_id 未注册策略。"""


class RequestTooLongError(RequestValidationError):
    """input_token_count 超出策略最大已验证长度，明确拒绝。"""


class ClosedCoordinatorError(BatchCoordinatorError):
    """协调器已 close，不再接受 submit。"""


class ResultCountMismatchError(BatchCoordinatorError):
    """执行返回向量数量与批次请求数不一致。"""

    def __init__(self, batch_id: str, expected: int, got: int) -> None:
        super().__init__(
            f"batch {batch_id}: expected {expected} outputs, got {got}"
        )
        self.batch_id = batch_id
        self.expected = expected
        self.got = got
