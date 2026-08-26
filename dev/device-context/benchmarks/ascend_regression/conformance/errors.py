#!/usr/bin/env python3
"""
torch_fl 统一错误对象与三维翻译（flagos/errors.py）

对应设备执行上下文职责（细项21·错误码翻译）与统一行为契约 F1-F4：
  - F1 三维翻译：类别（L1-L4）/ 位置（流/事件/任务）/ 根因（厂商原始信息）三投影
  - F2 分级处置：类别直接驱动处置策略（L1 重试 / L2 上抛 / L3 重放 / L4 恢复）
  - F4 根因保留：厂商原始错误码与描述原样保留

用法：
    from torch_fl.flagos.errors import FlagosError, translate_error, ErrorCategory
    try:
        ...
    except Exception as e:
        fe = translate_error(e, location="stream:0/op:matmul")
        # fe.category / fe.location / fe.root_cause
"""

import enum
import re
from typing import Optional


class ErrorCategory(enum.IntEnum):
    """统一错误四级分级（F2）。"""
    L1_RESOURCE = 1      # 资源类：可重试（带退避）
    L2_PARAM = 2         # 参数类：上抛调用方，不重试
    L3_EXECUTION = 3     # 执行类：任务重放（同上下文）
    L4_FATAL = 4         # 致命类：进入状态恢复流程（重建上下文）


# 昇腾 ACL 错误码 → 统一类别 映射（F1 类别投影的厂商侧依据）。
# 码值与含义以 ACL 官方错误码文档为准；此处覆盖探针实测与常用参数/资源/执行类。
# 实测样本：aclnnMatmulGetWorkspaceSize failed, ret=161002 → L2（参数非法）
ACL_ERR_TO_CATEGORY = {
    161001: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_DEVICE 设备非法
    161002: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_PARAM 参数非法
    161003: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_DATATYPE 数据类型非法
    161004: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_FORMAT 格式非法
    161005: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_OP_TYPE 算子类型非法
    161007: ErrorCategory.L1_RESOURCE, # 算子/内核缺失类（可重试资源类）
    161025: ErrorCategory.L3_EXECUTION,# 运行期执行参数类
    507021: ErrorCategory.L1_RESOURCE, # 设备内存不足类（示例，以文档为准）
}

# 消息关键词 → 类别（探针级粗分类，厂商错误码未命中时使用）
_MESSAGE_HINTS = [
    (re.compile(r"(shape|size mismatch|dimension|预期|形状)", re.I), ErrorCategory.L2_PARAM),
    (re.compile(r"(invalid (device|ordinal|data|op|param))", re.I), ErrorCategory.L2_PARAM),
    (re.compile(r"(out of (memory|resource)|allocat|名额|memory (alloc|fault))", re.I), ErrorCategory.L1_RESOURCE),
    (re.compile(r"(kernel|stream|event|runtime|execute|aclnn|aclnn\w+ failed)", re.I), ErrorCategory.L3_EXECUTION),
    (re.compile(r"(fatal|context (corrupt|invalid|damaged)|device (reset|lost))", re.I), ErrorCategory.L4_FATAL),
]


class FlagosError(Exception):
    """统一错误对象：类别 / 位置 / 根因 三投影（F1）。

    category   : ErrorCategory 四级分级
    location   : 错误归因到的流/事件/任务（框架层由调用方通过 location 参数提供；
                 运行时在途任务登记表就绪后由提交路径自动填充）
    root_cause : 厂商原始错误信息（异常类型 + 消息 + 错误码），原样保留（F4）
    """

    def __init__(self, category: ErrorCategory, root_cause: str,
                 location: Optional[str] = None, error_code: Optional[int] = None):
        super().__init__(f"[{category.name}] {root_cause}"
                         + (f" (location: {location})" if location else ""))
        self.category = category
        self.location = location
        self.root_cause = root_cause
        self.error_code = error_code

    @property
    def is_retryable(self) -> bool:
        """F2：L1 资源类可重试。"""
        return self.category == ErrorCategory.L1_RESOURCE

    @property
    def is_fatal(self) -> bool:
        """F2：L4 致命类需状态恢复。"""
        return self.category == ErrorCategory.L4_FATAL

    def to_dict(self) -> dict:
        """三维投影序列化（供可观测性事件流/监控诊断消费）。"""
        return {
            "category": self.category.name,
            "location": self.location,
            "root_cause": self.root_cause,
            "error_code": self.error_code,
        }


def _extract_acl_retcode(msg: str) -> Optional[int]:
    """从错误消息提取 ACL 错误码。

    兼容两种厂商错误形态：
      - torch_fl/aclnn 直调：aclnnMatmulGetWorkspaceSize failed, ret=161002
      - torch_npu/op-plugin：NPU function error: call aclnnMatmul failed, error code is 161002
    """
    m = re.search(r"ret\s*=\s*(\d+)", msg)
    if not m:
        m = re.search(r"error code is\s*(\d+)", msg)
    return int(m.group(1)) if m else None


def translate_error(exc: BaseException, location: Optional[str] = None) -> FlagosError:
    """把任意异常翻译为统一错误对象（F1 三维翻译）。

    优先级：厂商错误码（ret=XXXX → ACL_ERR_TO_CATEGORY）→ 消息关键词粗分类 → L3 默认。
    root_cause 原样保留异常类型与消息（F4）。
    """
    msg = str(exc)
    retcode = _extract_acl_retcode(msg)

    if retcode is not None and retcode in ACL_ERR_TO_CATEGORY:
        category = ACL_ERR_TO_CATEGORY[retcode]
    else:
        category = ErrorCategory.L3_EXECUTION
        for pattern, cat in _MESSAGE_HINTS:
            if pattern.search(msg):
                category = cat
                break

    return FlagosError(
        category=category,
        root_cause=f"{type(exc).__name__}: {msg}",
        location=location,
        error_code=retcode,
    )
