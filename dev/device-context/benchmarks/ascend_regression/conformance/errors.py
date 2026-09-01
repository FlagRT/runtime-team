#!/usr/bin/env python3
"""
torch_fl 统一错误对象与三维翻译（flagos/errors.py）

对应设备执行上下文职责（细项21·错误码翻译）与统一行为契约 F1-F4：
  - F1 三维翻译：类别（L1-L4）/ 位置（流/事件/任务）/ 根因（厂商原始信息）三投影
  - F2 分级处置：类别直接驱动处置策略（L1 重试 / L2 上抛 / L3 重放 / L4 恢复）
  - F4 根因保留：厂商原始错误码与描述原样保留

  - F5 分级可观测：区分「确定分级」与「保守兜底」，避免上层把兜底当定论

用法：
    from torch_fl.flagos.errors import FlagosError, translate_error, ErrorCategory
    try:
        ...
    except Exception as e:
        fe = translate_error(e, location="stream:0/op:matmul")
        # fe.category / fe.location / fe.root_cause
        # fe.mapped / fe.graded_by  —— 分级来源，见 F5

F5 分级可观测（2026-09-01 新增）：
    fe.mapped     True  = 错误码命中 ACL_ERR_TO_CATEGORY（确定分级）
                  False = 靠消息关键词或兜底（**保守分级，不可当定论**）
    fe.graded_by  "code_map"     = 厂商错误码映射表命中（最可信）
                  "message_hint" = 消息关键词粗分类（次可信，依赖消息文本）
                  "default"      = 无依据，兜底 L3_EXECUTION（最不可信）

    为什么需要：映射表覆盖率有限（当前 22/159 ≈ 14%），未覆盖的错误码会静默
    兜底为 L3_EXECUTION。若上层（尤其 D11 状态恢复决策）把兜底 L3 当作确定结论，
    可能做出错误处置（如对致命错误不触发恢复）。F5 让"未覆盖"这件事可见，
    上层可据此选择保守策略。**可观测性优先于覆盖率**——宁可知道自己不知道。
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
#
# 数据来源与维护方式（2026-09-01 起）：
#   · 昇腾运行时段（107xxx/207xxx/507xxx）：由 benchmarks/inference/gen_acl_error_map.py
#     从 CANN 头文件 rt_error_codes.h 提取（132 条，含宏名 + 官方语义注释），
#     规则建议分级 → 人工审核后录入。覆盖率与分级差异可用
#     benchmarks/inference/audit_error_map_coverage.py 复算。
#   · ACL 基础段（161xxx）：沿用实测样本，尚未接入头文件提取（待补）。
#   · 标注「实测裁决」的条目：规则置信度不足，但经真实触发实验定性。
#
# 实测样本：aclnnMatmulGetWorkspaceSize failed, ret=161002 → L2（参数非法）
ACL_ERR_TO_CATEGORY = {
    # ── ACL 基础段（161xxx，实测样本，待接入头文件提取）──
    161001: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_DEVICE 设备非法
    161002: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_PARAM 参数非法
    161003: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_DATATYPE 数据类型非法
    161004: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_FORMAT 格式非法
    161005: ErrorCategory.L2_PARAM,    # ACL_ERROR_INVALID_OP_TYPE 算子类型非法
    161007: ErrorCategory.L1_RESOURCE, # 算子/内核缺失类（可重试资源类）
    161025: ErrorCategory.L3_EXECUTION,# 运行期执行参数类

    # ── 昇腾运行时段（来源：CANN rt_error_codes.h，2026-09-01 审核录入）──
    # L4 致命：进入设备状态恢复流程（R2-R5 重建）
    207004: ErrorCategory.L4_FATAL,    # ACL_ERROR_RT_NO_DEVICE            no device
    # L1 资源：可重试（带退避）
    207005: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_RESOURCE_ALLOC_FAIL  resource alloc fail
    207009: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_NO_NOTIFY_RESOURCE   no notify resource
    207010: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_NO_MODEL_RESOURCE    no model resource
    207011: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_NO_CDQ_RESOURCE      no cdq resource
    507021: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_PROFILING_ERROR      profiling error
                                       #   （订正：旧注"设备内存不足类"系臆断，已按头文件订正）
    # L2 参数/契约：上抛调用方，不重试、不重放（前置条件缺失，重放必然再失败）
    107000: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_PARAM_INVALID        param invalid
    107002: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_CONTEXT_NULL         current context null
                                       #   （歧义待裁决：未 set_context = 契约违反 L2；若上下文已销毁应 L4）
    107004: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_MODEL_CONTEXT        model not in current context
    107008: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_ADDR_UNALIGNED       memory address unaligned
    107016: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_INVALID_MEMORY_TYPE  invalid memory type
    107017: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_INVALID_HANDLE       invalid handle
    107018: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_INVALID_MALLOC_TYPE  invalid malloc type
    107025: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_STREAM_UNJOINED      invalid capture model
    207000: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_FEATURE_NOT_SUPPORT  feature not support
    207019: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_FEATURE_NOT_SUPPORT_UPDATE_OP
    507025: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_RINGBUFFER_NOT_INIT  ringbuffer not init
    507031: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_LABEL_CONTEXT        label not in current context
    507040: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_INVALID_DIEID        invalid die id
    107015: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_STREAM_NO_CB_REG     callback not register to stream
                                       #   【实测裁决】A/B 单变量对照证实：对未 subscribe_report 的 stream
                                       #   投递 callback 即命中，属契约违反，L3 重放必然再失败。
                                       #   规则因语义含 stream 判 low 置信，此处以实测为准。

    # ── 人工裁决条目（2026-09-01）：规则判 low 置信 / 被消息关键词干扰，经语义分析定级 ──
    # 裁决原则：按**责任方归属**定级
    #   · 调用方用错（用法/契约违反）→ L2（重试与重放均无意义，应上抛）
    #   · 环境资源可恢复（等待释放）  → L1（可重试）
    #   · 执行期失败（算子内部/trap） → L3（留一次重放机会）
    #   · 硬件致命（AI Core 异常）    → L4（由 R2 探针评估决定是否真需重建，不会盲目重建）
    # 注：以下为语义裁决而非实测，若需更细粒度待实测验证。

    # L4 硬件致命
    507015: ErrorCategory.L4_FATAL,    # ACL_ERROR_RT_AICORE_EXCEPTION   aicore exception
    # L1 资源（等待释放后可重试）
    207007: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_NO_EVENT_RESOURCE  no event resource
    207008: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_NO_STREAM_RESOURCE no stream resource
    # L2 调用方用法错误（stream/event/task 契约违反）
    107003: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_STREAM_CONTEXT     stream not in current context
    107005: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_STREAM_MODEL       stream not in model
    107006: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID
    107030: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_CAPTURE_MODE_NOT_SUPPORT
    507009: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT
    # L3 硬件执行期陷阱（kernel 越界访问类：重放可复现，但不需重建设备）
    507042: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW
    507043: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW
    507044: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW
    507045: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW
    # L3 aclnn 算子内部错误：责任在算子包/部署配置，非调用方参数。
    #   重试无效（配置错误是持久的）、重建设备无据，归执行类。
    #   统一归 L3 还顺带修正 561001 —— 它曾被 _MESSAGE_HINTS 的 "shape" 关键词误判为 L2。
    561001: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_INFERSHAPE_ERROR
    561103: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_NULLPTR
    561104: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE
    561106: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_INVALID_IMPL_MODE
    561107: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND
    561109: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND
    561110: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_JSON_FORMAT_INVALID
    561111: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_JSON_DTYPE_INVALID
    561112: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND
    561113: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_OP_FILE_INVALID
    561114: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND
    561115: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH
    561117: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_INPUT_JSON_IS_NULL
    561118: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID
    561119: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID

    # ── 规则高置信条目固化（2026-09-01）──
    # 这些条目当前分级恰好与兜底/关键词结果一致，但那是**巧合**而非显式声明。
    # 显式写入以防 _MESSAGE_HINTS 或兜底值变化导致静默漂移（可观测性优先）。
    #
    # L4 硬件计算单元异常：与 507015 AICORE_EXCEPTION 对齐 —— 同类硬件单元的
    #   exception 不应有等级差，统一交 R2 探针评估是否真需重建（探针通过则不重建）。
    #   注：*_TRAP_EXCEPTION 是 kernel 越界访问（软件问题）→ 仍归 L3，不触发设备重建。
    507018: ErrorCategory.L4_FATAL,    # ACL_ERROR_RT_AICPU_EXCEPTION        aicpu exception
    507035: ErrorCategory.L4_FATAL,    # ACL_ERROR_RT_VECTOR_CORE_EXCEPTION  vector core exception
    507049: ErrorCategory.L4_FATAL,    # ACL_ERROR_RT_FFTS_PLUS_EXCEPTION    ffts+ exception
    # L1 资源
    207001: ErrorCategory.L1_RESOURCE, # ACL_ERROR_RT_MEMORY_ALLOCATION      memory allocation (OOM)
    # L2 参数
    107001: ErrorCategory.L2_PARAM,    # ACL_ERROR_RT_INVALID_DEVICEID       invalid device id
    # L3 执行（timeout / abort / 内部错误 / trap 越界 / 算子内部）
    107007: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL
    107011: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_SUBSCRIBE
    107019: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_WAIT_TIMEOUT
    107020: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_TASK_TIMEOUT
    107022: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_DEVICE_TASK_ABORT
    107023: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_ABORT
    107027: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_CAPTURED
    107028: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_EVENT_CAPTURED
    107029: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_NOT_CAPTURED
    107031: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_CAPTURE_IMPLICIT
    107032: ErrorCategory.L3_EXECUTION,# ACL_ERROR_STREAM_CAPTURE_CONFLICT
    107033: ErrorCategory.L3_EXECUTION,# ACL_ERROR_STREAM_TASK_GROUP_STATUS
    107034: ErrorCategory.L3_EXECUTION,# ACL_ERROR_STREAM_TASK_GROUP_INTR
    107035: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_TASK_ABORT_STOP
    107036: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_CAPTURE_UNMATCHED
    361001: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_RUNTIME_ERROR
    507000: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_INTERNAL_ERROR
    507002: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_TASK_FULL
    507003: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_TASK_EMPTY
    507004: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_NOT_COMPLETE
    507006: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_EVENT_NOT_COMPLETE
    507011: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_MODEL_EXECUTE
    507012: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_REPORT_TIMEOUT
    507014: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICORE_TIMEOUT
    507016: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICORE_TRAP_EXCEPTION   （trap：kernel 越界）
    507017: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICPU_TIMEOUT
    507023: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_MODEL_ABORT_NORMAL
    507024: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_KERNEL_UNREGISTERING
    507027: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_KERNEL_LOOKUP
    507028: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_KERNEL_DUPLICATE
    507034: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_VECTOR_CORE_TIMEOUT
    507036: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION
    507046: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_STREAM_SYNC_TIMEOUT
    507047: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_EVENT_SYNC_TIMEOUT
    507048: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_FFTS_PLUS_TIMEOUT
    507050: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION
    507899: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_DRV_INTERNAL_ERROR
    507900: ErrorCategory.L3_EXECUTION,# ACL_ERROR_RT_AICPU_INTERNAL_ERROR
    507905: ErrorCategory.L3_EXECUTION,# ACL_ERROR_SNAPSHOT_LOCK_TIMEOUT
    507912: ErrorCategory.L3_EXECUTION,# ACL_ERROR_SNAPSHOT_CALLBACK_FAILED
    507913: ErrorCategory.L3_EXECUTION,# ACL_ERROR_SNAPSHOT_REGISTER_CALLBACK_FAILED
    561000: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER
    561002: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_TILING_ERROR
    561003: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_FIND_KERNEL_ERROR
    561101: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_CREATE_EXECUTOR
    561102: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR
    561105: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_KEY_CONFILICT
    561108: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_LOAD_JSON_FAILED
    561116: ErrorCategory.L3_EXECUTION,# ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE
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
    mapped     : True = 错误码命中映射表（确定分级）；False = 关键词/兜底（保守分级）
    graded_by  : "code_map" | "message_hint" | "default"（F5 分级来源）
    """

    def __init__(self, category: ErrorCategory, root_cause: str,
                 location: Optional[str] = None, error_code: Optional[int] = None,
                 mapped: bool = False, graded_by: str = "default"):
        super().__init__(f"[{category.name}] {root_cause}"
                         + (f" (location: {location})" if location else ""))
        self.category = category
        self.location = location
        self.root_cause = root_cause
        self.error_code = error_code
        self.mapped = mapped
        self.graded_by = graded_by

    @property
    def is_retryable(self) -> bool:
        """F2：L1 资源类可重试。"""
        return self.category == ErrorCategory.L1_RESOURCE

    @property
    def is_fatal(self) -> bool:
        """F2：L4 致命类需状态恢复。"""
        return self.category == ErrorCategory.L4_FATAL

    @property
    def is_grade_confident(self) -> bool:
        """F5：分级是否有确定依据（命中厂商错误码映射表）。

        为 False 时上层应按保守策略处置——尤其 D11 状态恢复决策，
        不应把兜底 L3 当作"确定不是致命错误"而跳过恢复评估。
        """
        return self.mapped

    def to_dict(self) -> dict:
        """投影序列化（含 F5 分级来源，供可观测性事件流/监控诊断消费）。"""
        return {
            "category": self.category.name,
            "location": self.location,
            "root_cause": self.root_cause,
            "error_code": self.error_code,
            "mapped": self.mapped,
            "graded_by": self.graded_by,
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
    """把任意异常翻译为统一错误对象（F1 三维翻译 + F5 分级可观测）。

    优先级：厂商错误码（ret=XXXX → ACL_ERR_TO_CATEGORY）→ 消息关键词粗分类 → L3 默认。
    root_cause 原样保留异常类型与消息（F4）。

    同时记录分级来源（F5）：
      - 命中映射表 → mapped=True,  graded_by="code_map"
      - 命中消息关键词 → mapped=False, graded_by="message_hint"
      - 两者皆无 → mapped=False, graded_by="default"（兜底 L3，不可当定论）
    """
    msg = str(exc)
    retcode = _extract_acl_retcode(msg)

    if retcode is not None and retcode in ACL_ERR_TO_CATEGORY:
        category = ACL_ERR_TO_CATEGORY[retcode]
        mapped, graded_by = True, "code_map"
    else:
        category = ErrorCategory.L3_EXECUTION
        mapped, graded_by = False, "default"
        for pattern, cat in _MESSAGE_HINTS:
            if pattern.search(msg):
                category = cat
                graded_by = "message_hint"
                break

    return FlagosError(
        category=category,
        root_cause=f"{type(exc).__name__}: {msg}",
        location=location,
        error_code=retcode,
        mapped=mapped,
        graded_by=graded_by,
    )
