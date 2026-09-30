#!/usr/bin/env python3
"""
统一运行时 API（runtime/__init__.py）

用户视角的全部入口。**同一份代码切换后端只改 use() 一行**：

    import runtime
    runtime.use("ascend")            # 或 "kunlun"（昆仑芯 P800）/ "cambricon"（寒武纪 MLU590）
                                     # 三家均为**厂商官方 torch 插件**路线（torch_npu / torch.cuda 兼容层 / torch_mlu）
    runtime.set_device(0)
    s = runtime.create_stream()
    fe = runtime.translate_error(exc, location="op:matmul")
    r = runtime.recover_device(0, mode="real")   # -> dict: {ordinal, mode, recovered, state, detail, ...}
    ok = r["recovered"]

    # 内存句柄（工作包 B）：只做句柄语义，不做池化策略
    h = runtime.allocate(4 * 1024 * 1024)        # -> 统一句柄 dict（不含厂商指针）
    runtime.free(h)                              # 重复释放会如实报错

    # 设备上下文生命周期（工作包 C）：能力未声明时如实报错
    ctx = runtime.context_create(0)
    runtime.context_count()
    runtime.context_destroy(ctx)                 # 只接受本层创建的句柄

对应职责子层：设备上下文 + 多流 Stream + 错误码翻译 + 状态恢复。
架构位置：上承算子层/编译层与模型转换器，下接多机多卡分布式训练与推理。
"""

from .api.errors import (
    DISPOSITION,
    ErrorCategory,
    FlagosError,
    translate_via_backend,
)
from .api.stream import Event, Stream
from .backends.base import (
    CONTEXT_HANDLE_KEYS,
    DEVICE_STATE_TOKENS,
    MEMORY_HANDLE_KEYS,
    RuntimeBackend,
)
from .backends import registry
from .backends.registry import (
    BackendNotFound,
    available,
    current,
    current_name,
    discover,
    get,
    register,
    set_current,
    use,
)

__all__ = [
    # 后端选择
    "use", "set_current", "current", "get", "available", "discover", "register",
    "current_name", "BackendNotFound", "RuntimeBackend", "registry",
    # 错误分级
    "FlagosError", "ErrorCategory", "DISPOSITION", "translate_via_backend",
    "Stream", "Event",
    # 设备 / 流（转发到当前后端）
    "device_count", "set_device", "memory_stats",
    "create_stream", "create_event", "current_stream", "synchronize",
    "stream_priority_readback",
    "probe_device", "recover_device", "translate_error", "device_state",
    "set_device_state", "handle_error",
    # 内存句柄与生命周期（工作包 B）
    "allocate", "free", "memory_handle_count", "MEMORY_HANDLE_KEYS",
    # 设备上下文生命周期（工作包 C）
    "context_create", "context_destroy", "context_count", "context_query",
    "CONTEXT_HANDLE_KEYS",
    "DEVICE_STATE_TOKENS",
    # 审计（`.native` 逃生舱 / 退化路径）
    "native_accesses", "degradations",
]

__version__ = "0.1.0"


# ─────────────── 转发：全部调用当前后端 ───────────────

def device_count() -> int:
    return current().device_count()


def set_device(ordinal: int) -> None:
    current().set_device(ordinal)


def memory_stats(ordinal: int = 0) -> dict:
    """设备内存统计（规范三键 `total_mb` / `used_mb` / `free_mb`；
    声明了 `memory_alloc_stat` 的后端另给 `allocated_mb`）。"""
    return current().memory_stats(ordinal)


def create_stream(priority=None) -> Stream:
    """创建统一 Stream 对象（包装后端原生流）；**可选**指定流优先级。

    2026-09-29（工作包 C）：同时登记"该流创建于哪个本层上下文"，
    以便上下文销毁后在使用点**如实拦截**（厂商侧是静默的，见 backend.check_stream_usable）。

    2026-09-30（(A) 方案）：新增可选 `priority`（**只增不改**，默认 None = 既有行为逐位不变）。
    四道约束由后端基类**统一**执行（`backends/base.py::create_stream`）：
      ① 后端未声明 `stream_priority_control` ⇒ `NotImplementedError`（**显式拒绝**）；
      ② `priority` 非 int / 越出 `stream_priority_range()` ⇒ `ValueError`；
      ③ 创建后**必须回读校验**（回读 != 请求 ⇒ `RuntimeError`）——
         这是"参数被厂商**静默丢弃**"的唯一防线（910C 的 `torch.npu.Stream(priority=7)`
         正是这种情形：收下 kwarg 却回读恒 0）。
    ⭐ 官方语义：数值**越小优先级越高**（`stream_priority_range()` 返回 `(least, greatest)`，
    其中 `greatest` 是**数值最小**的那个）。能否**设置**请先查
    `info()["supports"]["stream_priority_control"]`——**能读范围 ≠ 能设置**。
    """
    b = current()
    native = b.create_stream(priority)
    b.note_stream_created(native)
    return Stream(b, native)


def stream_priority_readback(stream):
    """**回读**某条流的实际优先级；读不出返回 `None`（**不猜、不补零**）。

    兼容传入统一 `Stream` 或后端原生流对象。

    为什么要有这个入口（2026-09-30）：`torch.npu.Stream(priority=7)` 与
    `pyACL create_stream_with_config(priority=7)` 在 Python 侧**长得一模一样**，
    只有回读能把"参数真的进了设备"（7）与"被静默丢弃"（0）分开。
    `create_stream(priority=…)` 内部即用它对第 ③ 条做强制校验。
    """
    native = stream._native_obj if isinstance(stream, Stream) else stream
    return current().stream_priority_readback(native)


def create_event() -> Event:
    """创建统一 Event 对象（包装后端原生事件）。"""
    b = current()
    return Event(b, b.create_event())


def current_stream():
    return current().current_stream()


def synchronize(ordinal: int = 0, timeout_ms=None) -> None:
    current().synchronize(ordinal, timeout_ms=timeout_ms)


def translate_error(exc: BaseException, location: str = "") -> FlagosError:
    return translate_via_backend(exc, current(), location=location)


def probe_device(ordinal: int = 0) -> bool:
    return current().probe_device(ordinal)


def recover_device(ordinal: int = 0, mode: str = "probe", reason: str = "") -> dict:
    return current().recover_device(ordinal, mode=mode, reason=reason)


def device_state(ordinal: int = 0):
    return current().device_state(ordinal)


def set_device_state(ordinal: int, state, reason: str = "") -> str:
    """**驱动**设备四态状态机（本层账本），返回新状态 **token 字符串**。

    与 `device_state()`（只读查询）成对：后者答"**现在是什么态**"，本接口答"**把它置成什么态**"。

    为什么要补这个入口（2026-09-29 A2 压测实测）：`recover_device(mode="real")` **只在设备处于
    `isolated` 时**才真重建，而此前**公开面没有任何入口能把设备置为 `isolated`** ⇒ 只走公开 API 时
    `real` 永远走"无需重建"分支，「某卡 L4 故障 → 设备级恢复」这条链**在公开面上不可触发**。

    ⚠️ **这是本层的隔离账本，不等于厂商设备的真实状态**：把健康设备标成 `isolated`
    只影响**本层**的调度 / 恢复判定（演练与混沌注入靠它），**不会**让硬件出错；
    真实故障的隔离仍应由评估（探针失败）驱动。取值域见 `DEVICE_STATE_TOKENS`；
    **非法取值 ⇒ `ValueError`**（不静默）。
    """
    return current().set_device_state(ordinal, state, reason=reason)


def handle_error(exc: BaseException, ordinal=None, location: str = "", mode: str = "probe"):
    """**五段式错误处理编排**：错误 → 统一错误对象（R1）→ 评估（R2）→
    隔离 / 重建（R3 / R4）→ 重放（R5）。

    返回**统一类型**的 `FlagosError`（与 `translate_error` 同型）；其 `recovery_decision`
    记录流程事件（`captured` / `evaluated: …` / `recovered` / `replay_ready`）供监控与可观测消费。
    `mode` 与 `recover_device` 同语义，**默认 `"probe"`**（进程内安全）；传 `"real"` 才真重建。

    为什么要补这个入口（2026-09-29）：上层此前只能拿到"分级"（`translate_error`）与"重建"
    （`recover_device`）两个零件，**中间那段（评估 → 隔离 → 重放）只能自己拼** —— 而那正是
    R 系列契约的实现。本入口把五段一次性收敛到本层，避免每个上层各写一套（三套口径必然漂移）。

    ⚠️ **已知边界（如实标注，未擅自扩张）**：`replay_ready` 表示"重放集合已可消费"，其内容取决于
    上层是否登记过**在途任务** —— 该登记入口（`mark_inflight` / `finish_inflight`）**目前未公开**，
    故现阶段 `replay_tasks` 恒为空列表。
    """
    return current().handle_error(exc, ordinal=ordinal, location=location, mode=mode)


# ─────────────── 内存句柄与生命周期（工作包 B）───────────────

def allocate(size_bytes: int, ordinal: int = 0) -> dict:
    """申请设备内存，返回**统一内存句柄**（公共字段见 `MEMORY_HANDLE_KEYS`）。

    只提供句柄语义（申请/释放），**不含池化、碎片、峰值、扩容策略** —— 那些属显存方向。
    """
    return current().allocate(size_bytes, ordinal=ordinal)


def free(handle) -> None:
    """释放 `allocate()` 返回的句柄。**重复释放/非本层句柄会报错**（不静默）。"""
    current().free(handle)


def memory_handle_count() -> int:
    """当前在世的内存句柄数（泄漏判据的取数入口）。"""
    return current().memory_handle_count()


# ─────────────── 设备上下文生命周期（工作包 C）───────────────

def context_create(ordinal: int = 0) -> dict:
    """新建设备上下文，返回统一句柄；后端未声明 `context_lifecycle` 时如实报错。"""
    return current().context_create(ordinal=ordinal)


def context_destroy(handle) -> None:
    """销毁本层创建的设备上下文句柄；**对非本层句柄一律拒绝**。"""
    current().context_destroy(handle)


def context_query() -> dict:
    """查询**此刻进程实际生效的设备上下文**（**只读**，不改状态）。

    与 `context_count()` 的分工：后者答"**本层造了几个**"，本接口答"**此刻在哪个上下文上**"
    —— 对**框架自建**的上下文同样有意义（P800 就是这种情况：上下文由 XPytorch 自建，
    本层一个都不造，但上层仍需要知道"我在哪个上下文上、它归谁管"）。

    固定键：`queryable / present / ordinal / flags / managed_by / reason`。
    `managed_by` 取 `"unified"`（本层创建）/ `"external"`（厂商或框架自建）/ `None`。
    """
    return current().context_query()


def context_count() -> int:
    """本层当前在世的设备上下文数（不含进程默认上下文）。"""
    return current().context_count()


# ─────────────── 审计（`.native` 逃生舱 / 退化路径）───────────────

def native_accesses() -> dict:
    """`.native` 逃生舱取用审计：`{"total": n, "by_kind": {...}}`。

    取用即视为绑定该厂商；本计数用于事后定位"可移植性被破坏"的位置。
    """
    return current().native_accesses()


def degradations() -> dict:
    """能力缺失导致的**退化路径**审计（如 `record_stream` 不可用 → 保守同步）。"""
    return current().degradations()


def _auto_discover():
    """导入时自动发现已安装后端（失败静默，用户可显式 use() 触发）。"""
    try:
        discover()
    except Exception:
        pass


_auto_discover()
