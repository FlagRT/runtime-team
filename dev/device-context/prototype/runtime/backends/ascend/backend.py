#!/usr/bin/env python3
"""
昇腾（Ascend）后端实现（runtime/backends/ascend/backend.py）

对应周计划 W2 任务 3：把 910C 阶段已交付的资产注册进统一框架。

关键原则：**不重复实现，直接复用已验证模块**——
  - 错误码翻译  → 复用 conformance/errors.py（108 条映射 + F5 可观测）
  - 状态恢复    → 复用 conformance/recovery.py（R1-R5 + rebuild_mode 真实重建）
  - 设备状态    → 复用 conformance/device_state.py（四态机）
本文件只做"接口适配"：把已有能力包装成 RuntimeBackend 的标准形态。

底层：torch_npu（昇腾原生 PyTorch 扩展），本层不触碰算子分发。
"""

import ctypes
from pathlib import Path
from typing import Optional

from ...api.errors import ErrorCategory, FlagosError
from ..base import RuntimeBackend, state_token

#: 真实 `libascendcl.so` 的**进程内句柄**缓存（`/proc/self/maps` 里的那份）。
_LIB_CACHE = {}


def _real_libascendcl():
    """取**已加载的真实** `libascendcl.so`（找不到返回 None）。

    ⚠️ 为什么不用 `ctypes.CDLL("libascendcl.so")`（2026-09-30 踩到）：
    CANN 工具链里同时有**真实库**与**stub 库**（供编译期用），直接按名加载会**抓到 stub**，
    调用返回 `rc=100039`（stub library cannot be used for execution）——
    **症状是"调用失败"，很容易被误读成"接口不可用"**。
    正确做法：从 `/proc/self/maps` 取 pyACL **自己已经加载**的那份真实库路径再 `dlopen`。

    ⚠️ 只在需要**头文件里有、pyACL 未暴露**的入口时才走这条路（如 `aclrtStreamGetPriority`）。
    """
    if "lib" in _LIB_CACHE:
        return _LIB_CACHE["lib"]
    lib = None
    try:
        seen = set()
        with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if "libascendcl.so" in ln:
                    path = ln.split()[-1]
                    if path.startswith("/") and path not in seen:
                        seen.add(path)
                        try:
                            lib = ctypes.CDLL(path)
                            break
                        except OSError:
                            continue
    except OSError:
        lib = None
    _LIB_CACHE["lib"] = lib
    return lib

# conformance 目录（已有资产所在）：device-context/benchmarks/ascend_regression/conformance
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"

# conformance 模块用 IntEnum 分级（L1=1..L4=4），统一层用字符串枚举 —— 转换表
_INT_TO_CATEGORY = {
    1: ErrorCategory.L1_RESOURCE,
    2: ErrorCategory.L2_PARAM,
    3: ErrorCategory.L3_EXECUTION,
    4: ErrorCategory.L4_FATAL,
}


class AscendBackend(RuntimeBackend):
    """昇腾后端：torch_npu + 已有 conformance 资产。"""

    name = "ascend"
    device_type = "npu"

    # 已具备的能力（conformance 会据此生成 stub-skip 报告）
    _capabilities = {
        "device", "memory", "stream", "event",
        # ⚠️ 2026-09-29（契约不变式 I1 判据暴露，台账第 18 条）：**弃用别名不得列入声明集**。
        # `sync_timeout` 自 2026-09-28 起统一为 `bounded_sync` 的**别名**（见基类
        # `_CAPABILITY_ALIASES`），三家 `_CAPABILITY_KEYS` 均含它 ⇒ `supports("sync_timeout")`
        # 在任何实例上都为 True（兼容入口保留）。但把它当成**在册能力**列进 `info()["capabilities"]`
        # 只有本家这么做 ⇒ 同一份下游代码在不同芯片上读到**不同的在册集合**（18 vs 13/10）——
        # 而这不是真实能力差异。**别名是兼容入口，不是在册能力。**
        "bounded_sync",           # 统一键名：有界同步（与 kunlun / cambricon 对齐）
        "error_map",               # 109 条 ACL 错误码映射（2026-09-22 增补通用段 500000）
        "recovery_probe", "recovery_real",  # 探针重试 + 真实重建
        "device_state",            # 四态机
        "graph_capture",           # torch.npu.graph
        "stream_priority",         # 范围**可读**：实测 (least, greatest) = (7, 0)，0 最高
                                   # ⚠️ 2026-09-30（A2/B2 v2）：**能读范围 ≠ 能设置** ——
                                   #   设置受阻（torch_npu 丢参数 + 无 ExternalStream），
                                   #   故下面只声明 readback，**不声明** stream_priority_control。
        "stream_priority_readback",  # 回读某条流的优先级（ctypes 调真实 libascendcl 的
                                   #   `aclrtStreamGetPriority`；pyACL 未暴露该入口）
                                   #   实测：普通流回读 0；pyACL 建的流回读 == 传入值 ✅
        "multidevice",
        # ── 2026-09-29（工作包 B/C）新增 ──
        "memory_alloc",            # pyACL acl.rt.malloc/free（**不是** torch.npu.caching_allocator_*）
        "memory_alloc_stat",       # memory_stats()["allocated_mb"]（分配器视角）
        "record_stream",           # 跨流内存保护（Tensor.record_stream 可用）
        "context_lifecycle",       # pyACL create/set/get/destroy_context（真机实测可用）
        # 2026-09-29（工作包 C·对齐 P800 第五轮）：**只读观测**。与 context_lifecycle
        # 分开声明 —— 有的栈能管生命周期、有的栈只允许观测（平台单上下文）。
        # 本家实测：get_context / get_primary_ctx_state 可用（见 `_ctx_query_raw`）。
        "context_query",
        # ── 2026-09-29（A2 收尾）新增 ──
        "device_state_control",  # 四态**驱动**（`runtime.set_device_state()`）；本层账本，非厂商能力
    }

    #: 能力**全集**（已知能力名，`info()["supports"]` 按此逐项 True/False 呈现）。
    #: 2026-09-22 补齐：原先 ascend 未定义本清单、也未覆写 `info()`，
    #: 于是它的元信息比其他三家**薄一大截**（无 `supports` 映射）——**四家对称性缺口**。
    #: 本清单 = 三实例**同一份规范键集合**（含 `sync_timeout`，但它是**别名**：
    #: `info()["supports"]` 里会出现（兼容下游按历史键名查询），**不再进 `capabilities`**）。
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync", "sync_timeout",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
        # 2026-09-30（B2 v2 落地，(A) 方案）：优先级能力**拆两把钥匙**
        #   —— 与 `device_state` / `device_state_control`、`context_query` / `context_lifecycle` 同构：
        #      **能读 ≠ 能改**。两键同名的家族一次性补齐（未声明即为 False，如实呈现）。
        "stream_priority_control", "stream_priority_readback",
        # 2026-09-29（工作包 B/C）新增键（未声明即为 False，如实呈现）
        "memory_alloc", "memory_alloc_stat", "record_stream", "context_lifecycle",
        # 2026-09-29（工作包 C·P800 专项）：**只读观测**能力键，与 context_lifecycle
        # 分开 —— 有的栈能管上下文生命周期，有的栈只允许观测（平台单上下文）。
        "context_query",
        # 2026-09-29（A2 收尾）：四态机的 **驱动** 入口（原只有查询 `device_state`）。
        # 芯片无关 —— 走本层共享状态机（进程内账本，不依赖厂商原语），故三家一致声明。
        "device_state_control",
    )

    def info(self) -> dict:
        """元信息（与另三家同款结构：name / framework / torch / device_count / supports / …）。

        2026-09-22 补：本方法原缺 ⇒ `info()` 走基类默认，只给 name/device_type/capabilities，
        **没有 `supports` 映射**。下游"读后端即知能力"的统一体验在 ascend 上不成立。
        """
        issues = self.known_issues()
        return {
            "name": self.name,
            # 2026-09-29（B3 复核）：本方法**原缺 `device_type`** ⇒ `info()["device_type"]`
            # 直接 KeyError，而契约与 smoke 都写「info 至少含 name/device_type/capabilities」。
            # 未被发现的原因：矩阵取的是**类属性**、smoke 该判据只对 **stub** 跑 ⇒ 真实后端无人守。
            "device_type": self.device_type,
            **self._base_info_fields(),
            "framework": "torch_npu",
            "torch": self.torch.__version__,
            "device_count": self.device_count(),
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
            # 有界同步的**真实实现路径**（本家走 pyACL；不可用时降级，如实暴露原因）
            "bounded_sync_scope": "pyACL synchronize_*_with_timeout（真中断）；"
                                  "pyACL 不可用时降级为普通同步",
            "acl_available": self.acl is not None,
            "acl_unavailable_reason": self.acl_unavailable_reason,
            "known_issues": [x.get("id") for x in issues],
            "evidence_level": (
                "已在 910C 真机长期验证（conformance 13+6、两条腿、错误闭环）；"
                "2026-09-22 补齐 info()/known_issues 并修「非超时码冒充超时」"
            ),
        }

    def __init__(self):
        self._torch = None
        self._acl = None
        #: pyACL 不可用时的**如实原因**（供上层与证据读取；不可用时 `_acl` 为 False）
        self._acl_reason = ""
        self._errors = None
        self._recovery = None
        self._device_state = None
        self._conformance_loaded = False

    # ─────────────── 延迟加载（避免在无 torch_npu 环境 import 失败）───────────────

    @property
    def torch(self):
        if self._torch is None:
            import torch
            import torch_npu  # noqa: F401  触发 npu 设备注册
            self._torch = torch
        return self._torch

    #: pyACL 中**真正表示"同步/等待超时"**的错误码（来源 CANN `rt_error_codes.h`）。
    #: **只有这些码**才允许映射为 `TimeoutError`；其他非零码必须按错误码表如实分级 ——
    #: 否则会把"参数/上下文/驱动错误"冒充成"超时"，让下游按 L3 的 `replay` 去重放（动作是错的）。
    _ACL_SYNC_TIMEOUT_CODES = frozenset({107019, 107020, 507046, 507047})

    @property
    def acl(self):
        """pyACL；**不可用时返回 None**（有界同步降级为普通同步）。

        ⚠️ **2026-09-22 修（原型内部缺陷，910C 真机暴露）：必须检查 `acl.init()` 的返回值。**
        原先只看"有没有抛异常"，于是当 `acl.init()` **返回非 0** 时（实测宿主带卡容器并发超限时
        返回 **500000 = `ACL_ERROR_INTERNAL_ERROR`**，定义见 CANN `acl/acl_base_rt.h`），
        本属性仍把一个**未初始化的句柄**当成可用 ⇒ 后续每次有界同步都拿到虚假 rc，
        并被错误地报告成"同步超时"（实测 rc=107000 = `ACL_ERROR_RT_PARAM_INVALID`）。
        一句话：**一个 ACL 初始化失败被伪装成了"设备同步超时"**，下游会据此 take 错误的处置
        （L3→`replay` 重放，而正确动作是 L2→`raise`）。

        ⇒ 现在把「init 非 0」与「取不到设备数」都视为**不可用**：降级为普通同步并**如实标注**，
        原因留在 `acl_unavailable_reason` 里。
        """
        if self._acl is None:
            try:
                import acl
                rc = acl.init()
                # ⚠️ `100002 = ACL_ERROR_REPEAT_INITIALIZE`（CANN `acl/acl_base.h`）：
                #    torch_npu / Torch-FL 等先初始化过 ACL 时，本处再 init 会返回它。
                #    **它不是错误**；若当失败，会把「pyACL 可用」误判为不可用，
                #    于是有界同步静默降级为普通同步（能力凭空丢失）。
                #    ⇒ 与第 9 条缺陷同源：**非零 rc 必须逐码确认语义**。
                if rc not in (0, 100002):
                    self._acl = False
                    extra = ("；ACL_ERROR_INTERNAL_ERROR —— 宿主带卡容器并发超限时的典型码"
                             if rc == 500000 else "")
                    self._acl_reason = f"acl.init() 返回 {rc}{extra}"
                else:
                    cnt = acl.rt.get_device_count()
                    # pyACL 的多数接口返回 (值, rc) 二元组；rc 非 0 视为不可用
                    if isinstance(cnt, tuple):
                        cnt, crc = cnt[0], cnt[1]
                        if crc != 0:
                            self._acl = False
                            self._acl_reason = f"acl.rt.get_device_count() 返回 rc={crc}"
                            return self._acl or None
                    if not cnt:
                        self._acl = False
                        self._acl_reason = "acl.rt.get_device_count() = 0（无可用设备上下文）"
                    else:
                        self._acl = acl
                        self._acl_reason = ""
            except Exception as e:
                self._acl = False  # 标记不可用，避免重复尝试
                self._acl_reason = f"import acl 失败：{type(e).__name__}: {str(e)[:80]}"
        return self._acl or None

    @property
    def acl_unavailable_reason(self) -> str:
        """pyACL 不可用的如实原因（可用时为空串）。"""
        _ = self.acl                      # 触发一次探测
        return self._acl_reason

    def _raise_bounded_rc(self, rc: int, what: str, timeout_ms: int) -> None:
        """把 pyACL 的**非零 rc** 转为正确的异常类型（**不冒充超时**）。

        - rc ∈ `_ACL_SYNC_TIMEOUT_CODES` → `TimeoutError`（真有界语义）
        - 其他 rc → 经 `translate_error` **按错误码表如实分级**并抛出
          （例：107000 = `ACL_ERROR_RT_PARAM_INVALID` ⇒ L2_PARAM ⇒ disposition `raise`）
        """
        if rc in self._ACL_SYNC_TIMEOUT_CODES:
            raise TimeoutError(f"{what}超时（{timeout_ms}ms），pyACL rc={rc}")
        fe = self.translate_error(
            RuntimeError(f"{what}失败，error code is {rc}"),
            location=f"{self.name}:bounded_sync")
        raise fe

    def _load_conformance(self):
        """按需导入已有 conformance 模块（错误码/恢复/状态）。"""
        if self._conformance_loaded:
            return
        import sys
        sys.path.insert(0, str(_CONFORMANCE_DIR))
        import errors as _errors
        import recovery as _recovery
        import device_state as _device_state
        self._errors = _errors
        self._recovery = _recovery
        self._device_state = _device_state
        self._conformance_loaded = True

    # ─────────────── 设备（职责 D2）──────────────

    def device_count(self) -> int:
        return self.torch.npu.device_count()

    def set_device(self, ordinal: int) -> None:
        self.torch.npu.set_device(ordinal)

    # ─────────────── 内存（职责 D3）──────────────

    def memory_stats(self, ordinal: int) -> dict:
        torch = self.torch
        prev = torch.npu.current_device()
        if prev != ordinal:
            torch.npu.set_device(ordinal)
        try:
            free, total = torch.npu.mem_get_info()
        except Exception:
            # 老版本 torch_npu 可能没有 mem_get_info
            total = torch.npu.get_device_properties(ordinal).total_memory
            free = total - torch.npu.memory_allocated(ordinal)
        finally:
            if prev != ordinal:
                torch.npu.set_device(prev)
        out = {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }
        # 工作包 B-2：新增 `allocated_mb`（分配器视角）—— 三键语义不变，只增键；
        # 取不到时**省略**该键（半成品 0 值比没有键更误导）。
        try:
            out["allocated_mb"] = int(torch.npu.memory_allocated(ordinal) / 1024 / 1024)
        except Exception:
            pass
        return out

    # ─────────────── 内存句柄与生命周期（职责 D3 · 工作包 B-1）───────────────

    def _alloc_raw(self, size_bytes: int, ordinal: int):
        """用 pyACL 申请**原始设备内存**，返回厂商指针（不暴露给上层）。

        ⚠️ 实测（910C，2026-09-29）：**不能**用 `torch.npu.caching_allocator_alloc`
        —— 该名字在本栈上只是继承了 `torch.cuda` 的实现（torch_npu 未覆写），
        调用会走 `torch.cuda.current_device()` → `_cuda_init()`，直接报
        `RuntimeError: Found no NVIDIA driver on your system`。
        **`hasattr` 为真 ≠ 可用**（与本方向台账第 6/11 条同族）⇒ 本家走 pyACL。
        """
        acl = self.acl
        if acl is None:
            raise self.translate_error(
                RuntimeError("内存句柄：pyACL 不可用（"
                             + (self.acl_unavailable_reason or "原因未记录") + "）"),
                location="memory:allocate")
        torch = self.torch
        prev = torch.npu.current_device()
        if prev != ordinal:
            torch.npu.set_device(ordinal)
        try:
            ptr, ret = acl.rt.malloc(int(size_bytes), 0)
        finally:
            if prev != ordinal:
                torch.npu.set_device(prev)
        if ret != 0:
            raise self.translate_error(
                RuntimeError(f"acl.rt.malloc failed, error code is {ret}"),
                location="memory:allocate")
        return ptr

    def _free_raw(self, ptr, handle: dict) -> None:
        """释放 pyACL 指针。

        ⚠️ 实测：`acl.rt.free()` 对**二次释放 / 非法指针静默返回 0**（不报错）。
        因此"重复释放必须报错"**由基类的句柄登记表保证**，不依赖厂商原语
        —— 这正是统一层把"静默"变"显式"的一个具体落点。
        """
        ret = self.acl.rt.free(ptr)
        if ret != 0:
            raise self.translate_error(
                RuntimeError(f"acl.rt.free failed, error code is {ret}"),
                location="memory:free")

    # ─────────────── 设备上下文生命周期（职责 D2/D1 · 工作包 C）───────────────

    def _ctx_create_raw(self, ordinal: int):
        """pyACL 创建设备上下文；实测 `create_context` 返回 `(handle, ret)` 元组。"""
        acl = self.acl
        if acl is None:
            raise self.translate_error(RuntimeError("设备上下文：pyACL 不可用"),
                                       location="context:create")
        res = acl.rt.create_context(int(ordinal))
        ctx, ret = res if isinstance(res, tuple) else (res, 0)
        if ret != 0:
            raise self.translate_error(
                RuntimeError(f"acl.rt.create_context failed, error code is {ret}"),
                location="context:create")
        return ctx

    def _ctx_set_raw(self, ctx, handle: dict) -> None:
        ret = self.acl.rt.set_context(ctx)
        if ret != 0:
            raise self.translate_error(
                RuntimeError(f"acl.rt.set_context failed, error code is {ret}"),
                location="context:set")

    def _ctx_destroy_raw(self, ctx, handle: dict) -> None:
        """销毁上下文。

        实测（910C，2026-09-29）：销毁后 pyACL **自动回落到进程默认上下文**，设备仍可用。
        ⚠️ 但**在其上建立的流"当场使用不报错"**，直到**进程退出清理阶段**才暴露
        `The stream is not in the current context` / `Stream destroy failed … 107003`
        —— 属"**静默延迟暴露**"，本层需主动拦截（见职责审计的绑定语义判据）。
        """
        ret = self.acl.rt.destroy_context(ctx)
        if ret != 0:
            raise self.translate_error(
                RuntimeError(f"acl.rt.destroy_context failed, error code is {ret}"),
                location="context:destroy")

    def _ctx_query_raw(self, ordinal: int = 0):
        """只读读回**当前生效的设备上下文**（pyACL）。

        实测（910C，2026-09-29）：
          · `acl.rt.get_context(dev)` 返回 **`(ctx, ret)` 元组**（与 `create_context` 同款形状），
            3 次连续调用**稳定一致** ⇒ 可用于同一性比对；
          · **销毁上下文后** 返回 **`(0, 107002)`**（`ACL_ERROR_RT_CONTEXT_NULL`）
            ⇒ **厂商如实报错**，不需要本层兜；
          · `acl.rt.get_primary_ctx_state(dev)` 返回**三元组**：设备 0 → `(1, 0, 0)`
            （primary 已存在）、设备 1 → `(0, 0, 0)`、越界设备 7 → `(0, 0, 107001)`。
          · pyACL **无 flags 概念** ⇒ `flags` 如实置 `None`（不臆造）。

        入参 `ordinal` **被忽略**：契约要的是「**此刻实际生效**的那个上下文」，
        故一律向厂商问"当前设备上的上下文"（多设备下由 `acl.rt.get_device()` 决定）。
        """
        acl = self.acl
        if acl is None:
            return None
        try:
            dev = acl.rt.get_device()
            dev = int(dev[0]) if isinstance(dev, tuple) else int(dev)
        except Exception:                                     # noqa: BLE001
            dev = 0
        res = acl.rt.get_context(dev)
        ctx, ret = res if isinstance(res, tuple) else (res, 0)
        if int(ret) != 0 or not ctx:
            return None
        return {"ctx": ctx, "ordinal": dev, "flags": None}

    def peek_current_device(self) -> int:
        """当前默认设备序号（**只读**，用于 record_stream 的保守同步）。"""
        try:
            return int(self.torch.npu.current_device())
        except Exception:
            return 0

    # ─────────────── 执行 / 多流 Stream（职责 D4/D5）──────────────

    def _create_stream_raw(self, priority=None):
        """默认路径：`torch.npu.Stream()` —— **与历史版本逐位一致**。

        `priority is not None` 分支**正常走不到**：本后端不声明 `stream_priority_control`，
        基类门禁会先抛 `NotImplementedError`。此处仍**显式抛错**（而不是静默忽略）——
        万一有人误把能力键加上，这里会**当场失败**，而不是悄悄给出一条无优先级的流。

        为什么本家做不了（2026-09-30 实测，6 条证据；详见 `known_issues()`）：
        pyACL 能建带优先级的流且**可回读**（0/3/7 全部保留），但那条流是**裸 ACL 句柄**，
        而本栈**没有任何入口**能把它包成"torch 可用的流"
        （无 `torch.npu.ExternalStream`；`Stream(stream_ptr=H)` 静默忽略；
        `getStreamFromExternal` 未暴露到 Python）。流若不能进 torch 执行上下文，设置了也白设。
        """
        if priority is not None:
            raise NotImplementedError(
                f"ascend：**无法**在保持『流可被 torch 执行上下文使用』的前提下设置优先级"
                f"（请求 priority={priority}）。\n"
                f"  · pyACL `aclrtCreateStreamWithConfig` 确实接受并保留该参数，但它产出的是"
                f"**裸 ACL 句柄**，本栈无入口将其包回 torch 流；\n"
                f"  · `torch.npu.Stream(priority=…)` 收下了这个 kwarg 却**静默丢弃**"
                f"（`aclrtStreamGetPriority` 回读恒 0）；\n"
                f"  · 上游诉求：torch_npu 补 `ExternalStream`（同文件已有 `ExternalEvent`），"
                f"或把 `Stream(stream_ptr=…)` 真正接上（现值被静默忽略）。\n"
                f"  · 证据：`910C/probes/prio_readback_910c_npu_20260930.log`、"
                f"`prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md`。")
        return self.torch.npu.Stream()

    def create_event(self):
        """优先复用已有 NpuEventAdapter（补 wait_host + 未 record query 修正），
        保证与 910C 阶段的事件语义契约完全一致；不可用时退回原生 Event。"""
        self._load_conformance()
        try:
            from npu_events import NpuEventAdapter
            return NpuEventAdapter()
        except Exception:
            return self.torch.npu.Event()

    def current_stream(self):
        return self.torch.npu.current_stream()

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 非空时使用 pyACL 有界等待（超时抛 TimeoutError）。"""
        if timeout_ms is None:
            self.torch.npu.synchronize()
            return
        acl = self.acl
        if acl is None:
            # 无 pyACL 时降级为普通同步（调用方需知悉：不再是"有界"的）
            self.torch.npu.synchronize()
            return
        # set_device 的 rc 也必须检查：实测上下文无效时它返回 107002，
        # 若忽略，后面的同步会拿到 107000 并被误报成"超时"（整条误归因链的中间环）
        rc_dev = acl.rt.set_device(ordinal)
        if rc_dev != 0:
            self._raise_bounded_rc(rc_dev, f"有界同步前 set_device({ordinal})", timeout_ms)
        rc = acl.rt.synchronize_device_with_timeout(timeout_ms)
        if rc != 0:
            self._raise_bounded_rc(rc, f"设备 {ordinal} 同步", timeout_ms)

    # ─────────────── 多流 Stream 支撑（供 api/stream.py 封装）───────────────

    def stream_context(self, native_stream):
        return self.torch.npu.stream(native_stream)

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流（pyACL synchronize_stream_with_timeout）。

        注意：需要流的底层句柄（torch Stream 的 .npu_stream 属性），
        且任务与同步必须在同一流上才会触发超时（同步空流会立即返回）。
        """
        acl = self.acl
        if acl is None:
            native_stream.synchronize()
            return
        handle = getattr(native_stream, "npu_stream", None)
        if handle is None:
            native_stream.synchronize()
            return
        rc = acl.rt.synchronize_stream_with_timeout(handle, timeout_ms)
        if rc != 0:
            self._raise_bounded_rc(rc, "流同步", timeout_ms)

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待：轮询 query，避免无限阻塞。"""
        import time
        deadline = time.time() + timeout_ms / 1000.0
        while True:
            try:
                if native_event.query():
                    return True
            except Exception:
                # 未 record 的事件 query 语义由契约定义；此处按"未完成"处理
                pass
            if time.time() >= deadline:
                return False
            time.sleep(0.001)

    # ─────────────── 错误码翻译（职责 D10）──────────────

    # 供 smoke/conformance 做"含厂商码必走 code_map"的正向验证（各家自带样例；
    # 无样例的后端不设此属性，判据会如实 SKIP 正向检查，只验诚实性）
    SAMPLE_CODED_ERROR = "device reset failed, error code is 507015"

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """翻译为**统一** FlagosError。

        注意：conformance 模块的历史 FlagosError 使用 IntEnum 分级（L1=1..L4=4）
        且不含 disposition/retryable 等统一语义字段。为保证对外类型一致，
        这里统一转换为 runtime.api.errors.FlagosError。
        """
        self._load_conformance()
        fe = self._errors.translate_error(exc, location=location)
        u = self._to_unified(fe)
        # 2026-09-22（对称复跑暴露的不对称）：kunlun 在后端侧回填 backend 名，
        # ascend 此前只靠 api 层 translate_via_backend 回填 —— 直接调后端方法时
        # fe.backend 为 None。FlagosError.backend 是文档化字段，后端侧必须回填。
        u.backend = self.name
        return u

    @staticmethod
    def _to_unified(fe) -> FlagosError:
        """历史 FlagosError → 统一 FlagosError（IntEnum → 统一枚举）。"""
        cat = getattr(fe, "category", None)
        if isinstance(cat, int) and not isinstance(cat, ErrorCategory):
            cat = _INT_TO_CATEGORY.get(int(cat), ErrorCategory.L3_EXECUTION)
        graded_by = getattr(fe, "graded_by", "default")
        return FlagosError(
            category=cat if isinstance(cat, ErrorCategory) else ErrorCategory.L3_EXECUTION,
            root_cause=getattr(fe, "root_cause", str(fe)),
            location=getattr(fe, "location", "") or "",
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=bool(
                getattr(fe, "is_grade_confident", graded_by == "code_map")
            ),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
        )

    # ─────────────── 状态恢复（职责 D11）──────────────

    def probe_device(self, ordinal: int) -> bool:
        self._load_conformance()
        return self._recovery.probe_device(ordinal, device=self.device_type)

    def _recover_device_impl(self, ordinal: int, mode: str = "probe",
                             reason: str = "") -> dict:
        """设备重建。mode: probe / real / hybrid（与 recovery.rebuild_mode 一致）。

        统一返回 dict，recovered 语义 = **设备当前可用**（与另两家后端一致）。

        2026-09-09 修正：底层 recovery.recover_device 仅在设备处于 ISOLATED
        状态时才执行恢复，否则直接 return False —— 这会让「设备本来就正常、
        无需重建」被上层误读为「恢复失败」。此处补充状态与探活判定，
        用 detail 区分「无需重建 / 恢复成功 / 恢复失败」。
        """
        self._load_conformance()
        state = None
        try:
            state = state_token(self._device_state.query_device_state(ordinal))
        except Exception:
            state = "unknown"

        ok = self._recovery.recover_device(
            ordinal,
            reason=reason or f"runtime: rebuild({mode})",
            device=self.device_type,
            rebuild_mode=mode,
        )
        # ⭐ 2026-09-29（第 21 条）：把"是否**真的执行了**销毁/重置/重建序列"作为**事实**
        #   回报给基类（私有键 `_rebuilt`，由基类转成 `context_recreated`）。
        #   不要用 `ok` 反推 —— `ok=True` 也可能来自"设备本来就可用"或"探针重试成功"。
        _path = self._recovery.last_rebuild_path()
        _rebuilt = (_path == "aclrtResetDevice")
        if ok:
            if _rebuilt:
                _detail = f"rebuild_mode={mode}：真实重建成功（aclrtResetDevice 序列已执行）"
            else:
                _detail = (f"rebuild_mode={mode}：探针重试成功（**未**执行销毁/重建；"
                           f"设备本就可用或探针已恢复）")
            return {"ordinal": ordinal, "mode": mode, "recovered": True,
                    "state": state, "detail": _detail, "_rebuilt": _rebuilt}

        # 未执行/重建失败：以探活结果判定设备是否实际可用
        try:
            alive = bool(self.probe_device(ordinal))
        except Exception:
            alive = False
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive, "state": state,
            "detail": (f"rebuild_mode={mode}：设备状态={state}，" +
                       ("无需重建，探活可用" if alive else "探活不可用，恢复失败")),
            # 本分支**没有**执行重建 ⇒ 如实回报 False（基类据此定 `context_recreated`）
            "_rebuilt": False,
        }

    # ─────────────── 可选能力 ───────────────

    def stream_priority_range(self):
        """流优先级区间 **(least, greatest) 2 元组**，语义与 CUDA 一致（0 最高、7 最低）。

        ⚠️ **2026-09-30 修（契约形状，跨实例不一致）**：pyACL 的
        `aclrtDeviceGetStreamPriorityRange` 返回**三元组** `(least, greatest, rc)`，
        本方法此前**直接透传三元组**；而 cambricon 的同名接口返回 **2 元组**
        ⇒ **同一份下游代码在两家读到不同形状**（真机实测 `(7, 0, 0)`）。
        现归一为契约规定的 2 元组；形状由离线判据守住。

        ⚠️ **能读范围 ≠ 能设置**：本家范围可读（实测 `(7, 0)`），但**设置受阻**
        （见 `_create_stream_raw` 与 `known_issues()`）；故本家**不声明**
        `stream_priority_control`。
        """
        acl = self.acl
        if acl is None:
            return None
        try:
            res = acl.rt.device_get_stream_priority_range()
        except Exception:
            return None
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            return int(res[0]), int(res[1])
        return None

    def _stream_priority_read_raw(self, handle):
        """**厂商原语层**：ACL 流句柄 → 优先级（读不出返回 None）。

        ⚠️ 单独成方法是为了**可注入/可替换**：离线自检需要在**无设备**时驱动这条链路
        （"能造可控假原语就别 SKIP"——否则判据只能空转或跳过）。
        """
        lib = _real_libascendcl()
        if lib is None:
            return None
        try:
            fn = lib.aclrtStreamGetPriority
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            got = ctypes.c_uint32(0xFFFFFFFF)
            rc = int(fn(ctypes.c_void_p(int(handle)), ctypes.byref(got)))
            return int(got.value) if rc == 0 else None
        except Exception:                                         # noqa: BLE001
            return None

    def stream_priority_readback(self, native_stream):
        """回读该流的优先级（`aclrtStreamGetPriority`）；读不出返回 None。

        为什么必须能回读（2026-09-30）：`torch.npu.Stream(priority=7)` 与
        `pyACL create_stream_with_config(priority=7)` 在 Python 侧**长得一模一样**，
        只有回读能把"参数真的进去了"（7）与"被静默丢弃"（0）分开。
        """
        handle = getattr(native_stream, "npu_stream", None)
        if not handle:
            return None
        return self._stream_priority_read_raw(handle)

    def device_state(self, ordinal: int):
        """查询设备四态（AVAILABLE/DEGRADED/ISOLATED/DESTROYED）。"""
        self._load_conformance()
        return self._device_state.query_device_state(ordinal)

    def known_issues(self) -> list:
        """已知问题（本家此前为空 —— 2026-09-22 910C 真机实测后登记第一条）。

        ⚠️ 措辞纪律：这条是**宿主资源约束**，不是厂商缺陷，也不是本层代码缺陷；
        但它的**症状极具误导性**（看起来像"设备同步超时"），故必须让接入者在第一步就看见。
        """
        return [{
            "id": "ASCEND-ACL-INIT-CONCURRENCY",
            "severity": "medium",
            "scope": "宿主层面的带卡容器并发名额（同机所有使用者共享）",
            "condition": "宿主已有较多带卡容器处于 Up（本次实测为 4 个他人容器）时，再起容器",
            "symptom": (
                "acl.init() 返回 **500000（ACL_ERROR_INTERNAL_ERROR）**；"
                "随后 acl.rt.set_device() 返回 107002（ACL_ERROR_RT_CONTEXT_NULL）、"
                "有界同步返回 107000（ACL_ERROR_RT_PARAM_INVALID）；"
                "控制台另有 `Failed to obtain the console log level` 与"
                "「Different containers share the same device」提示。"
                "**看起来像设备故障或同步超时，实为宿主名额已满、ACL 根本没初始化成功**"
            ),
            "repro_rate": "确定性（2026-09-22 在 910C 上复现 3/3 次，含重起容器后）",
            "root_cause_layer": "宿主/驱动资源限制（非算子、非通信、非本层代码）",
            "workaround": (
                "释放一个带卡容器名额（协调对应使用者 docker stop）后**立即恢复**；"
                "或改用另一台宿主机。**不要在代码里加重试** —— 那是宿主约束，重试无用"
            ),
            "workaround_risk": "无（不涉及改动）；但要注意 `npu-smi` 会报 `Health: OK`，"
                               "**芯片健康不等于名额有空**",
            "report_to": "知会同机其他使用者/管理员（无需上报厂商）",
            "evidence": "本方向 2026-09-22 实测：acl.init rc=500000、set_device rc=107002、"
                        "sync rc=107000；错误码定义见 CANN `acl/acl_base_rt.h` 与 "
                        "`acl/error_codes/rt_error_codes.h`",
        }, {
            "id": "ASCEND-STREAM-PRIORITY-SET-BLOCKED",
            "severity": "low",
            "scope": "流优先级**设置**（`create_stream(priority=…)`）；范围读取与回读不受影响",
            "condition": "在本栈（CANN 9.0.0 / torch_npu 2.11.0 / 910C）请求带优先级的流",
            "symptom": (
                "`create_stream(priority=…)` 抛 `NotImplementedError`。"
                "**不是设备故障，也不是本层没实现完 —— 是插件层缺一个入口**："
                "① `torch.npu.Stream(priority=7)` 收下 kwarg 却**静默丢弃**"
                "（`aclrtStreamGetPriority` 回读恒 0）；"
                "② 无 `torch.npu.ExternalStream`（同文件里有 `ExternalEvent`，**无 Stream 版**）；"
                "③ `torch_npu._C` 无任何 `*External*` 符号 —— 两条**同族但不同**的假通路："
                "`Stream(stream_ptr=H)` **被接受但静默忽略**（拿到的是流池里另一条流）；"
                "`Stream(stream_id=H)`/`_npu_setStream(stream_id=H)` 则**接受任何整数、"
                "到使用点才抛**（见 ⑥）；"
                "④ `libtorch_npu.so` 导出 `c10_npu::getStreamFromExternal(void*, signed char)` 与 "
                "`c10_npu::setCurrentNPUStream(...)`，**未暴露到 Python**"
                "（全量扫 `.so`：无任何其它库导入该符号；包内 Python 零引用）"
                "⇒ **这是「Python 绑定缺失」，不是「做不到」**；"
                "⑤ `acl_rt.h` 无 priority setter ⇒ 无法「先建后改」"
                "（`aclrtSetStreamAttribute` 的枚举只有 "
                "`FAILURE_MODE / FLOAT_OVERFLOW_CHECK / USER_CUSTOM_TAG / CACHE_OP_INFO`）；"
                "⑥ `Stream(stream_id=H)` **不校验** H，但**任何使用点**都抛 "
                "`INTERNAL ASSERT FAILED at \"../torch_npu/csrc/core/npu/NPUStream.cpp\":371, "
                "please report a bug to PyTorch. Unrecognized stream … (I didn't recognize the "
                "stream type)` ⇒ 把「用法错误」表述成「请向框架报 bug」。"
                "⇒ 保持『流可被 torch 执行上下文使用』的前提下，**无法**让 priority 生效"
                "（pyACL 建的带优先级流是裸 ACL 句柄，包不回 torch 流）"
            ),
            "repro_rate": ("确定性（2026-09-30 在 910C 上逐项实测；其中 ②③ 为 12 个候选 kwarg 的"
                           "穷举结果：仅 `stream_ptr` 被接受且被忽略，其余一律 TypeError）。"
                           "**2026-10-08 自证审计复核**（`audit_stream_priority_ascend.py`）："
                           "结论不变，并补齐「逐档位设备回读」与「使用点断言原文」两项原始读数；"
                           "⚠️ 同族复查中修掉一处**探针自身缺陷**：`hasattr/getattr` 探 "
                           "`Stream.priority` 会**抛** `RuntimeError`（`hasattr` 只吞 "
                           "`AttributeError`）⇒ 存在性探测必须 `try/except BaseException` 分类回报"),
            "root_cause_layer": "厂商 PyTorch 插件（torch_npu）接口面缺失；本层已如实降级不声明",
            "workaround": (
                "① 需要『参数真的进设备』的取证场景：直接用 pyACL "
                "`aclrtCreateStreamWithConfig`（**裸流，不能跑 torch 算子**）；"
                "② 需要跨实例可移植：用 `create_stream()`（不指定优先级），"
                "并用 `supports(stream_priority_control)` **提前判分支**"
            ),
            "workaround_risk": ("裸 ACL 流**不可用于 torch 执行上下文**；且官方文档说明"
                                "**Atlas 训练系列上 priority 属「预留参数、暂不使用」**"
                                "⇒ 即便打通入口，训练产品上也未必有调度效果"
                                "（**「能设」与「有效果」是两件事**）"),
            "report_to": "torch_npu / CANN 上游（诉求：补 `ExternalStream`，或让 "
                         "`Stream(stream_ptr=…)` 真正生效；现状是**静默忽略**，极易误判）",
            "evidence": "`910C/probes/prio_readback_910c_npu_20260930.log`（回读对照）、"
                        "`910C/probes/prio_api_surface_910c_npu_20260930.log`（接口面与插件源码）、"
                        "`prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md` §5、"
                        "`910C/probes/audit_20261008_out/ascend_audit.log`（2026-10-08 自证审计："
                        "ACL 保留 0/3/7 + 逐档位设备回读全 0 + 四处使用点断言原文）",
        }]


def build() -> AscendBackend:
    """注册表自动发现使用的工厂函数。"""
    return AscendBackend()
