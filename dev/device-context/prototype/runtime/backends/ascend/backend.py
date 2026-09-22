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

from pathlib import Path
from typing import Optional

from ...api.errors import ErrorCategory, FlagosError
from ..base import RuntimeBackend

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
        "sync_timeout",            # pyACL synchronize_*_with_timeout（历史键名）
        "bounded_sync",           # 统一键名：有界同步（与 flagos 对齐）
        "error_map",               # 109 条 ACL 错误码映射（2026-09-22 增补通用段 500000）
        "recovery_probe", "recovery_real",  # 探针重试 + 真实重建
        "device_state",            # 四态机
        "graph_capture",           # torch.npu.graph
        "stream_priority",         # least=7 / greatest=0
        "multidevice",
    }

    #: 能力**全集**（已知能力名，`info()["supports"]` 按此逐项 True/False 呈现）。
    #: 2026-09-22 补齐：原先 ascend 未定义本清单、也未覆写 `info()`，
    #: 于是它的元信息比其他三家**薄一大截**（无 `supports` 映射）——**四家对称性缺口**。
    #: 本清单 = 另三家那 12 项 + 本家历史键名 `sync_timeout`（保留以免下游读不到）。
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync", "sync_timeout",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
    )

    def info(self) -> dict:
        """元信息（与另三家同款结构：name / framework / torch / device_count / supports / …）。

        2026-09-22 补：本方法原缺 ⇒ `info()` 走基类默认，只给 name/device_type/capabilities，
        **没有 `supports` 映射**。下游"读后端即知能力"的统一体验在 ascend 上不成立。
        """
        issues = self.known_issues()
        return {
            "name": self.name,
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
                if rc != 0:
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
        return {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }

    # ─────────────── 执行 / 多流 Stream（职责 D4/D5）──────────────

    def create_stream(self):
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

    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "") -> dict:
        """设备重建。mode: probe / real / hybrid（与 recovery.rebuild_mode 一致）。

        统一返回 dict，recovered 语义 = **设备当前可用**（与 flagos 后端一致）。

        2026-09-09 修正：底层 recovery.recover_device 仅在设备处于 ISOLATED
        状态时才执行恢复，否则直接 return False —— 这会让「设备本来就正常、
        无需重建」被上层误读为「恢复失败」。此处补充状态与探活判定，
        用 detail 区分「无需重建 / 恢复成功 / 恢复失败」。
        """
        self._load_conformance()
        state = None
        try:
            state = str(self._device_state.query_device_state(ordinal))
        except Exception:
            state = "unknown"

        ok = self._recovery.recover_device(
            ordinal,
            reason=reason or f"runtime: rebuild({mode})",
            device=self.device_type,
            rebuild_mode=mode,
        )
        if ok:
            return {"ordinal": ordinal, "mode": mode, "recovered": True,
                    "state": state, "detail": f"rebuild_mode={mode}：重建成功"}

        # 未执行/重建失败：以探活结果判定设备是否实际可用
        try:
            alive = bool(self.probe_device(ordinal))
        except Exception:
            alive = False
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive, "state": state,
            "detail": (f"rebuild_mode={mode}：设备状态={state}，" +
                       ("无需重建，探活可用" if alive else "探活不可用，恢复失败")),
        }

    # ─────────────── 可选能力 ───────────────

    def stream_priority_range(self):
        acl = self.acl
        if acl is None:
            return None
        try:
            return acl.rt.device_get_stream_priority_range()
        except Exception:
            return None

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
        }]


def build() -> AscendBackend:
    """注册表自动发现使用的工厂函数。"""
    return AscendBackend()
