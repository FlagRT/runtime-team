#!/usr/bin/env python3
"""FlagOS 后端（torch_fl / flagos 设备后端适配）。

适用环境：锁定训练镜像（flagrt/ascend-operator-runtime-comm），其设备后端为
torch_fl 的 flagos（镜像明确禁止 torch_npu 与 Torch-FL 运行时共存）。

设计原则与其他后端一致：**不重复实现已验证能力，只做适配**。
- 错误码翻译复用 conformance/errors.py（与昇腾后端同源）
- 设备状态/恢复复用 conformance/device_state.py（能力受限时声明不支持）
- 流/事件直接返回框架原生对象（方法面已与统一封装对齐：
  wait_event / wait_stream / synchronize / record / query / wait）
"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import DISPOSITION, ErrorCategory, FlagosError
from ..base import RuntimeBackend

# conformance 目录（已有资产所在）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


class FlagosEventAdapter:
    """torch_fl(flagos) Event 的统一语义适配（与 NpuEventAdapter 同构）。

    补齐两点与统一事件契约的语义缺口：
      - E3：未 record 事件 query() 误报完成 → recorded 跟踪修正（返回 False）
      - E2v2：主机有界等待 wait_host（query 轮询，永不永久阻塞）
    """

    def __init__(self, *args, **kwargs):
        self._ev = None
        self._args, self._kwargs = args, kwargs
        self._recorded = False

    def _ensure(self):
        if self._ev is None:
            m = _current_flagos_module()
            self._ev = m.Event(*self._args, **self._kwargs)
        return self._ev

    def record(self, stream=None):
        r = self._ensure().record(stream)
        self._recorded = True
        return r

    def wait(self, stream=None):
        return self._ensure().wait(stream)

    def synchronize(self):
        r = self._ensure().synchronize()
        self._recorded = True
        return r

    def query(self):
        if not self._recorded:
            return False
        return self._ensure().query()

    def wait_host(self, timeout_ms=None):
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while not self.query():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.002)
        return True

    def elapsed_time(self, end_event):
        return self._ensure().elapsed_time(getattr(end_event, "_ev", end_event))

    def __getattr__(self, item):
        return getattr(self._ensure(), item)


_CURRENT: dict = {}


def _current_flagos_module():
    """返回已加载的 torch.flagos 模块（由 FlagosBackend 在加载时注入）。"""
    mod = _CURRENT.get("mod")
    if mod is None:
        import torch_fl  # noqa: F401
        import torch
        mod = torch.flagos
        _CURRENT["mod"] = mod
    return mod


class FlagosBackend(RuntimeBackend):
    """基于 torch_fl(flagos) 的后端实现。"""

    name = "flagos"
    device_type = "flagos"   # 设备串前缀：flagos:0

    def __init__(self) -> None:
        self._torch = None
        self._mod = None          # torch.flagos
        self._errors_mod = None
        self._device_state = None  # 共享的四态机（标准 import，见 _load_device_state）
        self._loaded = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        if self._loaded:
            return
        # 镜像约束：必须先 import torch_fl，再 import torch（否则 PrivateUse1 被占用）
        import torch_fl  # noqa: F401
        import torch

        self._torch = torch
        self._mod = torch.flagos
        _CURRENT["mod"] = self._mod
        if hasattr(self._mod, "init"):
            try:
                self._mod.init()
            except Exception:
                pass
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    @property
    def mod(self):
        self._load()
        return self._mod

    def _load_errors(self):
        if self._errors_mod is None:
            import importlib.util
            import sys

            sys.path.insert(0, str(_CONFORMANCE_DIR))
            spec = importlib.util.spec_from_file_location(
                "dc_conformance_errors", _CONFORMANCE_DIR / "errors.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._errors_mod = mod
        return self._errors_mod

    # ───────────── 设备 ─────────────
    def device_count(self) -> int:
        return int(self.mod.device_count())

    def set_device(self, ordinal: int) -> None:
        m = self.mod
        if hasattr(m, "set_device"):
            m.set_device(ordinal)
        else:
            self.torch.flagos.device(ordinal)

    def memory_stats(self, ordinal: int) -> dict:
        """显存统计。**统一字段必须含 `total_mb` / `used_mb` / `free_mb`**（接口约定 §1.2）。

        数据来源分两层，**如实分层**（2026-09-22 真机暴露后修）：

          ① `torch.flagos.memory_stats()` —— 只有**进程级**计数
             （`allocated_bytes` / `reserved_bytes` / 各类 call 计数），
             **不含设备总量**。实测 torch_fl 的 `get_device_properties(0).total_memory`
             恒为 **0**（该字段未填充）⇒ 框架侧拿不到总量。
          ② **pyACL** `acl.rt.get_mem_info(ordinal)` —— **设备级** `(free, total)`，
             与 `ascend` / `kunlun` 两个后端**同口径**（能反映同卡上他人占用）。
             实测 910C：free 60.9 GiB / total 61.3 GiB，与 `npu-smi` 一致。

        ⇒ 统一字段取 **②（设备级）**，原始字段保留 **①（进程级）**；
          pyACL 不可用时**降级为进程级**并如实标注（不冒充设备级）。

        说明：本后端仍**不以 torch_npu 为依赖**（与 Torch-FL 互斥那条约束照旧）；
        pyACL 是 CANN 的 C 层 Python 绑定，与 torch 设备后端无关（`ascend` 后端也用它）。
        """
        raw = dict(self.mod.memory_stats() or {})
        out = dict(raw)

        # ── ② 设备级（首选口径）──
        free_b, total_b, reason = self._acl_mem_info(ordinal)
        if total_b:
            out["total_mb"] = int(total_b / 1024 / 1024)
            out["free_mb"] = int(free_b / 1024 / 1024)
            out["used_mb"] = max(0, out["total_mb"] - out["free_mb"])
            out["memory_scope"] = "device"          # 与 ascend / kunlun 同口径
        else:
            # ── ① 降级：只能用进程级 reserved 顶 used；总量给不出来（通常为 0）──
            total = raw.get("total_bytes") or raw.get("total_mb", 0) * 1024 * 1024
            used_b = raw.get("reserved_bytes", raw.get("allocated_bytes", 0))
            out["total_mb"] = int(total / 1024 / 1024)
            out["used_mb"] = int(used_b / 1024 / 1024)
            out["free_mb"] = max(0, out["total_mb"] - out["used_mb"])
            out["memory_scope"] = "process"         # ⚠️ 口径降级，如实标注
            out["memory_degraded_reason"] = reason
        return out

    def _acl_mem_info(self, ordinal: int):
        """经 pyACL 取**设备级** `(free_bytes, total_bytes, reason)`。

        失败时返回 `(0, 0, 原因)` —— **不抛异常、不静默**：调用方据此如实降级。
        注意：必须先 `set_device(ordinal)`，否则 `get_mem_info` 返回
        `107002 = ACL_ERROR_RT_CONTEXT_NULL`（2026-09-22 实测）。
        """
        try:
            import acl  # 延迟导入：无 CANN 环境时本后端仍可用（降级为进程级）
        except Exception as e:
            return 0, 0, f"pyACL 不可导入（{type(e).__name__}）"
        try:
            rc = acl.init()
            # ⚠️ `100002 = ACL_ERROR_REPEAT_INITIALIZE`（CANN `acl/acl_base.h`，2026-09-22 查证）
            #    是「**重复初始化**」，**不是错误**：本进程里 torch_fl 已经初始化过 ACL，
            #    此处再 init 必然返回该码。
            #    实测教训：把它当失败 ⇒ memory_stats 降级为进程级、`total_mb=0` ⇒
            #    连带把错误闭环的 OOM 注入退化成 `torch.empty(0)`（见 F14）。
            #    ⇒ 纪律：**非零 rc 的语义必须逐码确认，不能一律当失败**。
            #      （与第 9 条缺陷同源：那里是把 rc≠0 一律当"超时"。）
            if rc not in (0, 100002):
                return 0, 0, f"acl.init rc={rc}"
            rc = acl.rt.set_device(ordinal)
            if rc != 0:
                return 0, 0, f"acl.rt.set_device({ordinal}) rc={rc}"
            r = acl.rt.get_mem_info(ordinal)
            if isinstance(r, tuple) and len(r) >= 2:
                free_b, total_b = int(r[0]), int(r[1])
                if total_b > 0:
                    return free_b, total_b, ""
                return 0, 0, "acl.rt.get_mem_info 返回 total=0"
            return 0, 0, f"acl.rt.get_mem_info 返回形态异常：{r!r}"
        except Exception as e:
            return 0, 0, f"pyACL 调用异常（{type(e).__name__}: {str(e)[:60]}）"

    def probe_device(self, ordinal: int) -> bool:
        try:
            m = self.mod
            if hasattr(m, "is_available") and not m.is_available():
                return False
            dev = f"flagos:{ordinal}"
            x = self.torch.zeros(2, 2, device=dev)
            return float(x.sum().item()) == 0.0
        except Exception:
            return False

    # ───────────── 流 / 事件 ─────────────
    def create_stream(self):
        return self.mod.Stream()

    def create_event(self):
        # 统一语义适配（未 record 不误报完成 + 主机有界等待）
        return FlagosEventAdapter()

    def current_stream(self):
        return self.mod.current_stream()

    def stream_context(self, native_stream):
        return self.mod.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        # flagos 原生同步为阻塞式；有界语义由上层 pyACL 路径（如可用）补充
        self.mod.synchronize()

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        native_stream.synchronize()

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        deadline = time.time() + (timeout_ms / 1000.0)
        while time.time() < deadline:
            try:
                if native_event.query():
                    return True
            except Exception:
                return False
            time.sleep(0.001)
        return False

    # ───────────── 已知问题（上游/环境）─────────────
    def known_issues(self) -> list:
        """本后端已知的上游问题（须已实测，注明复现率与证据）。

        2026-09-22 新增第 1 条：910C 真机上按变体矩阵实测到的 torch_fl 事件语义缺口。
        """
        return [
            {
                "id": "FLAGOS-EVENT-QUERY-SEMANTICS",
                "severity": "high",
                "scope": "统一 Event 契约的 `query()` / `wait_host()`（事件完成判定）",
                "condition": "事件被 record 到一个**其上已有工作**的流之后（含默认流），"
                             "或 record 到显式流上时",
                "symptom": (
                    "`Event.query()` **不反映完成状态**：流上有工作时恒返回 `False`（实测 4/4），"
                    "且**在 `ev.synchronize()` 成功返回之后仍然返回 `False`**（语义自相矛盾）；"
                    "空流场景下时真时假（实测 3 次中 1 次全 False）。"
                    "⇒ 依赖 `query()` 的 `wait_host()` 会**假超时**（返回 False 而实际已完成）"
                ),
                "repro_rate": (
                    "流上有工作：**确定性 100%**（4/4，含同步后）；空显式流：**不稳定**"
                    "（3 轮中 1 轮全 False）。矩阵与原始输出见 "
                    "`910C/probes/ev_matrix_20260922.log`、`ev_matrix2_20260922.log`"
                ),
                "root_cause_layer": "厂商运行时（torch_fl 的 Event 实现 / 与 CANN event 语义映射）",
                "workaround": (
                    "① **不要用 `query()` 判定完成**；需要等完成时用阻塞式 `Event.synchronize()`；"
                    "② 本方向的统一 `wait_host()` 保持**有界不阻塞**（行为正确），但必须把它的 "
                    "`False` 读作「**未确认完成**」而非「确认未完成」—— 在 flagos 上存在假阴性；"
                    "③ 依赖事件完成判定的流水线请改用「流同步 + 显式依赖」（S-2 路径），"
                    "该路径在 flagos 上已验证通过"
                ),
                "workaround_risk": (
                    "`Event.synchronize()` 是**阻塞式无限等待**，会丢失统一契约的「有界」语义 ⇒ "
                    "只能由调用方按场景自行取舍，运行时不代做"
                ),
                "report_to": "上报 torch_fl / FlagOS 厂商（属跨芯片共性问题，非寒武纪/昇腾特有）",
                "evidence": "910C/probes/ev_matrix_20260922.log、910C/probes/ev_matrix2_20260922.log；"
                            "smoke 第 [6] 节 `Event.record + wait_host 有界返回` 项不稳定复现",
            },
        ]

    # ───────────── 错误翻译 ─────────────
    _INT_TO_CATEGORY = {
        1: ErrorCategory.L1_RESOURCE,
        2: ErrorCategory.L2_PARAM,
        3: ErrorCategory.L3_EXECUTION,
        4: ErrorCategory.L4_FATAL,
    }

    def _to_unified(self, fe) -> FlagosError:
        """后端负责把历史类型（IntEnum 分级）转成统一 FlagosError。"""
        graded_by = getattr(fe, "graded_by", "unknown")
        cat = getattr(fe, "category", None)
        if not isinstance(cat, ErrorCategory):
            try:
                cat = self._INT_TO_CATEGORY.get(int(cat), ErrorCategory.L3_EXECUTION)
            except Exception:
                cat = ErrorCategory.L3_EXECUTION
        return FlagosError(
            category=cat,
            root_cause=getattr(fe, "root_cause", str(fe)),
            location=getattr(fe, "location", "") or "",
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=bool(
                getattr(fe, "is_grade_confident", graded_by == "code_map")),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )

    # 复用 conformance/errors.py 的 ACL 码表（与 ascend 同源），样例码同 ascend
    SAMPLE_CODED_ERROR = "device reset failed, error code is 507015"

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location)
        u = self._to_unified(fe)
        # 2026-09-22：与 ascend/kunlun 对齐，后端侧回填 backend 名（此前缺失）
        u.backend = self.name
        return u

    # ───────────── 恢复 ─────────────
    def recover_device(self, ordinal: int, mode: str = "probe",
                       **kwargs: Any) -> dict:
        """flagos 后端当前仅提供 probe 级恢复（real 重建需设备级原语，暂不支持）。"""
        ok = self.probe_device(ordinal)
        return {
            "ordinal": ordinal,
            "mode": mode,
            "recovered": bool(ok),
            "detail": "flagos 后端：probe 级探活恢复（real/hybrid 暂未支持）",
        }

    # ───────────── 能力声明 ─────────────
    #: 设备四态查询（2026-09-22 补实现，第 7 个跨后端缺陷）
    def _load_device_state(self):
        """按需加载 `conformance/device_state` 资产（进程内设备四态机）。

        ⚠️ **必须用标准 `import`（共享 `sys.modules`），不能用 importlib 独立模块名加载**
        —— 与 `kunlun` / `cambricon` 同一纪律。`device_state` 是**有状态单例**
        （模块级 `_ensure(ordinal)` 持有每台设备的四态、转换事件与订阅者）；
        若像本类的 `_load_errors()` 那样用 `spec_from_file_location("dc_xxx", ...)` 加载，
        会得到**两份状态机** —— conformance/上层设置的状态后端查不到，反之亦然。
        （`errors` 是无状态纯函数，两份无所谓；但那正是「第 4 个跨后端框架缺陷」的成因，
        **不应效仿**。）
        """
        if self._device_state is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            import device_state as _device_state
            self._device_state = _device_state
        return self._device_state

    def device_state(self, ordinal: int):
        """查询设备四态：`available` / `degraded` / `isolated` / `destroyed`。

        **2026-09-22 补实现（第 7 个跨后端缺陷）**：本类原先在 `_capabilities` 里
        **声明了 `device_state` 却没有对应方法** —— 与 2026-09-20 在 `kunlun` 上发现的
        「声明与实现不符」是**同一类**问题。处置口径与当时一致：**补实现，不是删声明**
        （四态机是芯片无关的共享资产，本后端复用它与另三家同一份实现、同一套语义）。

        暴露路径：`backend_offline_check.py --backend flagos` 第 [6] 组
        （此前该工具只为 cambricon 内置 stub，flagos 从未被自检过 ⇒ 该缺陷一直未被拦到）。

        边界（如实标注）：本后端只声明 `recovery_probe`；四态**转换**由上层/监控方向驱动，
        本方法只负责**查询**，恢复执行走 `recover_device(mode="probe")`。
        """
        return self._load_device_state().query_device_state(ordinal)

    # 能力声明：与 ascend 后端同一套键名（此前键名不一致，已统一）
    #: 能力**全集**（已知能力名，`info()["supports"]` 按此逐项 True/False 呈现；
    #: 与 kunlun / cambricon 同一份清单 —— 清单本身是"已知能力"，不代表本后端支持）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
    )

    #: 本后端**声明支持**的能力（不支持/未验证的一律不写进来 —— 如实声明，不伪造）
    _capabilities = {
        "device", "memory", "stream", "event",
        "error_map",            # 复用 conformance/errors.py 错误码映射
        "recovery_probe",       # probe 级恢复
        "device_state",         # 2026-09-22 补实现后声明成立
        "multidevice",
        # 不支持：bounded_sync（原生同步为阻塞式）/ recovery_real / stream_priority
    }

    def supports(self, capability: str) -> bool:
        """能力查询（键名与 ascend 后端一致，便于上层统一判断）。"""
        return capability in getattr(self, "_capabilities", set())

    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            "framework": "torch_fl(flagos)",
            "torch": self._torch.__version__,
            "device_count": self.device_count(),
            # 2026-09-22 修（同一次自检暴露的第二处）：原先这里手写了一份键名清单
            # （"stream_priority" / "device_rebuild_real" / "device_rebuild_probe" /
            #   "error_code_map" / "bounded_sync"），与 `_capabilities` 里的键名
            # （"stream_priority" / "recovery_real" / "recovery_probe" / "error_map" / …）
            # **对不上** ⇒ `info()["supports"]` 对已声明能力恒报 False，读者会得出
            # 完全相反的结论。
            # 改为与 kunlun / cambricon **同款**：按 `_CAPABILITY_KEYS` 全集逐项呈现，
            # 清单不再是手写的第二份真相 ⇒ 从结构上杜绝再次漂移。
            # 显存统计的口径（统一字段取自设备级；降级时如实标注）
            "memory_stats_scope": (
                "device —— pyACL `acl.rt.get_mem_info`，与 ascend / kunlun 同口径；"
                "pyACL 不可用时降级为 process（返回值内 `memory_scope` 与"
                "`memory_degraded_reason` 同步标注）"
            ),
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
        }


def build() -> FlagosBackend:
    return FlagosBackend()
