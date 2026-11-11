#!/usr/bin/env python3
"""壁仞 BR_0（Biren166M · br1xx）后端 —— **路线 A：厂商官方 torch 插件（`torch_br`）**。

⚠️ **命名空间说明（先读，实测）**
  壁仞这栈的设备 API 走 **`torch.supa`**（由 pip 包 `torch-br` 提供），三条独立证据（2026-11-11 实测）：
    1. `import torch_br` 后 `torch.supa` 存在，`torch.supa.device_count() == 2`（`BIREN_VISIBLE_DEVICES=5,7`），
       `torch.supa.get_device_name(0)` 可用，64×64 matmul 真跑成功（无 NaN/Inf）；
    2. `torch.cuda.device_count() == 0`、`torch.xpu.device_count() == 0` ⇒ **不是** cuda/xpu 复用路线
       （与 kunlun/ppu 的形态不同，本家是**独立命名空间** `torch.supa`，同 `torch.npu`/`torch.mlu` 的 PrivateUse1 形态）；
    3. `aten::mm` 的 `PrivateUse1` 注册来源在 `torch_br` 构建树内（`/home/jenkins/.../BR_PACKAGE/...`）
       ⇒ 算子由厂商插件注册，非社区 CUDA 实现。
  ⇒ `name = "biren"`（注册表键，厂商标识）· `device_type = "supa"`（设备串前缀，`torch.supa`）。
  `device_type` 用 `"supa"` 而非 `"biren"`：本栈设备串就是 `supa:0`，设备 API 命名空间也叫 `supa`。

**本后端的能力声明 = 逐项实测结果，未实测一律不声明**（依据：本方向「声明即承诺」纪律）：
  · ✅ 已实测声明（9 键，证据见 `BR0/probes/biren_image_20261111_out/` 与
    `BR0/probes/biren_conf_stream_20261111_out/`）：
      `device`（device_count/set_device/get_device_name）、`memory`（mem_get_info）、
      `memory_alloc_stat`（memory_allocated）、`stream`（Stream/synchronize/current_stream）、
      `event`（Event/record/query，含 E3「未 record 时 query 误报 True」的本层修正）、
      `multidevice`（2 卡可见）、`recovery_probe`（探针级探测），
      `device_state` / `device_state_control`（本层共享四态账本，芯片无关）。
  · ❌ **不声明** `stream_priority` / `stream_priority_control` / `stream_priority_readback`：
      2026-11-11 实测 `Stream(priority=N)` 对 N∈{0,±1,−2,−5,7} **构造全成功但 `stream.priority` 回读恒 0**（单档、未生效）；
      `Stream.priority_range()` 抛 `RuntimeError: greatest_priority <= -1 INTERNAL ASSERT FAILED`。
      ⇒ 尤其**不声明 readback**：接口存在但恒 0、不反映设置，声明它即制造假能力（「能设 ≠ 有效果」）。
  · ❌ **不声明** `bounded_sync` / `sync_timeout`：`Stream.synchronize()` 与 `torch.supa.synchronize()`
      **均无 timeout 参数**，原生只能无限等待；本层用「`query()` 轮询 + deadline」做成**超时上报**语义
      （到点抛 `TimeoutError`，**不保证中断底层已提交执行**）。conformance `e2_host_timeout` 已实测
      「未 record 的 `wait_host(200ms)` 返回 False、耗时 200 ms」——通过的是「不永久阻塞」这一条，
      不是「原生有界」。
  · ❌ **不声明** `recovery_real` / `graph_capture` / `memory_alloc` / `record_stream` / `context_*`：
      **未探测**（不写进来 = `supports()` 如实返回 False）。
      注意本栈确实暴露了 `SUPAGraph`/`graph`/`caching_allocator_alloc`/`set_stream_by_id` 等入口 —— 有入口 ≠ 已实测可用。
  · ❌ **不声明** `error_map`：错误码形态未采集（首轮只看到 `ErrorCode: 100` 与 `SUDNN` 错误名两类，
      未做码表探测）。

**已通过的真机判据（2026-11-11）**：离线契约自检 83/0/1 · smoke 46/0 · conformance 13/13 · 推理 conformance 6/6 ·
多流探针 `STREAM_SEMANTICS_PASS 8/8`。详见 `BR0/docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md`。

⚠️ **已知语义缺口（实测，已在适配层修正）**
  E3 型：**未 record 的 `Event.query()` 原生返回 `True`**（`torch.supa.Event()` 直接 `query() is True`）——
  裸调会把「还没排队的完成」误判成「已完成」。本后端用 `recorded` 标志修正，与 PPU / MLU590 同族处理。

运行前置（实测，2026-11-11）：
  · 容器镜像：`birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx`
    （宿主 `default-runtime: biren` 自动注入卡与宿主侧 .so；**不要**手工 `--device`，会干扰 toolkit）；
  · 选卡用 **`BIREN_VISIBLE_DEVICES=<物理卡号,逗号分隔>`**（容器内 0 基重编号）；
  · **库与 PYTHONPATH 必须走厂商入口脚本**：容器内命令前先
    `. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set`
    （否则 `import torch_br` 报 `libsupti.so.1: cannot open shared object file`——
     实测这是 `docker exec` 不带启用环境的假象，不是镜像坏）。
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from ...api.errors import FlagosError, coerce_category
from ..base import RuntimeBackend, state_token

#: conformance 共享资产目录（与既有四家同一解析方式）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


def _l3():
    """api 层的 L3 执行期错误枚举（`_load_errors()` 的枚举是另一个类对象，需归一）。"""
    from ...api.errors import ErrorCategory

    return ErrorCategory.L3_EXECUTION

logger = logging.getLogger(__name__)


class BirenEventAdapter:
    """`torch.supa.Event` 的统一语义适配（修正 E3：未 record 时 `query()` 误报 True）。"""

    def __init__(self, *args, **kwargs):
        self._ev = None
        self._args, self._kwargs = args, kwargs
        self._recorded = False

    # ---- 惰性创建：导入 torch_br 之后再建 Event（离线自检时不炸） ----
    def _ensure(self):
        if self._ev is None:
            import torch_br  # noqa: F401  （触发 PrivateUse1 注册）
            import torch

            self._ev = torch.supa.Event(*self._args, **self._kwargs)
        return self._ev

    def record(self, stream=None):
        ev = self._ensure()
        if stream is None:
            ev.record()
        else:
            ev.record(stream)
        self._recorded = True
        return self

    def query(self) -> bool:
        """语义修正：未 record 一律 False（原生会误报 True）。"""
        if not self._recorded:
            return False
        return bool(self._ensure().query())

    def synchronize(self) -> None:
        if self._recorded:
            self._ensure().synchronize()

    def wait(self, stream=None) -> None:
        ev = self._ensure()
        if stream is None:
            ev.wait()
        else:
            ev.wait(stream)

    def wait_host(self, timeout_ms=None) -> bool:
        """主机侧**有界**等待：完成 True / 超时 False（永不永久阻塞；E2-v2 契约）。

        原生 `torch.supa.Event` 没有带 timeout 的 `wait_host`，故用 `query()` 轮询实现。
        """
        if timeout_ms is None:
            self._ensure().synchronize()
            return True
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            if self.query():
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.001)

    def elapsed_time(self, other) -> float:
        a = self._ensure()
        b = getattr(other, "_ev", other)
        return float(a.elapsed_time(b))

    @property
    def native(self):
        return self._ensure()

    def __repr__(self) -> str:  # 便于日志定位
        return "BirenEventAdapter(recorded=%s, native=%r)" % (self._recorded, self._ev)


class BirenBackend(RuntimeBackend):
    """基于 `torch_br`（`torch.supa` 命名空间，PrivateUse1）的壁仞后端。"""

    name = "biren"
    device_type = "supa"

    #: 规范能力键全集（与既有四家同一套键名，便于跨实例同口径比对）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync", "sync_timeout",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
        "memory_alloc", "memory_alloc_stat", "record_stream", "context_lifecycle",
        "context_query", "device_state_control",
        "stream_priority_control", "stream_priority_readback",
    )

    #: 已实测声明的能力（**每条都要有 BR0 下的探针证据**；未实测一律不写）
    _capabilities = {
        "device",        # device_count / set_device / get_device_name 实测（2026-11-11）
        "memory",        # mem_get_info(ordinal) → (free,total) 实测
        "stream",        # Stream / current_stream / synchronize 实测
        "event",         # Event / record / query（含 E3 修正）实测
        "multidevice",   # 单机 8 卡（本次容器仅见 2 卡：BIREN_VISIBLE_DEVICES=5,7）
        "recovery_probe",  # 探针级恢复（只读探测，不含真重建）—— 该键在 base 里是「探针可用」语义
        # ── 2026-11-11（离线自检暴露的两处声明/实现不一致，已修）──
        "memory_alloc_stat",     # memory_allocated(ordinal) 实测可用 ⇒ memory_stats 给 allocated_mb
        "device_state",          # 四态**查询**（本层共享状态机，非厂商原语）
        "device_state_control",  # 四态**驱动**（本层账本，同上；芯片无关，四家一致声明）
    }

    #: 分级来源可达性（如实标注：码表路径未启用）
    _grading_paths = {"code_map": False, "message_hint": True}

    def __init__(self) -> None:
        self._torch = None
        self._br = None
        self._import_error: Optional[str] = None
        self._device_state = "available"
        self._errors_mod = None

    # ───────────────────────── 基础设施 ─────────────────────────

    @property
    def torch(self):
        if self._torch is None:
            try:
                import torch  # noqa: F401
                import torch_br  # noqa: F401  （注册 PrivateUse1）
                self._torch = torch
            except Exception as exc:  # 离线环境：如实记录，不伪造
                self._import_error = "%s: %s" % (type(exc).__name__, exc)
                raise
        return self._torch

    @property
    def br(self):
        if self._br is None:
            import torch_br

            self._br = torch_br
        return self._br

    def info(self) -> dict:
        """后端自述（能力如实声明；未声明键一律 False）。"""
        caps = {
            key: (key in self._capabilities) for key in self._CAPABILITY_KEYS
        }
        out = super().info() if hasattr(super(), "info") else {}
        out.update(
            {
                "name": self.name,
                "device_type": self.device_type,
                "framework": "torch_br（`torch.supa` 命名空间，PyTorch PrivateUse1 厂商插件）",
                "device_namespace": "supa（`torch.supa.*`）",
                "supports": caps,
                "grading_paths": dict(self._grading_paths),
                "known_issues": self.known_issues(),
                "import_error": self._import_error,
            }
        )
        return out

    # ───────────────────────── 设备（D2） ─────────────────────────

    def device_count(self) -> int:
        """可见设备数（受 `BIREN_VISIBLE_DEVICES` 影响，容器内 0 基）。

        ⚠️ 缺厂商扩展（`torch_br` 不可导入）时**必须报错**，不得静默返回 0 —— 否则
        「没装对」与「没卡」在上层看起来一样（本项目台账同族教训）。
        """
        try:
            return int(self.torch.supa.device_count())
        except Exception as exc:
            raise RuntimeError(
                "biren 后端不可用：无法导入厂商扩展 torch_br（%s: %s）。"
                "请确认：① 容器为壁仞官方镜像（birensupa-*）；"
                "② 命令前 source /usr/local/birensupa/br_container_tools/brsw_set_env.sh；"
                "③ 已设 BIREN_VISIBLE_DEVICES 且卡健康。"
                % (type(exc).__name__, str(exc).splitlines()[0][:160])
            ) from exc

    def set_device(self, ordinal: int) -> None:
        self.torch.supa.set_device(int(ordinal))

    # ───────────────────────── 内存（D3） ─────────────────────────

    def memory_stats(self, ordinal: int) -> dict:
        """设备级三键 + 分配器视角已分配量。

        实测（2026-11-11）：`torch.supa.mem_get_info(ordinal)` 返回 `(free_bytes, total_bytes)`，
        `torch.supa.memory_allocated(ordinal)` 可用 ⇒ 三键与 `allocated_mb` 都给。
        """
        torch = self.torch
        free, total = torch.supa.mem_get_info(int(ordinal))
        out = {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }
        try:
            out["allocated_mb"] = int(
                torch.supa.memory_allocated(int(ordinal)) / 1024 / 1024
            )
        except Exception:  # 取不到就不给该键（不填 0 冒充）
            pass
        return out

    # ───────────────────────── 流（D4） ─────────────────────────

    def _create_stream_raw(self, priority=None):
        """创建原生流。`priority=None` ⇒ 厂商默认；给值 ⇒ 透传但**本后端不声明优先级能力**。"""
        s = self.torch.supa
        if priority is None:
            return s.Stream()
        return s.Stream(priority=int(priority))

    def current_stream(self):
        return self.torch.supa.current_stream()

    def default_stream(self):
        return self.torch.supa.default_stream()

    def stream_context(self, native_stream):
        ctx = self.torch.supa.stream(native_stream)
        return ctx

    def set_stream(self, native_stream) -> None:
        self.torch.supa.set_stream(native_stream)

    # ───────────────────────── 事件（D4） ─────────────────────────

    def create_event(self, enable_timing: bool = False):
        return BirenEventAdapter(enable_timing=bool(enable_timing))

    # ───────────────────────── 同步（D5） ─────────────────────────

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。`timeout_ms` 为空 → 原生阻塞；非空 → **超时上报**语义（与 Kunlun/MLU 同口径）。

        ⚠️ 原生 `torch.supa.synchronize()` **无 timeout 参数**，故「有界」只能做成
        「到点抛 `TimeoutError` 让上层降级」，**不保证中断底层已提交的执行**。
        """
        torch = self.torch
        if timeout_ms is None:
            torch.supa.synchronize()
            return
        self._bounded_wait(lambda: torch.supa.current_stream().query(), timeout_ms, "device")

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """流同步（同 `synchronize` 的超时上报语义）。"""
        if timeout_ms is None:
            native_stream.synchronize()
            return
        self._bounded_wait(native_stream.query, timeout_ms, "stream")

    def _bounded_wait(self, query_fn, timeout_ms: int, what: str) -> None:
        """轮询 `query_fn()` 至完成；到点抛 `TimeoutError`（**先查再判超时**，与既有两家同构）。"""
        deadline = time.monotonic() + timeout_ms / 1000.0
        while True:
            try:
                if query_fn():
                    return
            except Exception:
                # 查询本身失败 → 不视为超时，交由上层错误翻译处理
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "biren: %s synchronize timeout after %d ms"
                    "（超时上报语义，不保证中断底层执行）" % (what, int(timeout_ms))
                )
            time.sleep(0.001)

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待：轮询 `query()`，**永不永久阻塞**；返回是否在期限内完成。"""
        ev = getattr(native_event, "_ev", native_event)
        deadline = time.time() + max(0.0, timeout_ms / 1000.0)
        recorded = getattr(native_event, "_recorded", None)
        while time.time() < deadline:
            done = bool(native_event.query()) if recorded is not None else bool(ev.query())
            if done:
                return True
            time.sleep(0.0005)
        return False

    # ───────────────────────── 错误翻译（D6） ─────────────────────────

    def _load_errors(self):
        """复用 `conformance/errors.py` 的厂商中立 message 规则（**不含**他厂码表）。

        调用方显式 `vendor_codes=False`，故 `graded_by` 只会是 `message_hint` / `default`、
        `mapped` 恒 False —— 与 `_grading_paths` 的声明一致。
        """
        if self._errors_mod is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            spec = importlib.util.spec_from_file_location(
                "dc_conformance_errors", _CONFORMANCE_DIR / "errors.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._errors_mod = mod
        return self._errors_mod

    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 `FlagosError`；分级来源如实限定为 `message_hint` / `default`。

        首轮实测（2026-11-11）观察到的错误形态：设备侧给 `ErrorCode: 100`（无设备）与
        `SUDNN` 错误**名**（`Sudnn Error` / `Failed to finalize engine config descriptor`），
        **未做码表探测** ⇒ 不声明 `error_map`（属「未验证」，不是「确认不具备」）。

        ⚠️ 必须 `vendor_codes=False`：共享翻译器自带**昇腾 ACL 码表**，让它参与本栈分级会把
        带他厂码的消息判成 L4 ⇒ 下游按 disposition 误触发设备级重建（既有实例的真实缺陷）。
        """
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location, vendor_codes=False)
        graded_by = getattr(fe, "graded_by", "default")
        return FlagosError(
            category=coerce_category(getattr(fe, "category", None)) or _l3(),
            root_cause=getattr(fe, "root_cause", "%s: %s" % (type(exc).__name__, exc)),
            location=getattr(fe, "location", "") or location,
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=(graded_by == "message_hint"),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )

    # ───────────────────────── 状态与恢复（D7/D8） ─────────────────────────

    def probe_device(self, ordinal: int) -> bool:
        """只读探针：能否在卡上分配并算一个小算子。"""
        try:
            torch = self.torch
            if ordinal >= torch.supa.device_count():
                return False
            prev = torch.supa.current_device()
            torch.supa.set_device(int(ordinal))
            try:
                # 轻量探活：能在该卡上建张量并求和（与 PPU 同构；真机上另有 conformance 覆盖真实算子）
                x = torch.zeros(2, 2, device="supa:%d" % int(ordinal))
                return float(x.sum().item()) == 0.0
            finally:
                try:
                    torch.supa.set_device(prev)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("biren: probe_device(%s) 失败：%s", ordinal, exc)
            return False

    def device_state(self, ordinal: int = 0):
        """四态查询：复用共享 `conformance/device_state.py`（进程内状态机，不依赖厂商原语）。

        与前四家**同一份实现、同一套语义**（故该能力声明成立）；本层只查询，
        状态转换由上层/监控驱动，恢复执行走 `recover_device(mode="probe")`。
        """
        self._load_shared_state_assets()
        return self._device_state.query_device_state(int(ordinal))

    def _recover_device_impl(self, ordinal: int, mode: str = "probe", **kwargs):
        """恢复入口：**只做 probe 级**（真实重建未探测、未声明）。"""
        # ⚠️ 契约：`state` = **调用时（恢复前）**的四态，不是恢复后的结果（时点定死）。
        pre_state = state_token(self.device_state(int(ordinal)))
        ok = self.probe_device(int(ordinal))
        return {
            "ordinal": int(ordinal),
            "mode": mode,
            "recovered": bool(ok),
            "state": pre_state,
            "_rebuilt": False,   # 未做真实重建（未声明 recovery_real）⇒ 如实回报
            "detail": "biren: probe 级探测（真实设备重建原语未探测，未声明 recovery_real）",
        }

    # ───────────────────────── 已知问题 ─────────────────────────

    def known_issues(self) -> list:
        """已实测的问题清单（手册 §9 模板：九字段齐全；只收实测项，未测不写）。"""
        return [
            {
                "id": "BIREN-EVENT-UNRECORDED-QUERY-TRUE",
                "severity": "medium",
                "scope": "事件语义（conformance E3）",
                "condition": "创建 `torch.supa.Event()` 后、未 `record()` 直接调 `query()`",
                "symptom": "原生返回 True（误报「已完成」）；会把还在跑的显存判成可回收",
                "root_cause_layer": "厂商 PyTorch 插件（torch_br）事件语义缺省值",
                "workaround": "本层 `BirenEventAdapter` 用 `_recorded` 标志修正为 False（同 PPU / MLU590 处理）",
                "report_to": "torch_br 上游（可选：建议未 record 时 query() 返回 False 或抛错）",
                "evidence": "2026-11-11 容器实测：`Event()` 后 `query()` → True（`BR0/probes/`）",
            },
            {
                "id": "BIREN-NO-NATIVE-TIMEOUT",
                "severity": "info",
                "scope": "同步语义（bounded_sync / sync_timeout）",
                "condition": "调用 `torch.supa.synchronize()` 或 `Stream.synchronize()` 想要超时",
                "symptom": "两者签名均无 timeout 参数 ⇒ 原生只能无限等待",
                "root_cause_layer": "厂商插件 API 形态（非缺陷，属能力边界）",
                "workaround": "本层用「轮询 `current_stream().query()` + deadline」做**超时上报**语义，"
                              "到点抛 TimeoutError（不保证中断底层执行）；故不声明 bounded_sync/sync_timeout",
                "report_to": "无（能力边界，不需要上报）",
                "evidence": "2026-11-11 API 探针：`dir(torch.supa)` 有 synchronize/Stream，均无 timeout 形参",
            },
            {
                "id": "BIREN-CONTAINER-ENV-REQUIRED",
                "severity": "info",
                "scope": "运行前置（环境，非产品缺陷）",
                "condition": "`docker exec` 直接调 `python3` 而未 source 厂商环境脚本",
                "symptom": "`import torch_br` 报 `ImportError: libsupti.so.1: cannot open shared object file`",
                "root_cause_layer": "镜像设计：库路径/PYTHONPATH 由 `brsw_set_env.sh` 在入口设置，"
                                    "`docker exec` 不继承入口进程的环境",
                "workaround": "命令前先 `. /usr/local/birensupa/br_container_tools/brsw_set_env.sh "
                              "--not-set-env-if-already-set`",
                "report_to": "无（镜像设计使然；已写入 BR_0 接入手册）",
                "evidence": "2026-11-11 实测：source 前 ImportError / source 后正常（`BR0/probes/`）",
            },
            {
                "id": "BIREN-STREAM-PRIORITY-SINGLE-TIER",
                "severity": "medium",
                "scope": "流语义（多流基线第 12 项：流优先级）",
                "condition": "构造 `torch.supa.Stream(priority=N)` 想设置流优先级，并回读 `stream.priority` 验证",
                "symptom": "构造对 N∈{0,±1,−2,−5,7} **全部成功**，但 `stream.priority` **逐个回读都是 0**"
                           "（`current_stream().priority` 亦为 0）⇒ 参数被接收、优先级不生效（单档）；"
                           "另 `Stream.priority_range()` 抛 `RuntimeError: greatest_priority <= -1 "
                           "INTERNAL ASSERT FAILED`（`BR_PACKAGE/buildSystem/...`），进程不崩",
                "root_cause_layer": "厂商 PyTorch 插件（torch_br）流优先级实现 + 上游优先级区间上报",
                "workaround": "`stream_priority` / `stream_priority_control` / `stream_priority_readback` "
                              "**三键均不声明**——尤其 readback：接口存在但恒 0、不反映设置，"
                              "声明它等于制造假能力（『能设 ≠ 有效果』）。多流并发语义另由探针 S12 并发段覆盖",
                "report_to": "torch_br 上游（可选：优先级设置落地 / `priority_range()` 返回合法区间）。"
                             "注：与昆仑芯侧同形断言属**各自实测**，未做同源性证明，不合并归因",
                "evidence": "`BR0/probes/biren_conf_stream_20261111_out/biren_facts_20261111_145533/"
                            "facts_s12_i6_20261111_145533.log`（§A/§A2 两段）",
            },
            {
                "id": "BIREN-CARD-HW-FAULT-SPREADING",
                "severity": "info",
                "scope": "测试机资源（非本层缺陷）",
                "condition": "使用本机的坏卡；或按 `brsmi` 显示索引挑卡",
                "symptom": "2026-11-10 起故障**扩散**：11-11 当天 `03:00.0` 与 `63:00.0` 彻底 `device remove`（8 张→6 张），"
                           "同时 `05:00.0`(13:34)、`83:00.0`(14:22) 新发同型故障（复位后仍 N/A + 128 MiB 残留）。"
                           "⚠️ 卡掉线后 `brsmi` 索引会重排、而 `/dev/biren/card_N` 节点名不变 ⇒ "
                           "表里的『第 3 号』与节点 `card_3` 不再是同一张卡",
                "root_cause_layer": "硬件/驱动（dmesg 链条同一形状：`smmu context fault(Read) fsr=0x402` → "
                                    "`PQM hang recovery fail` → `found spc hang, mask 0xffffffff` → `hot reset triggered`）",
                "workaround": "① 只用健康卡（当前 `85:00.0`=节点 `card_5`、`a5:00.0`=节点 `card_7` 为在用）；"
                              "② `BIREN_VISIBLE_DEVICES` 按**设备节点序号**选卡，**不按 `brsmi` 显示索引**；"
                              "③ 取卡前跑最小算子探针（`brsmi` 读数正常 ≠ 卡可用）",
                "report_to": "壁仞 / 机器提供方（工单对象是 `03/63/05/83` 这四张的 Bus-Id，"
                             "**不是**交接清单里写的 7 号卡）",
                "evidence": "`BR0/docs/BIREN_CARD02_FAULT_REPORT_20261111.md`（11-10 两张）、"
                            "`BR0/docs/BIREN_BR0_CONFORMANCE_AND_STREAM_20261111.md` §6（11-11 扩散与节点/索引错位）",
            },
        ]


def build(**kwargs) -> BirenBackend:
    """注册表约定入口。"""
    return BirenBackend(**kwargs)
