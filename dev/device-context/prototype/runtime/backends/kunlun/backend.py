#!/usr/bin/env python3
"""昆仑芯 P800 后端（XPytorch / torch.cuda 适配）。

⚠️ **命名空间说明（务必先读）**
  昆仑芯 P800 这栈上设备 API **走 `torch.cuda`，不是 `torch.xpu`**。四条独立证据：
    1. 编译标志：`torch.__config__.show()` → `USE_CUDA=ON, ..., USE_XPU=OFF`
    2. 运行时：`torch.xpu.is_available() == False`（`Torch not compiled with XPU enabled`）
    3. 官方佐证：FlagTree xpu3.6 单测 `third_party/xpu/python/test/unit/conftest.py`
       的 `--device` **默认值就是 `'cuda'`**
    4. 厂商运行口径：选卡用 `CUDA_VISIBLE_DEVICES`（实测 `=2` → 1 卡）
  机制：XPytorch（`/env/xpytorch-*.run`）+ `torch_xray` 符号重写，把 CUDA 命名空间调用
  重定向到 P800；启动时可见 `XCCL .../libbkcl.so loaded` 与 `SYMBOL_REWRITE torch success`。

  因此本后端：
    - `name = "kunlun"`        ← 厂商标识（我们注册表里的键）
    - `device_type = "cuda"`   ← 真实可用的设备串前缀（`torch.device("cuda:0")` 有效）
  **两者不同名是刻意为之**：后端名按厂商，设备串按命名空间。
  `kunlun` 与未来的 `nvidia` 共用 `torch.cuda`，故厂商判别不能用命名空间，
  需用厂商特征（`torch_xray`/`torch_xmlir` 模块、`/proc/kunlun/`、`xpu-smi`、通信库）。

**运行前置**（缺一不可）
  - `conda activate python310_torch29_cuda`（镜像默认 `python3` 是 conda base 3.13，**不是**目标环境）
  - `export XPU_EVENT_KL3_ENABLE=1`（官方手册要求）

**已知上游缺陷（已对外提交，见接入方案 §7.4）**
  - `torch.cuda.Stream.priority_range()` 触发 PyTorch `INTERNAL ASSERT FAILED
    at c10/cuda/CUDAStream.h:188`（稳定复现）→ 本后端 `stream_priority_range()` 直接返回 None，
    **不得透传该调用**
  - 厂商错误码不透出到 Python 异常（仅退出钩子偶见 `error code= 101`）→
    `translate_error` 的 `code_map` 路径**不可达**，只能走 `message_hint`

接入定义（接口约定 §2）：新增一家芯片 = 实现本接口 + 跑通 conformance 13 例 + 6 例。
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import FlagosError, coerce_category
from ..base import RuntimeBackend, state_token

logger = logging.getLogger(__name__)

# conformance 目录（既有资产所在）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


class KunlunEventAdapter:
    """`torch.cuda.Event` 的统一语义适配（与 NpuEventAdapter / FlagosEventAdapter 同构）。

    补齐两点与统一事件契约的语义缺口：
      - **E3：未 record 的 Event 调 `query()` 原生返回 `True`（误报"已完成"）** ——
        实测昆仑芯如此；用 `recorded` 标志修正为 `False`。
      - **E2v2：主机侧有界等待 `wait_host(timeout_ms)`** —— 原生 Event 无此方法，
        用 `query()` 轮询实现，**永不永久阻塞**。
    """

    def __init__(self, *args, **kwargs):
        self._ev = None
        self._args, self._kwargs = args, kwargs
        self._recorded = False

    def _ensure(self):
        if self._ev is None:
            import torch
            self._ev = torch.cuda.Event(*self._args, **self._kwargs)
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
        # E3 修正：未 record 不得报"已完成"
        if not self._recorded:
            return False
        return self._ensure().query()

    def wait_host(self, timeout_ms: Optional[int] = None) -> bool:
        """主机侧**有界**等待：完成返回 True，超时返回 False（不永久阻塞）。"""
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000.0
        while not self.query():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.001)
        return True

    def elapsed_time(self, end_event):
        return self._ensure().elapsed_time(getattr(end_event, "_ev", end_event))

    def __getattr__(self, item):
        return getattr(self._ensure(), item)


class KunlunBackend(RuntimeBackend):
    """基于 XPytorch（`torch.cuda`）的昆仑芯 P800 后端。"""

    name = "kunlun"
    device_type = "cuda"   # 见模块 docstring：设备串按命名空间，不按厂商

    #: 规范能力键全集（用于 info() 的自洽报告；键名与 ascend / cambricon 对齐）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync", "sync_timeout",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
        # 2026-09-29（工作包 B/C）新增键（未声明即为 False，如实呈现）
        "memory_alloc", "memory_alloc_stat", "record_stream", "context_lifecycle",
        # 2026-09-29（工作包 C·P800 专项）：**只读观测**能力键，与 context_lifecycle
        # 分开 —— 有的栈能管上下文生命周期，有的栈只允许观测（平台单上下文）。
        "context_query",
        # 2026-09-29（A2 收尾）：四态机的 **驱动** 入口（原只有查询 `device_state`）。
        # 芯片无关 —— 走本层共享状态机（进程内账本，不依赖厂商原语），故三家一致声明。
        "device_state_control",
        # 2026-09-30（B2 v2 落地，(A) 方案）：优先级能力**拆两把钥匙**
        #   —— 与 `device_state` / `device_state_control`、`context_query` / `context_lifecycle` 同构。
        "stream_priority_control", "stream_priority_readback",
    )

    #: 本后端**声明支持**的能力（不支持的一律不写进来，不伪造）
    _capabilities = {
        "device",
        "memory",
        "stream",
        "event",
        "bounded_sync",      # 主机侧等待真有界；流同步为"超时上报"语义，见 synchronize_stream
        "recovery_probe",    # 探针级恢复
        "device_state",      # 四态**查询**（复用 conformance 的进程内状态机，见 device_state()）
        # ── 2026-09-29（A2 收尾）新增 ──
        "device_state_control",  # 四态**驱动**（`runtime.set_device_state()`）；本层账本，非厂商能力
        "graph_capture",     # torch.cuda.graph（2026-09-20 实测 GRAPH_CAPTURE_PASS 5/5）
        "multidevice",       # 单机 8 卡
        # ── 2026-09-29（工作包 B）新增 ──
        "memory_alloc",      # torch.cuda.caching_allocator_alloc + caching_allocator_delete
                             #   （实测：0 → 4 MiB → 0，且非法/重复指针**如实报错**）
        "memory_alloc_stat",  # memory_stats()["allocated_mb"]（memory_allocated）
        "record_stream",     # Tensor.record_stream 可用
        # ── 2026-09-29（工作包 C·本家专项）：上下文**只读观测** ──
        # 实测：libcuda.so.1（= libxpucuda.so.515.58.kunlun，XPytorch 实际用的驱动）
        # 提供完整 cuCtx* 系列，且 XPytorch 自己在用（torch 初始化后 cuCtxGetCurrent 非 0）。
        # 但平台**只允许一个**上下文（第二次 create → rc=2），且它由框架自建 ——
        # 本层若抢先建会让 torch 起不来（invalid device ordinal）
        # ⇒ 只声明**只读**观测，**绝不**声明/调用创建销毁。
        "context_query",
        # ── 2026-09-30（B2 v2 落地，(A) 方案）：流优先级 ──
        # **能读 ≠ 能改**（同 `device_state` / `device_state_control` 的拆法）。
        "stream_priority",           # 范围可读：真原语 `cuCtxGetStreamPriorityRange`
                                     #   实测返回 (0, 0) = **退化单点**
        "stream_priority_readback",  # 单条流回读：`cuStreamGetPriority`（实测可用）
        # ── 以下**不支持**，故不声明 ──
        # "error_map"       : 无厂商错误码（Python 层不可得）→ 只有 message_hint 分级
        # "recovery_real"   : 无设备级重置/重建原语（实测 torch.cuda 只有内存统计类 reset*）
        # "stream_priority_control" : 区间是**退化单点**（实测 (0, 0)）⇒ 设置**不产生任何区分**，
        #   与其宣称"可设置"，不如如实不声明（同 `context_lifecycle` 在本平台的处置）。
        #   ⚠️ 旧注释曾把原因写成"priority_range() 触发 PyTorch INTERNAL ASSERT" ——
        #   那是 **torch API 路径**的问题；底层真原语**安全可读**（纪律 ⑰：根因表述本身可能错）。
    }

    #: 分级来源可达性（如实标注：code_map 路径在昆仑芯不可达）
    _grading_paths = {"code_map": False, "message_hint": True}

    # ── 上下文**只读**观测（工作包 C·P800 专项，2026-09-29）────────────────────
    #
    # 为什么只读、为什么不实现 context_lifecycle：四组真机判别实验（报告 §10）
    #   ① 平台**只允许一个**上下文 —— 已有上下文时 `cuCtxCreate_v2` 返回 `rc=2`；
    #   ② 该上下文由 XPytorch/XRE **自建** —— 本层抢先在 torch 之前建 ⇒ torch 报
    #      `CUDA error: invalid device ordinal`；**销毁本层建的上下文后 torch 立即恢复**
    #   ③ 对 torch 自己的上下文，厂商**拒绝**销毁（`cuCtxDestroy_v2` → `201`
    #      INVALID_CONTEXT）⇒ 危险操作在厂商侧已兜住；
    #   ④ 只读接口（GetCurrent/GetDevice/GetFlags/Synchronize）**全部可用且安全**
    #      （调用后 torch 计算仍为 512.0）。
    # ⇒ 本类**绝不调用 `cuCtxCreate_v2` / `cuCtxDestroy_v2`**，只做只读观测。

    #: 惰性加载的驱动库句柄（类级缓存；`False` = 已确认不可用，避免反复尝试）
    _ctx_driver = None

    def _driver_handle(self):
        """惰性取得**驱动库**句柄。

        用 soname `libcuda.so.1`，**不用**绝对路径、更不用别处的副本：
        XPytorch 已加载的就是这一个（`/proc/self/maps` 实测为
        `xcudart/lib/libxpucuda.so.515.58.kunlun`，而 `xcudart/lib/libcuda.so.1`
        正是指向它的符号链接）⇒ 同名复用**同一份**，不引入第二份副本。

        ⚠️ 反面教材（本轮踩到）：先前误从 `triton/backends/xpu/xpu3/so/` 取同名库
        ⇒ 触发 "Libraries loaded from different directories!" 版本错配，
        torch 直接报 `CUDA_ERROR_NOT_INITIALIZED`。
        **同栈的库必须走 soname，不要拿任意副本路径。**
        """
        if type(self)._ctx_driver is None:
            import ctypes
            try:
                type(self)._ctx_driver = ctypes.CDLL("libcuda.so.1")
            except OSError:
                type(self)._ctx_driver = False
        return type(self)._ctx_driver or None

    def _ctx_query_raw(self, ordinal: int = 0):
        """只读读回当前生效的上下文（句柄/设备号/标志）；无则 `None`。"""
        import ctypes
        lib = self._driver_handle()
        if lib is None:
            return None
        cur = ctypes.c_void_p(0)
        try:
            rc = lib.cuCtxGetCurrent(ctypes.byref(cur))
        except AttributeError:
            return None
        if int(rc) != 0 or not cur.value:
            return None
        dev = ctypes.c_int(-1)
        flg = ctypes.c_uint(0)
        rc_d = lib.cuCtxGetDevice(ctypes.byref(dev))
        rc_f = lib.cuCtxGetFlags(ctypes.byref(flg))
        return {"ctx": int(cur.value),
                "ordinal": int(dev.value) if int(rc_d) == 0 else None,
                "flags": int(flg.value) if int(rc_f) == 0 else None}

    def __init__(self) -> None:
        self._torch = None
        self._errors_mod = None
        self._device_state = None
        self._loaded = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        self._torch = torch
        if not os.environ.get("XPU_EVENT_KL3_ENABLE"):
            logger.warning(
                "环境变量 XPU_EVENT_KL3_ENABLE 未设置；官方手册要求测试前 "
                "export XPU_EVENT_KL3_ENABLE=1，否则部分路径行为未定义"
            )
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    def _load_errors(self):
        """复用 conformance/errors.py 的统一翻译骨架（含厂商中立的 message 规则）。

        注意：其中的 `ACL_ERR_TO_CATEGORY`（108 条昇腾错误码）在昆仑芯**永不命中**，
        实际只用 `_MESSAGE_HINTS` 消息规则 —— 故 `graded_by` 只会是
        `message_hint` 或 `default`，`mapped` 恒为 False。已如实反映在
        `_grading_paths` 与 `info()` 中。
        """
        if self._errors_mod is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            spec = importlib.util.spec_from_file_location(
                "dc_conformance_errors", _CONFORMANCE_DIR / "errors.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._errors_mod = mod
        return self._errors_mod

    def _load_device_state(self):
        """按需加载 conformance/device_state 资产（设备四态机）。

        ⚠️ **必须用标准 `import`（与 ascend 一致、共享 `sys.modules`），
        不能用 importlib 独立模块名加载。**

        原因：`device_state` 是**有状态的进程内单例**（模块级 `_ensure(ordinal)` 持有
        每个设备的四态、转换事件与订阅者）。若像 `_load_errors()` 那样用
        `spec_from_file_location("dc_xxx", ...)` 加载成独立模块，就会得到**两份状态机**
        —— conformance / 上层设置的状态，后端查不到；后端设置的状态，上层也看不到。

        （`errors` 是无状态纯函数，两份无所谓——但那正是「第 4 个跨后端框架缺陷」
        的成因，**不应效仿**。）
        """
        if self._device_state is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            import device_state as _device_state
            self._device_state = _device_state
        return self._device_state

    # ───────────── 设备 ─────────────
    def device_count(self) -> int:
        return int(self.torch.cuda.device_count())

    def set_device(self, ordinal: int) -> None:
        self.torch.cuda.set_device(ordinal)

    def memory_stats(self, ordinal: int) -> dict:
        """归一化为 {total_mb, used_mb, free_mb}。

        ⚠️ 实测：昆仑芯 `torch.cuda.memory_stats()` **返回空 dict**，不可用；
        必须走 `mem_get_info()`（实测 `(103045660672, 103079215104)` 准确）+ `memory_allocated()`。
        """
        torch = self.torch
        prev = torch.cuda.current_device()
        if prev != ordinal:
            torch.cuda.set_device(ordinal)
        try:
            free, total = torch.cuda.mem_get_info(ordinal)
            used = total - free
        finally:
            if prev != ordinal:
                try:
                    torch.cuda.set_device(prev)
                except Exception:
                    pass
        out = {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int(used / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }
        # 工作包 B-2：新增 `allocated_mb`（分配器视角，取不到则省略该键）。
        try:
            out["allocated_mb"] = int(torch.cuda.memory_allocated(ordinal) / 1024 / 1024)
        except Exception:
            pass
        return out

    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：分配 2x2 零张量并求和，不干扰业务。"""
        try:
            torch = self.torch
            if not torch.cuda.is_available():
                return False
            prev = torch.cuda.current_device()
            if prev != ordinal:
                torch.cuda.set_device(ordinal)
            try:
                x = torch.zeros(2, 2, device=f"cuda:{ordinal}")
                return float(x.sum().item()) == 0.0
            finally:
                if prev != ordinal:
                    try:
                        torch.cuda.set_device(prev)
                    except Exception:
                        pass
        except Exception:
            return False

    # ─────────────── 内存句柄与生命周期（职责 D3 · 工作包 B-1）───────────────

    def _alloc_raw(self, size_bytes: int, ordinal: int):
        """XPytorch 兼容层的原始设备内存分配。

        实测（P800，2026-09-29）：本家 `caching_allocator_alloc` 是**真覆写**
        （与 910C 的 `torch.npu` 不同 —— 那边只是继承 `torch.cuda` 的实现、一调就报
        `Found no NVIDIA driver`）；释放原语名是 **`caching_allocator_delete`**，
        **不存在** `caching_allocator_free`（`hasattr` 为 False）。
        """
        torch = self.torch
        prev = torch.cuda.current_device()
        if prev != ordinal:
            torch.cuda.set_device(ordinal)
        try:
            return torch.cuda.caching_allocator_alloc(int(size_bytes))
        finally:
            if prev != ordinal:
                try:
                    torch.cuda.set_device(prev)
                except Exception:
                    pass

    def _free_raw(self, ptr, handle: dict) -> None:
        """实测：`caching_allocator_delete` 对非法/重复指针**如实报错**
        （`RuntimeError: invalid device pointer`）—— 与 pyACL 的静默恰好相反；
        统一层仍统一登记，保证跨家行为一致。
        """
        self.torch.cuda.caching_allocator_delete(ptr)

    def peek_current_device(self) -> int:
        """当前默认设备序号（**只读**，用于 record_stream 的保守同步）。"""
        try:
            return int(self.torch.cuda.current_device())
        except Exception:
            return 0

    # ───────────── 流 / 事件 ─────────────
    def _create_stream_raw(self, priority=None):
        """默认路径：`torch.cuda.Stream()` —— **与历史版本逐位一致**。

        `priority is not None` 分支**正常走不到**（本后端不声明 `stream_priority_control`，
        基类门禁先抛 `NotImplementedError`）；此处仍**显式抛错**，避免误加能力键后静默放行。

        为什么本家做不了（2026-09-30 实测）：CUDA 兼容层三个入口都在，但
        **`cuCtxGetStreamPriorityRange` 返回 `least=0, greatest=0`**
        ⇒ **优先级空间退化为单点**；且 `cuStreamCreateWithPriority(..., prio=-1)`
        **回读仍为 0**（不保留传入值）。设备只报一个档位 ⇒ 设置**不产生任何区分**，
        与其宣称"可设置"，不如**如实不声明**（同 `context_lifecycle` 在 P800 的处置）。
        """
        if priority is not None:
            raise NotImplementedError(
                f"kunlun：本设备**优先级空间退化为单点**（`cuCtxGetStreamPriorityRange` "
                f"返回 least=0, greatest=0）⇒ 设置 priority={priority} 不产生任何区分，"
                f"故**如实不声明** `stream_priority_control`。\n"
                f"  · 三个 CUDA 兼容入口都在（Range/CreateWithPriority/GetPriority），"
                f"不是『接口没实现』，而是『设备只报一个档位』；\n"
                f"  · `cuStreamCreateWithPriority(prio=-1)` **回读仍为 0**（不保留传入值）；\n"
                f"  · 证据：`P800/probes/prio_readback_kunlun_20260930.log`。")
        return self.torch.cuda.Stream()

    def create_event(self):
        # 统一语义适配：未 record 不误报完成（E3）+ 主机侧有界等待（E2v2）
        return KunlunEventAdapter()

    def current_stream(self):
        return self.torch.cuda.current_stream()

    def stream_context(self, native_stream):
        return self.torch.cuda.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 为空 → 原生阻塞同步；非空 → 轮询上报超时。"""
        torch = self.torch
        if timeout_ms is None:
            torch.cuda.synchronize()
            return
        self._bounded_wait(lambda: torch.cuda.current_stream().query(), timeout_ms, "device")

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 `TimeoutError`。

        ⚠️ **语义边界（如实标注）**：实测昆仑芯 `Stream.synchronize()` 签名为
        `(self) -> None`，**无 timeout 参数**，且无中断原语。因此这里的「有界」是
        **超时上报**语义 —— 超时抛 `TimeoutError` 让上层得以降级/记录，
        但**不保证中断底层已提交的执行**。
        规范当前只写了「超时抛 TimeoutError」，未区分「真中断」与「超时上报」，
        已作为修订建议提交（接入方案 §7.3）。
        """
        if timeout_ms is None:
            native_stream.synchronize()
            return
        self._bounded_wait(native_stream.query, timeout_ms, "stream")

    def _bounded_wait(self, query_fn, timeout_ms: int, what: str) -> None:
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
                    f"kunlun: {what} synchronize timeout after {timeout_ms} ms"
                    "（超时上报语义，不保证中断底层执行）"
                )
            time.sleep(0.001)

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待事件；完成 True / 超时 False（永不永久阻塞）。"""
        if hasattr(native_event, "wait_host"):
            return bool(native_event.wait_host(timeout_ms))
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                if native_event.query():
                    return True
            except Exception:
                return False
            time.sleep(0.001)
        return False

    # ───────────── 可选能力 ─────────────
    def stream_priority_range(self):
        """流优先级区间 **(least, greatest) 2 元组**；本家如实返回 **`(0, 0)`（退化单点）**。

        ⚠️ **2026-09-30 更正根因表述（纪律 ⑰：能力缺失的根因表述本身可能是错的）**：
        旧版本返回 `None`（表述为"不支持"），理由记的是"调 `torch.cuda.Stream.priority_range()`
        会触发 PyTorch `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188`"。
        复核后发现：那是 **torch API 路径**的问题（XPytorch 把非法区间报给了 torch），
        **底层真原语安全可读** —— 直调 `cuCtxGetStreamPriorityRange` 返回 **`(0, 0)`**
        且**不触发断言** ⇒ "不支持"这个结论**方向对、表述错**：设备报的是
        **"只有一个档位"**，而不是"没有这个接口"。现改为如实回报。

        ⚠️ **不声明 `stream_priority_control` 的依据**：区间是单点 ⇒ 设置无区分意义。
        这与 `context_lifecycle`（平台单上下文 ⇒ 只声明只读 `context_query`）**同构**。
        """
        import ctypes
        lib = self._driver_handle()
        if lib is None:
            return None
        least = ctypes.c_int(0)
        greatest = ctypes.c_int(0)
        try:
            fn = lib.cuCtxGetStreamPriorityRange
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            rc = int(fn(ctypes.byref(least), ctypes.byref(greatest)))
        except AttributeError:
            return None
        if rc != 0:
            return None
        return int(least.value), int(greatest.value)

    def _stream_priority_read_raw(self, handle):
        """**厂商原语层**：CUDA 流句柄 → 优先级（读不出返回 None）。

        ⚠️ 单独成方法是为了**可注入/可替换**（离线自检在无设备时要能驱动这条链路）。
        """
        import ctypes
        lib = self._driver_handle()
        if lib is None or not handle:
            return None
        try:
            fn = lib.cuStreamGetPriority
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            got = ctypes.c_int(0x7FFFFFFF)
            rc = int(fn(ctypes.c_void_p(int(handle)), ctypes.byref(got)))
            return int(got.value) if rc == 0 else None
        except AttributeError:
            return None

    def stream_priority_readback(self, native_stream):
        """回读该流的优先级；读不出返回 None。

        实测（2026-09-30）：普通流回读 **0**；`cuStreamCreateWithPriority(prio=-1)`
        建出来的流回读**仍是 0**（不保留）—— 正是"设备只有一个档位"的直接证据。
        """
        handle = getattr(native_stream, "cuda_stream", None)
        if not handle:
            return None
        return self._stream_priority_read_raw(handle)

    # ───────────── 错误翻译 ─────────────
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 `FlagosError`。分级来源如实限定为 **`message_hint` / `default`**。

        2026-09-22 真机实测：昆仑芯**不把厂商错误码透出为数字码** ⇒ 无可查的码表，
        故**不声明 `error_map` 能力**（是「确认不具备」，不是「未验证」）。

        ⚠️ **2026-09-29 修（工作包 A 实验暴露，§11-① 的补全）**：此前是**事后降级** ——
        调共享翻译器（其码表为**昇腾 ACL 码表**）之后，只把 `graded_by` / `mapped` /
        `error_code` 三个**置信度字段**降下来，而 **`category` 仍冻结在共享码表给出的值**。
        实测后果（P800）：喂一条携带昇腾码的消息即得 **L4_FATAL / device_recovery** ——
        而契约 §1.4 明令「下游必须按 `disposition` 处理」⇒ 会**误触发设备级重建**；
        且 `errors.py` 自身规则是「无依据兜底 L3」，行为却**升级**为 L4 ⇒ 违反自身规则。

        还有更隐蔽的一层：`message_hint_unexpected` 这个标记本身也是错的 ——
        共享翻译器在**码表命中时即返回**、**从未评估消息规则**，所以降级后并没有人重算
        `message_hint`；`category` 只是"恰好"与消息规则一致或**不一致**，属**巧合而非声明**。

        ⇒ 现改为**从源头不使用外来码表**（`vendor_codes=False`）：`graded_by` / `mapped` /
        `error_code` / **`category`** 四者天然一致，无需任何事后修补 ——
        「降级必须整组一致」由此从"逐个字段记得改"变成"结构上不可能不一致"。
        """
        errors = self._load_errors()
        # ⚠️ `vendor_codes=False` 是本后端**正确性的关键**：本家无厂商码表，
        #    昇腾 ACL 码表不得参与本家的分级判定。
        fe = errors.translate_error(exc, location=location, vendor_codes=False)
        graded_by = getattr(fe, "graded_by", "default")
        return FlagosError(
            # 2026-09-20 修复（P800 推理腿暴露）：`_load_errors()` 用 importlib 把
            # conformance/errors.py 加载为**独立模块**，其 ErrorCategory 是 IntEnum
            # （L1=1..L4=4），与 api 层枚举**不是同一个类对象** —— 即使取值相同也不相等，
            # 直接透传会让 `FlagosError.disposition` 的 `DISPOSITION[cat]` 查表 KeyError。
            # 故经 `coerce_category` 按数值/名称归一（ascend 用的是同源 _INT_TO_CATEGORY）。
            category=coerce_category(getattr(fe, "category", None)) or _l3(),
            root_cause=getattr(fe, "root_cause", f"{type(exc).__name__}: {exc}"),
            location=getattr(fe, "location", "") or location,
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            # 无厂商码可依据 → 除"命中消息规则"外均不标为高置信
            is_grade_confident=(graded_by == "message_hint"),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )


    # ───────────── 恢复 ─────────────
    def _recover_device_impl(self, ordinal: int, mode: str = "probe",
                             reason: str = "", **kwargs: Any) -> dict:
        """设备重建。统一返回 dict，`recovered` 语义 = **设备当前可用**。

        ⚠️ 实测：`torch.cuda` 上**没有设备级重置/重建原语** ——
        `reset*` 系列全是内存统计类（`reset_peak_memory_stats` 等），
        无 `reset_device` / context 重建。故 **`real` 模式不支持**，
        一律以探活结果判定设备是否可用，并在 `detail` 中如实说明。
        """
        alive = self.probe_device(ordinal)

        # 2026-09-28（职责响应审计暴露）：接口约定 §1.5 的返回契约要求
        # `{ordinal, mode, recovered, state, detail}` **五键齐全**。
        # 本后端此前只返回四键（缺 `state`），与 ascend 不对称 ⇒
        # 下游（监控方向）按契约读 `state` 在本家会拿不到 —— 属对称性缺陷，已补。
        try:
            state = state_token(self.device_state(ordinal))
        except Exception:
            state = "unknown"
        if mode not in ("probe", "hybrid"):
            return {
                "ordinal": ordinal, "mode": mode, "recovered": alive,
                "state": state,
                "detail": ("昆仑芯无设备级重置/重建原语（torch.cuda 仅内存统计类 reset*，"
                           "无 reset_device/context 重建）→ real 模式不支持；"
                           "已按探活结果判定，如需 real 需上游提供原语"),
            }
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive,
            "state": state,
            "detail": f"probe 级探活：设备当前{'可用' if alive else '不可用'}",
        }

    def device_state(self, ordinal: int):
        """查询设备四态：`available` / `degraded` / `isolated` / `destroyed`。

        复用 conformance 的 device_state 资产（**进程内状态机，不依赖厂商原语**），
        与 ascend 同一份实现与同一套语义，故本后端的 `device_state` 能力声明成立。

        边界（如实标注）：昆仑芯侧无设备级重置/重建原语（见 `recover_device`），
        本后端只声明 `recovery_probe` —— 四态**转换**由上层/监控方向驱动
        （`set_device_state`），本方法只负责**查询**；恢复执行走
        `recover_device(mode="probe")`。
        """
        return self._load_device_state().query_device_state(ordinal)

    # ───────────── 能力声明 ─────────────
    # 2026-09-28：`supports()` 覆写已删除 —— 三家实现完全同款（`cap in self._capabilities`）
    # ⇒ 收敛到基类唯一实现，别名归一（`sync_timeout` → `bounded_sync`）只在基类维护一处。

    # ───────────── 已知上游/环境问题（给接入方直接可读）─────────────
    #: 只收录**已实测**的问题；每条注明复现率、归属层与证据位置。
    #: 目的是「release 给其他子方向」时，接入方读到后端即可获知坑与临时规避。
    _KNOWN_ISSUES = [
        {
            "id": "KUNLUN-KL3-EVENT-SYNC-HANG",
            "severity": "high",
            "scope": "多进程设备侧集合通信（flagcx / BKCL）",
            "condition": "环境变量 XPU_EVENT_KL3_ENABLE=1 **且**存在设备侧集合通信",
            "symptom": (
                "概率性永久挂死。自旋点三处：① dist.all_reduce 内部 "
                "c10d::flagcxBackend::syncStream → CUDAEvent::record → cudaEventRecordWithFlags；"
                "② 上层 torch.cuda.synchronize → cudaDeviceSynchronize；③ 通信域首次初始化"
                "（bkcl::init_rank / net_socket_all_gather）。前两处自旋于厂商 libcuda.so"
                "（实为 libxpucuda.so）内，用户态 100% CPU"
            ),
            "repro_rate": "≈89%（18 次运行 16 次挂死；本轮基线 4/4 = 100%）；挂死步数游走，与数据量/张量形状/reduce op/用卡对/同步间隔均无关",
            "root_cause_layer": "厂商 CUDA 兼容运行时 —— libxpucuda.so / XRE 的 KL3 事件机制与设备事件同步原语的交互",
            "ruled_out": "算子层（FlagGems 未参与，探针全程裸 torch.distributed）、编译层（无 triton 编译，BKCL 为预编译内核）均已排除",
            "workaround": (
                "在**不依赖 FlagGems** 的验证路径中不设置 XPU_EVENT_KL3_ENABLE"
                "（实测 8 次运行 0 次挂死，且 all_reduce 结果真值 2^120 精确匹配）。"
                "本方向训练腿（transformers 纯 torch）与推理腿（vLLM）均不依赖 FlagGems，故适用"
            ),
            "workaround_risk": (
                "该变量是 FlagGems kunlunxin 官方推荐值（tools/env.sh、"
                "src/flag_gems/backends.yaml、CI P800.yml 三处均设 1）；关闭可能影响厂商设备事件上报。"
                "**走 FlagGems 路径时不适用本规避**。注意 flagcx 侧无法规避："
                "syncStream 是流序正确性所必需（backend_flagcx.cpp:424，14 处集合通信各调一次）"
            ),
            "report_to": "昆仑芯（XPytorch / XRE）—— 我方已上报，对应本方向对外提交项 C3",
            "evidence": "prototype/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md",
        },
    ]

    def known_issues(self) -> list:
        return [dict(x) for x in self._KNOWN_ISSUES]

    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            # 公共字段改由基类唯一来源提供（原为三家各手写一份 ⇒ 同一份清单三个来源）
            **self._base_info_fields(),
            "device_type": self.device_type,
            "framework": "XPytorch (torch.cuda 兼容层) + torch_xray 符号重写",
            "torch": self._torch.__version__,
            "torch_build": {"USE_CUDA": True, "USE_XPU": False},
            "device_count": self.device_count(),
            # 与 _capabilities 同一套键名（自洽，不重复出现 A 键声明/B 键查询的问题）
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
            "error_grading": dict(self._grading_paths),
            "bounded_sync_scope": "主机侧等待（event.wait_host）真有界；流同步为超时上报语义",
            "vendor_discriminator": [
                "torch_xray / torch_xmlir 模块存在",
                "/proc/kunlun 存在",
                "xpu-smi 存在",
                "通信库 libbkcl.so（XCCL/BKCL）",
            ],
            "known_upstream_defects": [
                "Stream.priority_range() → PyTorch INTERNAL ASSERT (c10/cuda/CUDAStream.h:188)",
                "厂商错误码不透出到 Python 异常（仅退出钩子偶见 error code=101）",
                "XPU_EVENT_KL3_ENABLE=1 且存在设备侧集合通信时，设备事件同步概率性永久挂死（≈89%，"
                "自旋于厂商 libcuda.so）—— 详见 known_issues[0] 与 "
                "prototype/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md",
            ],
            # 结构化版本（含复现率/归属层/临时规避/上报对象），供接入方机器可读消费
            "known_issues": self.known_issues(),
        }


def _l3():
    from ...api.errors import ErrorCategory
    return ErrorCategory.L3_EXECUTION


def build() -> KunlunBackend:
    return KunlunBackend()
