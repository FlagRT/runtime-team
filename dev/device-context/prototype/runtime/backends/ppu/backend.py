#!/usr/bin/env python3
"""平头哥 PPU（真武 ZW810E）后端 —— **路径 B：复用 `torch.cuda` 命名空间**。

⚠️ **命名空间说明（先读）**
  平头哥这栈上设备 API **走 `torch.cuda`，不是 `torch.xpu`**。三条独立证据（本机实测）：
    1. `torch.cuda.is_available() == True`、`device_count() == 16`、
       `get_device_name(0) == "PPU-ZW810E"`、`torch.cuda` 上真跑 matmul 成功；
    2. **`torch.xpu.device_count() == 0`** ⇒ 本栈**不是** XPU 路线（`torch.xpu` 模块只是
       stock torch 带进来的 Intel XPU，看不到 PPU）；
    3. 元数据一致：`torch.version.cuda == "13.0"`，镜像内 `CUDA_SDK_VER=cuda-13.0`、
       `CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK`；厂商侧工具 `nvidia-smi` 是指向 `ppu-smi` 的符号链接。
  ⇒ `name = "ppu"`（厂商标识，我们注册表里的键）· `device_type = "cuda"`（设备串前缀）。
  **两者不同名是刻意的**：`ppu` 与 `kunlun`（同样复用 `torch.cuda`）因此必须靠
  **厂商特征**区分，不能靠命名空间 —— 见 `info()["vendor_discriminator"]`。

**本后端的能力声明 = 逐项实测结果**（探测脚本见 `PPU/probes/`）：
  · 声明见 `_capabilities`，每条都在注释里写明**证据来源**；
  · ❌ **不声明** `error_map`（实测：错误只给**错误名** `hggcErrorInvalidValue`，**无数值码**）；
  · ❌ **不声明** `recovery_real`（实测：`cuDevicePrimaryCtxReset_v2` 返回 rc=0，但**之后
     本进程设备操作全部 `hggcErrorInvalidValue`** ⇒ 该原语会导致进程内设备不可用，
     不能当作可用的设备级重建路径）。

运行前置（实测）：
  · 容器需挂 `/dev/alixpu`、`/dev/alixpu_ctl` 与要用的 `/dev/alixpu_ppu<N>`（**非 privileged**）；
  · ⚠️ **必须覆盖 `NCCL_SOCKET_IFNAME`** —— 基座镜像把它写死成 `eth0`，而本机**没有 eth0**
    （真实网卡 `bond0`）⇒ 不覆盖则 `init_process_group` 能建组、**真实 all_reduce 失败**。
"""
from __future__ import annotations

import ctypes
import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import FlagosError, coerce_category
from ..base import RuntimeBackend, state_token

logger = logging.getLogger(__name__)

#: conformance 目录（共享资产所在；与既有后端同一解析方式）
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


class PpuEventAdapter:
    """`torch.cuda.Event` 的统一语义适配。

    补齐两处与统一事件契约的语义缺口（**都是本机实测确认存在的缺口**）：
      - **E3：未 record 的 Event 调 `query()` 原生返回 `True`（误报"已完成"）** ——
        实测 `torch.cuda.Event().query() is True`（工具/探针确认）；用 `recorded` 标志修正为 `False`。
      - **E2v2：主机侧有界等待 `wait_host(timeout_ms)`** —— 原生 `Event.wait` 签名为
        `(self, stream=None) -> None`（**无 timeout 参数**）⇒ 用 `query()` 轮询实现，
        **永不永久阻塞**。
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
        """主机侧**有界**等待：完成 `True` / 超时 `False`（**不永久阻塞**）。"""
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


class PpuBackend(RuntimeBackend):
    """基于平头哥 PPU（真武 ZW810E，`torch.cuda` 兼容栈）的后端。"""

    name = "ppu"
    device_type = "cuda"   # 见模块 docstring：设备串按命名空间，不按厂商

    #: 规范能力键全集（键名与 ascend / kunlun / cambricon 对齐；用于 `info()` 自洽报告）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync", "sync_timeout",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
        "memory_alloc", "memory_alloc_stat", "record_stream", "context_lifecycle",
        "context_query", "device_state_control",
        "stream_priority_control", "stream_priority_readback",
    )

    #: 本后端**声明支持**的能力 —— **每一项都有本机实测证据**，不支持的一律不写进来
    _capabilities = {
        "device",             # torch.cuda.device_count()=16 / get_device_name="PPU-ZW810E" / matmul 成功
        "memory",             # mem_get_info(0) = (98022, 98304) MiB
        "memory_alloc",       # caching_allocator_alloc/delete 真调用：512 → +4 MiB → 512（PASS）
        "memory_alloc_stat",  # memory_allocated(0) 可用（上项即用它取证）
        "record_stream",      # Tensor.record_stream 真调用 + synchronize 通过（PASS）
        "stream",             # torch.cuda.Stream / current_stream / stream(上下文管理器) 均可用
        "event",              # torch.cuda.Event 可用；**E3 缺口由 PpuEventAdapter 修正**
        "bounded_sync",       # 主机侧等待用轮询实现，真有界（见 wait_event_host）
        "recovery_probe",     # probe_device（设备上真算 2x2 零张量）
        "device_state",       # 四态查询（共享状态机，芯片无关）
        "device_state_control",  # 四态驱动（共享状态机，芯片无关）
        # ── 上下文：本机实测「能建多个 + 能切换 + 只销毁自己的 + torch 全程可用」 ──
        # 证据（PPU/probes/probe_ppu_capability_decisions2.py）：
        #   create_A rc=0 / create_B rc=0（B≠A）/ set_current(A) rc=0 / destroy_A rc=0
        #   / set_current(B) rc=0 / torch_under_B=4.0 / destroy_B rc=0 / torch_after_restore=4.0
        "context_lifecycle",
        "context_query",      # cuCtxGetCurrent/GetDevice/GetFlags rc=0，present=True，调完 torch 仍可用
        # ── 流优先级：**三把钥匙全可用，且区间非单点**（本机实测，本栈第二个多档实例）──
        # 证据：`Stream.priority_range() = (0, -3)`；C API 扫描 flag∈{0,1} 全部 rc=0 且
        #   回读在 [-3,0] 内**等于请求**（越界静默夹取：-8→-3、8→0）；
        #   `torch.cuda.Stream(priority=p)` 后**用 C API 独立回读 == 请求**（-3/-2/-1/0 全中）
        #   ⇒ 参数**真进设备**（不是构造参数回显）。
        "stream_priority",
        "stream_priority_control",
        "stream_priority_readback",
        "graph_capture",      # torch.cuda.graph 真捕获 + replay + 结果对照 GRAPH_CAPTURE: PASS
        "multidevice",        # 单机 16 卡（本容器挂了 2 张）
        # ── 以下**不支持**，故不声明 ──
        # "error_map"     : **已确认不具备** —— 实测设备侧错误只给错误**名**
        #                   （`hggcErrorInvalidValue` / `CUDA error: invalid device ordinal`），
        #                   **不透出数值码** ⇒ 无码表可建，只能走 message_hint 分级。
        # "recovery_real" : **已确认不可用** —— `cuDevicePrimaryCtxReset_v2` 存在且返回 rc=0，
        #                   但**之后本进程设备操作全部 hggcErrorInvalidValue**（含 set_device 后重试）
        #                   ⇒ 该原语不是可用的设备级重建路径，本层**不得**把它包装成"能重建"。
    }

    #: 分级来源可达性（如实标注：本栈无数值码 ⇒ code_map 路径不可达）
    _grading_paths = {"code_map": False, "message_hint": True}

    #: 惰性加载的驱动库句柄（类级缓存；`None` = 未尝试，`False` = 已确认不可用）
    #: ⚠️ 名字必须是 **`_ctx_driver`** —— 离线自检工具按这个约定把**可控假驱动库**
    #:    注入进来，从而在无设备时也能真实驱动「优先级设置/回读 + 上下文生命周期」链路
    #:    （手册纪律：**能造可控假原语就别 SKIP**）。用别的名字 ⇒ 工具找不到 ⇒ 判据退化。
    _ctx_driver = None

    def __init__(self) -> None:
        self._torch = None
        self._errors_mod = None
        self._device_state = None
        self._loaded = False
        self._touched = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        self._torch = torch
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    def _touch_device(self) -> None:
        """**触碰设备**（只做一次）。

        ⚠️ 为什么必须显式做（本项目已三次踩到，见《接入手册》坑 18）：
        本栈的厂商 C API（`cu*`）在**尚无当前上下文**时返回 `rc=3`
        （`CUDA_ERROR_NOT_INITIALIZED`）—— 实测直接 `ctypes` 调
        `cuCtxGetStreamPriorityRange` 得 `rc=3`，紧跟着建流全部失败。
        ⇒ 凡要调 `cu*` 之前，先做一次设备侧最小操作把上下文带起来。
        """
        if self._touched:
            return
        torch = self.torch
        try:
            float(torch.zeros(1, device=f"{self.device_type}:0").sum().item())
        except Exception:                                        # noqa: BLE001
            pass
        self._touched = True

    def _driver_handle(self):
        """惰性取得**驱动库**句柄（本栈：`libcuda.so.1` 就是 PPU 的 CUDA 兼容层）。

        ⚠️ 只按 **soname** 取，**不拿任意副本路径** —— 同栈内引入第二份同名库副本会造成
        版本错配（既有实例实测过 `CUDA_ERROR_NOT_INITIALIZED`）。
        """
        if type(self)._ctx_driver is None:
            self._touch_device()                                  # 见 _touch_device 的说明
            try:
                type(self)._ctx_driver = ctypes.CDLL("libcuda.so.1")
            except OSError:
                type(self)._ctx_driver = False
        return type(self)._ctx_driver or None

    def _load_errors(self):
        """复用 `conformance/errors.py` 的统一翻译骨架（厂商中立的 message 规则）。

        注意：其中的昇腾码表在本栈**永不命中**；本后端调用时显式传
        `vendor_codes=False`（见 `translate_error`），故 `graded_by` 只会是
        `message_hint` 或 `default`，`mapped` 恒为 False —— 已如实反映在 `_grading_paths`。
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
        """按需加载 `conformance/device_state` 资产（设备四态机）。

        ⚠️ **必须用标准 `import`（共享 `sys.modules`）**，不能用 importlib 独立模块名加载：
        该模块是**有状态的进程内单例**，独立加载会得到**两份状态机**
        （上层设的状态后端查不到，后端设的状态上层看不到）—— 本项目审计台账第 23 条。
        """
        if self._device_state is None:
            sys.path.insert(0, str(_CONFORMANCE_DIR))
            import device_state as _device_state
            self._device_state = _device_state
        return self._device_state

    # ───────────── 设备（13 抽象之一部分）─────────────
    def device_count(self) -> int:
        return int(self.torch.cuda.device_count())

    def set_device(self, ordinal: int) -> None:
        self.torch.cuda.set_device(ordinal)

    def peek_current_device(self) -> int:
        """当前默认设备序号（**只读**，用于 record_stream 的保守同步）。"""
        try:
            return int(self.torch.cuda.current_device())
        except Exception:                                        # noqa: BLE001
            return 0

    def memory_stats(self, ordinal: int) -> dict:
        """归一化为 `{total_mb, used_mb, free_mb}`（+ 尝试附加 `allocated_mb`）。

        实测：本栈 `mem_get_info(ordinal)` 返回 `(free, total)` 且准确
        （`(98022, 98304)` MiB）；`memory_allocated(ordinal)` 可用。
        `memory_stats()` 亦返回非空 dict，但键为 `active.*` 明细，**不适合直接取三键**，
        故统一走 `mem_get_info` + `memory_allocated`（与既有实例同一口径）。
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
                except Exception:                                # noqa: BLE001
                    pass
        out = {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int(used / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }
        # 分配器视角；取不到则**省略该键**（契约：不得填 0 冒充）
        try:
            out["allocated_mb"] = int(torch.cuda.memory_allocated(ordinal) / 1024 / 1024)
        except Exception:                                        # noqa: BLE001
            pass
        return out

    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：在目标设备上真算一个 2x2 零张量并求和。"""
        try:
            torch = self.torch
            if not torch.cuda.is_available():
                return False
            prev = torch.cuda.current_device()
            if prev != ordinal:
                torch.cuda.set_device(ordinal)
            try:
                x = torch.zeros(2, 2, device=f"{self.device_type}:{ordinal}")
                return float(x.sum().item()) == 0.0
            finally:
                if prev != ordinal:
                    try:
                        torch.cuda.set_device(prev)
                    except Exception:                            # noqa: BLE001
                        pass
        except Exception:                                        # noqa: BLE001
            return False

    # ─────────────── 内存句柄与生命周期 ───────────────
    def _alloc_raw(self, size_bytes: int, ordinal: int):
        """厂商原始设备内存分配：`caching_allocator_alloc`。

        实测（本机）：`hasattr` 为 True，且真调用有效 ——
        `memory_allocated` 从 512 B → 4194816 B（**正好 +4 MiB**）→ 释放后回到 512 B。
        ⚠️ 释放原语名是 **`caching_allocator_delete`**；`caching_allocator_free`
        在本栈**不存在**（`hasattr` 为 False）⇒ 不得伪造。
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
                except Exception:                                # noqa: BLE001
                    pass

    def _free_raw(self, ptr, handle: dict) -> None:
        self.torch.cuda.caching_allocator_delete(ptr)

    # ───────────── 上下文生命周期 ─────────────
    #
    # 本机实测（`probe_ppu_capability_decisions2.py::Q2`，子进程隔离）：
    #   create_A rc=0（成为当前）/ create_B rc=0（B≠A，成为当前）/ `cuCtxSetCurrent(A)` rc=0
    #   / `cuCtxDestroy_v2(A)` rc=0 / 切回 B rc=0 / **在 B 下 torch 仍算出 4.0**
    #   / destroy B rc=0 / 恢复原上下文后 torch 仍算出 4.0
    # ⇒ 与 P800 相反：本栈**允许**本层自建多个上下文并切换，且**不影响** torch。
    #
    # ⚠️ 仍然只允许销毁**本层创建**的句柄（安全契约，见 base.py）：对进程默认上下文
    #    `cuCtxDestroy_v2` 的行为**未验证**，本层一律拒绝（基类 `_handle_id` 已守住）。

    def _ctx_create_raw(self, ordinal: int):
        lib = self._driver_handle()
        if lib is None:
            raise RuntimeError("ppu：驱动库 `libcuda.so.1` 不可得 ⇒ 无法创建上下文")
        lib.cuCtxCreate_v2.restype = ctypes.c_int
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_uint, ctypes.c_int]
        h = ctypes.c_void_p(0)
        rc = int(lib.cuCtxCreate_v2(ctypes.byref(h), ctypes.c_uint(0), ctypes.c_int(int(ordinal))))
        if rc != 0 or h.value is None:
            raise RuntimeError(f"ppu：`cuCtxCreate_v2(ordinal={ordinal})` 失败 rc={rc} ⇒ 如实报错")
        return int(h.value)

    def _ctx_set_raw(self, ctx, handle: dict) -> None:
        lib = self._driver_handle()
        if lib is None:
            raise RuntimeError("ppu：驱动库不可得 ⇒ 无法切换上下文")
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        rc = int(lib.cuCtxSetCurrent(ctypes.c_void_p(int(ctx))))
        if rc != 0:
            raise RuntimeError(f"ppu：`cuCtxSetCurrent` 失败 rc={rc} ⇒ 如实报错")

    def _ctx_destroy_raw(self, ctx, handle: dict) -> None:
        lib = self._driver_handle()
        if lib is None:
            raise RuntimeError("ppu：驱动库不可得 ⇒ 无法销毁上下文")
        lib.cuCtxDestroy_v2.restype = ctypes.c_int
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        rc = int(lib.cuCtxDestroy_v2(ctypes.c_void_p(int(ctx))))
        if rc != 0:
            raise RuntimeError(f"ppu：`cuCtxDestroy_v2` 失败 rc={rc} ⇒ 如实报错")

    def _ctx_query_raw(self, ordinal: int = 0):
        """**只读**读回当前生效的上下文（句柄/设备号/标志）；无则 `None`。

        实测：`cuCtxGetCurrent/GetDevice/GetFlags` 三者 rc=0，且调用后 torch 仍可计算
        ⇒ 只读入口安全（本方法**绝不**调用创建/销毁原语）。
        """
        lib = self._driver_handle()
        if lib is None:
            return None
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cur = ctypes.c_void_p(0)
        try:
            rc = int(lib.cuCtxGetCurrent(ctypes.byref(cur)))
        except AttributeError:
            return None
        if rc != 0 or cur.value is None:
            return None
        dev, flg = ctypes.c_int(-1), ctypes.c_uint(0)
        lib.cuCtxGetDevice.restype = ctypes.c_int
        lib.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cuCtxGetFlags.restype = ctypes.c_int
        lib.cuCtxGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        rc_d = int(lib.cuCtxGetDevice(ctypes.byref(dev)))
        rc_f = int(lib.cuCtxGetFlags(ctypes.byref(flg)))
        return {"ctx": int(cur.value),
                "ordinal": int(dev.value) if rc_d == 0 else None,
                "flags": int(flg.value) if rc_f == 0 else None}

    def _same_ctx(self, a, b) -> bool:
        """上下文同一性：本栈上下文以**整型句柄**表示 ⇒ 直接比数值。"""
        try:
            if a is None or b is None:
                return False
            return int(a) == int(b)
        except Exception:                                        # noqa: BLE001
            return False

    # ───────────── 流 / 事件 ─────────────
    #: 建流 flag。⚠️ 本栈**只接受 0 / 1** —— 实测 flags∈{2,4} 对**任意** priority
    #: 都返回 `rc=1`（INVALID_VALUE），现象看起来像"优先级不被支持"，实为 flag 取值
    #: 不被该兼容层接受（NVIDIA 语义里 2 并不是合法 flag）。取 **1**（NON_BLOCKING）。
    _STREAM_FLAG_NON_BLOCKING = 1

    def _create_stream_raw(self, priority=None):
        """`priority is None` ⇒ `torch.cuda.Stream()`（**厂商默认，与历史版本逐位一致**）。

        `priority=<int>` ⇒ **厂商 C API 建流 + `torch.cuda.ExternalStream` 包回 torch**：
        `cuStreamCreateWithPriority(h, flag=1, prio)` → `ExternalStream(h)`。

        为什么走 C API（本机实测**两条路都可用**，选 C API 是判据形态决定的）：
          · 本层「能设置」的判据是「**请求 == 设备侧回读**」，回读走 `cuStreamGetPriority`；
            由 C API 建流 ⇒ "读回的就是刚设的那条"在链路上闭合；且离线自检工具可按
            **`_ctx_driver` 约定**注入可控假原语，把这条判据**真跑起来**
            （纪律：**能造可控假原语就别 SKIP**）；
          · 与其余「多档」实例（寒武纪 MLU590）保持**同一处置形态**，便于跨实例对照。
        实测（本机，`flag=1`）：`prio ∈ {-3,-1,0}` 建流 rc=0、回读**逐值等于请求**、
        `ExternalStream` 包装成功且在其上真跑 matmul 得 512.0、`cuStreamDestroy_v2` rc=0。
        （对照参考：`torch.cuda.Stream(priority=…)` 路本机也可用、回读同样一致 ——
         本层选前者是为**可证据化**，不是后者不可用。）

        ⚠️ **该流归本层拥有**（`_register_owned_stream`）⇒ 用完须
        `runtime.release_stream(stream)`，否则泄漏一条设备级流；包装失败时
        **先销毁刚建的流再抛错**（不泄漏）。

        取值域由基类先用 `stream_priority_range()` 校验（越界 ⇒ `ValueError`）。
        实测越界值在厂商侧是**静默夹取**（-8→-3、8→0）—— 正因如此才**必须**由本层在
        更前面把越界挡成显式错误，不能让「夹取」冒充「按请求设置」。
        """
        torch = self.torch
        if priority is None:
            return torch.cuda.Stream()

        lib = self._driver_handle()
        if lib is None:
            raise RuntimeError(
                "ppu：驱动库 `libcuda.so.1` 不可得 ⇒ 无法在**指定优先级**下建流。\n"
                "  · 本路径需要 `cuStreamCreateWithPriority`（厂商 CUDA 兼容层）；\n"
                "  · 不指定优先级请用 `create_stream()`（走 torch 原生路径）。")
        create = getattr(lib, "cuStreamCreateWithPriority", None)
        if create is None:
            raise RuntimeError(
                "ppu：驱动库缺少 `cuStreamCreateWithPriority` ⇒ 无法在**指定优先级**下建流。\n"
                "  · 本层**不降级**（不会静默返回一条无优先级的流）；"
                "不指定优先级请用 `create_stream()`。")
        create.restype = ctypes.c_int
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        handle = ctypes.c_void_p(0)
        rc = int(create(ctypes.byref(handle),
                        ctypes.c_uint(self._STREAM_FLAG_NON_BLOCKING),
                        ctypes.c_int(int(priority))))
        if rc != 0 or handle.value is None:
            raise RuntimeError(
                f"ppu：`cuStreamCreateWithPriority(priority={priority})` 失败 "
                f"rc={rc} handle={handle.value!r} ⇒ 如实报错，不返回假流。")

        try:
            native = torch.cuda.ExternalStream(handle.value,
                                               device=torch.cuda.current_device())
        except BaseException:
            self._destroy_stream_raw(handle.value)      # 不留泄漏
            raise
        self._register_owned_stream(native, handle.value)
        return native

    def _destroy_stream_raw(self, handle) -> bool:
        """销毁由**本层**创建的流（`cuStreamDestroy_v2`）；原语不可得返回 `False`（不假装成功）。"""
        lib = self._driver_handle()
        if lib is None or not handle:
            return False
        fn = getattr(lib, "cuStreamDestroy_v2", None)
        if fn is None:
            return False
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p]
        return int(fn(ctypes.c_void_p(int(handle)))) == 0

    def create_event(self):
        # 统一语义适配：未 record 不误报完成（E3）+ 主机侧有界等待（E2v2）
        return PpuEventAdapter()

    def current_stream(self):
        return self.torch.cuda.current_stream()

    def stream_context(self, native_stream):
        return self.torch.cuda.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步；`timeout_ms` 非空时走**有界轮询**（超时抛 `TimeoutError`）。"""
        torch = self.torch
        if timeout_ms is None:
            torch.cuda.synchronize()
            return
        self._bounded_wait(lambda: torch.cuda.current_stream().query(), timeout_ms, "device")

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 `TimeoutError`。

        ⚠️ **语义边界（如实标注，与既有实例同口径）**：本栈 `Stream.synchronize()`
        签名为 `(self) -> None`（**无 timeout 参数**，实测），且无中断原语
        ⇒ 这里的「有界」是**超时上报**语义 —— 超时抛 `TimeoutError` 让上层得以降级/记录，
        但**不保证中断底层已提交的执行**。
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
            except Exception:                                    # noqa: BLE001
                # 查询本身失败 ⇒ 不视为超时，交由上层错误翻译处理
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ppu: {what} synchronize timeout after {timeout_ms} ms"
                    "（超时上报语义，不保证中断底层执行）")
            time.sleep(0.001)

    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待事件；完成 `True` / 超时 `False`（**永不永久阻塞**）。"""
        if hasattr(native_event, "wait_host"):
            return bool(native_event.wait_host(timeout_ms))
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                if native_event.query():
                    return True
            except Exception:                                    # noqa: BLE001
                return False
            time.sleep(0.001)
        return False

    # ───────────── 可选能力：流优先级（三把钥匙）─────────────
    def stream_priority_range(self):
        """流优先级区间，返回契约要求的 **`(least, greatest)` 2 元组**。

        本机实测：`torch.cuda.Stream.priority_range() == (0, -3)`；
        C API `cuCtxGetStreamPriorityRange` → `rc=0, least=0, greatest=-3`（两者一致）
        ⇒ **本栈有 4 档**（0 最低、-3 最高），**不是退化单点**。

        ⚠️ 形状纪律：必须返回 **2 元组**（不得把厂商三元组透传出去）；基类
        `_priority_bounds()` 会归一为 `(min, max)`（本栈 = `(-3, 0)`）再判取值域。
        取不到时返回 `None`（**不猜、不补零**）。
        """
        lib = self._driver_handle()
        if lib is None:
            return None
        # ⚠️ 用 **getattr 显式探针**判断符号是否存在，其余异常**一律传播** ——
        #    不能把「给 restype 赋值失败」（调用方式 bug）与「库没这个符号」（如实不支持）
        #    混为一谈且静默吞掉（既有实例的实测教训）。
        fn = getattr(lib, "cuCtxGetStreamPriorityRange", None)
        if fn is None:
            return None
        least, greatest = ctypes.c_int(0), ctypes.c_int(0)
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        if int(fn(ctypes.byref(least), ctypes.byref(greatest))) != 0:
            return None
        return (int(least.value), int(greatest.value))

    def stream_priority_readback(self, native_stream):
        """**独立回读**某条流的实际优先级（`cuStreamGetPriority`）；读不出返回 `None`。

        为什么必须走 C API 而不读 `.priority` 属性：属性值是**构造参数回显**，
        在"厂商静默丢弃参数"的栈上会**永远读不出问题**（既有实例实测：属性恒为请求值、
        设备侧其实没生效）。本栈两者恰好一致，但**判据仍取设备侧读数** —— 只有这样，
        "参数真的进了设备"才是被证明的。
        """
        handle = getattr(native_stream, "cuda_stream", None)
        if not handle:
            return None
        return self._stream_priority_read_raw(handle)

    def _stream_priority_read_raw(self, handle):
        """**厂商原语层**：流句柄 → 优先级（读不出返回 `None`）。

        ⚠️ 单独成方法是为了**可注入 / 可替换**（离线无设备时，自检工具要能按
        `_ctx_driver` 约定注入可控假原语来驱动这条链路）。
        同 `stream_priority_range()`：**显式探针 + 其余异常传播**，不静默吞错。
        """
        lib = self._driver_handle()
        if lib is None or not handle:
            return None
        fn = getattr(lib, "cuStreamGetPriority", None)
        if fn is None:
            return None
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        got = ctypes.c_int(0x7FFFFFFF)
        rc = int(fn(ctypes.c_void_p(int(handle)), ctypes.byref(got)))
        return int(got.value) if rc == 0 else None

    # ───────────── 错误翻译 ─────────────
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 `FlagosError`；分级来源如实限定为 `message_hint` / `default`。

        本机实测：设备侧错误**不透出数值码** —— 越界设备下标得到
        `AcceleratorError("CUDA error: invalid device ordinal")`，显存超配得到
        `OutOfMemoryError("CUDA out of memory. Tried to allocate …")`，
        另有厂商错误**名**（如 `hggcErrorInvalidValue`）。全程**没有可查的数字码**
        ⇒ **不声明 `error_map` 能力**（是「**已确认不具备**」，不是「未验证」）。

        ⚠️ 因此必须传 `vendor_codes=False`：共享翻译器自带的是**昇腾 ACL 码表**，
        若让它参与本栈分级，会把带他厂码的消息判成 L4 ⇒ 下游按 disposition 去
        **误触发设备级重建**（既有实例的实测缺陷）。传 `False` 后
        `graded_by`/`mapped`/`error_code`/`category` 四者**结构上不可能不一致**。
        """
        errors = self._load_errors()
        fe = errors.translate_error(exc, location=location, vendor_codes=False)
        graded_by = getattr(fe, "graded_by", "default")
        return FlagosError(
            # `_load_errors()` 把 conformance/errors.py 加载为**独立模块**，其 ErrorCategory
            # 与 api 层的不是同一个类对象（同值也不相等）⇒ 必须归一，否则 disposition 查表 KeyError。
            category=coerce_category(getattr(fe, "category", None)) or _l3(),
            root_cause=getattr(fe, "root_cause", f"{type(exc).__name__}: {exc}"),
            location=getattr(fe, "location", "") or location,
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            is_grade_confident=(graded_by == "message_hint"),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            backend=self.name,
        )

    # ───────────── 恢复 ─────────────
    def _recover_device_impl(self, ordinal: int, mode: str = "probe",
                             reason: str = "", **kwargs: Any) -> dict:
        """设备重建。统一返回 dict（基类会再附加上下文维度）。

        ⚠️ **本栈 `real` 模式不可用，且理由是实测的**：
        `cuDevicePrimaryCtxReset_v2` 符号存在、调用返回 **rc=0**，但**之后本进程的所有
        设备操作都失败**（`AcceleratorError: CUDA error: invalid argument`，含重新
        `set_device` 后再试）⇒ 它不是可用的"设备级重建"路径（用它等于把进程弄坏）。
        torch 侧 `reset*` 系列全是内存统计类（无 `reset_device`）。
        ⇒ 如实不声明 `recovery_real`；`real` 请求按探活结果回答并在 `detail` 写明原因。
        """
        alive = self.probe_device(ordinal)
        try:
            state = state_token(self.device_state(ordinal))
        except Exception:                                        # noqa: BLE001
            state = "unknown"
        if mode not in ("probe", "hybrid"):
            return {
                "ordinal": ordinal, "mode": mode, "recovered": alive,
                "state": state,
                "detail": ("平头哥 PPU：无**可用**的设备级重建原语 —— "
                           "`cuDevicePrimaryCtxReset_v2` 虽返回 rc=0，但调用后本进程设备操作"
                           "全部报 `hggcErrorInvalidValue`（实测，含 set_device 后重试）；"
                           "torch 侧 reset* 均为内存统计类 ⇒ `real` 如实不支持，"
                           "已按探活结果判定"),
            }
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive,
            "state": state,
            "detail": f"probe 级探活：设备当前{'可用' if alive else '不可用'}",
        }

    def device_state(self, ordinal: int):
        """查询设备四态 `available`/`degraded`/`isolated`/`destroyed`。

        复用 conformance 的 `device_state` 共享资产（**进程内状态机，不依赖厂商原语**）
        ⇒ 与其余实例同一份实现、同一套语义，故本后端的 `device_state` 声明成立。
        """
        return self._load_device_state().query_device_state(ordinal)

    # ───────────── 已知上游/环境问题（给接入方直接可读）─────────────
    #: 只收录**已实测**的问题；每条注明复现率、归属层与证据位置。
    _KNOWN_ISSUES = [
        {
            "id": "PPU-NCCL-SOCKET-IFNAME-ETH0-MISMATCH",
            "severity": "high",
            "scope": "多卡集合通信（torch.distributed 后端 `nccl`）",
            "condition": "使用基座镜像内建的 `NCCL_SOCKET_IFNAME=eth0`（未覆盖该变量）",
            "symptom": ("`init_process_group('nccl')` **能成功建组**，但第一次真实集合通信即失败："
                        "`DistBackendError: NCCL error … ncclInternalError: Internal check failed "
                        "(init.cc:411 -> 3)`；带 `NCCL_DEBUG=INFO` 可见根因 "
                        "`NCCL ERROR Bootstrap : no socket interface found`"
                        "（先打印 `NCCL_SOCKET_IFNAME set to eth0`）"),
            "root_cause": ("**本机没有 `eth0`** —— 真实网卡是 `bond0`（`192.168.5.112/26`）。"
                           "该变量由基座镜像写死，与本机网卡名不符。"),
            "root_cause_layer": "环境/镜像（非厂商缺陷、非本层缺陷）",
            "reproduce_rate": "100%（未覆盖变量时）",
            "workaround": ("启动容器时显式覆盖，如 `docker run -e NCCL_SOCKET_IFNAME=bond0 …`；"
                           "实测 `bond0` / `lo` / 清空该变量**三者也都能通过**，"
                           "只有镜像默认的 `eth0` 不通。"),
            "workaround_risk": ("低。取 `bond0` 是「真实网卡、语义确定」；`lo` 仅适用单机多卡"
                                "（本机 16 卡全在一台机器内）。"),
            "report_to": "共享机使用者 / 镜像维护方（**本方向已在自己的容器内规避**）",
            "evidence": ("PPU/probes/probe_allreduce_2card 运行记录 + "
                         "`hliu553-dc-dev.run.sh`（含覆盖说明）；"
                         "对照组：同机他人容器 `flagscale-train` / `flagos-adrec` 的 env 同样是 `eth0`"),
        },
        {
            "id": "PPU-PRIMARY-CTX-RESET-BREAKS-PROCESS",
            "severity": "medium",
            "scope": "设备级恢复路径（`recover_device(mode=\"real\")`）",
            "condition": "调用 `cuDevicePrimaryCtxReset_v2(0)`",
            "symptom": ("调用本身返回 **rc=0**（看起来成功），但**之后本进程的所有设备操作失败**："
                        "`AcceleratorError: CUDA error: invalid argument`"
                        "（`hggcErrorInvalidValue`），重新 `set_device` 后再试仍然失败"),
            "root_cause_layer": "厂商运行时 / 驱动",
            "reproduce_rate": "1/1（单次实测，子进程隔离）",
            "workaround": ("**不要用它做设备级重建**；本层如实不声明 `recovery_real`，"
                           "需要恢复时走 `probe`（进程内安全）。"),
            "workaround_risk": "无（不使用即无风险）",
            "report_to": "平头哥运行时（待确认是否为预期语义）",
            "evidence": "PPU/probes/probe_ppu_capability_decisions2_20261010.log（R2 段）",
        },
        {
            "id": "PPU-STREAM-CREATE-FLAG-ONLY-0-1",
            "severity": "low",
            "scope": "厂商 C API `cuStreamCreateWithPriority`",
            "condition": "`flags` 取 2 或 4（CUDA 语义里的 `CU_STREAM_NON_BLOCKING`/其它）",
            "symptom": "对**任意** priority 值一律返回 `rc=1`（INVALID_VALUE）—— 看起来像「优先级不被支持」，实为 flag 取值不被接受",
            "root_cause_layer": "厂商运行时（CUDA 兼容层对 flag 的取值域与 NVIDIA 不同）",
            "reproduce_rate": "100%（flag∈{2,4}）；flag∈{0,1} 时全部 rc=0",
            "workaround": "本层只用 **`torch` 原生路径**建流（不自己调该 C API）；确需直调时 **flags 取 1**",
            "workaround_risk": "低",
            "report_to": "平头哥运行时（兼容层 flag 语义与 NVIDIA 不一致）",
            "evidence": "PPU/probes/probe_ppu_stream_priority_20261010.log（FLAG=2/4 段）",
        },
    ]

    def known_issues(self) -> list:
        return [dict(x) for x in self._KNOWN_ISSUES]

    # ───────────── 元信息 ─────────────
    def info(self) -> dict:
        self._load()
        return {
            "name": self.name,
            "device_type": self.device_type,
            # 公共字段由基类唯一来源提供（capabilities / native_accesses / degradations）
            **self._base_info_fields(),
            "framework": "平头哥 PPU CUDA 兼容栈（复用 torch.cuda 命名空间）",
            "torch": self._torch.__version__,
            # ⚠️ 外层也要 getattr：`torch.version` 本身可能不存在（离线自检的 stub 就没有），
            #    写成 `getattr(self._torch.version, "cuda", None)` 会先在外层抛 AttributeError
            #    （2026-10-10 由离线自检 [7] 段抓到 —— 工具的"入口兜底"把它记成 1 条失败而没崩整轮）。
            "torch_cuda_version": getattr(getattr(self._torch, "version", None), "cuda", None),
            "device_count": self.device_count(),
            # 与 `_capabilities` 同一套键名（防止"手写第二份键名清单"造成漂移）
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
            "error_grading": dict(self._grading_paths),
            "bounded_sync_scope": "主机侧等待（event.wait_host）真有界；流同步为超时上报语义",
            "vendor_discriminator": [
                "设备名含 `PPU-`（实测 `get_device_name(0) == 'PPU-ZW810E'`）",
                "驱动库 `libcuda.so.1` 内含 `pccl` 字样（`libtorch_cuda.so` 亦含）",
                "集合通信库 `libpccl.so.2`（torch 侧后端名仍为 `nccl`，见下）",
                "宿主工具 `ppu-smi`（且 `nvidia-smi` 是其符号链接）",
                "设备节点 `/dev/alixpu_ppu<N>` 与内核模块 `alixpu`",
            ],
            "known_upstream_defects": [
                "错误只透出**错误名**（如 `hggcErrorInvalidValue`），**无数值码** ⇒ 无法建码表，"
                "`error_map` 如实不声明（**已确认不具备**）",
                "`cuDevicePrimaryCtxReset_v2` 返回 rc=0 但会破坏本进程设备可用性 ⇒ `recovery_real` 如实不声明",
                "`cuStreamCreateWithPriority` 仅接受 flags∈{0,1}（2/4 返回 INVALID_VALUE）",
            ],
            "dist_backend": {
                "name": "nccl",
                "note": ("torch 侧后端名 = `nccl`（实测 `init_process_group('nccl')` 可用、"
                         "2 卡 all_reduce 通过）；⚠️ **底层库叫 PCCL** —— "
                         "**库名 ≠ 后端名**，拼字符串时勿混。"
                         "`pccl`/`hccl`/`ucc` 均为 `Unknown backend type`，`xccl` 未编译进本 torch。"),
            },
            "known_issues": self.known_issues(),
        }


def _l3():
    from ...api.errors import ErrorCategory
    return ErrorCategory.L3_EXECUTION


def build() -> PpuBackend:
    return PpuBackend()
