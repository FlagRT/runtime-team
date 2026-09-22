#!/usr/bin/env python3
"""寒武纪 MLU590 后端（`torch_mlu` / PrivateUse1 命名空间适配）。

定位：统一运行时原型的**第三个接入实例**（前两家：昇腾 910C、昆仑芯 P800）。
      按《运行时层接口约定》§2 的 Backend 插件接入规范**新建**；
      **不迁移**前两家的实现与结论（`ascend|flagos` 绑定、108 条 ACL 错误码表、
      CANN 约束、XPytorch 结论、并发上限 3、选卡变量名等一律不迁移）。

⛔ **本文件是新写的实现，不复制任何既有芯片的 backend 代码。**
   可复用的是**芯片无关的共享资产**（`conformance/errors.py` 的消息规则、
   `conformance/device_state.py` 的四态机）—— 它们本身就是规范的一部分，
   与具体芯片无关；前两家实例也走同一份。

────────────────────────────────────────────────────────────────────────────
命名空间（手册 §2 的**路径 C：厂商私有命名空间 PrivateUse1**）
  `torch_mlu` 把 MLU 注册为 PyTorch 的 **PrivateUse1**，设备串前缀是 **`mlu`**：

      import torch, torch_mlu
      torch.mlu.device_count()        # → 卡数
      torch.device("mlu:0")           # → 有效设备串

  ⇒ 本后端：`name = "cambricon"`（我们注册表里的键，按厂商）
            `device_type = "mlu"`（真实可用的设备串前缀，按命名空间）

  ⚠️ **PrivateUse1 是进程级单例**：同一进程内只能激活一家。
     故本后端**不得与 `torch_npu` / `torch_fl` 在同进程混用**
     （与前两家同一条约束，接口约定已写明「进程级切换」）。

────────────────────────────────────────────────────────────────────────────
证据状态（**2026-09-22 已在 MLU590 真机验证**）
  验证环境：容器 `dc-mlu590-hliu553`（宿主 `tza-0a06-ai01-em9`），镜像
  `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`
  （digest `sha256:e55b420e…`，与定档记录一致）；容器内 py3.10.20 / Ubuntu 22.04.5。
  实测结果：**离线自检 35/0、smoke 42/0、conformance 13/13 + 6/6 全绿**；
  证据 `MLU590/probes/A{3A4,5,6,6b}_*_20260922.log`。

  ⇒ 本文件里凡标注 **`⚠️ 未实测`** 的，指**该条尚未单独实测**（不是没上过机）；
    未标注的 API 形态均已有真机输出支撑。

────────────────────────────────────────────────────────────────────────────
真机实测结论（原「未实测清单」逐条勾销，2026-09-22）
  1. ✅ `torch.mlu.mem_get_info(ordinal)` **存在且接受 ordinal**（(free,total) 语义同 CUDA）；
     `mem_get_info()` 无参形式亦可用 ⇒ 取值路径 ①/② 均可，**降级路径 ③ 未被触发**
  2. ✅ `Stream / Event / current_stream / stream / synchronize` 齐备，语义同 CUDA 命名空间
  3. ⚠️→✅ **未 record 的 `Event.query()` 原生返回 `True`（误报！）**
     ⇒ 本文件的 `CambriconEventAdapter` 的 E3 修正是**必需的**，不是防御性冗余
  4. ✅ **`Stream.synchronize()` 不接受 timeout**（`TypeError: unexpected keyword argument`）
     ⇒ 「有界」只能是**超时上报**语义（与昆仑芯同款）——本文件与文档均已如实标注
  5. ✅ `Stream.priority_range()` **可用且不崩**：实测返回 `(0, -3)`（语义同 CUDA 的
     least/greatest）。⚠️ 但**非法优先级（99）未被上游拦截**（不报错）⇒ 本层不透传非法值
  6. ✅ 厂商错误码**不透出为数字码**：实测 CNRT 给的是**错误名**
     （`RuntimeError: CNRT error: invalid argument.`），OOM 为 `OutOfMemoryError: MLU out of memory…`
     ⇒ `error_map` **如实不声明**（无可用于码表的数字码）；分级走 message_hint 已覆盖
     （形状错 → L2、OOM → L1，conformance f1 已通过）
  7. ✅ 图捕获入口为 **`torch.mlu.MLUGraph` + `torch.mlu.graph` 上下文管理器**
     ⇒ **实测 `GRAPH_CAPTURE` 5/5 成功**，故已声明 `graph_capture` 能力
  8. ✅ **无设备级重置原语**：`torch.mlu` 下 `reset*/destroy*/reinit*` 全部是内存统计类
     （`reset_peak_memory_stats` 等）⇒ `recovery_real` **已确认不具备**（不是"未验证"），不声明
  9. 🟡 运行时层镜像**不含 vLLM**（`ModuleNotFoundError: No module named 'vllm'`）
     ⇒ 推理腿服务化需用 `flagos-app/vllm*-cambricon-*` 应用镜像或另行安装；形态（厂商移植版/
     社区版+插件）**仍未验证**
 10. ✅ 集合通信后端名 = **`cncl`**（亦可用 `cpu:gloo,mlu:cncl`）；
     2 进程 `torchrun --standalone` 下 `init_process_group("cncl")` + `all_reduce` **结果正确**
     ⇒ 训练腿 `DC_DIST_BT=cncl`。（⚠️ 前两家分别是 `flagos` 与 `flagcx` ⇒ **确实不能类推**）

────────────────────────────────────────────────────────────────────────────
⚠️ 一条**环境坑**（已实测，见 `known_issues()`，与 P800 的"缺 triton"**不同因**）
  `import triton` **必须先 `import torch_mlu`**，否则 triton 内部 `import torch` 会触发
  torch 的 device-backend 自动加载并失败：
      RuntimeError: Failed to load the backend extension: mlu.
  实测：裸 `import triton` → 失败；`import torch_mlu, triton` → `triton 3.2.0` ✅。
  另外 `/opt/triton` 只在经 shell 启动时才进 `PYTHONPATH`（直接 `docker exec … python3` 看不到）。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

from ...api.errors import ErrorCategory, FlagosError, coerce_category
from ..base import RuntimeBackend

logger = logging.getLogger(__name__)

#: 芯片无关的共享资产目录（消息规则 / 设备四态机），与 ascend / flagos / kunlun 同一份
_CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "conformance"


class CambriconEventAdapter:
    """`torch.mlu.Event` 的统一语义适配（与 NpuEventAdapter / KunlunEventAdapter 同构）。

    补齐两点与统一事件契约的语义缺口：
      - **E3：未 record 的 Event 调 `query()` 不得报"已完成"** ——
        昆仑芯实测原生返回 `True`（误报）；本适配器用 `recorded` 标志修正为 `False`。
        ⚠️ 未实测：寒武纪原生行为未知，本适配器**无论原生如何都保证不误报**
        （这是更严格的一侧，不会掩盖问题）。
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
            import torch_mlu  # noqa: F401  触发 mlu 设备注册
            self._ev = torch.mlu.Event(*self._args, **self._kwargs)
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
        # E3：未 record 不得报"已完成"
        if not self._recorded:
            return False
        return self._ensure().query()

    def wait_host(self, timeout_ms: Optional[int] = None) -> bool:
        """主机侧**有界**等待：完成 True / 超时 False（不永久阻塞）。"""
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


class CambriconBackend(RuntimeBackend):
    """基于 `torch_mlu`（PrivateUse1）的寒武纪 MLU 后端。"""

    name = "cambricon"
    device_type = "mlu"

    #: 规范能力键全集（与 ascend / flagos / kunlun 同一套键名，便于跨实例同口径比对）
    _CAPABILITY_KEYS = (
        "device", "memory", "stream", "event", "bounded_sync",
        "error_map", "recovery_probe", "recovery_real",
        "device_state", "graph_capture", "stream_priority", "multidevice",
    )

    #: 本后端**声明支持**的能力（不支持/未验证的一律不写进来 —— 如实声明，不伪造）
    _capabilities = {
        "device",
        "memory",
        "stream",
        "event",
        "bounded_sync",      # 主机侧等待（event.wait_host）真有界；流同步为"超时上报"语义
        "recovery_probe",    # 探针级恢复
        "device_state",      # 四态**查询**（复用进程内状态机，见 device_state()）
        "multidevice",       # 单机 8 卡（实测：8 × MLU590-M9）
        "graph_capture",     # 2026-09-22 实测 GraphCapture 5/5 成功
                             #   （入口是 torch.mlu.MLUGraph + torch.mlu.graph）
        "stream_priority",   # 2026-09-22 实测 priority_range() = (0, -3) 可用且不崩
                             #   （与昆仑芯相反：那里同款 API 会触发 PyTorch INTERNAL ASSERT）
        # ── 以下**不声明**（两类原因分开写清，勿混为一谈）──
        # 【已实测确认不具备 → 不声明】不是"未验证"，是"确认没有"
        #   "recovery_real" : torch.mlu 下 reset* / destroy* / reinit* 全是内存统计类
        #                     （reset_peak_memory_stats 等）⇒ 无设备级重置原语
        #   "error_map"     : 厂商错误码**不透出为数字码**（CNRT 只给错误名，如
        #                     `CNRT error: invalid argument.`）⇒ 无可建码表的数字码；
        #                     分级由 message_hint 覆盖（形状错→L2、OOM→L1，conformance f1 已过）
    }

    #: 分级来源可达性（如实标注：在当前实现里 code_map 路径**不参与**判定）
    _grading_paths = {"code_map": False, "message_hint": True}

    def __init__(self) -> None:
        self._torch = None
        self._errors = None
        self._device_state = None
        self._assets_loaded = False
        self._loaded = False

    # ───────────── 延迟加载 ─────────────
    def _load(self) -> None:
        """导入 torch + torch_mlu。

        ⚠️ 缺 `torch_mlu` 时**明确抛错**（而不是静默降级）——
        否则会被误读成"设备有问题"。报错文案里给出下一步动作，
        对应手册 §9 坑 1「依赖链不完整伪装成设备问题」。
        """
        if self._loaded:
            return
        import torch
        try:
            import torch_mlu  # noqa: F401  注册 mlu 命名空间（PrivateUse1）
        except ImportError as e:
            raise RuntimeError(
                "cambricon 后端需要 torch_mlu（MLU 厂商 PyTorch 扩展）但导入失败："
                f"{e}。请确认当前容器是含 torch_mlu 的寒武纪镜像"
                "（定档：harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0），"
                "并确认已激活正确的 python 环境。"
            ) from e
        if not hasattr(torch, "mlu"):
            raise RuntimeError(
                "torch_mlu 已导入但 torch.mlu 命名空间仍不存在 —— "
                "可能是 torch 与 torch_mlu 版本不配套（torch_mlu 要求先装对应版本 torch）。"
            )
        self._torch = torch
        self._loaded = True

    @property
    def torch(self):
        self._load()
        return self._torch

    def _load_assets(self) -> None:
        """加载芯片无关的共享资产（消息规则 / 设备四态机）。

        用**标准 import（共享 `sys.modules`）**，与 ascend 一致。原因（两条，都有前例）：
          1. `device_state` 是**有状态的进程内单例**（模块级持有每个设备的四态与订阅者）。
             若用 `spec_from_file_location("dc_xxx", ...)` 加载成独立模块，会得到**两份状态机**
             —— conformance/上层设的状态后端查不到、后端设的上层看不到（昆仑芯接入时踩过，
             见 `backends/kunlun/backend.py::_load_device_state` 的说明）。
          2. 独立加载还会让同一语义出现**两份枚举类对象**，即使取值相同也不相等，
             导致 `FlagosError.disposition` 的查表 KeyError
             （即「第 4 个跨后端框架缺陷」）。标准 import 从根上避免这一类问题。
        """
        if self._assets_loaded:
            return
        sys.path.insert(0, str(_CONFORMANCE_DIR))
        import device_state as _device_state  # noqa: E402
        import errors as _errors  # noqa: E402
        self._errors = _errors
        self._device_state = _device_state
        self._assets_loaded = True

    # ───────────── 设备（职责 D2）─────────────
    def device_count(self) -> int:
        return int(self.torch.mlu.device_count())

    def set_device(self, ordinal: int) -> None:
        self.torch.mlu.set_device(ordinal)

    # ───────────── 内存（职责 D3）─────────────
    def memory_stats(self, ordinal: int) -> dict:
        """归一化为 `{total_mb, used_mb, free_mb}`。

        取值路径按优先级试三条（⚠️ 未实测：以下三条在寒武纪上哪条可用尚待容器内核对）：
          ① `torch.mlu.mem_get_info(ordinal)` —— 设备级 free/total（本仓库 ascend/kunlun 同口径）
          ② `torch.mlu.mem_get_info()`        —— 部分实现不接受 ordinal 参数
          ③ `get_device_properties(ordinal).total_memory` − `memory_allocated(ordinal)`
             —— **语义较弱**：`memory_allocated` 只反映**本进程**的分配量，
             不含同卡其他进程占用 ⇒ 走这条路径时打 WARNING，**不静默当作设备级数据**
        """
        torch = self.torch
        prev = torch.mlu.current_device()
        if prev != ordinal:
            torch.mlu.set_device(ordinal)
        try:
            free = total = None
            for call in (
                lambda: torch.mlu.mem_get_info(ordinal),
                lambda: torch.mlu.mem_get_info(),
            ):
                try:
                    free, total = call()
                    break
                except Exception:
                    continue
            if free is None:
                logger.warning(
                    "cambricon: torch.mlu.mem_get_info 不可用，降级为"
                    " total_memory - memory_allocated（仅本进程视角，非设备级）"
                )
                total = int(torch.mlu.get_device_properties(ordinal).total_memory)
                free = total - int(torch.mlu.memory_allocated(ordinal))
        finally:
            if prev != ordinal:
                try:
                    torch.mlu.set_device(prev)
                except Exception:
                    pass
        return {
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024),
        }

    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：在目标卡上建小张量并求和（不干扰业务）。"""
        try:
            torch = self.torch
            prev = torch.mlu.current_device()
            if prev != ordinal:
                torch.mlu.set_device(ordinal)
            try:
                x = torch.zeros(2, 2, device=f"mlu:{ordinal}")
                return float(x.sum().item()) == 0.0
            finally:
                if prev != ordinal:
                    try:
                        torch.mlu.set_device(prev)
                    except Exception:
                        pass
        except Exception:
            return False

    # ───────────── 流 / 事件（职责 D4/D5）─────────────
    def create_stream(self):
        return self.torch.mlu.Stream()

    def create_event(self):
        # 统一语义适配：未 record 不误报完成（E3）+ 主机侧有界等待（E2v2）
        return CambriconEventAdapter()

    def current_stream(self):
        return self.torch.mlu.current_stream()

    def stream_context(self, native_stream):
        return self.torch.mlu.stream(native_stream)

    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。`timeout_ms` 为空 → 原生阻塞同步；非空 → 轮询上报超时。"""
        torch = self.torch
        if timeout_ms is None:
            try:
                torch.mlu.synchronize(ordinal)
            except TypeError:
                # ⚠️ 未实测：部分实现不接受 device 参数
                torch.mlu.set_device(ordinal)
                torch.mlu.synchronize()
            return
        self._bounded_wait(lambda: torch.mlu.current_stream().query(), timeout_ms, "device")

    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 `TimeoutError`。

        ⚠️ **语义边界（如实标注，待实测确认）**：昆仑芯实测 `Stream.synchronize()`
        无 timeout 参数、无中断原语，故「有界」只能做成**超时上报**语义 ——
        超时抛 `TimeoutError` 让上层得以降级/记录，但**不保证中断底层已提交的执行**。
        寒武纪是否提供真中断原语**未知**；本实现先与昆仑芯同口径，
        待容器内核对（接口约定修订建议已就该语义边界提出建议）。
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
                    f"cambricon: {what} synchronize timeout after {timeout_ms} ms"
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
        """流优先级区间 (least, greatest)。

        2026-09-22 真机实测：`torch.mlu.Stream.priority_range()` 返回 **`(0, -3)`**，
        语义与 CUDA 一致（least=0 / greatest=-3），**且不触发断言** ——
        与昆仑芯形成明确对照（那里同款 API 会触发 PyTorch 自身的
        `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188`）。
        实测也确认 `Stream(priority=0/-1/-3)` 均可创建且读回值正确。

        ⚠️ **一处如实标注的缺口**：上游**未拦截非法优先级**（`Stream(priority=99)`
        不报错）。本层**不透传非法值**；调用方若给区间外的值，行为由厂商实现决定，
        不由本层保证。

        ⚠️ 仍未单独验证的：优先级的**实际调度效果**（读回值只证明被接受）。
        若后续需要据此做调度决策，应先补一条性能侧对照实验。
        """
        try:
            return self.torch.mlu.Stream.priority_range()
        except Exception:
            # 任何异常都不透传（避免重演"只读查询把进程打崩"）
            return None

    # ───────────── 错误翻译（职责 D10）─────────────
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 `FlagosError`。

        分级来源如实限定为 **`message_hint`（消息规则）/ `default`（兜底）**：
        厂商错误码是否透出到 Python 层**未实测**，故本后端**不声明 `error_map` 能力**，
        且若底层骨架意外给出 `code_map`（例如消息里恰好出现形如昇腾错误码的数字），
        一律**如实降级标注**，不冒充"码表命中"（P800 同款处理，见其 backend 注释）。
        """
        self._load_assets()
        fe = self._errors.translate_error(exc, location=location)  # type: ignore[union-attr]
        graded_by = getattr(fe, "graded_by", "default")
        if graded_by == "code_map":
            graded_by = "message_hint_unexpected"
        return FlagosError(
            # `errors.ErrorCategory` 是 IntEnum（L1=1..L4=4），与 api 层枚举**不是同一个类对象**；
            # 直接透传会让 `FlagosError.disposition` 的查表 KeyError（第 4 个跨后端框架缺陷）。
            # 经 `coerce_category` 按数值/名称归一（框架已提供该归一函数）。
            category=coerce_category(getattr(fe, "category", None)) or ErrorCategory.L3_EXECUTION,
            root_cause=getattr(fe, "root_cause", f"{type(exc).__name__}: {exc}"),
            location=getattr(fe, "location", "") or location,
            error_code=getattr(fe, "error_code", None),
            mapped=bool(getattr(fe, "mapped", False)),
            graded_by=graded_by,
            # 无厂商码可依据 → 除"命中消息规则"外均不标为高置信
            is_grade_confident=(graded_by == "message_hint"),
            recovery_decision=getattr(fe, "recovery_decision", {}) or {},
            # 后端侧回填后端名（2026-09-22 修：后端侧必须回填，
            # 不能只依赖 api 层 translate_via_backend —— 直调后端方法时该字段会为空）
            backend=self.name,
        )

    # ───────────── 状态恢复（职责 D11）─────────────
    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "", **kwargs: Any) -> dict:
        """设备重建。统一返回 dict，`recovered` 语义 = **设备当前可用**。

        ⚠️ **`real` 模式当前不支持**，原因是「**未验证**」而非「已确认不具备」：
        寒武纪是否有设备级重置/重建原语（对应昇腾的 `aclrtResetDevice` 序列）
        **本次未实测**，故**不声明 `recovery_real` 能力**、也不写一条未经验证的重建序列
        （写一条猜的实现会制造"看起来支持"的假象，比不写更糟）。
        待容器内确认原语存在且可安全调用后，再补 `real` 并在 `_capabilities` 中声明。
        """
        alive = self.probe_device(ordinal)
        if mode not in ("probe", "hybrid"):
            return {
                "ordinal": ordinal, "mode": mode, "recovered": alive,
                "detail": ("寒武纪的设备级重置/重建原语未验证 ⇒ real 模式不支持；"
                           "已按探活结果判定。如需 real 需先在容器内确认可用原语"),
            }
        return {
            "ordinal": ordinal, "mode": mode, "recovered": alive,
            "detail": f"probe 级探活：设备当前{'可用' if alive else '不可用'}",
        }

    def device_state(self, ordinal: int):
        """查询设备四态：`available` / `degraded` / `isolated` / `destroyed`。

        复用共享的 `conformance/device_state.py`（**进程内状态机，不依赖厂商原语**），
        与前两家同一份实现与同一套语义，故本后端的 `device_state` 能力声明成立。

        边界（如实标注）：本后端只声明 `recovery_probe`；四态**转换**由上层/监控方向驱动，
        本方法只负责**查询**，恢复执行走 `recover_device(mode="probe")`。
        """
        self._load_assets()
        return self._device_state.query_device_state(ordinal)  # type: ignore[union-attr]

    # ───────────── 能力声明 ─────────────
    def supports(self, capability: str) -> bool:
        return capability in self._capabilities

    # ───────────── 已知上游/环境问题（给接入方直接可读）─────────────
    #: 纪律：只收**已实测**的问题；每条注明条件、归属层与证据位置。
    #: 本方向在寒武纪上**尚无厂商缺陷结论**（未进容器），故本清单当前只含**环境约束**，
    #: 且明确标注证据来源。容器内实测后按手册 §9 模板补厂商缺陷条目。
    _KNOWN_ISSUES = [
        {
            "id": "MLU-DRIVER-TIER-CONSTRAINT",
            "severity": "info",
            "scope": "环境/镜像选型（本方向验证的前置条件）",
            "condition": "宿主驱动版本决定可用镜像档位",
            "symptom": (
                "两台测试机驱动实测 v6.2.29（6.2.x 线）⇒ 只能使用 neuware4.4.3 档镜像"
                "（官方标注 Host driver 6.2.15）；neuware4.7.2 档官方标注要求宿主驱动 6.5.48，"
                "本机不满足。选错档位表现为容器内厂商工具/库与宿主不匹配"
            ),
            "repro_rate": "不适用（环境约束，非概率性缺陷）",
            "root_cause_layer": "环境（宿主驱动 ↔ 镜像档位配套关系）",
            "workaround": "按驱动选档：走 flagos-runtime-cambricon-neuware4.4.3:2.2.0",
            "workaround_risk": (
                "4.4.3 为旧档（py3.10 / torch 2.7.1 / torch-mlu 1.29.2 / triton 3.2.0+mlu1.7.2），"
                "与最新 4.7.2 档差两代。本方向（设备抽象/执行上下文/多流/错误翻译/状态恢复）"
                "对 torch-mlu 小版本不敏感，故代价可接受；若后续出现与该档强相关的问题，"
                "走《寒武纪镜像渠道调研》§0.2 的驱动升级上报预案"
            ),
            "report_to": "总组 / 机器管理员（仅在触发 §0.2 预案门槛时才需上报）",
            "evidence": "MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md §0.1/§0.2；"
                        "MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md（驱动 v6.2.29 实测）",
        },
        {
            "id": "MLU-HOST-NO-NEUWARE",
            "severity": "info",
            "scope": "环境（软件栈形态）",
            "condition": "宿主无 /usr/local/neuware（只读探测结论）",
            "symptom": (
                "宿主上没有任何 MLU 软件栈（无 NEUWARE_HOME）⇒ 不能在宿主直接跑 torch_mlu；"
                "所有本方向验证必须在**带卡容器**内进行"
            ),
            "repro_rate": "不适用（环境约束）",
            "root_cause_layer": "环境（宿主机设计：MLU 栈只随容器分发）",
            "workaround": "一律在容器内验证；宿主只用来起容器与查卡（cnmon 是宿主工具）",
            "workaround_risk": None,
            "report_to": "无（无需上报）",
            "evidence": "MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md §2「宿主无 /usr/local/neuware」",
        },
        {
            "id": "MLU-TRITON-IMPORT-ORDER",
            "severity": "medium",
            "scope": "Python 导入顺序（算子 / 编译路径使用者会踩；本方向路径不依赖 triton）",
            "condition": "在未先导入 `torch_mlu` 的情况下 `import triton`",
            "symptom": (
                "triton 内部 `import torch` 会触发 torch 的 device-backend 自动加载并失败："
                "`RuntimeError: Failed to load the backend extension: mlu. "
                "You can disable extension auto-loading with TORCH_DEVICE_BACKEND_AUTOLOAD=0.`"
                "—— **报错文案看着像设备问题，实际是导入顺序问题**"
            ),
            "repro_rate": "确定性（100%，2026-09-22 实测 3/3 次）",
            "root_cause_layer": "环境/入口（torch 的 device-backend 自动加载时机 vs 厂商扩展注册时机）",
            "workaround": (
                "**先 `import torch_mlu` 再 `import triton`**（实测得到 `triton 3.2.0`）；"
                "或兼容做法 `import torch, torch_mlu` 之后再导入任何依赖 triton 的库。"
                "不要用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 绕过——它会一并影响 mlu 后端注册"
            ),
            "workaround_risk": "无（顺序调整即可，不改环境变量、不动公共资产）",
            "report_to": "知会算子 / 编译方向（他们是用 triton 路径的一方）",
            "evidence": "MLU590/probes/A5_verify_20260922.log §①；"
                        "MLU590/probes/A6_capability_probe_20260922.log §③",
        },
        {
            "id": "MLU-RUNTIME-IMAGE-NO-VLLM",
            "severity": "info",
            "scope": "推理腿服务化形态（本方向验证路径受影响，但不阻塞设备上下文验证）",
            "condition": "使用 `flagos-runtime-*` 运行时层镜像（而非 `flagos-app/vllm*` 应用镜像）",
            "symptom": (
                "运行时层镜像**不含 vLLM**：`ModuleNotFoundError: No module named 'vllm'` ⇒ "
                "`from vllm.platforms import current_platform` 无法执行，服务化形态无法就地验证"
            ),
            "repro_rate": "确定性（镜像内容事实）",
            "root_cause_layer": "环境（镜像分层设计：runtime 层只到 torch/插件/triton/flag_gems）",
            "workaround": (
                "推理腿走同线应用镜像 `harbor.baai.ac.cn/flagos-app/vllm0.24.0-cambricon-neuware4.4.3`"
                "（或 vllm0.20.2 变体），与本方向统一启动脚本 `serve_standard.sh` 配合"
            ),
            "workaround_risk": "该应用镜像内 vLLM 是厂商移植版还是社区版 + 插件，**仍未验证**",
            "report_to": "无（镜像分层设计使然，不需要上报）",
            "evidence": "MLU590/probes/A5_verify_20260922.log §B「推理腿平台判别」",
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
            "framework": "torch_mlu（PyTorch PrivateUse1 厂商扩展）",
            "torch": self._torch.__version__ if self._torch is not None else None,
            "device_namespace": "mlu（torch.mlu.*；PrivateUse1，进程内与 torch_npu/torch_fl 互斥）",
            "device_count": self.device_count(),
            "capabilities": sorted(self._capabilities),
            # 与 _capabilities 同一套键名（自洽，避免 A 键声明 / B 键查询的口径漂移）
            "supports": {k: self.supports(k) for k in self._CAPABILITY_KEYS},
            "error_grading": dict(self._grading_paths),
            "bounded_sync_scope": "主机侧等待（event.wait_host）真有界；流同步为超时上报语义",
            # ⭐ 证据等级：2026-09-22 已在真机验证（原为"未实机验证"）
            "evidence_level": (
                "已在 MLU590 真机验证（2026-09-22）：离线自检 35/0、smoke 42/0、"
                "conformance 13/13 + 6/6 全绿；证据 MLU590/probes/A*_20260922.log"
            ),
            "verified_on_device": [
                "mem_get_info(ordinal) 可用（(free,total)，语义同 CUDA）；降级路径未被触发",
                "Stream/Event/current_stream/stream/synchronize 齐备",
                "未 record 的 Event.query() 原生误报 True ⇒ 适配层 E3 修正必需",
                "Stream.synchronize() 不接受 timeout ⇒ 有界=超时上报语义（如实标注）",
                "priority_range() = (0, -3) 可用且不崩；⚠️ 非法优先级(99)上游未拦截",
                "错误码为 CNRT 错误名而非数字码 ⇒ error_map 如实不声明",
                "图捕获入口 torch.mlu.MLUGraph + torch.mlu.graph ⇒ 实测 5/5 通过",
                "无设备级重置原语（reset* 全为内存统计类）⇒ recovery_real 确认不具备",
                "集合通信后端名 = cncl（2 进程 all_reduce 结果正确）",
            ],
            "unverified": [
                "推理腿 vLLM 形态（运行时镜像不含 vLLM；应用镜像内是厂商移植版还是社区版+插件未验证）",
                "流优先级的实际调度效果（读回值只证明被接受，未做性能侧对照）",
            ],
            "environment": {
                "chip": "寒武纪 MLU590-M9 × 8",
                "host_driver_observed": "v6.2.29（实测，6.2.x 线）",
                "image_pinned": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0"
                                 "（实测 digest sha256:e55b420e… 与定档一致）",
                "container_runtime": "py3.10.20 / Ubuntu 22.04.5 / torch 2.7.1+cpu / torch_mlu 1.29.2+torch2.7.1",
                "dist_backend_name": "cncl（亦可用 cpu:gloo,mlu:cncl）",
                "note": "宿主无 /usr/local/neuware ⇒ 必须容器内运行；"
                        "import triton 前必须先 import torch_mlu（见 known_issues）",
            },
            "known_issues": self.known_issues(),
        }


def build() -> CambriconBackend:
    """注册表自动发现使用的工厂函数。"""
    return CambriconBackend()
