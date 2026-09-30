#!/usr/bin/env python3
"""
Backend 插件接口规范 v0.1（runtime/backends/base.py）

对应职责（本方向负责子层）：
  - 设备上下文：设备句柄与生命周期、内存句柄与生命周期、执行句柄
  - 多流 Stream：Stream / Event 抽象与同步语义
  - 错误码翻译：厂商错误码 → L1-L4 统一分级
  - 设备状态恢复：四态监控 + 五段式恢复

设计原则：
  1. 厂商已有原生 PyTorch 扩展（torch_npu / 昆仑芯 SDK）负责算子分发，
     本层**不重复实现算子分发**，只统一"设备抽象与流语义"。
  2. 新增一家芯片 = 实现本接口 + 跑通 conformance。这是"统一接口"的可验收定义。
  3. v0.1 为原型期，允许破坏性变更（每季度评审一次）。
  4. **能力如实声明**：没实测到的一律不写进 `_capabilities`；未声明的能力被调用时
     **如实报错**（`NotImplementedError`），**不得静默退化**。

参考：vendor 插件目录模式（backends/<vendor>/），
      但本层位于其上层——设备抽象层而非算子 dispatch 层。
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..api.errors import FlagosError


#: 设备四态**规范 token**（= `conformance/device_state.py::DeviceState` 的 `.value`）。
#: 契约承诺：`recover_device()["state"]` **必须**是本元组取值之一（见接口约定 §1.5）。
DEVICE_STATE_TOKENS = ("available", "degraded", "isolated", "destroyed")

#: 内存句柄的**公共字段**（上层只允许依赖这些键；厂商指针一律**不进句柄**）。
#: 2026-09-29（工作包 B）：句柄不含 `ptr` —— 暴露厂商指针等于绕过 `.native` 逃生舱纪律，
#: 会把"换芯片不改代码"的主张悄悄破坏掉。指针只留在本层的内部登记表里。
MEMORY_HANDLE_KEYS = ("handle_id", "kind", "backend", "ordinal", "size_bytes")

#: 上下文句柄的**公共字段**（与内存句柄**同一套 `handle_id` 命名**）。
#: ⚠️ 2026-09-29：初版曾把上下文句柄的 id 字段写成 `context_id`，而取句柄的公共口只认
#: `handle_id` ⇒ **上下文永远销毁不掉**（判据当场抓到）。这正是台账 ⑫b 家族的坑：
#: **同一概念在本层的对外形态必须唯一** ⇒ 统一为 `handle_id` + 用 `kind` 区分种类。
CONTEXT_HANDLE_KEYS = ("handle_id", "kind", "backend", "ordinal")


def state_token(state) -> str:
    """把设备四态规范化为**契约 token 字符串**。

    2026-09-29 修（工作包 A 实验暴露）：三家 `recover_device` 原写 `str(state)`，
    而 `DeviceState` 是 `enum.Enum`（**不是** `str, Enum` / `StrEnum`）
    ⇒ `str(member)` 给的是 `'DeviceState.AVAILABLE'`，**不是**四态规范取值；
    下游按契约比较 `state == "available"` 会失败。
    同模块 `DeviceStatus.snapshot()` 用的却是 `.value` ⇒ **同仓两套约定**，此处归一。
    """
    v = getattr(state, "value", None)
    return str(v) if v is not None else str(state)


class RuntimeBackend(ABC):
    """厂商后端插件接口 v0.1。

    实现者需提供：name / device_type 两个类属性，以及下列全部抽象方法。
    """

    #: 后端名，用户通过 runtime.use(name) 选择（如 "ascend" / "kunlun"）
    name: str = ""
    #: 用户可见的设备串前缀（如 "npu" / "xpu"）
    device_type: str = ""

    # ───────────────────────── 设备（职责 D2）─────────────────────────

    @abstractmethod
    def device_count(self) -> int:
        """可用设备数量。"""

    @abstractmethod
    def set_device(self, ordinal: int) -> None:
        """绑定当前进程/线程的默认设备。"""

    # ───────────────────────── 内存（职责 D3）─────────────────────────

    @abstractmethod
    def memory_stats(self, ordinal: int) -> dict:
        """返回 {"total_mb": int, "used_mb": int, "free_mb": int}。

        2026-09-29（工作包 B-2）：**在保持上述三键不变的前提下**，鼓励额外提供
        `"allocated_mb"`（分配器视角：本进程已分配量），并声明能力 `memory_alloc_stat`。
        取不到时**省略该键**，不得填 0 冒充。三键语义与口径以本方向为准（显存方向采集）。
        """

    # ─────────────── 内存句柄与生命周期（职责 D3 · 工作包 B-1）───────────────
    #
    # 设计要点（三条，均源自 2026-09-29 的真机实测）：
    #   1. **只做句柄语义**，不做池化 / 碎片 / 峰值 / 扩容策略（那些属显存方向）；
    #   2. **成对释放**：句柄必须经 `free()` 释放；**重复释放 / 非本层句柄必须报错**；
    #   3. ⭐ **把厂商的"静默"变成显式错误**：实测 pyACL 的 `acl.rt.free()` 对
    #      **二次释放/非法指针静默返回 0**（不报错）；昆仑芯在 torch 层报错，但只覆盖
    #      它自己的分配器。统一层**统一登记句柄、统一拦截**，跨两家行为一致。

    #: 句柄登记表：`handle_id -> {"handle": 公共句柄, "ptr": 厂商指针}`（**每实例独立**）
    _alloc_handles: Optional[dict] = None

    def allocate(self, size_bytes: int, ordinal: int = 0) -> dict:
        """申请一段设备内存，返回**统一内存句柄**（公共字段见 `MEMORY_HANDLE_KEYS`）。

        - 未声明能力 `memory_alloc` 的后端 ⇒ **如实报错**（不静默退化为空操作）；
        - `size_bytes` 必须为正整数，否则 `ValueError`（参数类问题不该走到厂商层）；
        - 释放必须走 `free()`；**重复释放会报错**（见 `free`）。
        """
        if not self.supports("memory_alloc"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'memory_alloc' ⇒ 不支持内存句柄语义"
                "（**如实不具备**，不等于已确认该芯片不具备；需真机探测后补声明）。"
                "调用前请先 `supports('memory_alloc')` 判定。")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ValueError(f"size_bytes 必须为正整数，收到 {size_bytes!r}")

        ptr = self._alloc_raw(int(size_bytes), int(ordinal))
        seq = int(self.__dict__.get("_handle_seq", 0)) + 1
        self.__dict__["_handle_seq"] = seq
        handle = {"handle_id": seq, "kind": "memory", "backend": self.name,
                  "ordinal": int(ordinal), "size_bytes": int(size_bytes)}
        self._reg("_alloc_handles")[seq] = {"handle": handle, "ptr": ptr}
        return handle

    def free(self, handle) -> None:
        """释放 `allocate()` 返回的句柄。

        **重复释放 / 非本层句柄 ⇒ 报错**（契约要求：不得静默接受）。
        """
        if not self.supports("memory_alloc"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'memory_alloc' ⇒ 不支持内存句柄语义")
        hid = self._handle_id(handle, "_alloc_handles", "内存")
        entry = self._reg("_alloc_handles").pop(hid)
        self._free_raw(entry["ptr"], entry["handle"])

    def memory_handle_count(self) -> int:
        """当前在世的内存句柄数（审计用；泄漏判据的取数入口）。"""
        return len(self._reg("_alloc_handles"))

    def _alloc_raw(self, size_bytes: int, ordinal: int):
        """子类实现：调用厂商原语分配，返回厂商指针（不暴露给上层）。"""
        raise NotImplementedError(f"后端 '{self.name}' 未实现 _alloc_raw")

    def _free_raw(self, ptr, handle: dict) -> None:
        """子类实现：调用厂商原语释放。"""
        raise NotImplementedError(f"后端 '{self.name}' 未实现 _free_raw")

    # ─────────── 设备上下文生命周期（职责 D2/D1 · 工作包 C）───────────
    #
    # 现状澄清（2026-09-29）：原型的 `stream_context` 是**切流**的上下文管理器，
    # 与"设备上下文（context）的创建/销毁"是两件事；而 `recover_device(mode="real")`
    # 内部其实会 destroyContext → ResetDevice → 重建，**执行了但完全不暴露**。
    #
    # 安全契约（重要）：**只允许销毁本层自己创建的上下文句柄**。
    # 对非本层句柄（尤其是进程默认上下文）**一律拒绝并报错** —— 误毁默认上下文
    # 会让整个进程的设备不可用，属不可逆破坏。
    #
    # ⭐ 真机实测（910C · 2026-09-29）：上下文**销毁后**，其上建立的流
    # **当场使用不报错（静默成功）**，直到**进程退出清理阶段**才暴露
    # `The stream is not in the current context` / `Stream destroy failed, … 107003`。
    # ⇒ 本层必须**主动拦截**（见 `check_context_binding`），不能依赖厂商事后暴露。

    #: 上下文登记表：`handle_id -> {"handle": 公共句柄, "ctx": 厂商上下文, "ordinal": n}`
    _ctx_handles: Optional[dict] = None

    def context_create(self, ordinal: int = 0) -> dict:
        """在指定设备上**新建**一个设备上下文，返回统一句柄。

        未声明能力 `context_lifecycle` 的后端 ⇒ 如实报错。
        """
        if not self.supports("context_lifecycle"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'context_lifecycle' ⇒ 不支持设备上下文"
                "生命周期（**如实不具备**：该栈未暴露上下文创建/销毁入口）。")
        ctx = self._ctx_create_raw(int(ordinal))
        seq = int(self.__dict__.get("_ctx_seq", 0)) + 1
        self.__dict__["_ctx_seq"] = seq
        handle = {"handle_id": seq, "kind": "context", "backend": self.name,
                  "ordinal": int(ordinal)}
        self._reg("_ctx_handles")[seq] = {"handle": handle, "ctx": ctx}
        self.__dict__["_current_ctx_id"] = seq        # 新建即成为当前上下文
        self._refresh_ctx_health()
        return handle

    def context_destroy(self, handle) -> None:
        """销毁 `context_create()` 返回的句柄。

        ⚠️ **只接受本层创建的句柄**：对非本层句柄（含进程默认上下文）**拒绝并报错**。
        """
        if not self.supports("context_lifecycle"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'context_lifecycle' ⇒ 不支持设备上下文生命周期")
        hid = self._handle_id(handle, "_ctx_handles", "上下文")
        entry = self._reg("_ctx_handles").pop(hid)
        self._ctx_destroy_raw(entry["ctx"], entry["handle"])
        # 标记为已销毁（用 dict 当集合：`_reg()` 统一返回 dict，避免两套容器语义）
        self._reg("_dead_ctx")[hid] = True
        if self.__dict__.get("_current_ctx_id") == hid:
            self.__dict__["_current_ctx_id"] = None
        self._refresh_ctx_health()

    def context_set(self, handle) -> None:
        """把某个**本层创建的**上下文切换为当前上下文。

        为什么需要它：`context_create` 只负责"造出来"，"切到它上面执行"是另一件事
        （pyACL 的 `create_context` 与 `set_context` 是两个调用）。没有 `set` 就无法做
        "多上下文隔离"的验证，也无法在销毁 A 之后切回 B。

        ⚠️ 只接受本层句柄 —— 不接受厂商原生上下文对象（那属于 `.native` 逃生舱场景）。
        """
        if not self.supports("context_lifecycle"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'context_lifecycle' ⇒ 不支持设备上下文生命周期")
        hid = self._handle_id(handle, "_ctx_handles", "上下文")
        self._ctx_set_raw(self._reg("_ctx_handles")[hid]["ctx"], handle)
        self.__dict__["_current_ctx_id"] = hid
        self._refresh_ctx_health()

    def current_context_id(self):
        """当前生效的**本层上下文** id（非本层上下文时返回 None）。"""
        return self.__dict__.get("_current_ctx_id")

    def note_stream_created(self, native_stream) -> None:
        """登记"该流是在哪个**本层上下文**下创建的"（无则为 None）。

        为什么需要：实测（910C）上下文销毁后，**在其上创建的流"当场使用不报错"**，
        直到**进程退出清理阶段**才暴露 `stream not in current ctx`（107003）
        —— 属"静默延迟暴露"。契约要求「销毁后使用其流必须**明确**」⇒
        本层必须自己把这条关联记下来，才能在使用点上**主动拦截**。
        """
        try:
            self._reg("_stream_ctx")[id(native_stream)] = self.current_context_id()
        except Exception:                                     # noqa: BLE001
            pass

    def check_stream_usable(self, native_stream) -> None:
        """使用点拦截：流所属上下文已销毁 ⇒ **如实报错**（不得静默成功）。"""
        rel = self._reg("_released_streams").get(id(native_stream))
        if rel is not None:
            raise RuntimeError(
                f"该流已被 `release_stream()` 释放（厂商句柄 {int(rel):#x}）⇒ 不可继续使用。"
                "本层在『厂商 C API 建流 + 包装』路径上创建的流由**本层拥有**"
                "（实测：torch 不拥有它 —— 丢弃包装对象后句柄仍可用）"
                "⇒ 用完后须显式 `runtime.release_stream(stream)`。")
        cid = self._reg("_stream_ctx").get(id(native_stream))
        if cid is not None and cid in self._reg("_dead_ctx"):
            raise RuntimeError(
                f"该流创建于已销毁的设备上下文（context_id={cid}）⇒ 不可继续使用。"
                "厂商栈在此处**不会立刻报错**（实测：直到进程退出清理阶段才暴露 107003），"
                "本层按契约在**使用点**如实拦截；请重新创建上下文与流。")

    def could_intercept_stream_binding(self) -> bool:
        """本层是否具备"上下文↔流绑定"的拦截能力（供判据/上报使用）。"""
        return self.supports("context_lifecycle")

    def _refresh_ctx_health(self) -> None:
        """预留钩子：上下文存活面变化后需要刷新时可覆写。"""
        return None

    def context_query(self) -> dict:
        """查询**此刻进程实际生效的设备上下文**（**只读**，不改状态）。

        与 `context_count()` 的分工（本接口存在的理由）：
          · `context_lifecycle` 系接口回答「**本层造了几个**」；
          · `context_query` 回答「**此刻实际在哪个上下文上**」——
            对**外部/框架自建**的上下文同样有意义（哪怕本层一个都没造）。

        为什么必须把两件事分开（2026-09-29 深夜实测，四组判别实验）：
          · 910C：`acl.rt.create_context` 可建多个，本层可管（已声明 `context_lifecycle`）；
          · P800：**平台只允许一个上下文**（已有上下文时第二次 `cuCtxCreate_v2` 返回 `rc=2`），
            而它由 XPytorch/XRE **自建**；统一层若抢先去建，torch 反而起不来
            （实测 `CUDA error: invalid device ordinal`；而**销毁本层建的上下文后 torch 立即恢复**）。
        ⇒ 「上下文由谁拥有」在各栈差异极大。上层不该被迫写分支 ⇒ 本接口把差异
          **写进字段**（`managed_by`），而不是让上层去猜 —— 这正是本层"让分歧显式
          且可消费"的定位。

        返回（**固定键**，便于跨后端比对；取不到的键如实置 None，**不填 0 冒充**）：
          · `queryable`  : 本后端是否具备该查询能力
          · `present`    : 此刻是否存在生效上下文
          · `ordinal`    : 上下文绑定的设备序号（取不到 ⇒ None）
          · `flags`      : 上下文的创建标志（取不到 ⇒ None）
          · `managed_by` : `"unified"`（本层创建）/ `"external"`（厂商或框架自建）/ None
          · `reason`     : 不可查询 / 不存在 / 取不到字段的**具体原因**（便于诊断，不吞掉）
        """
        if not self.supports("context_query"):
            return {"queryable": False, "present": None, "ordinal": None, "flags": None,
                    "managed_by": None,
                    "reason": f"后端 '{self.name}' 未声明能力 'context_query'"
                              "（该栈无上下文只读查询入口）"}
        try:
            raw = self._ctx_query_raw(0)
        except Exception as e:                                # noqa: BLE001
            return {"queryable": False, "present": None, "ordinal": None, "flags": None,
                    "managed_by": None,
                    "reason": f"上下文查询失败：{type(e).__name__}: {e}"}
        if not raw:
            return {"queryable": True, "present": False, "ordinal": None, "flags": None,
                    "managed_by": None,
                    "reason": "此刻无生效上下文（厂商侧尚未建立，或本进程尚未触碰设备）"}
        # `managed_by` 判定：拿厂商上下文对象与**本层登记表**比对（仅比对、不外泄）
        managed = "external"
        for _hid, entry in (self._reg("_ctx_handles") or {}).items():
            if entry.get("ctx") is not None and self._same_ctx(entry["ctx"], raw.get("ctx")):
                managed = "unified"
                break
        return {"queryable": True, "present": True,
                "ordinal": raw.get("ordinal"), "flags": raw.get("flags"),
                "managed_by": managed, "reason": ""}

    def _ctx_query_raw(self, ordinal: int = 0):
        """子类实现：**只读**查询当前生效的厂商上下文。

        返回 `{"ctx": <厂商上下文对象>, "ordinal": int|None, "flags": int|None}`，
        或 `None`（无生效上下文）。
        ⚠️ 返回结构里的 `ctx` **仅供本层比对，不得对外暴露**（不放进任何公开返回）。
        """
        raise NotImplementedError(f"后端 '{self.name}' 未实现 _ctx_query_raw")

    def _same_ctx(self, a, b) -> bool:
        """判断两个厂商上下文对象是否同一（各栈形态不同 ⇒ 子类可覆写）。"""
        try:
            return a == b
        except Exception:                                     # noqa: BLE001
            return False

    def context_count(self) -> int:
        """本层当前在世的设备上下文数（不含进程默认上下文）。"""
        if not self.supports("context_lifecycle"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 'context_lifecycle' ⇒ 无上下文计数")
        return len(self._reg("_ctx_handles"))

    def context_snapshot(self) -> dict:
        """上下文维度**快照**（供 `recover_device` 等上报；不支持时如实置 None）。

        只增不改：不改变任何既有返回契约（见接口约定 §1.5 的只增不改原则）。
        """
        if not self.supports("context_lifecycle"):
            return {"context_supported": False, "context_count": None}
        try:
            return {"context_supported": True, "context_count": self.context_count()}
        except Exception as e:                                    # noqa: BLE001
            return {"context_supported": True, "context_count": None,
                    "context_error": f"{type(e).__name__}: {e}"}

    def _ctx_create_raw(self, ordinal: int):
        """子类实现：调用厂商上下文创建原语。"""
        raise NotImplementedError(f"后端 '{self.name}' 未实现 _ctx_create_raw")

    def _ctx_destroy_raw(self, ctx, handle: dict) -> None:
        """子类实现：调用厂商上下文销毁原语。"""
        raise NotImplementedError(f"后端 '{self.name}' 未实现 _ctx_destroy_raw")

    # ───────────────────── 执行 / 多流 Stream（职责 D4/D5）──────────────────────

    # ───────────────── 流优先级：**能读范围 ≠ 能设置**（2026-09-30）─────────────────
    #
    # 来由（2026-09-30 B2 v2 实测 + 官方文档）：统一 API 若照抄 torch 的构造签名
    # `Stream(priority=…)`，会得到一个"**看起来设了优先级、其实没设**"的流 ——
    # 在 910C 上 `torch.npu.Stream(priority=7)` 经 `aclrtStreamGetPriority` **回读恒 0**
    # （参数在 torch_npu 插件层被丢弃）；而 pyACL `aclrtCreateStreamWithConfig` 的 0/3/7
    # **全部可回读**。⇒ 本层不能把"厂商构造函数收了这个 kwarg"当成"能力已具备"。
    #
    # 契约（本层的承诺，**只增不改**）：
    #   ① `create_stream()`（priority=None）= **厂商默认**，既有行为逐位不变；
    #   ② `priority` 非 int（含 bool）⇒ `ValueError` —— 与能力无关，最先判；
    #   ③ `priority` 越出 `stream_priority_range()` 的区间 ⇒ `ValueError`
    #      （**判在能力门禁之前**：越界是**调用方错误**，与「本后端能不能设置」是两件事；
    #        顺序颠倒会把越界误报成 `NotImplementedError`，掩盖更基本的问题 —— 2026-09-30 修）；
    #   ④ **单点区间的等价请求：放行**（2026-09-30 新增）——
    #      区间 `least == greatest` ⇒ 设备只有一档 ⇒ 请求值**必然等于厂商默认优先级**
    #      ⇒ 产出流的优先级**可回读验证等于请求** ⇒ 不存在信息损失；
    #      此时报错是**过度保守**（与「静默放行」是同一类失真的两个方向），故放行，但**必须带回读证据**。
    #   ⑤ 其余 `priority` 非 None 的情形：后端**未声明 `stream_priority_control`** ⇒
    #      `NotImplementedError`（**显式拒绝**，绝不静默返回一条无优先级的流）；
    #   ⑥ 声明了 `stream_priority_control` 的后端：创建后**必须回读校验** ⇒ 不一致即 `RuntimeError`
    #      —— 这是「参数被厂商**静默丢弃**」的唯一防线。
    #
    # ⚠️ 为什么「放行」只在**单点**成立：多档区间下任何「部分放行」都会让下游误判能力边界
    #    （以为能设 0 就能设 7）⇒ 一律交回能力门禁。
    # ⚠️ 声明 `stream_priority_control` 的后端**必须同时**声明 `stream_priority_readback`
    #    （否则第 ⑥ 条无法执行 ⇒ 无法证明参数真的进去了）。该耦合由离线判据守住。

    @abstractmethod
    def _create_stream_raw(self, priority=None):
        """子类实现：**厂商建流原语**。`priority=None` 表示"不指定"（厂商默认）。

        ⚠️ 实现纪律：`priority is not None` 且本后端未声明 `stream_priority_control` 时，
        **必须显式抛错**（通常基类门禁已先抛）—— **绝不能静默忽略**该参数。

        ⚠️ 例外（2026-09-30 新增）：**单点区间**（`least == greatest`）下「**等于该唯一档位**」的
        请求由**基类直接放行**（等价厂商默认，且带回读校验），**不会**落到本方法 ⇒
        本方法只处理「**真的需要按请求选档**」的情形。
        """

    def _priority_bounds(self):
        """把 `stream_priority_range()` 规整为 `(low, high)`；不可得返回 `None`。

        ⚠️ **形状纪律**（2026-09-30）：契约是 **2 元组** `(least, greatest)`，其中
        `greatest` 是**数值最小**的那个（官方：0 最高、7 最低）⇒ 区间取 `min/max`，
        **不按原序**。pyACL 的 `aclrtDeviceGetStreamPriorityRange` 返回**三元组**
        `(least, greatest, rc)`，后端**必须**归一 —— 否则同一份下游代码在不同家读到
        不同形状（本轮修掉的一处真实跨实例不一致）。
        """
        rng = self.stream_priority_range()
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            return None
        try:
            a, b = int(rng[0]), int(rng[1])
        except (TypeError, ValueError):
            return None
        return (min(a, b), max(a, b))

    # ─────────── 本层**拥有**的流：登记 / 释放（2026-09-30，(A) 方案配套）───────────
    #
    # 为什么需要：走「厂商 C API 建流 + 包装成 torch 流」时，**torch 不拥有**这条流
    # （P800 实测：丢弃包装对象 + gc 之后句柄**仍可计算**）⇒ 不显式销毁就**泄漏一条设备级流**
    # （设备流总数有限：910C 实测可用流上限 1979）。
    # 而厂商自己建的流（`torch.npu.Stream()` / `torch.cuda.Stream()`）由厂商 / torch 拥有
    # ⇒ 本层释放必须是 **no-op**，**不得越权销毁**别人的流。

    def _register_owned_stream(self, native_stream, handle) -> None:
        """登记「本层拥有、需显式释放」的流（`id(native)` → 厂商句柄）。"""
        try:
            self._reg("_owned_streams")[id(native_stream)] = int(handle)
        except Exception:                                     # noqa: BLE001
            pass

    def owns_stream(self, native_stream) -> bool:
        """该流是否由**本层**创建并拥有（决定 `release_stream` 是否真的销毁）。"""
        return id(native_stream) in self._reg("_owned_streams")

    def release_stream(self, native_stream) -> bool:
        """释放由**本层拥有**的流；厂商拥有的流 ⇒ **no-op 并返回 `False`**。

        返回值语义（**不假装成功**）：
          · `True`  = 本层创建的那条流**真的被销毁**（厂商销毁原语返回成功）；
          · `False` = 本层不拥有（未做任何事，逐位保持原有语义）或销毁原语不可得。

        幂等：重复释放返回 `False`、不报错；但**再使用**已释放的流会被
        `check_stream_usable()` 当场拦下（契约：使用已销毁对象必须明确）。
        """
        reg = self._reg("_owned_streams")
        handle = reg.pop(id(native_stream), None)
        if handle is None:
            return False
        self._reg("_released_streams")[id(native_stream)] = int(handle)
        return bool(self._destroy_stream_raw(handle))

    def _destroy_stream_raw(self, handle) -> bool:
        """**厂商原语层**：销毁由本层创建的流。基类默认 `False`（未实现 ⇒ 不假装成功）。"""
        return False

    def create_stream(self, priority=None):
        """创建并返回该后端的 Stream 对象；**可选**指定流优先级。

        `priority=None`（默认）⇒ 厂商默认行为，与历史版本**逐位一致**。
        `priority=<int>` ⇒ 走上面 ①–⑥ 六条（**顺序**本身就是契约的一部分）。

        为什么不做成"传了就当没传"（2026-09-30 的教训）：
        静默忽略参数会产出**最贵的假象** —— 上层以为"高优先级流已建好"，
        据此做调度决策，而设备侧根本没有这回事。**宁可报错，不给假象。**

        ⚠️ 但"宁可报错"**不等于"报错越多越诚实"**（2026-09-30 修）：请求落在
        **单点区间**上时（如 P800 的 `(0, 0)`），请求值**必然等于厂商默认优先级**
        ⇒ 结果可回读验证 ⇒ 此时报错是**过度保守**，把「可满足的请求」说成「不支持」，
        同样是失真（方向相反）。故第 ④ 条**放行**，并**强制**带回读证据。
        """
        if priority is None:
            return self._create_stream_raw(None)

        # ① 类型（与能力无关、成本为零）：bool 是 int 的子类，先挡掉
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise ValueError(
                f"priority 必须是 int 或 None，收到 {type(priority).__name__}：{priority!r}")

        # ② 取值域（**先于能力门禁**：越界是调用方错误，与「能不能设置」是两件事）
        bounds = self._priority_bounds()
        if bounds is not None and not (bounds[0] <= priority <= bounds[1]):
            raise ValueError(
                f"priority={priority} 越出本设备区间 {bounds}"
                f"（官方语义：数值**越小优先级越高**；`stream_priority_range()` 返回 "
                f"(least, greatest)，本层已归一为 (low, high)）")

        # ③ **单点区间的等价请求：放行**（但必须带回读证据 —— 放行不是免校验）
        if (bounds is not None and bounds[0] == bounds[1] == priority
                and self.supports("stream_priority_readback")):
            native = self._create_stream_raw(None)
            got = self.stream_priority_readback(native)
            if got is not None and int(got) == int(priority):
                return native
            raise RuntimeError(
                f"单点区间 {bounds} 的等价放行**未通过校验**：请求 {priority}，回读 {got!r}。\n"
                f"  · 含义：本后端自称区间只有一档，但**厂商默认流的优先级与该档位不一致**"
                f" ⇒ 区间值不可信（**声明与实现不一致**），故不予放行；\n"
                f"  · 处置：修区间读数或修默认流 —— **不要靠放宽判据**（同族纪律：厂商缺陷不得改判据变绿）。")

        # ④ 声明即承诺：未声明 ⇒ 显式拒绝（不是静默降级）
        if not self.supports("stream_priority_control"):
            raise NotImplementedError(
                f"后端 '{self.name}' 未声明能力 `stream_priority_control` ⇒ 拒绝 "
                f"create_stream(priority={priority!r})。\n"
                f"  · 本层**能读**优先级范围（stream_priority_range() = "
                f"{self.stream_priority_range()!r}），但**不等于能设置**；\n"
                f"  · 本层宁可报错，也不返回一条『看起来设了优先级、其实没设』的流；\n"
                f"  · 该实例的具体原因与证据见 `info()['known_issues']` 与芯片目录报告；\n"
                f"  · 不指定优先级请用 `create_stream()`。")

        # ⑤ 声明了 control ⇒ 区间必须可解析（否则无法做取值域校验）
        if bounds is None:
            raise RuntimeError(
                f"后端 '{self.name}' 声明了 `stream_priority_control`，但 "
                f"`stream_priority_range()` 给不出可解析的 2 元组区间"
                f"（实际 = {self.stream_priority_range()!r}）"
                f" ⇒ **声明与实现不一致**（设置无法做取值域校验）。")

        # ⑥ 创建 + **回读校验**（本层不允许静默退化）
        native = self._create_stream_raw(int(priority))
        got = self.stream_priority_readback(native)
        if got is None:
            raise RuntimeError(
                f"后端 '{self.name}' 声明了 `stream_priority_control`/`stream_priority_readback`，"
                f"但创建后**读不回**优先级 ⇒ 无法证明参数真的进去了"
                f" ⇒ **声明与实现不一致**（设置必须可校验）。")
        if int(got) != int(priority):
            raise RuntimeError(
                f"流优先级**未生效**：请求 {priority}，设备**回读** {got}。\n"
                f"  · 含义：厂商在该路径上**接受但静默丢弃**了 priority"
                f"（910C 的 `torch.npu.Stream(priority=…)` 正是这种情形：回读恒 0）；\n"
                f"  · 本层**不掩盖**这一点：要么换一条真能把参数送进设备的路径，"
                f"要么如实不声明该能力。\n"
                f"  · 取证：`stream_priority_readback()` 与芯片目录报告。")
        return native

    @abstractmethod
    def create_event(self):
        """创建并返回该后端的 Event 对象。"""

    @abstractmethod
    def current_stream(self):
        """返回当前默认流。"""

    @abstractmethod
    def synchronize(self, ordinal: int, timeout_ms: Optional[int] = None) -> None:
        """设备同步。timeout_ms 非空时应为有界等待（超时抛 TimeoutError），
        这是长驻服务避免整体 hang 死的关键能力。"""

    # ── 多流 Stream 支撑（供 api/stream.py 封装使用）──

    @abstractmethod
    def stream_context(self, native_stream):
        """返回切换到该流的上下文管理器（with 使用）。"""
        # 使用点拦截（工作包 C·绑定语义）：见 check_stream_usable。

    @abstractmethod
    def synchronize_stream(self, native_stream, timeout_ms: int) -> None:
        """有界等待指定流；超时抛 TimeoutError。"""

    @abstractmethod
    def wait_event_host(self, native_event, timeout_ms: int) -> bool:
        """主机侧有界等待事件；返回 True 表示已完成，超时返回 False。"""

    # ── 跨流内存保护（工作包 B-3）：`record_stream` 能力位 + 保守同步路径 ──

    def peek_current_device(self) -> int:
        """**只读**返回当前默认设备序号（不切换设备）。

        用途：`conservative_stream_sync()` 需要知道该同步哪个设备。
        默认实现返回 0（单设备场景安全）；三家均覆写为厂商的 `current_device()`。
        """
        return 0

    def conservative_stream_sync(self, native_stream=None) -> None:
        """`record_stream` 不可用时的**保守同步**：同步当前设备后再让内存被复用。

        宁可慢，不可坏 —— 跨流内存被提前回收会变成**数据竞争**（最难查的一类问题）。
        本方法把"不支持"从"运行时 `getattr` 失败"变成"一条可预判的退化路径"。
        """
        self.synchronize(self.peek_current_device())

    # ───────────────────────── 错误码翻译（职责 D10）─────────────────────────

    @abstractmethod
    def translate_error(self, exc: BaseException, location: str = "") -> FlagosError:
        """厂商错误 → 统一 FlagosError（含 mapped / graded_by 可观测字段）。"""

    # ───────────────────────── 状态恢复（职责 D11）─────────────────────────

    @abstractmethod
    def probe_device(self, ordinal: int) -> bool:
        """轻量探活：区分可继续与需重建。健康设备应返回 True。"""

    def recover_device(self, ordinal: int, mode: str = "probe",
                       reason: str = "") -> dict:
        """设备重建（**公共入口**，统一返回 dict 且**只增不改**）。

        本方法在基类实现，负责在子类实现之上**统一附加"上下文维度"**，
        避免三家各写一遍（写成三遍就必然出现三套口径）：
            `context_supported` / `context_count` / `context_recreated`
        —— 契约五键 `{ordinal, mode, recovered, state, detail}` **保持不变**。

        2026-09-09 统一返回类型为 dict（此前有的返回 bool、有的 dict，上层无法统一处理）。
        """
        rec = self._recover_device_impl(ordinal, mode=mode, reason=reason)
        if isinstance(rec, dict):                                 # Mock/自定义实现可返回非 dict
            snap = self.context_snapshot()
            rec["context_supported"] = snap["context_supported"]
            rec["context_count"] = snap["context_count"]
            # ⭐ 2026-09-29 修正（第 21 条）：**只认实现如实回报的事实**，不再用 `mode + recovered` 反推。
            #   原实现 `bool(mode == "real" and rec.get("recovered"))` 把「设备当前可用」
            #   当成「重建动作已执行」，于是在**未执行任何重建**的路径上也报 True：
            #     · ascend 健康设备上调 `real`（同一次返回的 `detail` 自己写着"无需重建"）
            #     · kunlun / cambricon **未声明 `recovery_real`**（明说 real 不支持）
            #   ⇒ 同一次返回内 `detail` 与 `context_recreated` 自相矛盾，违反 I1「诚实声明」/ I4「降级可观测」。
            #   现由实现在**做决策的同一处**记录实际路径，并通过私有键 `_rebuilt` 回报；
            #   **未回报视为未重建**（宁可不声明，不臆造）。
            _rebuilt = bool(rec.pop("_rebuilt", False))
            rec["context_recreated"] = bool(_rebuilt and mode in ("real", "hybrid"))
            if snap.get("context_error"):
                rec["context_error"] = snap["context_error"]
        return rec

    @abstractmethod
    def _recover_device_impl(self, ordinal: int, mode: str = "probe",
                             reason: str = "") -> dict:
        """子类实现：设备重建的**实际动作**，返回 dict（至少含契约五键所需的业务字段）。

        实现者不必关心上下文维度 —— 基类 `recover_device()` 会统一附加。

        mode:
          - "probe"  探针重试近似（默认保底，进程内安全）
          - "real"   真实重建（CANN 官方序列：destroyEvent→destroyStream→
                     destroyContext→aclrtResetDevice→setDevice→重建）
          - "hybrid" 先探针（快路径），失败后真实重建

        注意：real 模式会重置当前进程默认上下文；多进程共享设备时其他进程不受影响，
        但本进程需重新 set_device。生产默认启用前需多卡多进程压力测试。
        """

    # ───────── 状态机驱动 + 错误处理编排（2026-09-29 新增 · **只增不改**）─────────
    #
    # 【为什么放在基类唯一实现】四态状态机是**芯片无关的共享资产**
    #   （`conformance/device_state.py` —— 进程内账本，不依赖任何厂商原语）⇒ 三家实现必然逐字相同。
    #   项目已有先例：`supports()` 也从三家各写一份**收敛到基类唯一实现** —— 写成三份必然漂移。
    #
    # 【为什么现在才补（来由如实记录）】`recover_device(mode="real")` **只在设备处于 ISOLATED 时**
    #   才真重建，而在此之前**公开面没有任何入口能把设备置为 ISOLATED** ⇒ 只走公开 API 时
    #   `real` 永远走"无需重建"分支，「某卡 L4 故障 → 设备级恢复」这条链**在公开面上不可触发**
    #   （2026-09-29 A2 多卡压测实测发现）。补上本组入口后该链可由公开 API 完整驱动：
    #     生产：捕获异常 → `handle_error()`（R1 分级 → R2 评估 → R3 隔离 → R4 重建 → R5 重放）
    #     演练 / 监控：`set_device_state()` 直接驱动四态
    def _load_shared_state_assets(self):
        """按需加载共享状态机 / 恢复资产。

        ⚠️ 必须用**扁平名**导入（与各后端 `_load_conformance()` 同一形式、同一目录）；
        换成包路径会得到**另一个模块实例**（进程内两份状态机、各说各话），见审计台账第 23 条。
        """
        if getattr(self, "_shared_state_assets_loaded", False):
            return
        import sys
        from pathlib import Path
        conf_dir = str(Path(__file__).resolve().parents[1] / "conformance")
        if conf_dir not in sys.path:
            sys.path.insert(0, conf_dir)
        import device_state as _device_state
        import recovery as _recovery
        self._device_state = _device_state
        self._recovery = _recovery
        self._shared_state_assets_loaded = True

    def set_device_state(self, ordinal: int, state, reason: str = "") -> str:
        """**驱动**四态状态机（本层账本），返回新状态 **token 字符串**。

        - `state` 接受**对外 token**（`available` / `degraded` / `isolated` / `destroyed`）
          或共享枚举成员；**非法取值 ⇒ `ValueError`**（不静默）。
        - 幂等：同态转换不产生事件（与状态机语义一致）。
        - ⚠️ **这是本层的隔离账本，不等于厂商设备的真实状态**：把健康设备标成 `isolated`
          只影响**本层的调度 / 恢复判定**（演练与混沌注入靠它），**不会**让硬件出错；
          反过来，真实故障的隔离仍应由 R2 评估（探针失败）驱动。
        - **三芯片一致可用**（共享资产，不依赖厂商原语）⇒ 无 `supports()` 分支。
        """
        self._load_shared_state_assets()
        ds = self._device_state
        if isinstance(state, str):
            tok = state.strip().lower()
            if tok not in DEVICE_STATE_TOKENS:
                raise ValueError(f"未知设备状态 {state!r}；取值域 = {DEVICE_STATE_TOKENS}")
            for member in ds.DeviceState:
                if member.value == tok:
                    state = member
                    break
        elif not isinstance(state, ds.DeviceState):
            raise ValueError("state 必须是对外 token 字符串，或共享枚举 DeviceState 成员；"
                             f"收到 {type(state).__name__}")
        new = ds.set_device_state(
            ordinal, state,
            reason or f"runtime: set_device_state({getattr(state, 'value', state)})")
        return state_token(new)

    def handle_error(self, exc: BaseException, ordinal=None,
                     location: str = "", mode: str = "probe"):
        """**五段式错误处理编排**：错误 → 统一错误对象（R1）→ 评估（R2）→
        隔离/重建（R3/R4）→ 重放（R5）。

        返回**统一类型**的 `FlagosError`（与 `runtime.translate_error` 同型）；
        `recovery_decision` 记录流程事件（供监控 / 可观测消费）。
        `mode` 语义与 `recover_device` 一致，**默认 `"probe"`**（进程内安全）；传 `"real"` 才真重建。

        ⚠️ 编排**复用共享编排器**（不在此另写一份，避免出现第二套口径），但**译码器传本后端自己的**
        `translate_error` —— 以继承其**码表归属**规则。若沿用共享译码器的默认（昇腾 ACL 码表），
        未声明 `error_map` 的后端会把携带他厂码的消息误判成 L4 ⇒ **误触发设备级重建**
        （与审计台账第 16 条同一缺陷族）。
        """
        self._load_shared_state_assets()
        return self._recovery.handle_error(
            exc, ordinal=ordinal, location=location or "",
            device=self.device_type, rebuild_mode=mode,
            translate_fn=self.translate_error)

    # ───────────────────────── 可选能力（默认不支持）──────────────────────────

    def stream_priority_range(self):
        """流优先级范围 **(least, greatest) 2 元组**；不支持返回 None。

        ⚠️ **形状是契约的一部分**（2026-09-30 修）：返回**必须**是 2 元组或 None ——
        不得把厂商的**三元组**（如 pyACL 的 `(least, greatest, rc)`）直接透传出去。
        官方语义：`greatest` 是**数值最小**的那个 ⇒ 0 最高、7 最低。
        """
        return None

    def stream_priority_readback(self, native_stream):
        """**回读**某条流的实际优先级；读不出返回 None（**不猜、不补零**）。

        用途：把"参数真的进了设备"与"构造函数收了这个 kwarg"**分开**。
        2026-09-30 实测：`torch.npu.Stream(priority=7)` 回读 **0**（插件层丢弃），
        而 pyACL `aclrtCreateStreamWithConfig` 的 0/3/7 **全部可回读**
        —— 没有这条回读，两种情形在 Python 侧**看起来一模一样**。
        """
        return None

    def known_issues(self) -> list:
        """本后端已知的**上游/环境**问题清单（默认空）。

        用途：接入方（其他子方向）读到后端即可获知该环境的坑与临时规避，
        无需翻文档。每项为 dict，建议字段：
            id / severity / scope / condition / symptom / root_cause_layer /
            workaround / workaround_risk / report_to / evidence

        纪律：只描述**已实测**的问题，须注明复现率与证据位置；不得把推测写成结论。
        """
        return []

    #: 历史 / 弃用能力键 → 规范键（**别名一律归一到规范键判定**，保证跨后端一致）。
    #: 2026-09-28（职责响应审计暴露）：`sync_timeout` 是 ascend 的历史键名，与规范键
    #: `bounded_sync` 同义。此前只有 ascend 的 `_CAPABILITY_KEYS` 含它 ⇒ 下游若用该键判定，
    #: 在 kunlun / cambricon 上会得到 False —— **同一份代码跨芯片行为不一致**。
    #: 现统一为别名：三家 `_CAPABILITY_KEYS` 均含该键，取值随规范键。
    _CAPABILITY_ALIASES = {"sync_timeout": "bounded_sync"}

    def supports(self, capability: str) -> bool:
        """能力查询，便于 conformance 做 stub-skip 报告。**含历史键名别名归一。**"""
        cap = getattr(self, "_CAPABILITY_ALIASES", {}).get(capability, capability)
        return cap in getattr(self, "_capabilities", set())

    # ───────────────────────── `.native` 逃生舱审计（工作包 B-4）─────────────────────────
    #
    # 修订建议 §3：`.native` 定为"显式逃生舱"，一旦上层取用，该处代码即视为
    # **绑定该厂商**，不得再声称跨芯片可移植 ⇒ 必须能**事后溯源**。
    # 本层做法：公开属性 `.native` **计数**；层内部一律走私有 `_native_obj`，不计数。

    def note_native_access(self, kind: str) -> None:
        """记一次 `.native` 取用（由 `api/stream.py` 的公开属性调用）。"""
        acc = self.__dict__.setdefault("_native_accesses", {"total": 0, "by_kind": {}})
        acc["total"] += 1
        acc["by_kind"][kind] = acc["by_kind"].get(kind, 0) + 1

    def native_accesses(self) -> dict:
        """`.native` 取用审计：`{"total": n, "by_kind": {...}}`（只读副本）。"""
        acc = self.__dict__.get("_native_accesses") or {"total": 0, "by_kind": {}}
        return {"total": int(acc["total"]), "by_kind": dict(acc["by_kind"])}

    def reset_native_accesses(self) -> None:
        """清零审计计数（便于"本轮跑完看取用次数"的取数方式）。"""
        self.__dict__["_native_accesses"] = {"total": 0, "by_kind": {}}

    # ───────────────── 能力缺失的**退化路径**审计（工作包 B-3）─────────────────
    #
    # 与上面的 `.native` 审计**分开计数**，因为两者性质相反：
    #   `.native` 取用 = 破坏可移植性（坏账）；退化路径 = 用性能换正确性（可接受的降级）。
    # 但两者都必须**可观测**：否则"静默退化"又回来了（台账第 ⑨ 条家族）。

    def note_degradation(self, kind: str) -> None:
        """记一次因能力缺失而走的退化路径（如 `record_stream` 不可用 → 保守同步）。"""
        acc = self.__dict__.setdefault("_degradations", {"total": 0, "by_kind": {}})
        acc["total"] += 1
        acc["by_kind"][kind] = acc["by_kind"].get(kind, 0) + 1

    def degradations(self) -> dict:
        """退化路径审计：`{"total": n, "by_kind": {...}}`（只读副本）。"""
        acc = self.__dict__.get("_degradations") or {"total": 0, "by_kind": {}}
        return {"total": int(acc["total"]), "by_kind": dict(acc["by_kind"])}

    def reset_degradations(self) -> None:
        self.__dict__["_degradations"] = {"total": 0, "by_kind": {}}

    # ───────────────────────── 元信息 ─────────────────────────

    def _base_info_fields(self) -> dict:
        """所有后端 `info()` 都应并入的公共字段（新增能力时**一处生效**）。"""
        return {
            "capabilities": sorted(getattr(self, "_capabilities", set())),
            "native_accesses": self.native_accesses(),
            "degradations": self.degradations(),
        }

    def info(self) -> dict:
        out = {"name": self.name, "device_type": self.device_type}
        out.update(self._base_info_fields())
        return out

    # ───────────────────────── 内部工具 ─────────────────────────

    def _reg(self, name: str) -> dict:
        """取（或惰性创建）**每实例独立**的登记表。"""
        reg = getattr(self, name, None)
        if reg is None:
            reg = {}
            setattr(self, name, reg)
        return reg

    #: 登记表名 → 期望的句柄种类（**防两类句柄互相误用**）
    _REG_KINDS = {"_alloc_handles": "memory", "_ctx_handles": "context"}

    def _handle_id(self, handle, reg_name: str, what: str) -> int:
        """校验并取出句柄 id；**任何不合契约的句柄都如实报错，绝不静默接受**。"""
        if not isinstance(handle, dict) or "handle_id" not in handle:
            raise ValueError(
                f"{what}句柄格式不合法：期望 allocate()/context_create() 返回的 dict，"
                f"收到 {type(handle).__name__}。**不接受厂商原始指针/句柄**"
                "（需绕过统一层请走 `.native` 显式逃生舱并计入审计）。")
        want = self._REG_KINDS.get(reg_name)
        if want and handle.get("kind") not in (None, want):
            raise ValueError(
                f"{what}句柄种类不匹配：拿到 kind={handle.get('kind')!r}，期望 {want!r}"
                "—— **内存句柄与上下文句柄不得互相误用**（如实报错，不静默接受）。")
        hid = handle.get("handle_id")
        if hid not in self._reg(reg_name):
            raise ValueError(
                f"{what}句柄 {hid} 不属于本后端或**已被释放**（重复释放/跨后端误用）"
                "—— 如实报错，不静默接受。")
        return hid
