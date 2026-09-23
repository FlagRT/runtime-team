#!/usr/bin/env python3
"""后端**离线契约自检**（无设备）—— 新芯片接入时"先自查、再上机"。

═══════════════════════════════════════════════════════════════════════════════
【它解决什么问题】
  《新芯片接入手册》第 4 步（实现 backend）与第 5 步（跑 conformance）之间有一段空档：
  代码写完了但**机器/容器还没到位**（权限未开、镜像未取到）。
  这段时间里，最容易犯的一类错是**实现层面的**：
    · 抽象方法没实现全（要到实例化才炸）
    · 有界同步其实没上界（超时参数被忽略）
    · 事件语义没修（未 record 的 query() 误报"已完成"）
    · 错误翻译冒充码表命中（mapped/graded_by 说谎）
    · 能力声明与实际实现不一致（`info().capabilities` 与 `supports()` 打架）
  这些问题**不需要真实芯片就能查出来** —— 本脚本用 stub 厂商命名空间把后端"空跑"一遍。

【它不能证明什么（重要，勿外推）】
  ⛔ **不能替代 conformance**，也**不能证明后端在真机上能用**。
  它只证明"实现逻辑与契约形态正确"。凡涉及**厂商 API 真实形态**的结论
  （`torch.mlu.mem_get_info` 是否存在、`Stream.synchronize` 是否收 timeout、
   厂商错误码是否透出、集合通信后端名是什么……）**必须上机实测**。
  脚本输出的最后一行会重复这一点，避免被脱离语境引用。

【用法】
    python3 prototype/scripts/backend_offline_check.py --backend cambricon
    python3 prototype/scripts/backend_offline_check.py --backend cambricon --proto <prototype 目录>

【给新厂商加 stub（一家一个函数，约 40 行）】
  在下方 `_STUBS` 里加一个 `_stub_<vendor>()`：
    · 造一个 `torch` 模块（至少要有 `zeros`）+ 该厂商命名空间的假实现
      （count / set_device / current_device / Stream / Event / current_stream /
        stream / synchronize / 显存查询 / get_device_properties）
    · 把厂商扩展模块（如 `torch_mlu`）塞进 `sys.modules` 触发"注册"
  参考 `_stub_cambricon`。**stub 要刻意做得"坏"一点**（例如让未 record 的
  `Event.query()` 无条件返回 True），这样才能验出适配层有没有真的在做修正。

【判据来源】
  每条检查都对应《接口约定》或 conformance 用例的同口径判据，
  在代码里以注释标明（F1 / E2-v2 / E3 / R1-R5 / 能力自洽）。
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time
import traceback
import types
from pathlib import Path

PASS, FAIL = 0, 0


#: 被 stub 能力边界跳过（非失败）的检查计数
SKIPPED = 0


def skip(name, reason):
    """显式跳过：**stub 无法真实模拟**时才用，必须写清原因，不得用来掩盖失败。"""
    global SKIPPED
    SKIPPED += 1
    print(f"  [SKIP] {name} —— {reason}")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ═════════════════════ 离线性保障：阻断真实厂商运行时 ═════════════════════
#: 被阻断的真实厂商运行时导入（同时也是"本自检确实离线"的证据）
_BLOCKED_IMPORTS: list = []


class _VendorRuntimeBlocker:
    """阻断**真实**厂商运行时绑定的导入，保证本自检真的"离线"。

    ⚠️ **为什么必须有它（2026-09-22 实测教训）**：本脚本的 stub 只替换了 **torch 命名空间**，
    但真机上厂商绑定常常**本来就在 `PYTHONPATH` 上**（实测 910C 容器里 CANN 的 `acl` 可直接 import）。
    于是 `--backend ascend` 会走到**真实 pyACL**路径，进而访问硬件：

        acl.init() → 500000（宿主带卡容器并发名额用尽）
        set_device → 107002（CONTEXT_NULL）
        synchronize_device_with_timeout → 107000（PARAM_INVALID）
        ⇒ 后端抛 TimeoutError ⇒ 自检**直接 traceback、连汇总都打不出**

    三个后果都很坏：① 自检在真机上根本用不了；② 结论被真实环境状态污染（假阴性）；
    ③ "离线"这个前提是假的。
    `sys.modules` 优先于 `sys.meta_path`，故我们**注入的假模块仍然生效**（如 ascend 的假 `acl`）。
    """

    # 注：`torch_fl` 属路线 B（已归档删除），此处**仍然保留在阻断清单里** ——
    # 阻断清单的语义是"厂商运行时一个都不许碰"，多列一家只会更安全（防误引入）。
    VENDOR_ROOTS = ("acl", "torch_npu", "torch_fl", "torch_mlu", "torch_npu_ops",
                    "torch_mlu_ops", "torch_xmlir", "torch_xray", "flagcx", "cncl")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.VENDOR_ROOTS:
            _BLOCKED_IMPORTS.append(name)
            raise ImportError(
                f"[offline-check] 已阻断真实厂商运行时 `{name}` 的导入（本自检必须离线运行）")
        return None


def _install_blocker() -> None:
    if not any(isinstance(f, _VendorRuntimeBlocker) for f in sys.meta_path):
        sys.meta_path.insert(0, _VendorRuntimeBlocker())


def _host_vendor_bindings() -> list:
    """探测宿主 `sys.path` 上**真实存在**的厂商绑定模块名。

    用 `PathFinder` 直接搜 `sys.path`（**绕过 `sys.modules`**），这样才能区分两种情形：
      · "本机没有厂商绑定" —— 与
      · "本机有，但已被我们注入的假模块/阻断器挡在门外"
    这两者的含义完全不同，**不能混为一谈**（2026-09-22 实测：910C 容器里真实 `acl` 确实在
    `PYTHONPATH` 上，而当时提示语写成"本机无厂商绑定"，是错的）。
    """
    from importlib.machinery import PathFinder
    found = []
    for root in _VendorRuntimeBlocker.VENDOR_ROOTS:
        try:
            if PathFinder.find_spec(root) is not None:
                found.append(root)
        except Exception:
            pass
    return sorted(found)


def _real_vendor_modules() -> list:
    """返回**以真实文件形式**加载的厂商运行时模块（我们的假模块没有 `__file__`）。"""
    out = []
    for name, mod in list(sys.modules.items()):
        if name.split(".")[0] in _VendorRuntimeBlocker.VENDOR_ROOTS:
            if getattr(mod, "__file__", None):
                out.append(name)
    return sorted(out)


def _make_fake_acl(init_rc=0):
    """假 pyACL —— 用**可控的返回码**离线驱动 ascend 的有界同步路径。

    这是第 9 个原型缺陷（"把 ACL 初始化/参数错误冒充成同步超时"）的回归判据载体：
    没有真机也能验证"非超时码不得被报成 TimeoutError"。
    返回 (假模块, 可变 rc 状态)。
    """
    rc = {"sync": 0, "set_device": 0}
    acl = types.ModuleType("acl")
    acl.__offline_stub__ = True
    acl.init = lambda: init_rc
    acl.finalize = lambda: 0
    acl.rt = types.SimpleNamespace(
        set_device=lambda o: rc["set_device"],
        synchronize_device_with_timeout=lambda ms: rc["sync"],
        synchronize_stream_with_timeout=lambda h, ms: rc["sync"],
        get_device_count=lambda: (8, 0),
        device_get_stream_priority_range=lambda: (0, -1),
    )
    return acl, rc


# ═════════════════════ stub：厂商命名空间 ═════════════════════
def _vendor_stub(ns_name, vendor_modules=(), mem_mode="full"):
    """构造一个"厂商设备命名空间"stub —— 四家后端共用同一套实现。

    ns_name        设备命名空间名（`mlu` / `cuda` / `npu`）→ 挂成 `torch.<ns_name>`
    vendor_modules 需一并伪造的**顶层厂商模块**名（`torch_mlu` / `torch_npu`）
                   —— 后端里的 `import torch_mlu` 这类语句靠它们才能通过
    mem_mode       显存查询能力，用于验证后端的取值兜底链：
                     'full'   → `mem_get_info(ordinal)` 可用
                     'noarg'  → 只接受无参 `mem_get_info()`
                     'absent' → 没有 `mem_get_info`（须降级到 total_memory - memory_allocated）

    ⚠️ **stub 刻意做得"坏"一点**（见 `Event`）：未 record 的 `query()` 无条件返回 True。
    只有让底层"坏"，才能验出适配层有没有在真的做修正。

    2026-09-22 通用化：原先只有 `_stub_cambricon` 一家，导致 P800 / 910C 上
    `--backend kunlun|ascend` 直接被拒（工具自称通用却只能用一家）；三家共用本函数后
    同一份自检可在三个芯片实例上跑，新增厂商也只需一行注册。
    """
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+stub"
    ns = types.ModuleType(f"torch.{ns_name}")
    state = {"current": 0, "stream_query": True}

    class Stream:
        def __init__(self, *a, **k):
            pass
        def query(self):
            return state["stream_query"]
        def synchronize(self):
            pass
        def wait_event(self, e):
            pass
        def wait_stream(self, s):
            pass

    class Event:
        """**刻意做成坏实现**：未 record 也返回 True —— 用来验适配层的 E3 修正。"""
        def __init__(self, *a, **k):
            self._rec = False
        def record(self, stream=None):
            self._rec = True
        def wait(self, stream=None):
            pass
        def query(self):
            return True
        def synchronize(self):
            self._rec = True

    class Props:
        total_memory = 96 * 1024 ** 3

    class _Ctx:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    class _T:
        def __init__(self, v):
            self._v = v
        def sum(self):
            return _T(self._v)
        def item(self):
            return self._v

    class _Dev:
        """设备串对象：命名空间的 `device(ordinal)` 工厂返回它。"""
        def __init__(self, idx):
            self.type = ns_name
            self.index = idx
        def __str__(self):
            return f"{ns_name}:{self.index}"

    ns.device_count = lambda: 8
    ns.set_device = lambda o: state.__setitem__("current", o)
    ns.current_device = lambda: state["current"]
    ns.is_available = lambda: True
    # 部分后端的 memory_stats() 会直接读命名空间上的 `memory_stats()`
    ns.memory_stats = lambda: {"total_bytes": 96 * 1024 ** 3,
                               "allocated_bytes": 1 * 1024 ** 3}
    ns.Stream = Stream
    ns.Event = Event
    ns.current_stream = lambda: Stream()
    ns.stream = lambda s: _Ctx()
    ns.synchronize = lambda *a, **k: None
    ns.get_device_properties = lambda o: Props()
    ns.memory_allocated = lambda o: 1 * 1024 ** 3
    ns.device = lambda o=0: _Dev(o)
    if mem_mode == "full":
        ns.mem_get_info = lambda ordinal=None: (80 * 1024 ** 3, 96 * 1024 ** 3)
    elif mem_mode == "noarg":
        ns.mem_get_info = lambda: (80 * 1024 ** 3, 96 * 1024 ** 3)
    # 'absent'：不设该属性

    torch.zeros = lambda *a, **k: _T(0.0)
    setattr(torch, ns_name, ns)
    mods = {name: types.ModuleType(name) for name in vendor_modules}
    return torch, mods, state


def _stub_cambricon(mode="full"):
    """寒武纪 `torch.mlu`（PrivateUse1）—— 需伪造顶层 `torch_mlu`。"""
    return _vendor_stub("mlu", ("torch_mlu",), mode)


def _stub_kunlun(mode="full"):
    """昆仑芯走 `torch.cuda` 兼容层（XPytorch）—— 无额外顶层厂商模块。"""
    return _vendor_stub("cuda", (), mode)


def _stub_ascend(mode="full", acl_init_rc=0, with_fake_acl=True):
    """昇腾 `torch.npu` —— 需伪造顶层 `torch_npu`，并注入**假 pyACL**。

    `acl_init_rc` 可控：用 `acl_init_rc=500000` 验证"init 非 0 必须视为不可用"这条判据。
    """
    torch, mods, state = _vendor_stub("npu", ("torch_npu",), mode)
    if with_fake_acl:
        acl_mod, rc = _make_fake_acl(init_rc=acl_init_rc)
        mods["acl"] = acl_mod
        state["acl_rc"] = rc
    return torch, mods, state


#: 厂商 stub 注册表 —— 新厂商在这里加一项即可（三家已内置）
#:
#: 每项 = (stub_fn, device_type_期望, caps)。**caps 是"这个 stub 能真实验到什么"的声明**：
#: 语义空间的 stub 无法模拟真实算子与厂商运行时，凡是 stub 不能真实覆盖的判据，
#: 一律按 caps 走 `skip()` 显式跳过 —— **不得**因为"stub 不支持"而判 FAIL（那是误报），
#: 也**不得**因为跳过而把该后端说成"已通过"（跳过会单独计数）。
#:   stub_real_ops          能否真实跑 `probe_device`（需要在设备上做真计算）
#:   stub_bounded_sync      能否真实模拟"任务未完成 + timeout"（需要厂商超时原语）
#:   stub_mem_stats_shape   stub 的原始显存返回形状是否与真实厂商栈一致
#:   stub_vendor_extension  该后端是否有**独立可缺失的厂商扩展模块**（缺了才谈得上报错）
_STUBS = {
    "cambricon": (_stub_cambricon, "mlu", {}),
    "kunlun": (_stub_kunlun, "cuda", {"stub_vendor_extension": False}),
    "ascend": (_stub_ascend, "npu", {"stub_real_ops": False, "stub_bounded_sync": False}),
}


def fresh_import(proto_dir, backend, stub_fn, mode="full", with_vendor=True, **stub_kwargs):
    """清干净再导入：确保每次都是"全新发现"（注册表单例会跨用例残留）。

    · 装 **真实厂商运行时阻断器**（保证"离线"名副其实，见 `_VendorRuntimeBlocker`）
    · 清掉上轮残留（**含 `acl`** —— 真机上它可能已被真实加载过）
    · `**stub_kwargs` 透传给 stub（如 ascend 的 `acl_init_rc`）
    """
    _install_blocker()
    for m in [k for k in list(sys.modules)
              if k.startswith("runtime")
              or k.split(".")[0] in ("torch", "torch_mlu", "torch_npu", "torch_fl", "acl")]:  # torch_fl 同属阻断清单，见 VENDOR_ROOTS 注释
        del sys.modules[m]
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)
    torch, vendor_mods, state = stub_fn(mode, **stub_kwargs)
    sys.modules["torch"] = torch
    if with_vendor:
        sys.modules.update(vendor_mods)
    import runtime
    from runtime.backends.registry import clear
    clear()
    loaded = runtime.discover(names=(backend,), verbose=False)
    return runtime, loaded, state


def run(proto_dir, backend):
    stub_fn, dev_ns, caps = _STUBS[backend]
    print("=" * 74)
    print(f"后端离线契约自检：backend={backend}（device_type 期望={dev_ns}）")
    print("⚠️ 无真实设备：只验实现逻辑与契约形态，不验厂商 API 真实行为")
    if caps:
        print(f"ℹ️ 本家 stub 的能力边界（以下判据会被显式 SKIP）："
              f"{', '.join(f'{k}=False' for k in sorted(caps) if caps[k] is False)}")
    print("=" * 74)

    # ── 0. 离线性（本自检不得触碰真实厂商运行时）──
    print("\n[0] 离线性保障（真实厂商运行时必须被阻断）")
    _real = _real_vendor_modules()
    check("未以真实文件形式加载任何厂商运行时", not _real, str(_real))
    _host = _host_vendor_bindings()
    if _host:
        print(f"  [INFO] 本机 sys.path 上**存在**真实厂商绑定：{_host}"
              f" —— 已挡在门外（见下行），故本自检确实是离线的")
    else:
        print("  [INFO] 本机 sys.path 上未发现厂商绑定（本机天然离线）")
    if _BLOCKED_IMPORTS:
        print(f"  [INFO] 已被阻断器拦截的真实导入 {len(set(_BLOCKED_IMPORTS))} 处："
              f"{sorted(set(_BLOCKED_IMPORTS))}")
    if "acl" in sys.modules and getattr(sys.modules["acl"], "__offline_stub__", False):
        print("  [INFO] 已注入**假 pyACL**（ascend 的有界同步走它，真实 acl 不会被触及）")

    # ── 1. 发现 / 实例化（ABC 会在实例化时强制 13 个抽象方法齐全）──
    print("\n[1] 发现 / 实例化 / 抽象方法完整性")
    runtime, loaded, state = fresh_import(proto_dir, backend, stub_fn)
    check("registry.discover 能发现该后端", loaded == [backend], str(loaded))
    from runtime.backends.registry import use
    bk = use(backend)
    check("build() 可实例化（ABC 已强制 13 个抽象方法全部实现）", bk.name == backend)
    check(f"device_type == '{dev_ns}'", bk.device_type == dev_ns, bk.device_type)

    # ── 2. 设备域（D2/D3）──
    print("\n[2] 设备域：count / set_device / memory_stats / probe_device")
    check("device_count > 0", bk.device_count() > 0, str(bk.device_count()))
    bk.set_device(3)
    check("set_device 生效", state["current"] == 3)
    mem = bk.memory_stats(0)
    if caps.get("stub_mem_stats_shape", True):
        check("memory_stats 结构 = {total_mb,used_mb,free_mb}",
              set(mem) == {"total_mb", "used_mb", "free_mb"}, str(mem))
        check("memory_stats 值有效（>0 且 total=used+free 近似）",
              mem["total_mb"] > 0 and abs(mem["total_mb"] - mem["used_mb"] - mem["free_mb"]) <= 1,
              str(mem))
    else:
        skip("memory_stats 结构/取值",
             "该后端从厂商命名空间原样透传原始字段；stub 造的原始返回形状与真实厂商栈不一致，"
             "据此判定等于用假数据判真实现 —— 须真机跑 smoke 判定（smoke 已含同口径判据）")
    check("memory_stats 用完还原当前设备（不产生隐式副作用）", state["current"] == 3)
    if caps.get("stub_real_ops", True):
        check("probe_device 返回 True（真值路径）", bk.probe_device(0) is True)
    else:
        skip("probe_device 真值路径",
             "该后端的探活走 conformance/recovery 的真实设备计算，stub 无真实算子可跑；"
             "真机 conformance / smoke 已覆盖（r_recovery 用例）")

    # ── 3. 流 / 事件域（D4/D5）+ 有界同步 ──
    print("\n[3] 流 / 事件域 + 有界同步（接口约定 §1.3）")
    st = bk.create_stream()
    bk.create_event()
    check("create_stream / create_event / current_stream 可用",
          st is not None and bk.current_stream() is not None)
    with bk.stream_context(st):
        pass
    check("stream_context 可作上下文管理器", True)
    bk.synchronize(0)
    check("synchronize(timeout_ms=None) 原生阻塞路径不抛错", True)
    bk.synchronize(0, timeout_ms=0)
    check("已完成任务 + timeout_ms=0 → 正常返回（不误判超时）", True)
    if caps.get("stub_bounded_sync", True):
        state["stream_query"] = False
        try:
            bk.synchronize(0, timeout_ms=0)
            check("未完成任务 + timeout_ms=0 → 应抛 TimeoutError", False)
        except TimeoutError:
            check("未完成任务 + timeout_ms=0 → 抛 TimeoutError（有界成立）", True)
        try:
            bk.synchronize_stream(st, 0)
            check("synchronize_stream(timeout_ms=0) → 应抛 TimeoutError", False)
        except TimeoutError:
            check("synchronize_stream(timeout_ms=0) → 抛 TimeoutError（有界成立）", True)
        state["stream_query"] = False
        t0 = time.monotonic()
        try:
            bk.synchronize(0, timeout_ms=200)
        except TimeoutError:
            pass
        dt = (time.monotonic() - t0) * 1000
        check("有界同步真的按时返回（非永久阻塞）", 180 <= dt <= 900, f"{dt:.0f} ms")
        state["stream_query"] = True
    elif "acl_rc" in state:
        # 该家的有界同步走 pyACL —— stub 提供了**可控 rc 的假 acl**，于是可以**离线**验证
        # 「rc 分类」这条最关键的判据。这是第 9 个原型缺陷（把 acl.init/参数错误冒充成同步超时）
        # 的回归守卫：真机暴露的那个 bug，本可以在这里被拦下。
        rc_state = state["acl_rc"]

        rc_state["sync"] = 507046            # ACL_ERROR_RT_STREAM_SYNC_TIMEOUT（真超时码）
        try:
            bk.synchronize(0, timeout_ms=50)
            check("真超时码(507046) ⇒ 必须抛 TimeoutError", False)
        except TimeoutError:
            check("真超时码(507046) ⇒ TimeoutError（有界语义成立）", True)
        except Exception as e:
            check("真超时码(507046) ⇒ TimeoutError", False, f"{type(e).__name__}: {str(e)[:70]}")

        rc_state["sync"] = 107000            # ACL_ERROR_RT_PARAM_INVALID（L2，非超时）
        try:
            bk.synchronize(0, timeout_ms=50)
            check("非超时码(107000) ⇒ 必须抛错", False)
        except TimeoutError:
            check("非超时码(107000) ⇒ **不得**冒充 TimeoutError", False,
                  "被误报为超时 ⇒ 下游会按 L3 去 replay，正确动作是 L2 的 raise")
        except Exception as e:
            _cat = getattr(getattr(e, "category", None), "name", None)
            check("非超时码(107000) ⇒ 按错误码表分级为 L2_PARAM（不冒充超时）",
                  _cat == "L2_PARAM",
                  f"{type(e).__name__}: category={_cat} "
                  f"disposition={getattr(e, 'disposition', None)}")

        rc_state["sync"] = 0
        check("rc=0 ⇒ 正常返回（不误判）", bk.synchronize(0, timeout_ms=50) is None)

        # set_device 的 rc 也必须分级（实测：上下文无效时它先返回 107002）
        rc_state["set_device"] = 107002      # ACL_ERROR_RT_CONTEXT_NULL
        try:
            bk.synchronize(0, timeout_ms=50)
            check("set_device rc≠0 ⇒ 必须抛错", False)
        except TimeoutError:
            check("set_device rc≠0 ⇒ **不得**冒充 TimeoutError", False)
        except Exception as e:
            _cat = getattr(getattr(e, "category", None), "name", None)
            check("set_device rc≠0 ⇒ 按错误码表分级（L2_PARAM）", _cat == "L2_PARAM",
                  f"{type(e).__name__}: category={_cat}")
        rc_state["set_device"] = 0
    else:
        skip("有界同步三项（未完成抛 Timeout / 按时返回）",
             "该后端的有界同步不支持可控 rc 的假原语，stub 无法产生'任务未完成'状态；"
             "真机经 conformance e2/e2-b 与错误闭环已覆盖")

    # ── 4. 事件语义契约（E3 / E2-v2）──
    print("\n[4] 事件语义契约（conformance E3 / E2-v2）")
    ev = bk.create_event()
    check("E3：未 record 的 query() 为 False（原生坏行为已被适配层修正）",
          ev.query() is False, str(ev.query()))
    t0 = time.monotonic()
    r = ev.wait_host(200)
    dt = (time.monotonic() - t0) * 1000
    check("E2-v2：未 record wait_host(200) 返回 False 且 <1s（不永久阻塞）",
          r is False and dt < 1000, f"{r}, {dt:.0f}ms")
    ev.record(st)
    check("record 后 query() 为 True", ev.query() is True)
    check("record 后 wait_event_host 返回 True", bk.wait_event_host(ev, 100) is True)

    # ── 5. 错误翻译（F1：三投影 + 不冒充码表）──
    print("\n[5] 错误翻译（conformance F1）")
    fe = bk.translate_error(ValueError("bad param"), location="op:matmul")
    check("返回统一 FlagosError", isinstance(fe, runtime.FlagosError), type(fe).__name__)
    check("后端名已回填（后端侧回填，不依赖 api 层）", fe.backend == backend, fe.backend)
    check("disposition 可用（枚举已归一，无 KeyError）",
          fe.disposition in ("retry", "raise", "replay", "device_recovery"), fe.disposition)
    check("根因原样保留（F4）", bool(fe.root_cause))
    # 用 **PyTorch 真实文案** 做输入（conformance F1 注入的就是一个真实 matmul 形状错）：
    #   `torch.randn(3,4) @ torch.randn(5,6)` → "mat1 and mat2 shapes cannot be multiplied"
    # ⚠️ 真机上这条是否通过，取决于**厂商栈抛出的文案**是否含 shape/dimension 一类关键词
    #    （分级走的是消息规则，不是码表）；这属于 conformance 第 5 步的范畴，**本自查不能代替**。
    fe_shape = bk.translate_error(
        RuntimeError("mat1 and mat2 shapes cannot be multiplied (3x4 and 5x6)"),
        location="stream:0/op:matmul")
    check("F1：形状不匹配（真实 PyTorch 文案）→ L2_PARAM",
          fe_shape.category.name == "L2_PARAM", fe_shape.category.name)
    if bk.supports("error_map"):
        check("声明 error_map ⇒ 无码消息不得伪称 code_map",
              fe.graded_by != "code_map", fe.graded_by)
        sample = getattr(bk, "SAMPLE_CODED_ERROR", None)
        if sample:
            fc = bk.translate_error(RuntimeError(sample), location="self-check")
            check("声明 error_map ⇒ 含码样例必走 code_map",
                  fc.graded_by == "code_map", fc.graded_by)
        else:
            print("  [SKIP] 未提供 SAMPLE_CODED_ERROR，正向码表检查跳过（如实）")
    else:
        check("未声明 error_map ⇒ 不得冒充码表命中（诚实性）",
              fe_shape.graded_by != "code_map", fe_shape.graded_by)
        check("未声明 error_map ⇒ mapped 必须为 False（F1 判据要求）",
              fe_shape.mapped is False, str(fe_shape.mapped))
        # ⭐ 2026-09-22 新增（第 6 个跨后端缺陷的防回归判据）
        #   背景：底层共享翻译器 `conformance/errors.py` 的码表是**单一厂商（昇腾 ACL）码表**，
        #   抽取规则却很通用（`ret=<数字>` / `error code is <数字>`）。于是只要异常消息里
        #   恰好出现一个**码表内的数字**，它就会返回 `graded_by="code_map"` 且 `mapped=True`。
        #   对**没有厂商码表**的后端（kunlun / cambricon），这不是本厂商的码表命中 ⇒
        #   `graded_by` / `mapped` / `error_code` 必须**一起**降级。只改 `graded_by` 会留下
        #   「`mapped=True` 但 `graded_by≠code_map`」的自相矛盾，而 `mapped=True` 的契约含义是
        #   **确定分级** ⇒ 等于把保守推断冒充成定论（违反 I2）。
        #   本判据**只用消息文本、不碰设备** —— 正是"无设备先自查"该发挥作用的场合：
        #   该缺陷首次是在真机错误闭环里被发现的，而它本可以在上机前就被这里拦住。
        #   样例用 507015（共享码表内 → 会触发 code_map 路径），文案与错误闭环的注入一致。
        _foreign_code_msg = "AICORE exception, error code is 507015"
        ff = bk.translate_error(RuntimeError(_foreign_code_msg), location="self-check")
        check("码表内码串入 ⇒ graded_by 不得为 code_map",
              ff.graded_by != "code_map", ff.graded_by)
        check("码表内码串入 ⇒ mapped 必须为 False（否则=把保守推断冒充成确定分级）",
              ff.mapped is False, f"mapped={ff.mapped} graded_by={ff.graded_by}")
        check("码表内码串入 ⇒ error_code 必须为 None（该码非本厂商码）",
              ff.error_code is None, str(ff.error_code))

    # ── 6. 恢复与设备状态（R1-R5）──
    print("\n[6] 恢复与设备状态（conformance R1-R5）")
    rec = bk.recover_device(0, mode="probe")
    check("recover_device 返回 dict 且含 recovered（统一契约）",
          isinstance(rec, dict) and "recovered" in rec, str(rec)[:70])
    # 2026-09-22：原先这里直接调 `bk.device_state(0)`，若后端**声明了 device_state 却没实现**
    # 会抛 AttributeError 把整轮自检中斷（只留 traceback，连汇总都打不出来）——
    # 那正好把"声明与实现不符"这类**最重要**的缺陷变成一次崩溃。
    # 现改为捕获 AttributeError 并判 FAIL（**该判据的来由**：路线 B 后端曾声明
    # `device_state` 却无实现、整轮自检直接 traceback —— 该后端已随路线 B 删除，
    # 判据保留，防同类问题在任何厂商后端上复现）。
    try:
        ds = bk.device_state(0)
        check("device_state 返回四态之一",
              str(ds).split(".")[-1].lower() in
              ("available", "degraded", "isolated", "destroyed"), str(ds))
    except AttributeError as e:
        check("device_state 可调用（声明了该能力就必须有实现）", False,
              f"AttributeError: {e} —— 能力声明与实现不符（与 kunlun 2026-09-20 同类）")
    except Exception as e:
        check("device_state 可调用", False, f"{type(e).__name__}: {str(e)[:110]}")
    if not bk.supports("recovery_real"):
        rec2 = bk.recover_device(0, mode="real")
        check("未声明 recovery_real ⇒ real 模式如实说明不支持（不伪造重建）",
              isinstance(rec2, dict) and "detail" in rec2, rec2.get("detail", "")[:60])

    # ── 7. 能力声明自洽 + known_issues ──
    print("\n[7] 能力声明自洽 / 已知问题（stub-skip 报告基础）")
    info = bk.info()
    bad = sorted(k for k in info.get("capabilities", []) if not bk.supports(k))
    check("info.capabilities 与 supports() 自洽", not bad, str(bad))
    # 2026-09-22 新增：smoke 已有"info.supports 的各项取值与 supports() 一致"判据，
    # 但**键名漂移**不在其覆盖内 —— 若 info() 手写了第二份键名清单，两边都取 False，
    # 取值一致性照样成立，而读者从 info() 会看到"已声明能力全是 False"。
    # 故这里额外要求**键集合 == 能力全集**（有 `_CAPABILITY_KEYS` 用它，否则用 `_capabilities`）。
    _sup_keys = info.get("supports")
    if isinstance(_sup_keys, dict) and _sup_keys:
        _universe = getattr(bk, "_CAPABILITY_KEYS", None) or tuple(sorted(bk._capabilities))
        _missing = sorted(set(_universe) - set(_sup_keys))
        _extra = sorted(set(_sup_keys) - set(_universe))
        check("info()['supports'] 键集合 == 能力全集（防键名漂移）",
              not _missing and not _extra,
              f"缺 {_missing} / 多 {_extra}" if (_missing or _extra)
              else f"{len(_sup_keys)} 键")

    sup = info.get("supports")
    if isinstance(sup, dict):
        mism = sorted(k for k, v in sup.items() if bool(v) != bk.supports(k))
        check("info.supports 与 supports() 一致", not mism, str(mism))
    else:
        skip("info()['supports'] 一致性/键集合",
             "本后端的 info() 未提供 supports 映射（三家中仅个别如此）—— "
             "下游「读后端即知能力」的统一体验在这家不成立，建议补齐"
             "（2026-09-22 已为 ascend 补齐该缺口）")
    ki = bk.known_issues()
    _req = ("id", "severity", "scope", "condition", "symptom",
            "root_cause_layer", "workaround", "report_to", "evidence")
    check("known_issues 结构完整（若为空则跳过 —— 无缺陷也要如实说没有）",
          all(all(k in x for k in _req) for x in ki),
          f"{len(ki)} 条: {[x.get('id') for x in ki]}")
    if not bk.supports("stream_priority"):
        check("未声明 stream_priority ⇒ stream_priority_range() 返回 None",
              bk.stream_priority_range() is None)
    if not bk.supports("graph_capture"):
        print("  [INFO] 未声明 graph_capture：`torch.<ns>.graph` 存在性须容器内实测后决定")

    # ── 8. 缺厂商扩展时不得静默降级 ──
    print("\n[8] 缺厂商扩展时的行为（应报错并给出下一步，不得静默降级）")
    if caps.get("stub_vendor_extension", True):
        runtime2, loaded2, _ = fresh_import(proto_dir, backend, stub_fn, with_vendor=False)
        bk2 = runtime2.get(backend)
        try:
            bk2.device_count()
            check("缺厂商扩展时应报错", False, "未报错 —— 会静默降级，必须修")
        except Exception as e:
            msg = str(e)
            check("缺厂商扩展时抛错且文案可操作",
                  len(msg) > 20 and ("torch" in msg.lower()), f"{type(e).__name__}: {msg[:60]}")
    else:
        skip("缺厂商扩展时应报错",
             "该后端没有**独立可缺失的厂商扩展模块** —— 它的'扩展'是被替换过的 torch 本身"
             "（如昆仑芯 XPytorch），摘掉模块这一步在本语义空间下无法构造；"
             "真机上表现为 device_count()==0，由 registry/smoke 的 device_count 判据覆盖")

    print("\n" + "=" * 74)
    print(f"离线自检结果: {PASS} 通过 / {FAIL} 失败 / {SKIPPED} 跳过（stub 能力边界，非失败）")
    print("⚠️ 结论边界：本结果只证明实现逻辑与契约形态。")
    print("   厂商 API 真实形态 / 错误码 / 集合通信后端名 / conformance 通过与否，")
    print("   一律必须在真机容器内实测（见《新芯片接入手册》第 5 步）。")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


def symmetry(proto_dir):
    """`--all`：**跨后端对称性自检** —— 把各家的"对外可观测形态"摆在一起比对。

    动机：本方向的核心主张是"统一 API，换芯片只改一行"。**这条主张的敌人是不对称** ——
    某一家少了方法、少了字段、字段语义不同，都属于"统一 API"名不副实。
    2026-09-22 第 3 家接入的收口期，正是靠这一类对比挖出三处缺陷
    （历史实例：某后端声明 `device_state` 却无实现、`info()` 键名漂移、`ascend` 的 `info()` 缺 `supports`；
    前两者所在的后端已随路线 B 删除，判据保留以守护现役三家）。

    两类输出：
      · **硬判据**（计入 通过/失败）：所有后端都必须成立的不变式
      · **差异清单**（不计失败）：允许存在但**必须能说清理由**的差异（如 `device_type` 本就不同）
    """
    print("=" * 74)
    print("跨后端对称性自检（--all）：硬判据 + 差异清单")
    print(f"参与后端：{sorted(_STUBS)}")
    print("=" * 74)

    rows: dict = {}
    for backend, (stub_fn, dev_ns, caps) in _STUBS.items():
        try:
            runtime, loaded, state = fresh_import(proto_dir, backend, stub_fn)
            bk = runtime.get(backend)
            if bk is None:
                rows[backend] = {"error": "discover 未发现"}
                continue
            info = bk.info() or {}
            rows[backend] = {
                "device_type": bk.device_type,
                "capabilities": set(getattr(bk, "_capabilities", set())),
                "cap_keys": set(getattr(bk, "_CAPABILITY_KEYS", ()) or ()),
                "info_keys": sorted(info),
                "supports_keys": set(info.get("supports") or {}),
                "methods": {m: callable(getattr(bk, m, None)) for m in
                            ("info", "supports", "known_issues", "device_state",
                             "stream_priority_range", "recover_device", "translate_error")},
                "n_issues": len(bk.known_issues() or []),
                "sample_coded": getattr(bk, "SAMPLE_CODED_ERROR", None) is not None,
                "declares_error_map": bool(bk.supports("error_map")),
                "_bk": bk,
            }
        except Exception as e:
            rows[backend] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}

    ok_rows = {k: v for k, v in rows.items() if "error" not in v}
    print("\n[表] 各家可观测形态")
    hdr = f"  {'后端':<11}{'device_type':<12}{'能力数':<7}{'info 键':<9}{'supports 键':<11}{'known_issues':<13}{'码样例':<7}"
    print(hdr)
    for k in sorted(rows):
        v = rows[k]
        if "error" in v:
            print(f"  {k:<11}[加载失败] {v['error']}")
            continue
        print(f"  {k:<11}{v['device_type']:<12}{len(v['capabilities']):<7}"
              f"{len(v['info_keys']):<9}{len(v['supports_keys']):<11}"
              f"{v['n_issues']:<13}{'有' if v['sample_coded'] else '-':<7}")

    print("\n[硬判据] 所有后端都必须成立")
    if not ok_rows:
        check("至少有一家后端可加载", False, str({k: v.get("error") for k, v in rows.items()}))
    else:
        check("三家全部可加载（无异常）", len(ok_rows) == len(rows),
              str({k: v.get("error") for k, v in rows.items() if "error" in v}))

        # ① 对外必需方法齐备
        req = ("info", "supports", "known_issues", "device_state", "recover_device", "translate_error")
        miss = {k: [m for m in req if not v["methods"][m]] for k, v in ok_rows.items()}
        check("对外必需方法齐备（info/supports/known_issues/device_state/recover/translate）",
              all(not m for m in miss.values()),
              str({k: m for k, m in miss.items() if m}))

        # ② info() 必须提供 supports 映射（"读后端即知能力"的统一体验）
        nosup = [k for k, v in ok_rows.items() if not v["supports_keys"]]
        check("info() 均提供 supports 映射", not nosup, str(nosup))

        # ③ 声明的能力必须落在那家的能力全集内（防键名拼错）
        bad = {k: sorted(v["capabilities"] - v["cap_keys"])
               for k, v in ok_rows.items() if v["cap_keys"] and v["capabilities"] - v["cap_keys"]}
        check("声明能力 ⊆ 能力全集（防键名拼错）", not bad, str(bad))

        # ④ 声明 error_map 的家必须提供码样例（否则正向 code_map 检查无从下手）
        need = [k for k, v in ok_rows.items() if v["declares_error_map"] and not v["sample_coded"]]
        check("声明 error_map ⇒ 必须提供 SAMPLE_CODED_ERROR", not need, str(need))

        # ⑤ 同一能力在不同家的"声明与否"必须能解释 —— 这里只报差异，不判失败
        all_caps = sorted(set().union(*[v["capabilities"] for v in ok_rows.values()])) if ok_rows else []
        print("\n[差异清单] 允许存在，但每处都应有理由（不计失败）")
        for c in all_caps:
            holders = sorted(k for k, v in ok_rows.items() if c in v["capabilities"])
            if 0 < len(holders) < len(ok_rows):
                print(f"  · 能力 {c:<18} 仅 {holders} 声明")
        dts = sorted({v["device_type"] for v in ok_rows.values()})
        if len(dts) > 1:
            print(f"  · device_type 不同：{dts} —— **这是设计而非缺陷**"
                  f"（设备串按命名空间，不按厂商；见接口约定修订建议第 1 条）")
        sup_sets = {k: tuple(sorted(v["supports_keys"])) for k, v in ok_rows.items()}
        if len(set(sup_sets.values())) > 1:
            print("  · supports 键集合不一致（各家的「已知能力全集」略有差异，"
                  "如 ascend 保留历史键名 sync_timeout）")

    print("\n" + "=" * 74)
    print(f"对称性自检结果: {PASS} 通过 / {FAIL} 失败")
    print("⚠️ 结论边界：同上 —— 仅证明可观测形态的一致性，不证明真机可用。")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="后端离线契约自检（无设备）")
    ap.add_argument("--backend", default=None,
                    help=f"后端名；当前已内置 stub 的厂商：{sorted(_STUBS)}")
    ap.add_argument("--all", action="store_true",
                    help="跨后端对称性自检（比对四家的可观测形态，含硬判据）")
    ap.add_argument("--proto", default=None,
                    help="prototype 目录（默认取本脚本上一级）")
    args = ap.parse_args(argv)
    if not args.all and not args.backend:
        ap.error("需给出 --backend <名字> 或 --all")
    if args.backend and args.backend not in _STUBS:
        print(f"❌ 没有为 '{args.backend}' 内置 stub。已内置：{sorted(_STUBS)}")
        print("   加一家厂商 = 在 _STUBS 里加一个 stub 函数（见文件头说明）。")
        return 2
    proto = args.proto or str(Path(__file__).resolve().parents[1])
    if not os.path.isdir(os.path.join(proto, "runtime")):
        print(f"❌ {proto} 下没有 runtime/，请用 --proto 指定 prototype 目录")
        return 2
    if args.all:
        return _guarded(symmetry, proto)
    return _guarded(run, proto, args.backend)


def _guarded(fn, *a):
    """兜底：**未预期异常不得让整轮自检无汇总地崩掉**。

    2026-09-22 实测教训（两次）：① `device_state` 未实现 ⇒ AttributeError 整轮崩；
    ② 910C 真机上真实 `acl` 被 import ⇒ 后端抛 TimeoutError 整轮崩。
    两次都只留下 traceback、**连汇总都打不出**，等于把最重要的缺陷变成一次崩溃。
    ⇒ 统一在入口兜底：判 1 条失败 + 打印异常与末帧 + **照常输出汇总**。
    """
    global FAIL
    try:
        return fn(*a)
    except Exception as e:
        FAIL += 1
        tb = traceback.format_exc().strip().splitlines()
        print("\n[FATAL] 自检本身抛出未预期异常（已判 1 条失败；以下为定位信息）")
        print(f"        {type(e).__name__}: {str(e)[:220]}")
        print(f"        末帧: {tb[-2].strip() if len(tb) >= 2 else '(无)'}")
        print("\n" + "=" * 74)
        print(f"离线自检结果: {PASS} 通过 / {FAIL} 失败 / {SKIPPED} 跳过"
              f"（含 1 条因未预期异常判失败）")
        print("⚠️ 结论边界：本结果只证明实现逻辑与契约形态。")
        print("=" * 74)
        return 1


if __name__ == "__main__":
    sys.exit(main())
