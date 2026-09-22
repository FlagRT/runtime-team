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
import types
from pathlib import Path

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ═════════════════════════ stub：厂商命名空间 ═════════════════════
def _stub_cambricon(mode="full"):
    """寒武纪 `torch.mlu`（PrivateUse1）stub。

    mode 控制显存查询能力的三种情形，用于验证后端的取值兜底链：
      'full'   → `mem_get_info(ordinal)` 可用
      'noarg'  → 只接受无参 `mem_get_info()`
      'absent' → 没有 `mem_get_info`（须降级到 total_memory - memory_allocated）
    """
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0+stub"
    mlu = types.ModuleType("torch.mlu")
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

    mlu.device_count = lambda: 8
    mlu.set_device = lambda o: state.__setitem__("current", o)
    mlu.current_device = lambda: state["current"]
    mlu.Stream = Stream
    mlu.Event = Event
    mlu.current_stream = lambda: Stream()
    mlu.stream = lambda s: _Ctx()
    mlu.synchronize = lambda *a, **k: None
    mlu.get_device_properties = lambda o: Props()
    mlu.memory_allocated = lambda o: 1 * 1024 ** 3
    if mode == "full":
        mlu.mem_get_info = lambda ordinal=None: (80 * 1024 ** 3, 96 * 1024 ** 3)
    elif mode == "noarg":
        mlu.mem_get_info = lambda: (80 * 1024 ** 3, 96 * 1024 ** 3)
    # 'absent'：不设该属性

    torch.zeros = lambda *a, **k: _T(0.0)
    torch.mlu = mlu
    return torch, {"torch_mlu": types.ModuleType("torch_mlu")}, state


#: 厂商 stub 注册表 —— 新厂商在这里加一项即可
_STUBS = {
    "cambricon": (_stub_cambricon, "mlu"),
}


def fresh_import(proto_dir, backend, stub_fn, mode="full", with_vendor=True):
    """清干净再导入：确保每次都是"全新发现"（注册表单例会跨用例残留）。"""
    for m in [k for k in list(sys.modules)
              if k.startswith("runtime") or k in ("torch", "torch_mlu", "torch_npu")]:
        del sys.modules[m]
    if proto_dir not in sys.path:
        sys.path.insert(0, proto_dir)
    torch, vendor_mods, state = stub_fn(mode)
    sys.modules["torch"] = torch
    if with_vendor:
        sys.modules.update(vendor_mods)
    import runtime
    from runtime.backends.registry import clear
    clear()
    loaded = runtime.discover(names=(backend,), verbose=False)
    return runtime, loaded, state


def run(proto_dir, backend):
    stub_fn, dev_ns = _STUBS[backend]
    print("=" * 74)
    print(f"后端离线契约自检：backend={backend}（device_type 期望={dev_ns}）")
    print("⚠️ 无真实设备：只验实现逻辑与契约形态，不验厂商 API 真实行为")
    print("=" * 74)

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
    check("memory_stats 结构 = {total_mb,used_mb,free_mb}",
          set(mem) == {"total_mb", "used_mb", "free_mb"}, str(mem))
    check("memory_stats 值有效（>0 且 total=used+free 近似）",
          mem["total_mb"] > 0 and abs(mem["total_mb"] - mem["used_mb"] - mem["free_mb"]) <= 1,
          str(mem))
    check("memory_stats 用完还原当前设备（不产生隐式副作用）", state["current"] == 3)
    check("probe_device 返回 True（真值路径）", bk.probe_device(0) is True)

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

    # ── 6. 恢复与设备状态（R1-R5）──
    print("\n[6] 恢复与设备状态（conformance R1-R5）")
    rec = bk.recover_device(0, mode="probe")
    check("recover_device 返回 dict 且含 recovered（统一契约）",
          isinstance(rec, dict) and "recovered" in rec, str(rec)[:70])
    ds = bk.device_state(0)
    check("device_state 返回四态之一",
          str(ds).split(".")[-1].lower() in
          ("available", "degraded", "isolated", "destroyed"), str(ds))
    if not bk.supports("recovery_real"):
        rec2 = bk.recover_device(0, mode="real")
        check("未声明 recovery_real ⇒ real 模式如实说明不支持（不伪造重建）",
              isinstance(rec2, dict) and "detail" in rec2, rec2.get("detail", "")[:60])

    # ── 7. 能力声明自洽 + known_issues ──
    print("\n[7] 能力声明自洽 / 已知问题（stub-skip 报告基础）")
    info = bk.info()
    bad = sorted(k for k in info.get("capabilities", []) if not bk.supports(k))
    check("info.capabilities 与 supports() 自洽", not bad, str(bad))
    sup = info.get("supports")
    if isinstance(sup, dict):
        mism = sorted(k for k, v in sup.items() if bool(v) != bk.supports(k))
        check("info.supports 与 supports() 一致", not mism, str(mism))
    else:
        print("  [SKIP] info() 无 supports 映射（非必须，跳过一致性检查）")
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
    runtime2, loaded2, _ = fresh_import(proto_dir, backend, stub_fn, with_vendor=False)
    bk2 = runtime2.get(backend)
    try:
        bk2.device_count()
        check("缺厂商扩展时应报错", False, "未报错 —— 会静默降级，必须修")
    except Exception as e:
        msg = str(e)
        check("缺厂商扩展时抛错且文案可操作",
              len(msg) > 20 and ("torch" in msg.lower()), f"{type(e).__name__}: {msg[:60]}")

    print("\n" + "=" * 74)
    print(f"离线自检结果: {PASS} 通过 / {FAIL} 失败")
    print("⚠️ 结论边界：本结果只证明实现逻辑与契约形态。")
    print("   厂商 API 真实形态 / 错误码 / 集合通信后端名 / conformance 通过与否，")
    print("   一律必须在真机容器内实测（见《新芯片接入手册》第 5 步）。")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="后端离线契约自检（无设备）")
    ap.add_argument("--backend", required=True,
                    help=f"后端名；当前已内置 stub 的厂商：{sorted(_STUBS)}")
    ap.add_argument("--proto", default=None,
                    help="prototype 目录（默认取本脚本上一级）")
    args = ap.parse_args(argv)
    if args.backend not in _STUBS:
        print(f"❌ 没有为 '{args.backend}' 内置 stub。已内置：{sorted(_STUBS)}")
        print("   加一家厂商 = 在 _STUBS 里加一个 stub 函数（见文件头说明）。")
        return 2
    proto = args.proto or str(Path(__file__).resolve().parents[1])
    if not os.path.isdir(os.path.join(proto, "runtime")):
        print(f"❌ {proto} 下没有 runtime/，请用 --proto 指定 prototype 目录")
        return 2
    return run(proto, args.backend)


if __name__ == "__main__":
    sys.exit(main())
