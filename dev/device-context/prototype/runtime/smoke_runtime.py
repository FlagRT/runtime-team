#!/usr/bin/env python3
"""
运行时框架冒烟测试（本地，无需 NPU）

验证范围：
  1. 注册表机制：register / use / get / available / BackendNotFound
  2. 统一错误对象：分级 → 处置策略映射、可观测字段
  3. Mock 后端：证明"新增后端 = 实现接口 + 注册"可行（kunlun stub 同此路径）
  4. 真实后端（ascend）：若环境有 torch_npu 则一并验证，否则跳过（不算失败）
  5. **真实后端通用自检（后端无关）**：按注册表自动挑选可用后端，跑与厂商无关的契约检查；
     能力相关项按 `supports()` 如实 SKIP
     （2026-09-14 新增：昆仑芯接入时暴露「smoke 只覆盖昇腾、不覆盖其他后端」的缺口）

用法：
    python3 smoke_runtime.py                     # 第 5 节自动挑选可用真实后端
    python3 smoke_runtime.py --backend kunlun    # 指定后端
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime
from runtime.api.errors import ErrorCategory, FlagosError
from runtime.backends.base import RuntimeBackend
from runtime.backends.registry import BackendNotFound, available, clear, use

passed, failed = 0, 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {detail}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {detail}")


class _MockNativeStream:
    """模拟后端原生流对象（需具备 wait_event/wait_stream/synchronize）。"""
    def wait_event(self, event): pass
    def wait_stream(self, other): pass
    def synchronize(self): pass


class _MockNativeEvent:
    """模拟后端原生事件对象（需具备 record/query/synchronize）。"""
    def __init__(self):
        self._recorded = False

    def record(self, stream=None):
        self._recorded = True

    def query(self):
        return self._recorded

    def synchronize(self):
        self._recorded = True


class MockBackend(RuntimeBackend):
    """最小后端实现：证明接口可实现（kunlun stub 走同一路径）。"""
    name = "mock"
    device_type = "mock"
    _capabilities = {"device", "stream"}

    def __init__(self):
        self.devices = 2
        self.current = None

    def device_count(self): return self.devices
    def set_device(self, o): self.current = o

    def memory_stats(self, o):
        return {"total_mb": 65536, "used_mb": 1024, "free_mb": 64512}

    def create_stream(self): return _MockNativeStream()
    def create_event(self): return _MockNativeEvent()
    def current_stream(self): return "mock-current"

    def synchronize(self, ordinal, timeout_ms=None):
        if timeout_ms == 0:
            raise TimeoutError("mock timeout")

    # 多流支撑（新增接口）
    def stream_context(self, native_stream):
        import contextlib
        return contextlib.nullcontext()

    def synchronize_stream(self, native_stream, timeout_ms):
        if timeout_ms == 0:
            raise TimeoutError("mock stream timeout")

    def wait_event_host(self, native_event, timeout_ms):
        return True

    def translate_error(self, exc, location=""):
        return FlagosError(
            category=ErrorCategory.L2_PARAM,
            root_cause=str(exc),
            location=location,
            mapped=True,
            graded_by="code_map",
            is_grade_confident=True,
        )

    def probe_device(self, ordinal): return True
    def recover_device(self, ordinal, mode="probe", reason=""): return True


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="运行时框架冒烟测试（本地）")
    ap.add_argument("--backend", default=None,
                    help="指定第 [6] 节要自检的真实后端（默认自动挑选可用者）")
    args = ap.parse_args(argv)

    print("=== 运行时框架冒烟测试（本地）===\n")

    # ── 1. 注册表机制 ──
    print("[1] 注册表机制")
    clear()
    check("初始无后端", available() == [], f"got {available()}")

    try:
        use("mock")
        check("未注册时 use 应抛错", False)
    except BackendNotFound:
        check("未注册时 use 抛 BackendNotFound", True)

    runtime.register(MockBackend())
    check("注册后可见", "mock" in available(), f"got {available()}")

    b = use("mock")
    check("use 返回后端实例", b.name == "mock")
    check("current 一致", runtime.current().name == "mock")
    check("后端单例", runtime.current() is b)

    # ── 2. 转发 API ──
    print("\n[2] 统一 API 转发")
    check("device_count", runtime.device_count() == 2)
    runtime.set_device(1)
    check("set_device", b.current == 1)
    mem = runtime.memory_stats(0)
    check("memory_stats 结构", set(mem) == {"total_mb", "used_mb", "free_mb"}, str(mem))
    st = runtime.create_stream()
    ev = runtime.create_event()
    check("create_stream 返回统一 Stream",
          isinstance(st, runtime.Stream), type(st).__name__)
    check("Stream 持有后端引用", st.backend.name == "mock")
    check("create_event 返回统一 Event",
          isinstance(ev, runtime.Event), type(ev).__name__)
    check("Event 持有后端引用", ev.backend.name == "mock")
    ev.record(st)
    check("Event.record 可用", True)
    check("Event.wait_host 有界", ev.wait_host(timeout_ms=100) is True)

    try:
        runtime.synchronize(0, timeout_ms=0)
        check("有界同步超时抛错", False)
    except TimeoutError:
        check("有界同步超时抛 TimeoutError", True)

    # ── 3. 错误分级 ──
    print("\n[3] 统一错误分级")
    fe = runtime.translate_error(ValueError("bad param"), location="op:matmul")
    check("分级为 L2_PARAM", fe.category == ErrorCategory.L2_PARAM)
    check("处置 = 上抛", fe.disposition == "raise", fe.disposition)
    check("重试性 = False", fe.retryable is False)
    check("可观测字段", fe.mapped and fe.graded_by == "code_map")
    check("后端名回填", fe.backend == "mock", fe.backend)
    check("location 透传", fe.location == "op:matmul")
    check("__str__ 可读", "L2_PARAM" in str(fe), str(fe)[:60])

    # 分级 → 处置 全映射
    expect = {
        ErrorCategory.L1_RESOURCE: "retry",
        ErrorCategory.L2_PARAM: "raise",
        ErrorCategory.L3_EXECUTION: "replay",
        ErrorCategory.L4_FATAL: "device_recovery",
    }
    ok = all(
        FlagosError(category=c, root_cause="x").disposition == d
        for c, d in expect.items()
    )
    check("L1-L4 → 处置全映射", ok)

    # ── 4. 能力查询（stub-skip 报告基础）──
    print("\n[4] 能力查询")
    check("supports(device)", b.supports("device"))
    check("不支持的能力返回 False", not b.supports("graph_capture"))
    check("info 含 name/device_type/capabilities",
          set(b.info()) == {"name", "device_type", "capabilities"}, str(b.info()))

    # ── 5. 真实后端（若环境可用）──
    print("\n[5] 昇腾后端（需 torch_npu，不可用则跳过）")
    clear()
    loaded = runtime.discover(names=("ascend",), verbose=False)
    if "ascend" in loaded:
        ab = use("ascend")
        check("ascend 加载", ab.name == "ascend")
        try:
            n = ab.device_count()
            check("device_count > 0", n > 0, f"n={n}")
            if n:
                ab.set_device(0)
                m = ab.memory_stats(0)
                check("memory_stats 真实值", m["total_mb"] > 0, str(m))
                st = ab.create_stream()
                check("create_stream 真实对象", st is not None, type(st).__name__)
                check("probe_device", ab.probe_device(0) is True)
                fe2 = ab.translate_error(
                    RuntimeError("ACL stream sync timeout, error code is 507046"),
                    location="probe")
                check("真实错误码翻译（507046 → L3_EXECUTION）",
                      fe2.category == ErrorCategory.L3_EXECUTION,
                      f"{fe2.category.value} graded_by={fe2.graded_by} code={fe2.error_code}")
                check("统一类型（历史 IntEnum 已转换）",
                      isinstance(fe2.category, ErrorCategory), type(fe2.category).__name__)
                check("统一语义可用（disposition）",
                      fe2.disposition == "replay", fe2.disposition)
                check("可观测字段", fe2.mapped and fe2.graded_by == "code_map")

                # L4 类错误（设备级）应触发设备恢复语义
                fe3 = ab.translate_error(
                    RuntimeError("device reset failed, error code is 507015"),
                    location="probe")
                check("L4 错误 → device_recovery 语义",
                      fe3.category == ErrorCategory.L4_FATAL,
                      f"{fe3.category.value} → {fe3.disposition}")
        except ModuleNotFoundError as e:
            # 本地无 torch/torch_npu：不算失败，属环境缺失（真机在 910C 验证）
            print(f"  [SKIP] 本地缺少依赖（{e}），昇腾后端留待 910C 真机验证")
            print("         （判定：SKIP，不计入失败）")
        except Exception as e:
            check("ascend 调用异常", False, f"{type(e).__name__}: {e}")
    else:
        print("  [SKIP] 本地无 torch_npu，昇腾后端留待 910C 验证")

    # ── 6. 真实后端通用自检（后端无关；能力相关项按 supports 如实 SKIP）──
    print("\n[6] 真实后端通用自检（后端无关）")
    clear()
    try:
        from runtime.backends.registry import _KNOWN_BACKENDS as _known
    except Exception:
        _known = ("ascend", "flagos", "kunlun")
    loaded = runtime.discover(names=_known, verbose=False)
    order = [args.backend] if args.backend else ["kunlun", "ascend", "flagos"]
    picked = None
    for name in order:
        if name not in loaded:
            print(f"  [SKIP] {name}: 未发现（依赖缺失或未实现）")
            continue
        try:
            bk = use(name)
            if bk.device_count() > 0:
                picked = (name, bk)
                break
            print(f"  [SKIP] {name}: device_count=0")
        except Exception as e:
            print(f"  [SKIP] {name}: 初始化失败（{type(e).__name__}: {e}）")

    if picked is None:
        print("  [SKIP] 无可用真实后端，本节整体跳过（不计入失败）")
    else:
        name, bk = picked
        print(f"  → 选中后端: {name}  (device_type={bk.device_type})")
        try:
            cnt = bk.device_count()
            check("device_count > 0", cnt > 0, f"n={cnt}")
            bk.set_device(0)
            m = bk.memory_stats(0)
            check("memory_stats 结构", set(m) >= {"total_mb", "used_mb", "free_mb"}, str(m))
            check("memory_stats total_mb > 0", m.get("total_mb", 0) > 0, str(m))

            st = runtime.create_stream()
            ev = runtime.create_event()
            check("统一 Stream 可用", st is not None and st.backend.name == name,
                  type(st).__name__)
            check("统一 Event 可用", ev is not None and ev.backend.name == name,
                  type(ev).__name__)
            with st.context():
                pass
            check("Stream.context 上下文可用", True)
            ev.record(st)
            check("Event.record + wait_host 有界返回", ev.wait_host(timeout_ms=2000) is True)

            check("probe_device(0)", bk.probe_device(0) is True)
            rec = bk.recover_device(0, mode="probe")
            check("recover_device 返回 dict（统一契约）",
                  isinstance(rec, dict) and "recovered" in rec, str(rec))

            # 错误翻译：厂商错误码按**能力相关**处理（见 conformance/cases.py f1 的同一修正）
            fe = bk.translate_error(RuntimeError("CUDA error: invalid device ordinal"),
                                    location="smoke")
            check("translate_error 返回统一类型", isinstance(fe, FlagosError))
            check("translate_error 回填后端名", fe.backend == name, fe.backend)
            # 2026-09-22 修正（910C 对称复跑暴露）：本注入消息**不含厂商错误码**，
            # 却要求声明了 error_map 的后端必须走 code_map —— 判据不公平
            # （ascend 对无码消息正确地走了 message_hint，被误判失败）。
            # 改为两条诚实判据：①无码消息不得伪称 code_map；②含厂商码样例必须走 code_map
            # （样例由各家后端自带 SAMPLE_CODED_ERROR，无样例则如实 SKIP 正向检查）。
            sample = getattr(bk, "SAMPLE_CODED_ERROR", None)
            if bk.supports("error_map"):
                check("声明 error_map → 无码消息不伪称 code_map（诚实）",
                      fe.graded_by != "code_map", f"graded_by={fe.graded_by}")
                if sample:
                    fe_c = bk.translate_error(RuntimeError(sample), location="smoke")
                    check("声明 error_map → 含厂商码样例必走 code_map",
                          fe_c.graded_by == "code_map", f"graded_by={fe_c.graded_by}")
                else:
                    print("  [SKIP] 该后端未提供 SAMPLE_CODED_ERROR，正向 code_map 检查跳过（如实）")
            else:
                check("未声明 error_map → 分级来源非 code_map（如实）",
                      fe.graded_by != "code_map", f"graded_by={fe.graded_by}")

            # info() 与 supports() 自洽
            info = bk.info()
            caps = set(info.get("capabilities", []))
            bad = sorted(k for k in caps if not bk.supports(k))
            check("info.capabilities 与 supports() 自洽", not bad, str(bad))
            sup_map = info.get("supports")
            if isinstance(sup_map, dict):
                mismatch = sorted(k for k, v in sup_map.items() if bool(v) != bk.supports(k))
                check("info.supports 与 supports() 一致", not mismatch, str(mismatch))
        except ModuleNotFoundError as e:
            print(f"  [SKIP] {name} 依赖缺失（{e}）")
        except Exception as e:
            check(f"{name} 通用自检异常", False, f"{type(e).__name__}: {e}")

    print(f"\n=== 结果: {passed} 通过 / {failed} 失败 ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
