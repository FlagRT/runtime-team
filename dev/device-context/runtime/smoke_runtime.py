#!/usr/bin/env python3
"""
运行时框架冒烟测试（本地，无需 NPU）

验证范围：
  1. 注册表机制：register / use / get / available / BackendNotFound
  2. 统一错误对象：分级 → 处置策略映射、可观测字段
  3. Mock 后端：证明"新增后端 = 实现接口 + 注册"可行（kunlun stub 同此路径）
  4. 真实后端（ascend）：若环境有 torch_npu 则一并验证，否则跳过（不算失败）

用法：python3 smoke_runtime.py
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

    def create_stream(self): return "mock-stream"
    def create_event(self): return "mock-event"
    def current_stream(self): return "mock-current"

    def synchronize(self, ordinal, timeout_ms=None):
        if timeout_ms == 0:
            raise TimeoutError("mock timeout")

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


def main():
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
    check("create_stream", runtime.create_stream() == "mock-stream")
    check("create_event", runtime.create_event() == "mock-event")

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
                check("真实错误码翻译", fe2.category is not None,
                      f"{fe2.category.value} graded_by={fe2.graded_by}")
        except ModuleNotFoundError as e:
            # 本地无 torch/torch_npu：不算失败，属环境缺失（真机在 910C 验证）
            print(f"  [SKIP] 本地缺少依赖（{e}），昇腾后端留待 910C 真机验证")
            print("         （判定：SKIP，不计入失败）")
        except Exception as e:
            check("ascend 调用异常", False, f"{type(e).__name__}: {e}")
    else:
        print("  [SKIP] 本地无 torch_npu，昇腾后端留待 910C 验证")

    print(f"\n=== 结果: {passed} 通过 / {failed} 失败 ===")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
