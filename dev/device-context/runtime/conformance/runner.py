#!/usr/bin/env python3
"""
统一运行时 conformance 运行框架（runtime/conformance/runner.py）

与 910C 阶段的 conformance 的区别：**设备抽象全部由统一运行时 API 提供**，
不再在 runner 里直接 import 厂商扩展。

  - 用例文件（cases.py 13 例 / infer_cases.py 6 例）保持原样，
    它们只通过 ctx 访问设备（device/sync/event/stream/stream_ctx/current_stream）
  - ctx 由 runtime.use(backend) 组装 → 切换后端只改 --backend
  - 新增一家芯片 = 实现 RuntimeBackend + 跑本 runner

用法：
    python3 runner.py --backend ascend                      # 13 例
    python3 runner.py --backend ascend --cases infer_cases  # 6 例
    python3 runner.py --backend kunlun                      # stub（报告未实现能力）

从 benchmarks/ascend_regression/conformance 迁移而来；原位置保留为 910C 阶段归档。
"""

import argparse
import importlib
import json
import os
import signal
import sys
import traceback

# __file__ = <device-context>/runtime/conformance/runner.py
# 需要把 <device-context> 加入 sys.path 才能 `import runtime`
_HERE = os.path.dirname(os.path.abspath(__file__))          # runtime/conformance
_RUNTIME_DIR = os.path.dirname(_HERE)                        # runtime
_PKG_DIR = os.path.dirname(_RUNTIME_DIR)                     # device-context
sys.path.insert(0, _HERE)
sys.path.insert(0, _PKG_DIR)
# 共享资产目录（errors / device_state / recovery 等），避免用例导入依赖后端加载顺序
_ASSETS_DIR = os.path.join(_PKG_DIR, 'benchmarks', 'ascend_regression', 'conformance')
if os.path.isdir(_ASSETS_DIR):
    sys.path.insert(0, _ASSETS_DIR)


class CaseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise CaseTimeout("用例超时（120s）")


def _setup_backend(backend_name: str):
    """用统一运行时 API 组装设备抽象（替代原 runner 里直接 import 厂商扩展）。"""
    import torch

    import runtime
    from runtime.api.stream import Event as UnifiedEvent
    from runtime.api.stream import Stream as UnifiedStream

    backend = runtime.use(backend_name)          # ← 唯一与芯片相关的调用
    device = backend.device_type

    # 预热：统一 API 走一次设备操作
    torch.zeros(1, device=f"{device}:0")

    def _sync():
        backend.synchronize(0)

    def _stream_factory():
        return UnifiedStream(backend, backend.create_stream())

    def _event_factory():
        return UnifiedEvent(backend, backend.create_event())

    def _stream_ctx(s):
        # 兼容统一 Stream 与原生流
        return s.context() if isinstance(s, UnifiedStream) else backend.stream_context(s)

    def _current_stream():
        return UnifiedStream(backend, backend.current_stream())

    env = {
        "name": f"runtime:{backend.name}",
        "ver": getattr(runtime, "__version__", "0.1.0"),
        "count": backend.device_count,
    }
    return device, _sync, _event_factory, _stream_factory, _stream_ctx, _current_stream, env


def main():
    ap = argparse.ArgumentParser(description="统一运行时 conformance 测试框架（设备无关）")
    ap.add_argument("--backend", default="ascend",
                    help="运行时后端名（ascend / kunlun ...），由 registry 发现")
    ap.add_argument("--chip", default=None,
                    help="芯片标识（默认取后端名，仅用于结果标记）")
    ap.add_argument("--cases", default="cases", help="用例模块名（cases / infer_cases）")
    ap.add_argument("--out", default="conformance_result.json", help="结果 JSON 输出路径")
    args = ap.parse_args()

    try:
        (device, sync, event_cls, stream_cls,
         stream_ctx, current_stream, env) = _setup_backend(args.backend)
    except Exception as e:
        print(f"CONFORMANCE_ABORT: 后端 '{args.backend}' 初始化失败: {e}")
        return 1

    devs = env["count"]()
    if devs < 1:
        print(f"CONFORMANCE_ABORT: 无可用 {device} 设备")
        return 1

    import torch
    ctx = {
        "chip": args.chip or args.backend,
        "backend": args.backend,
        "devs": devs,
        "device": device,
        "torch": torch,
        "sync": sync,
        "event": event_cls,
        "stream": stream_cls,
        "stream_ctx": stream_ctx,
        "current_stream": current_stream,
    }

    mod = importlib.import_module(args.cases)
    cases = [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("case_")]
    cases.sort()

    chip = ctx["chip"]
    print(f"=== 统一运行时 conformance: backend={args.backend}, chip={chip}, "
          f"用例数={len(cases)} ===")
    print(f"[env] {env['name']}={env['ver']} devices={devs}")

    results = {
        "runner": "runtime/conformance",
        "backend": args.backend,
        "chip": chip,
        "env": {env["name"]: env["ver"], "devices": devs},
        "cases": {},
    }

    signal.signal(signal.SIGALRM, _alarm_handler)
    passed = 0
    for name, fn in cases:
        case_name = name[len("case_"):]
        signal.alarm(120)
        try:
            ok, detail = fn(ctx)
            status = "PASS" if ok else "FAIL"
        except CaseTimeout:
            ok, status, detail = False, "TIMEOUT", "用例超时 120s"
        except Exception as e:
            ok = False
            status = "ERROR"
            detail = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
        finally:
            signal.alarm(0)
        if ok:
            passed += 1
        results["cases"][case_name] = {"ok": bool(ok), "status": status, "detail": str(detail)}
        print(f"  [{status:7s}] {case_name:28s} {str(detail)[:80]}")

    total = len(cases)
    print(f"\n=== 汇总: {passed}/{total} 通过 ===")
    verdict = "CONFORMANCE_PASS" if passed == total else "CONFORMANCE_FAIL"
    results["verdict"] = verdict
    results["passed"] = passed
    results["total"] = total
    print(f"结论: {verdict}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
