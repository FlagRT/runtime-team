#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
conformance/runner.py — 跨芯片一致性测试套件·运行框架（昇腾为第一个基线）
═══════════════════════════════════════════════════════════════════════════════

【设计原则】（对应方案设计思路 D1/创新四）
  - 用例以"统一接口操作序列"描述，与芯片无关
  - 同一套用例运行于全部已接入插件（芯片），行为差异即缺陷
  - 允许性能差异，禁止语义差异

【用法】容器内、tf-venv-integration 激活、单进程：
    python runner.py --chip ascend --out conformance_ascend_result.json
  下一款芯片接入时，在对应环境运行同一套用例：
    python runner.py --chip cambricon --out conformance_cambricon_result.json
  然后比对两份 JSON 的 ok 字段：同一用例行为不一致 → 新芯片插件缺陷。

【用例约定】cases.py 中以 `def case_<name>(ctx) -> (ok, detail)` 定义；
  runner 自动收集并逐个执行；ctx 提供 flagos 设备与公共工具。

【硬约束】全程 torch_fl（flagos 设备），不 import torch_npu。
"""

import argparse
import importlib
import json
import signal
import sys
import traceback


class _CaseTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CaseTimeout()


def main():
    ap = argparse.ArgumentParser(description="跨芯片一致性测试套件运行框架")
    ap.add_argument("--chip", default="ascend", help="当前芯片标识（如 ascend/cambricon/kunlunxin）")
    ap.add_argument("--out", default="conformance_result.json", help="结果 JSON 输出路径")
    ap.add_argument("--cases", default="cases", help="用例模块名（默认 cases）")
    args = ap.parse_args()

    # 环境自检
    import torch
    import torch_fl
    from torch_fl import flagos
    devs = flagos.device_count()
    if devs < 1:
        print("CONFORMANCE_ABORT: 无可用 flagos 设备，先恢复环境")
        sys.exit(1)

    # 设备预热：源码版 torch_fl 的 pin_memory 依赖 flagos 设备已初始化，否则段错误
    torch.zeros(1, device="flagos")

    ctx = {
        "chip": args.chip,
        "devs": devs,
        "torch": torch,
        "torch_fl": torch_fl,
        "flagos": flagos,
        "sync": lambda: flagos.synchronize(),  # 统一同步原语（flagos 原生，非 torch.cuda）
    }

    # 收集用例
    mod = importlib.import_module(args.cases)
    cases = [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("case_")]
    cases.sort()
    print(f"=== conformance runner: chip={args.chip}, 用例数={len(cases)} ===")
    print(f"[env] torch_fl={getattr(torch_fl,'__version__','unknown')} devices={devs}")

    results = {"chip": args.chip, "env": {"torch_fl": getattr(torch_fl, "__version__", "unknown"), "devices": devs}, "cases": {}}
    signal.signal(signal.SIGALRM, _alarm_handler)
    for name, fn in cases:
        case_name = name[len("case_"):]
        signal.alarm(30)  # 单用例超时保护（30s），防止一个用例卡死拖垮整套
        try:
            ok, detail = fn(ctx)
            signal.alarm(0)
            results["cases"][case_name] = {"ok": bool(ok), "detail": str(detail)}
            print(f"  [{'PASS' if ok else 'FAIL'}] {case_name}: {detail}")
        except _CaseTimeout:
            results["cases"][case_name] = {"ok": False, "detail": "用例超时（30s）——疑似事件/同步阻塞，需排查"}
            print(f"  [TIMEOUT] {case_name}: 30s 超时（疑似阻塞）")
        except Exception as e:
            signal.alarm(0)
            results["cases"][case_name] = {"ok": False, "detail": f"EXCEPTION: {e}", "trace": traceback.format_exc().splitlines()[-1]}
            print(f"  [ERROR] {case_name}: {type(e).__name__}: {e}")

    passed = sum(1 for v in results["cases"].values() if v["ok"])
    results["summary"] = {"passed": passed, "failed": len(results["cases"]) - passed, "total": len(results["cases"])}
    print(f"\n=== 汇总: {passed}/{len(results['cases'])} 通过 ===")
    print(f"结论: {'CONFORMANCE_PASS' if passed == len(results['cases']) else 'CONFORMANCE_PARTIAL'} "
          f"(行为差异即缺陷；性能差异允许并另录)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {args.out}")


if __name__ == "__main__":
    main()
