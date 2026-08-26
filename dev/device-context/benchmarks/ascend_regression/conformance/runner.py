#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
conformance/runner.py — 跨芯片一致性测试套件·运行框架（设备无关）
═══════════════════════════════════════════════════════════════════════════════

【设计原则】（对应方案设计思路 D1/创新四）
  - 用例以"统一接口操作序列"描述，与芯片无关
  - 同一套用例运行于全部已接入后端（芯片），行为差异即缺陷
  - 允许性能差异，禁止语义差异

【用法】（容器内、对应后端环境激活、单进程）：
  A 线（torch_npu，主线）：
    python runner.py --chip ascend --backend npu --out conformance_ascend_aline_result.json
  B 线（torch_fl，预研支线）：
    python runner.py --chip ascend --backend flagos --out conformance_ascend_bfinal_result.json
  下一款芯片接入时，在对应环境运行同一套用例：
    python runner.py --chip cambricon --backend npu --out conformance_cambricon_result.json
  然后比对 JSON 的 ok 字段：同一用例行为不一致 → 新后端插件缺陷。

【ctx 约定】（cases.py 的用例通过 ctx 访问设备抽象）：
  ctx["device"]   设备字符串（"npu"/"flagos"/...）
  ctx["sync"]()   该设备主机同步原语
  ctx["event"]()  事件工厂（返回 record/wait/query/wait_host 兼容对象）
  ctx["backend"]  后端名；ctx["chip"] 芯片标识；ctx["devs"] 设备数

【硬约束】后端与设备一一对应：npu=torch_npu（不 import torch_fl），
  flagos=torch_fl（不 import torch_npu）。
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


def _setup_backend(backend):
    """按后端组装设备抽象（device/sync/event 工厂）。"""
    import torch
    if backend == "npu":
        import torch_npu
        from npu_events import NpuEventAdapter
        device = "npu"
        sync = torch.npu.synchronize
        event_cls = NpuEventAdapter  # 统一语义适配层（补 wait_host + query recorded 修正）
        env = {"name": "torch_npu", "ver": getattr(torch_npu, "__version__", "unknown"),
               "count": lambda: torch.npu.device_count()}
    elif backend == "flagos":
        import torch_fl
        from torch_fl import flagos
        device = "flagos"
        sync = flagos.synchronize
        event_cls = flagos.Event
        env = {"name": "torch_fl", "ver": getattr(torch_fl, "__version__", "unknown"),
               "count": lambda: flagos.device_count()}
    else:
        raise SystemExit(f"未知后端: {backend}")
    return device, sync, event_cls, env


def main():
    ap = argparse.ArgumentParser(description="跨芯片一致性测试套件运行框架（设备无关）")
    ap.add_argument("--chip", default="ascend", help="当前芯片标识（如 ascend/cambricon/kunlunxin）")
    ap.add_argument("--backend", default="npu", choices=["npu", "flagos"],
                    help="设备后端：npu=torch_npu（A 线主线）/ flagos=torch_fl（B 线预研支线）")
    ap.add_argument("--out", default="conformance_result.json", help="结果 JSON 输出路径")
    ap.add_argument("--cases", default="cases", help="用例模块名（默认 cases）")
    args = ap.parse_args()

    device, sync, event_cls, env = _setup_backend(args.backend)
    devs = env["count"]()
    if devs < 1:
        print(f"CONFORMANCE_ABORT: 无可用 {device} 设备，先恢复环境")
        sys.exit(1)

    import torch
    torch.zeros(1, device=device)  # 设备预热（torch_npu/flagos 均需首次设备操作初始化）

    ctx = {
        "chip": args.chip,
        "backend": args.backend,
        "devs": devs,
        "device": device,
        "torch": torch,
        "sync": sync,
        "event": event_cls,
    }

    mod = importlib.import_module(args.cases)
    cases = [(n, getattr(mod, n)) for n in dir(mod) if n.startswith("case_")]
    cases.sort()
    print(f"=== conformance runner: chip={args.chip}, backend={args.backend}, 用例数={len(cases)} ===")
    print(f"[env] {env['name']}={env['ver']} devices={devs}")

    results = {"chip": args.chip, "backend": args.backend,
               "env": {env["name"]: env["ver"], "devices": devs}, "cases": {}}
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
