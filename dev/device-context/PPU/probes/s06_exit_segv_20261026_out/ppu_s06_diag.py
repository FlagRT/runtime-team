"""s06 退出期 rc=139 的原生现场定位探针（PPU）。

用法：python3 ppu_s06_diag.py <case|all> <normal|gcfirst|os_exit>

目的：区分三种可能
  A. 段错误发生在**解释器最终化 / GC** 阶段（退出期 finalizer 调用已失效的原生句柄）
  B. 段错误发生在**用例执行中**（判据本身的 bug）
  C. 显式 gc.collect() 就能触发（= 有对象在回收时炸，便于二分）
"""
import gc
import importlib
import os
import sys

sys.path.insert(0, os.getcwd())
import runtime.conformance.runner as R  # noqa: E402  （导入即把 conformance 目录加入 sys.path）

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
EXIT = sys.argv[2] if len(sys.argv) > 2 else "normal"

(device, sync, event_cls, stream_cls, stream_ctx,
 current_stream, env) = R._setup_backend("ppu")

import torch  # noqa: E402

devs = env["count"]()
ctx = {
    "chip": "ppu", "backend": "ppu", "devs": devs, "device": device, "torch": torch,
    "sync": sync, "event": event_cls, "stream": stream_cls, "stream_ctx": stream_ctx,
    "current_stream": current_stream, "supports": env.get("supports"),
}

mod = importlib.import_module("contract_invariants")
ALL = ["i1_honest_declaration", "i2_no_fabrication",
       "i3_invalidated_governed", "i4_degradation_observable"]
names = ALL if MODE == "all" else [n for n in ALL if n.startswith(MODE)]
print(f"[probe] mode={MODE} exit={EXIT} cases={names}", flush=True)

for n in names:
    ok, detail = getattr(mod, "case_" + n)(ctx)
    print(f"  [{'PASS' if ok else 'FAIL'}] {n:26s} {str(detail)[:70]}", flush=True)

print("cases_done", flush=True)

if EXIT == "gcfirst":
    print("gc#1 ...", flush=True)
    print("  collected", gc.collect(), flush=True)
    ctx = None
    print("gc#2 ...", flush=True)
    print("  collected", gc.collect(), flush=True)
    print("gc_done", flush=True)

print("reached_end", flush=True)
sys.stdout.flush()
if EXIT == "os_exit":
    os._exit(0)
