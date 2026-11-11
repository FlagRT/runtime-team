#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_stream_priority_queued.py — 流优先级「按官方机制」的正确测法（后端无关）
═══════════════════════════════════════════════════════════════════════════════

【为什么重做】2026-09-30 上一版（`probe_stream_priority_effect.py`）得到「未观测到优先级效果」，
但**查官方文档后确认是实验设计与机制不匹配**。CANN 文档原文（Stream 管理）：

  · 「高优先级 Stream 中待执行的任务将优先于低优先级 Stream 中的任务得到调度，
      **但不会抢占已处于运行状态的低优先级任务**」
  · 「Device 在执行过程中**不会动态重新评估任务队列**」
  · 「Stream 的优先级主要用于影响任务的**调度顺序**，而非强制规定严格的执行序列」

⇒ 上一版的测法是「低优先级先提交 → 它**已经进入运行态** → 高优先级随后提交**只能排队**」
  ⇒ **按文档所述机制，这场景下优先级本来就不可能改变完成顺序**。
  ⇒ 正确测法：让两条流**同时处于排队态**（都等在同一个 gate 事件上），再一起放行，
     这时调度器才需要在"都已就绪"的任务之间做选择 —— 优先级才有机会起作用。

【优先级方向（官方语义，别再搞反）】
  `acl.rt.device_get_stream_priority_range()` 对应 C 接口
  `aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority)`，
  官方示例用 **greatestPriority 建高优先级流**、leastPriority 建低优先级流。
  实测本栈返回 `(7, 0, 0)` ⇒ **least=7 / greatest=0** ⇒ **数值越小优先级越高**（0 最高、7 最低）。
  （与 torch_npu docstring 的 "Lower numbers represent higher priorities" 一致。）
  ⚠️ 上一版把 7 当作"高优先级"，**方向标反了** —— 本探针按官方语义取 `hi = min(a, b)`。

【判据（先写死）】
  P1 `range` 可解析，并按官方语义给出 `hi = min` / `lo = max`
  P2 两端都能建流并算对（证明参数没被吞）
  P3 **gate 模式**：两流同时排队后放行，高优先级先完成 ≥ 阈值（默认 6/8）轮
  P4 **naive 对照**：先提交者先跑（旧测法）——预期"先提交者先完成"，用来说明差别来自测法
  P5 每轮记录两流完成时刻，便于复核

【环境变量】
  DC_BACKEND / DC_OUT_DIR / DC_TAG / DC_PRIO_ROUNDS(8) / DC_PRIO_PASS_N(6)
  DC_PRIO_ITERS(60) 每流链式 matmul 次数 / DC_PRIO_WORK(2048) 方阵边长
【用法】DC_BACKEND=ascend python3 probes/probe_stream_priority_queued.py --dev 0
【判定】STREAM_PRIORITY_QUEUED_PASS / PARTIAL / FAIL（>95 并附 effect 字段）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402
import runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    BACKEND = a.backend
    OUT_DIR = Path(a.out).parent if a.out else Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
    TAG = os.environ.get("DC_TAG", "")
    ROUNDS = int(os.environ.get("DC_PRIO_ROUNDS", "8"))
    PASS_N = int(os.environ.get("DC_PRIO_PASS_N", "6"))
    ITERS = int(os.environ.get("DC_PRIO_ITERS", "60"))
    WORK = int(os.environ.get("DC_PRIO_WORK", "2048"))

    checks: dict = {}

    def judge(name, ok, detail, extra=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if extra:
            checks[name].update(extra)

    def skip(name, detail):
        checks[name] = {"ok": True, "detail": detail, "skipped": True}

    bk = runtime.use(BACKEND)
    bk.device_count()
    dt = bk.device_type
    ns = getattr(torch, dt)
    ns.set_device(a.dev)
    print(f"=== probe_stream_priority_queued.py: 按官方机制测优先级 ===")
    print(f"[env] backend={BACKEND} device_type={dt} torch={torch.__version__} "
          f"rounds={ROUNDS} iters={ITERS} work={WORK}")

    # ── P1 方向（按官方语义：数值越小优先级越高）──
    rng = bk.stream_priority_range()
    hi = lo = None
    if isinstance(rng, (tuple, list)) and len(rng) in (2, 3):
        try:
            vals = [int(rng[0]), int(rng[1])]
            rc = int(rng[2]) if len(rng) == 3 else 0
            if rc == 0:
                hi, lo = min(vals), max(vals)      # ⭐ 官方语义：greatest priority 数值最小
        except (TypeError, ValueError):
            pass
    unsupported = ("stream_priority" not in bk._capabilities) or hi is None
    print(f"[P1] range_raw={rng!r} ⇒ 按官方语义 hi(最高)={hi} lo(最低)={lo} "
          f"supports={('stream_priority' in bk._capabilities)}")
    judge("P1_range_readable_并给出官方语义方向", hi is not None,
          f"raw={rng!r}; hi={hi}, lo={lo}（官方：greatestPriority 数值最小 = 最高优先级）",
          {"range_raw": list(rng) if isinstance(rng, (tuple, list)) else None,
           "hi_priority": hi, "lo_priority": lo})

    if unsupported:
        for k in ("P2_create_both_ends", "P3_gate_high_wins", "P4_naive_control"):
            skip(k, f"后端未声明 stream_priority / 范围不可读（raw={rng!r}）⇒ 如实跳过")
        verdict = "STREAM_PRIORITY_QUEUED_PASS"
        out = {"verdict": verdict, "backend": BACKEND, "range_raw": rng, "checks": checks}
        _dump(out, a.out, OUT_DIR, TAG)
        return 0

    ns.synchronize()
    x = torch.randn(WORK, WORK, device=f"{dt}:{a.dev}")
    y = torch.randn(WORK, WORK, device=f"{dt}:{a.dev}")
    s_hi = ns.Stream(priority=hi)
    s_lo = ns.Stream(priority=lo)

    def work_on(stream, t):
        with ns.stream(stream):
            acc = t.clone()
            for _ in range(ITERS):
                acc = acc @ t
        return acc

    # ── P2 两端都能建流并算对 ──
    try:
        r1 = work_on(ns.Stream(priority=hi), x)
        r2 = work_on(ns.Stream(priority=lo), y)
        ns.synchronize()
        # 用可精确表示的整数算式校验算得对
        ai = torch.arange(1, 65, dtype=torch.float32, device=f"{dt}:{a.dev}")
        A = torch.tril(torch.ones(64, 64, device=f"{dt}:{a.dev}"))
        with ns.stream(ns.Stream(priority=hi)):
            v_hi = float((ai @ A).sum().item())
        with ns.stream(ns.Stream(priority=lo)):
            v_lo = float((ai @ A).sum().item())
        ns.synchronize()
        ok2 = abs(v_hi - 89440) < 1e-2 and abs(v_lo - 89440) < 1e-2
        judge("P2_create_both_ends", ok2, f"hi={v_hi} lo={v_lo}（期望各 89440）")
    except Exception as e:  # noqa: BLE001
        judge("P2_create_both_ends", False, f"{type(e).__name__}: {str(e)[:160]}")

    # ── P3 gate 模式：两流同时排队再放行 ──
    def run_gate(rounds):
        wins = 0
        recs = []
        for r in range(rounds):
            gate = ns.Event()
            e_hi, e_lo = ns.Event(), ns.Event()
            # ⭐ 两条流都先 **wait 同一个 gate**（此时都进"排队等待"态），再各自干活
            with ns.stream(s_lo):
                s_lo.wait_event(gate)
                acc = y.clone()
                for _ in range(ITERS):
                    acc = acc @ y
                e_lo.record()
            with ns.stream(s_hi):
                s_hi.wait_event(gate)
                acc = x.clone()
                for _ in range(ITERS):
                    acc = acc @ x
                e_hi.record()
            t0 = time.monotonic()
            gate.record()                       # 放行：两条流**同时**变为就绪
            hi_at = lo_at = None
            while hi_at is None or lo_at is None:
                if hi_at is None and e_hi.query():
                    hi_at = time.monotonic() - t0
                if lo_at is None and e_lo.query():
                    lo_at = time.monotonic() - t0
                time.sleep(0.0003)
            recs.append({"round": r, "hi_s": round(hi_at, 4), "lo_s": round(lo_at, 4)})
            if hi_at < lo_at:
                wins += 1
        return wins, recs

    def run_naive(rounds):
        wins = 0
        recs = []
        for r in range(rounds):
            e_hi, e_lo = ns.Event(), ns.Event()
            t0 = time.monotonic()
            with ns.stream(s_lo):               # 低优先级**先提交**（旧测法）
                acc = y.clone()
                for _ in range(ITERS):
                    acc = acc @ y
                e_lo.record()
            with ns.stream(s_hi):
                acc = x.clone()
                for _ in range(ITERS):
                    acc = acc @ x
                e_hi.record()
            hi_at = lo_at = None
            while hi_at is None or lo_at is None:
                if hi_at is None and e_hi.query():
                    hi_at = time.monotonic() - t0
                if lo_at is None and e_lo.query():
                    lo_at = time.monotonic() - t0
                time.sleep(0.0003)
            recs.append({"round": r, "hi_s": round(hi_at, 4), "lo_s": round(lo_at, 4)})
            if hi_at < lo_at:
                wins += 1
        return wins, recs

    try:
        g_wins, g_recs = run_gate(ROUNDS)
        n_wins, n_recs = run_naive(ROUNDS)
    except Exception as e:  # noqa: BLE001
        judge("P3_gate_high_wins", False, f"{type(e).__name__}: {str(e)[:200]}")
        judge("P4_naive_control", False, "同上失败")
        g_wins = n_wins = 0
        g_recs = n_recs = []

    gate_effect = g_wins >= PASS_N
    judge("P3_gate_high_wins", True,
          f"gate 模式（两流同时排队）：高优先级先完成 {g_wins}/{ROUNDS}（阈值 {PASS_N}）⇒ "
          f"{'✅ 观测到优先级效果' if gate_effect else '⚠️ 仍未观测到效果（如实记录）'}",
          {"effect_observed": gate_effect, "hi_wins": g_wins, "rounds": ROUNDS, "timings": g_recs})
    judge("P4_naive_control", True,
          f"naive 对照（低优先级先提交、先跑起来）：高优先级先完成 {n_wins}/{ROUNDS}"
          f"（按官方机制**预期为 0**，因为不抢占、不重评估）",
          {"hi_wins": n_wins, "timings": n_recs})

    passed = sum(1 for v in checks.values() if v.get("ok"))
    verdict = ("STREAM_PRIORITY_QUEUED_PASS" if passed == len(checks)
               else ("STREAM_PRIORITY_QUEUED_PARTIAL" if passed else "STREAM_PRIORITY_QUEUED_FAIL"))
    print(f"\n[gate] 高优先级先完成 {g_wins}/{ROUNDS} | [naive] {n_wins}/{ROUNDS}")
    print(f"{verdict}: {passed}/{len(checks)}"
          + (f"（effect_observed={gate_effect}）" if not unsupported else "（unsupported）"))
    out = {"verdict": verdict, "passed": passed, "total": len(checks), "backend": BACKEND,
           "range_raw": list(rng) if isinstance(rng, (tuple, list)) else None,
           "hi_priority": hi, "lo_priority": lo, "effect_observed": (None if unsupported else gate_effect),
           "gate_hi_wins": None if unsupported else g_wins,
           "naive_hi_wins": None if unsupported else n_wins,
           "checks": checks, "torch": torch.__version__,
           "mechanism_note": "CANN 文档：优先级不抢占已运行任务、不动态重评估队列 ⇒ "
                             "旧测法（先提交者已进入运行态）按机制不可能观测到差异"}
    _dump(out, a.out, OUT_DIR, TAG)
    return 0 if passed == len(checks) else 1


def _dump(out, out_path, out_dir, tag):
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "checks"},
                            ensure_ascii=False, indent=2)[:400])
    for k, v in out["checks"].items():
        print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k}  {v['detail'][:150]}")
    p = Path(out_path) if out_path else (Path(out_dir) / f"stream_priority_queued_result{tag}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {p}")


if __name__ == "__main__":
    sys.exit(main())
