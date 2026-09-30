#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_stream_priority_effect.py — 工作包 D：流优先级「效果」验证（后端无关）
═══════════════════════════════════════════════════════════════════════════════

【为什么单独一个探针】
  现状只能读「优先级范围」（ascend 7/0、cambricon (0,-3)、kunlun None），但**范围可读 ≠ 效果可用**
  —— 多卡同机多流争用时"优先级是否真生效"此前**无任何判据回答**。本探针做两件事：
    P1 同一优先级层次上创建的流，其可观测形态（stream 属性/优先级查询原语）**可读且稳定**；
    P2 高/低优先级两条满载流争用同一设备时，**完成顺序是否与优先级一致**（多次重复取统计，不赌单次）。

【判据（先写死，不许事后解释）】
  P1-range       stream_priority_range() 读得到（ kunlun 预期 None ⇒ 如实标 unsupported，不计失败 ）
  P1-create-hi   用范围**高端值**能创建流、能执行计算并校验数值
  P1-create-lo   用范围**低端值**能创建流、能执行计算并校验数值
  P2-effect      满载争用 K 轮（默认 8），高优先级流先完成 **≥ 阈值**（默认 6/8 = 75%）轮
                 ⇒ 若 < 阈值：**如实记录"未观测到优先级效果"**（这也是一条结论，不是探针失败——
                 厂商可能在该场景不调度优先级；判据名固定 P2-effect，取值 effect_observed: true/false）

  ⚠️ 结论边界（如实标注）：本探针观测的是"**本场景下**优先级对完成顺序的影响"。测不出效果 ≠ 优先级
  无效（可能是负载形态、设备占用、调度策略等因素）；但"测不出"本身必须**如实记录**，
  这是「范围可读 ≠ 效果可用」这条结论的取数基础。

【配额上限（Q 段，与 probe_stream_quota.py 同口径的扩展）】
  本探针**不做**配额上限扫描（那是 probe_stream_quota.py 加 DC_QUOTA_N 参数的事，且会长时间占卡）。
  B2 的配额部分由 probe_stream_quota.py 以**逐档递增**方式跑（见报告），此处不重复。

【环境变量】
  DC_BACKEND / DC_OUT_DIR / DC_TAG   同其它探针
  DC_PRIO_ROUNDS   P2 争用轮数（默认 8）
  DC_PRIO_PASS_N   P2 判"有效"的先完成轮数阈值（默认 6）
  DC_PRIO_WORK     每轮每流的负载规模（matmul 边长，默认 2048）

【用法】
  DC_BACKEND=ascend python3 probe_stream_priority_effect.py

【判定】STREAM_PRIORITY_EFFECT_PASS / PARTIAL / FAIL（附 effect_observed 字段）
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent                     # prototype/
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402
import runtime  # noqa: E402

BACKEND = os.environ.get("DC_BACKEND", "ascend")
OUT_DIR = Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
TAG = os.environ.get("DC_TAG", "")
ROUNDS = int(os.environ.get("DC_PRIO_ROUNDS", "8"))
PASS_N = int(os.environ.get("DC_PRIO_PASS_N", "6"))
WORK = int(os.environ.get("DC_PRIO_WORK", "2048"))

checks: dict = {}


def main() -> int:
    print("=== probe_stream_priority_effect.py: 工作包 D 流优先级效果 ===")
    bk = runtime.use(BACKEND)
    n = bk.device_count()                   # 懒加载纪律：拼设备串前先触碰
    dt = bk.device_type
    print(f"[env] backend={BACKEND} device_type={dt} devices={n} torch={torch.__version__}")
    torch.zeros(1, device=f"{dt}:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── P1-range ──
    # ⚠️ 范围形态各栈不同：ascend pyACL 返回**三元组** (hi, lo, rc)（2026-09-30 实测 (7,0,0)）；
    #   约定取值域 = 「前两位是 (hi, lo)，末位若是 int 视作 rc」。只认 rc==0（或无 rc）。
    rng = bk.stream_priority_range()
    rng_repr = None
    rc = None
    if isinstance(rng, (tuple, list)) and len(rng) in (2, 3):
        try:
            if len(rng) == 3:
                hi_, lo_, rc_ = int(rng[0]), int(rng[1]), int(rng[2])
                rc = rc_
            else:
                hi_, lo_ = int(rng[0]), int(rng[1])
            if rc is None or rc == 0:
                lo, hi = lo_, hi_
                rng_repr = [lo, hi]
        except (TypeError, ValueError):
            rng_repr = None
    p1_range_ok = rng_repr is not None
    unsupported = ("stream_priority" not in bk._capabilities) or not p1_range_ok
    print(f"[P1-range] range={rng!r} → (lo,hi)={rng_repr} supports={('stream_priority' in bk._capabilities)}")
    checks["P1_range_readable"] = {
        "ok": p1_range_ok or ("stream_priority" not in bk._capabilities),
        "detail": f"raw={rng!r}；解析为 (lo,hi)={rng_repr}；"
                  f"supports={'stream_priority' in bk._capabilities}",
        "unsupported": "stream_priority" not in bk._capabilities,
    }

    # ── P1-create-hi / P1-create-lo（kunlun unsupported ⇒ 如实跳过）──
    ns = getattr(torch, dt)
    a_ref = torch.arange(1, 65, dtype=torch.float32, device=f"{dt}:0")
    x = a_ref.clone()
    expected = float((x @ torch.tril(torch.ones(64, 64, device=f"{dt}:0"))).sum().item())

    def _calc_on(prio) -> tuple[bool, str]:
        try:
            s = ns.Stream(priority=prio) if prio is not None else ns.Stream()
            with ns.stream(s):
                y = x @ torch.tril(torch.ones(64, 64, device=f"{dt}:0"))
            s.synchronize()
            v = float(y.sum().item())
            return abs(v - expected) < 1e-2, f"prio={prio} v={v:.1f}"
        except Exception as e:              # noqa: BLE001
            return False, f"prio={prio} {type(e).__name__}: {str(e)[:120]}"

    if unsupported or rng_repr is None:
        for k in ("P1_create_hi", "P1_create_lo"):
            checks[k] = {"ok": True, "skipped": True,
                         "detail": f"后端未声明 stream_priority / 范围不可读（range={rng!r}）⇒ 如实跳过"}
            print(f"[{k}] SKIP（unsupported）")
    else:
        ok_hi, d_hi = _calc_on(rng_repr[1])   # 高端值
        ok_lo, d_lo = _calc_on(rng_repr[0])   # 低端值
        checks["P1_create_hi"] = {"ok": ok_hi, "detail": d_hi}
        checks["P1_create_lo"] = {"ok": ok_lo, "detail": d_lo}
        print(f"[P1-create-hi] {d_hi} {'✅' if ok_hi else '❌'}")
        print(f"[P1-create-lo] {d_lo} {'✅' if ok_lo else '❌'}")

    # ── P2-effect：高/低优先级满载争用，比较完成顺序 ──
    effect_observed = None
    hi_wins = 0
    timings = []
    if unsupported or rng_repr is None:
        checks["P2_effect"] = {"ok": True, "skipped": True, "effect_observed": None,
                               "detail": "后端未声明 stream_priority ⇒ 如实跳过"}
        print("[P2-effect] SKIP（unsupported）")
    else:
        hi_p, lo_p = rng_repr[1], rng_repr[0]
        acc_l = acc_h = None
        wins_by_order = {}
        if hi_p == lo_p:
            checks["P2_effect"] = {"ok": True, "skipped": True, "effect_observed": None,
                                   "detail": f"范围两端相同（{rng_repr}）⇒ 无从比较，如实跳过"}
            print(f"[P2-effect] SKIP（range 两端相同 {rng_repr}）")
        else:
            # 同一设备、两条流并发：高优先级做同规模负载。比较"完成事件就绪顺序"。
            ns.synchronize()
            a = torch.randn(WORK, WORK, device=f"{dt}:0")
            b = torch.randn(WORK, WORK, device=f"{dt}:0")
            # ⚠️ 单次 matmul 在 910C 上 <2 ms 即完成（2026-09-30 实测 0.4–1.8 ms）⇒ 高优先级
            #   提交前低优先级已跑完，**根本没有形成争用**，测到的只是"无争用下的完成顺序"。
            #   ⇒ 每流做 ITERS 次链式 matmul 拉长时间窗（默认 60 次 ≈ 数十 ms），确保两流真正同时在跑。
            # ⭐ 反向对照（2026-09-30 加入）：把提交顺序反过来（高优先级先提交）。
            #   若"谁先提交谁先完成"在两种顺序下都成立 ⇒ 提交顺序主导、优先级无可观测效果；
            #   若反转后高优先级仍赢 ⇒ 说明 lo-first 那组是争用不够；若反转后高优先级仍输 ⇒ 更强的反证。
            ITERS = int(os.environ.get("DC_PRIO_ITERS", "60"))
            for r in range(ROUNDS):
                for order in ("lo_first", "hi_first"):
                    eh = ns.Event()
                    el = ns.Event()
                    sh_hi = ns.Stream(priority=hi_p)
                    sh_lo = ns.Stream(priority=lo_p)

                    def submit_lo():
                        nonlocal acc_l
                        with ns.stream(sh_lo):
                            acc_l = a.clone()
                            for _ in range(ITERS):
                                acc_l = acc_l @ a
                            el.record()

                    def submit_hi():
                        nonlocal acc_h
                        with ns.stream(sh_hi):
                            acc_h = b.clone()
                            for _ in range(ITERS):
                                acc_h = acc_h @ b
                            eh.record()

                    t0 = time.monotonic()
                    if order == "lo_first":
                        submit_lo(); submit_hi()
                    else:
                        submit_hi(); submit_lo()
                    hi_done_at = lo_done_at = None
                    while hi_done_at is None or lo_done_at is None:
                        if hi_done_at is None and eh.query():
                            hi_done_at = time.monotonic() - t0
                        if lo_done_at is None and el.query():
                            lo_done_at = time.monotonic() - t0
                        time.sleep(0.0005)
                    timings.append({"round": r, "order": order,
                                    "hi_s": round(hi_done_at, 4), "lo_s": round(lo_done_at, 4)})
                    if hi_done_at < lo_done_at:
                        hi_wins += 1
                        wins_by_order[order] = wins_by_order.get(order, 0) + 1
            # 判定口径（先写死）：只在 lo_first 方向计 hi_wins（高优先级后提交仍能赢才算效果）；
            # hi_first 方向的高优先级先完成预期内，只作对照。
            hi_wins_lo_first = wins_by_order.get("lo_first", 0)
            hi_wins_hi_first = wins_by_order.get("hi_first", 0)
            effect_observed = hi_wins_lo_first >= PASS_N
            submit_order_dominates = (hi_wins_lo_first == 0 and hi_wins_hi_first == ROUNDS)
            checks["P2_effect"] = {
                "ok": True,                  # 探针本身成功；效果结论在 effect_observed
                "effect_observed": effect_observed,
                "hi_wins_lo_first": hi_wins_lo_first, "rounds": ROUNDS, "pass_n": PASS_N,
                "hi_wins_hi_first": hi_wins_hi_first,
                "submit_order_dominates": submit_order_dominates,
                "detail": f"lo_first 方向高优先级先完成 {hi_wins_lo_first}/{ROUNDS}（阈值 {PASS_N}）⇒ "
                          f"{'观测到效果' if effect_observed else '未观测到优先级效果（如实记录）'}；"
                          f"对照 hi_first 方向 {hi_wins_hi_first}/{ROUNDS}"
                          + ("；两种顺序下均『谁先提交谁先完成』⇒ 提交顺序主导" if submit_order_dominates else ""),
                "timings": timings,
            }
            print(f"[P2-effect] lo_first 高优先级先完成 {hi_wins_lo_first}/{ROUNDS}；"
                  f"hi_first 对照 {hi_wins_hi_first}/{ROUNDS} ⇒ "
                  f"{'✅ 观测到效果' if effect_observed else '⚠️ 未观测到优先级效果（如实记录，非探针失败）'}")

    passed = sum(1 for v in checks.values() if v.get("ok"))
    total = len(checks)
    verdict = ("STREAM_PRIORITY_EFFECT_PASS" if passed == total
               else ("STREAM_PRIORITY_EFFECT_PARTIAL" if passed else "STREAM_PRIORITY_EFFECT_FAIL"))
    print(f"\n{verdict}: {passed}/{total} 子项通过"
          + (f"（effect_observed={effect_observed}）" if effect_observed is not None else ""))

    out = {"verdict": verdict, "passed": passed, "total": total,
           "backend": BACKEND, "range": rng_repr, "supports": not unsupported,
           "effect_observed": effect_observed, "checks": checks,
           "torch": torch.__version__}
    path = OUT_DIR / f"stream_priority_effect_result{TAG}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
