#!/usr/bin/env python3
"""流优先级「能设置」+ 流**释放**语义的真机验证（后端无关；2026-09-30 第二轮）。

回答三个问题（判据先写死，不事后调整）：
  R1 默认路径（`create_stream()`）造的流：**不得**被登记为本层拥有，`release_stream` 必须 no-op。
  R2 单点区间：`create_stream(唯一档位)` 走「**等价放行**」（等价厂商默认、可回读验证）
     ⇒ 该流由厂商拥有（不得登记、release 必须 no-op）。
  R3 **实现链路**：直接调后端私有 `_create_stream_raw(域内值)`（绕过基类门禁，**只为验证实现**）
     —— 若返回的流被登记为本层拥有 ⇒ `release_stream` 必须真销毁，且**销毁后使用必须报错**；
        若后端未声明 `stream_priority_control` ⇒ 如实抛错（SKIP，不是失败）。
  R4 越界值 ⇒ `ValueError`（取值域校验）。
  R5 若 R3 得到本层拥有的流：在其上**真跑一次算子**，证明"能设置"的流**真的能用**。

⚠️ 边界（如实，不外推）：
  · 本探针只证「本层接口 + 厂商原语在该机器上可用」，不证**调度效果**（那是另一件事，
    见 `prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md`）。
  · `_create_stream_raw` 是**私有**方法：此处调用是**取证**用途，不代表上层用法。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 本文件在 `prototype/probes/` 下 ⇒ 原型根 = `dirname(HERE)`**一层**。
#    2026-09-30 实测教训：写成 `dirname(dirname(HERE))` 多算一层 ⇒ 真机上
#    `ModuleNotFoundError: No module named runtime`（今天第二次犯同类 off-by-one，
#    另一次是 `runtime/demos/demo_unified.py`）。⇒ 根解析**必须实测**，
#    并优先用 `DC_ROOT` 显式覆盖。
ROOT = os.environ.get("DC_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rt = runtime.use(a.backend)
    rt.set_device(a.dev)
    dt = rt.device_type
    dev = f"{dt}:{a.dev}"
    x = torch.ones(4, 4, device=dev)
    expect = float(x.sum().item() + x.numel())      # 16 + 16 = 32

    checks: dict = {}

    def judge(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail, "skipped": False}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)

    def skip(name, detail):
        checks[name] = {"ok": True, "detail": detail, "skipped": True}
        print(f"  [SKIP] {name}  {detail}", flush=True)

    # 取后端实例（统一面只暴露函数，这里为取证需要实例）
    from runtime.backends import registry
    bk = registry.current() if hasattr(registry, "current") else None
    if bk is None:                                     # 兜底：取不到就**报错**，不静默继续
        raise RuntimeError("无法取得当前后端实例（registry.current() 不可用）")
    print(f"[env] backend={a.backend} device_type={dt} range={bk.stream_priority_range()!r} "
          f"control={bk.supports('stream_priority_control')}", flush=True)

    print("\n[R1] 默认路径的所有权", flush=True)
    try:
        # ⚠️ `runtime.use()` 返回的是**后端实例**（不是统一面）⇒ 其 `create_stream()` 直接给
        #    **厂商原生流**；统一包装在模块级 `runtime.create_stream()`。早先误用 `._native_obj`
        #    得到 AttributeError（接口认知修正，2026-09-30 真机踩到）。
        nat_plain = rt.create_stream()
        judge("R1a 默认路径造出的流**不得**被登记为本层拥有",
              not rt.owns_stream(nat_plain), f"owns={rt.owns_stream(nat_plain)}")
        rel = rt.release_stream(nat_plain)
        judge("R1b 厂商拥有的流 ⇒ release_stream 必须 no-op 返回 False",
              rel is False, f"release={rel!r}")
    except BaseException as e:                                           # noqa: BLE001
        judge("R1 默认路径的所有权与释放", False, f"{type(e).__name__}: {str(e)[:110]}")

    bounds = bk._priority_bounds()
    has_set = bk.supports("stream_priority_control")

    print("\n[R2] 单点区间：等价放行（仅当区间单点）", flush=True)
    if bounds and bounds[0] == bounds[1]:
        try:
            nat_eq = rt.create_stream(bounds[0])
            got = rt.stream_priority_readback(nat_eq)
            judge("R2a 单点区间下请求唯一档位 ⇒ 成功且**回读一致**（等价厂商默认）",
                  got == bounds[0], f"回读={got!r}（请求 {bounds[0]}）")
            judge("R2b 等价放行的流由**厂商拥有** ⇒ 不得登记、release 必须 no-op",
                  (not rt.owns_stream(nat_eq)) and (rt.release_stream(nat_eq) is False),
                  f"owns={rt.owns_stream(nat_eq)}")
        except BaseException as e:                                       # noqa: BLE001
            judge("R2 单点区间的等价放行", False, f"{type(e).__name__}: {str(e)[:110]}")
    else:
        skip("R2 单点区间的等价放行", f"本机区间 = {bounds!r}（非单点）⇒ 该分支不适用")

    print("\n[R3] 实现链路：私有 `_create_stream_raw(域内值)` 直调取证", flush=True)
    owned = None
    if not has_set:
        try:
            bk._create_stream_raw(bounds[0] if bounds else 0)
            judge("R3 未声明 control ⇒ 私有原语也必须**显式拒绝**（不得静默返回无优先级流）",
                  False, "未报错 —— 静默放行")
        except NotImplementedError:
            skip("R3 实现链路可用性",
                 "本后端**未声明** `stream_priority_control` ⇒ 私有原语如实拒绝（SKIP，非失败）")
        except BaseException as e:                                       # noqa: BLE001
            skip("R3 实现链路可用性",
                 f"未声明 control 且抛 {type(e).__name__}（如实；{str(e)[:70]}）")
    else:
        try:
            lvl = bounds[0] if bounds else 0
            nat = bk._create_stream_raw(lvl)
            got = bk.stream_priority_readback(nat)
            judge("R3a 私有原语造出的流**可回读且等于请求值**（参数真的进了设备）",
                  got == lvl, f"回读={got!r}（请求 {lvl}）")
            # 在该流上真跑一次算子（证明「能设置」的流**真的能用**）
            run_ok = False
            y_sum = None
            try:
                with runtime.Stream(bk, nat).context():
                    y = x + 1.0
                rt.synchronize(a.dev)
                y_sum = float(y.sum().item())
                run_ok = abs(y_sum - expect) < 1e-6
            except BaseException as e:                                   # noqa: BLE001
                print(f"    （算子执行失败：{type(e).__name__}: {str(e)[:90]}）", flush=True)
            judge("R3b 该流**真的能承载算子**（不是『看起来设了』的假流）",
                  run_ok, f"sum={y_sum!r}（期望 {expect}）")
            if bk.owns_stream(nat):
                owned = nat
                rel = bk.release_stream(nat)
                judge("R3c 本层拥有的流 ⇒ release_stream 必须**真的销毁**（返回 True）",
                      rel is True, f"release={rel!r}")
                try:
                    bk.check_stream_usable(nat)
                    judge("R3d 已释放的流**再使用**必须报错（使用已销毁对象必须明确）",
                          False, "未报错")
                except RuntimeError as e:
                    judge("R3d 已释放的流**再使用**必须报错（使用已销毁对象必须明确）",
                          ("release" in str(e) or "释放" in str(e)), f"RuntimeError: {str(e)[:80]}")
                except BaseException as e:                               # noqa: BLE001
                    judge("R3d 已释放的流**再使用**必须报错（使用已销毁对象必须明确）",
                          False, f"异常类型不合契约：{type(e).__name__}")
            else:
                skip("R3c/R3d 释放语义",
                     "该路径的流由**厂商/torch 拥有**（未登记）⇒ release 为 no-op，无需销毁")
        except BaseException as e:                                       # noqa: BLE001
            judge("R3 实现链路", False, f"{type(e).__name__}: {str(e)[:120]}")

    print("\n[R4] 取值域校验", flush=True)
    if bounds:
        for bad in (bounds[0] - 1, bounds[1] + 1):
            try:
                rt.create_stream(priority=bad)
                judge(f"R4 越界 priority={bad} ⇒ ValueError", False, "未报错")
            except ValueError:
                judge(f"R4 越界 priority={bad} ⇒ ValueError", True, "ValueError")
            except BaseException as e:                                   # noqa: BLE001
                judge(f"R4 越界 priority={bad} ⇒ ValueError", False,
                      f"异常类型不对：{type(e).__name__}")
    else:
        skip("R4 取值域校验", f"区间不可得（{bounds!r}）")

    failed = [k for k, v in checks.items() if not v["ok"]]
    skipped = [k for k, v in checks.items() if v.get("skipped")]
    verdict = ("STREAM_RELEASE_CONTROL_PASS" if not failed else "STREAM_RELEASE_CONTROL_FAIL")
    print(f"\n=== 汇总 ===\n  通过 {len(checks) - len(failed)} / 失败 {len(failed)} / 跳过 {len(skipped)}")
    for k in failed:
        print(f"  [FAIL] {k}: {checks[k]['detail'][:130]}")
    print(f"verdict = {verdict}")
    out = {"backend": a.backend, "device_type": dt, "range": bounds,
           "supports_control": has_set, "verdict": verdict,
           "passed": len(checks) - len(failed), "failed": len(failed),
           "skipped": len(skipped), "checks": checks}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[out] {a.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
