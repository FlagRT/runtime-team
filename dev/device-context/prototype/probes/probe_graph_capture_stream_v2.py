#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_graph_capture_stream_v2.py — graph capture 流语义验证（**后端无关 V2**）
═══════════════════════════════════════════════════════════════════════════════

【与 V1 的关系】
  V1 = `910C/distributed_training/ascend_regression/probe_graph_capture_stream.py`，
  **硬编码 torch_npu / torch.npu.***（约 20 处），只能在昇腾上跑。
  V1 **保留不删**（历史归档，910C 的既有结论仍由它复现）。
  本 V2 与 V1 的**判据完全同口径**（G1–G5 五项），仅把设备 API 前缀改成
  由统一运行时给出 —— 于是同一份逻辑可在 910C / P800 / MLU590 / 第 4 家起复用。

【判据（对齐《多流 Stream 验收基线》S-7「图捕获流语义」）】
  **契约内判据（计入 PASS/FAIL）**
  G1 capture 正确性：捕获 matmul → replay 结果与 eager 一致（rel_err < 1e-3）
  G2 replay 确定性：同输入两次 replay 结果**逐位一致**（rel_err == 0）
  G3 输入更新：写回同一地址后 replay 使用新值（rel_err < 1e-3）
  G5 显式 stream 参数：`graph(g, stream=s)` 形式可用且结果正确
  **宽容度观察项（不计入 PASS/FAIL）**
  G4* capture 区内**切换到未纳入捕获的流**（`with graph(g): with stream(s): …`）

【⚠️ 为什么 G4 降级为观察项（2026-09-22 实证，P800 上挖出）】
  G4 这种写法**本身在上游契约之外**：捕获区内的所有工作都应留在捕获流上，
  若要用副流，必须经 `graph(g, stream=s)`（= G5）或显式 fork/join 把它**纳入捕获**；
  只是在捕获区内 `with stream(s)` 切过去，该副流并未进入捕获状态。
  ⇒ 它失败**不是厂商缺陷**，也不该算"图捕获流语义"不支持；通过也不代表能力更强，
  只代表该运行时**更宽松**。实测三家：npu ✅ / mlu ✅（容忍） / XPytorch ❌
  （`AcceleratorError: CUDA error: unrecognized error code`，稳定复现）。
  与此前"捕获区内调 synchronize"是同一类：**先核调用方用法，再怀疑芯片**。

【条目隔离（踩过的坑）】
  一次**失败**的捕获会把分配器/捕获状态搞脏，导致后续条目**连带失败**
  （实测：P800 上 G4 失败后，本可单独通过的 G5 报
  `Offset increment outside graph capture encountered unexpectedly`）。
  ⇒ 本脚本：① 观察项 G4 **放在最后**跑；② 每个条目失败后做状态清理（synchronize + gc）。
  条目是否与顺序无关，用"逐项独立进程"复验过（见探针说明与各实例报告）。

【一条必须遵守的用法约束（P800 首测踩过，见 `P800/docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` §3.2）】
  **捕获区内不得调用 synchronize / 任何同步原语**。CUDA Graph 语义要求捕获期所有工作留在
  被捕获的流上；在捕获区内同步是**调用方契约违反**，会报出看起来像"厂商不支持"的错误。
  ⇒ 本脚本刻意**不在** `with dev.graph(g):` 内部做任何同步。

【环境变量】
  DC_BACKEND  运行时后端名（ascend / kunlun / cambricon）
  DC_DEV_API  设备 API 前缀兜底（无 runtime 时可用 "npu"/"cuda"/"mlu"）
  DC_OUT_DIR  结果输出目录（默认本目录）
  DC_TAG      结果文件名后缀（多后端对照时分离，如 "_mlu590"）

【用法】
  910C   ：DC_BACKEND=ascend    python3 probe_graph_capture_stream_v2.py
  P800   ：CUDA_VISIBLE_DEVICES=6,7 DC_BACKEND=kunlun     DC_TAG=_p800    python3 probe_graph_capture_stream_v2.py
  MLU590 ：MLU_VISIBLE_DEVICES=0   DC_BACKEND=cambricon  DC_TAG=_mlu590  python3 probe_graph_capture_stream_v2.py

【判定】GRAPH_CAPTURE_PASS / GRAPH_CAPTURE_FAIL（G1–G5 全通过才算 PASS）
【数值纪律】一律 `.item()` / `.cpu()` 后取标量比对，避免设备侧比较引入额外同步语义。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent                     # prototype/
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

BACKEND = os.environ.get("DC_BACKEND", "ascend")
OUT_DIR = Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
TAG = os.environ.get("DC_TAG", "")


def resolve_dev_api() -> str:
    """确定设备 API 前缀（"npu" / "cuda" / "mlu"），并顺带把后端切好。

    优先级：runtime 后端声明的 device_type > 环境变量 DC_DEV_API > 按后端名的默认映射。
    """
    try:
        import runtime  # noqa: E402
        runtime.use(BACKEND)
        dt = getattr(runtime.current(), "device_type", "")
        if dt:
            return dt
    except Exception as exc:                  # 无 runtime（纯 torch 环境）时兜底
        print(f"[env] runtime 不可用（{type(exc).__name__}），回退环境变量/默认映射")
    if os.environ.get("DC_DEV_API"):
        return os.environ["DC_DEV_API"]
    return {"ascend": "npu", "kunlun": "cuda",
            "cambricon": "mlu"}.get(BACKEND, "cuda")


DEV_API = resolve_dev_api()
dev = getattr(torch, DEV_API)
DEV = DEV_API


def _cleanup(dev) -> None:
    """失败捕获后的状态清理：同步 + 回收，避免弄脏后续条目（实测必要性见模块 docstring）。"""
    import gc
    try:
        dev.synchronize()
    except Exception:
        pass
    gc.collect()


def pick_graph_class():
    """按优先级寻找图对象类。

    各家命名不同（CUDA 传统是 `CUDAGraph`；昇腾 `NPUGraph`；寒武纪 `MLUGraph`），
    故按已知名依次试，最后兜底扫描"名字里带 graph 的大写属性"。
    """
    for name in ("CUDAGraph", "NPUGraph", "MLUGraph", "Graph"):
        cls = getattr(dev, name, None)
        if isinstance(cls, type):
            return name, cls
    cands = [a for a in dir(dev)
             if a[0].isupper() and "graph" in a.lower() and isinstance(getattr(dev, a, None), type)]
    if cands:
        return cands[0], getattr(dev, cands[0])
    return None, None


def main() -> int:
    print("=== probe_graph_capture_stream_v2.py: graph capture 流语义（后端无关 V2）===")
    print(f"[env] backend={BACKEND} dev_api={DEV_API} devices={dev.device_count()} "
          f"torch={torch.__version__}")
    torch.zeros(1, device=DEV)                 # 设备预热
    dev.set_device(0)

    gname, GCls = pick_graph_class()
    print(f"[env] 图对象类: {gname or '未找到'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    #: 宽容度观察项（不计入 verdict）—— 见模块 docstring 的 G4 说明
    INFORMATIONAL = ("G4_capture_stream_swap",)
    result: dict = {"verdict": "FAIL", "checks": {}, "informational": list(INFORMATIONAL),
                    "note": "", "backend": BACKEND, "dev_api": DEV_API,
                    "graph_class": gname, "torch": torch.__version__}
    n = 1024

    if GCls is None:
        result["note"] = (f"本后端命名空间 torch.{DEV_API} 下未找到图对象类 "
                          f"⇒ 图捕获不可用（如实标注，不伪造）")
        print(f"\n{result['note']}")
        _dump(result)
        return 1

    # 固定输入（避免随机影响确定性判定）
    x = torch.randn(n, n, device=DEV)
    w = torch.randn(n, n, device=DEV)
    eager_ref = (x @ w).sum().item()

    # ══════════ G1+G2+G3：标准 capture/replay 语义 ══════════
    g1_g2_g3_ok = False
    try:
        g = GCls()
        (x @ w).sum()                          # 预热 + 稳定内存池（graph 惯例）
        dev.synchronize()

        with dev.graph(g):
            # ⚠️ 捕获区内不得同步（见模块 docstring 的用法约束）
            y = (x @ w).sum()
        dev.synchronize()
        # capture 仅记录不执行，必须先 replay 才产生结果
        g.replay()
        dev.synchronize()
        val1 = y.item()
        g.replay()                             # G2：同输入再 replay（确定性）
        dev.synchronize()
        val2 = y.item()
        x.copy_(torch.randn(n, n, device=DEV))  # G3：换值后 replay（图内指针固定）
        g.replay()
        dev.synchronize()
        val3 = y.item()

        rel1 = abs(val1 - eager_ref) / max(abs(eager_ref), 1.0)
        rel2 = abs(val1 - val2) / max(abs(val1), 1.0)
        eager3 = (x @ w).sum().item()
        rel3 = abs(val3 - eager3) / max(abs(eager3), 1.0)

        g1_g2_g3_ok = rel1 < 1e-3 and rel2 == 0.0 and rel3 < 1e-3
        print(f"[G1] capture vs eager: val={val1:.6f} ref={eager_ref:.6f} rel_err={rel1:.2e} "
              f"{'✅' if rel1 < 1e-3 else '❌'}")
        print(f"[G2] replay 确定性: val1={val1:.6f} val2={val2:.6f} 差={rel2:.2e} "
              f"{'✅' if rel2 == 0.0 else '❌'}")
        print(f"[G3] 输入更新后 replay: val3={val3:.6f} eager3={eager3:.6f} rel_err={rel3:.2e} "
              f"{'✅' if rel3 < 1e-3 else '❌'}")
        result["checks"]["G1_capture_correct"] = {"ok": rel1 < 1e-3, "detail": f"rel_err={rel1:.2e}"}
        result["checks"]["G2_replay_deterministic"] = {"ok": rel2 == 0.0,
                                                      "detail": f"两次 replay 差={rel2:.2e}"}
        result["checks"]["G3_input_update_replay"] = {"ok": rel3 < 1e-3, "detail": f"rel_err={rel3:.2e}"}
    except Exception as e:
        print(f"[G1-G3] 异常: {type(e).__name__}: {str(e)[:200]}")
        result["checks"]["G1_capture_correct"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:150]}"}
        _cleanup(dev)   # 失败的捕获会弄脏状态 ⇒ 清理后再跑后续条目（否则会连带失败）

    # ══════════ G5：capture 显式 stream 参数 ══════════
    g5_ok = False
    try:
        g5 = GCls()
        s5 = dev.Stream()
        x5 = torch.randn(n, n, device=DEV)
        w5 = torch.randn(n, n, device=DEV)
        y5_ref = (x5 @ w5).sum().item()
        (x5 @ w5).sum()                        # 预热
        dev.synchronize()
        with dev.graph(g5, stream=s5):         # 显式传 stream（与默认流区分）
            y5 = (x5 @ w5).sum()
        dev.synchronize()
        g5.replay()
        dev.synchronize()
        rel5 = abs(y5.item() - y5_ref) / max(abs(y5_ref), 1.0)
        g5_ok = rel5 < 1e-3
        print(f"[G5] capture 显式 stream 参数: rel_err={rel5:.2e} {'✅' if g5_ok else '❌'}")
        result["checks"]["G5_capture_explicit_stream"] = {"ok": g5_ok, "detail": f"rel_err={rel5:.2e}"}
    except Exception as e:
        print(f"[G5] 异常: {type(e).__name__}: {str(e)[:200]}")
        result["checks"]["G5_capture_explicit_stream"] = {"ok": False,
                                                         "detail": f"{type(e).__name__}: {str(e)[:150]}"}
        _cleanup(dev)

    # ══════════ G4*：capture 区内切流（**宽容度观察项，不计入 verdict，放最后跑**）══════════
    #   理由见模块 docstring：该写法在上游契约之外；且失败的捕获会污染后续条目状态 ⇒ 必须最后跑。
    g4_ok = False
    try:
        g4 = GCls()
        s_cap = dev.Stream()
        x4 = torch.randn(n, n, device=DEV)
        w4 = torch.randn(n, n, device=DEV)
        y4_ref = (x4 @ w4).sum().item()
        (x4 @ w4).sum()                        # 预热
        dev.synchronize()
        with dev.graph(g4):
            with dev.stream(s_cap):            # 捕获期内切到命名流（仍不同步）
                y4 = (x4 @ w4).sum()
        dev.synchronize()
        g4.replay()
        dev.synchronize()
        val4 = y4.item()
        rel4 = abs(val4 - y4_ref) / max(abs(y4_ref), 1.0)
        g4_ok = rel4 < 1e-3
        print(f"[G4*] capture 内切流（**契约外用法**，观察项）: val={val4:.6f} ref={y4_ref:.6f} "
              f"rel_err={rel4:.2e} {'容  忍' if g4_ok else '不容忍'}")
        result["checks"]["G4_capture_stream_swap"] = {
            "ok": g4_ok, "informational": True,
            "detail": f"契约外用法；本运行时{'容忍' if g4_ok else '不容忍'}（rel_err={rel4:.2e}）"}
    except Exception as e:
        print(f"[G4*] capture 内切流（**契约外用法**，观察项）: 不容忍 —— "
              f"{type(e).__name__}: {str(e)[:120]}")
        result["checks"]["G4_capture_stream_swap"] = {
            "ok": False, "informational": True,
            "detail": f"契约外用法；本运行时不支持：{type(e).__name__}: {str(e)[:120]}"}

    all_ok = g1_g2_g3_ok and g5_ok
    judged = {k: v for k, v in result["checks"].items() if not v.get("informational")}
    passed = sum(1 for v in judged.values() if v.get("ok"))
    obs = result["checks"].get("G4_capture_stream_swap", {})
    result["verdict"] = "GRAPH_CAPTURE_PASS" if all_ok else "GRAPH_CAPTURE_FAIL"
    result["passed"] = passed
    result["total"] = len(judged)
    result["observed_contract_violating_usage"] = {
        "ok": bool(obs.get("ok")), "detail": obs.get("detail", "")}
    result["note"] = (f"graph capture 流语义（契约内）{passed}/{len(judged)} 通过"
                      f"（capture / replay 确定性 / 输入更新 / 显式 stream）；"
                      f"另附宽容度观察项 1 项（契约外用法，不计入判定）")
    print(f"\n{result['verdict']}: {result['note']}")
    _dump(result)
    return 0 if all_ok else 1


def _dump(result: dict) -> None:
    out = OUT_DIR / f"graph_capture_stream_result{TAG}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    sys.exit(main())
