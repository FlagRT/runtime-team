"""错误注入 → 恢复闭环（设备侧职责）· 原型验证 V1

我们（设备上下文）在这条链路上负责的三段：
  ① 注入的真实错误能否被**正确识别**（translate_error → L1–L4 + disposition）
  ② 恢复动作能否**执行**（recover_device 原语 / 重放 / 重试）
  ③ 恢复后**业务是否继续可用**

监控方向负责"何时注入、注入什么、恢复编排"；本脚本提供设备侧证据。

四类注入（2026-09-22 起，第 4 类**按后端分化**，见 `l4_injection_plan()`）：
  ① 形状不匹配 → 期望 L2_PARAM（参数类，raise）          ② 显存超限 → 期望 L1_RESOURCE（资源类，retry）
  ③ 流同步超时 → 期望 L3_EXECUTION（执行类，replay）      ④ 码串入 → 期望见下
      · 声明 `error_map` 的后端：用它**自己的**码样例，期望 `mapped=True` / `graded_by=code_map`
      · **未声明**的后端（kunlun / cambricon）：注入**共享码表（昇腾 ACL）内的数字码**，
        期望 `mapped=False` / `error_code=None` —— 这是一次**诚实性负向测试**：
        不得把他厂数字码当成本厂商的码表命中。
  ⚠️ 本期修正了一处**假证据**：③ 的文案原先硬编码「真实 507046」，而 507046 是**昇腾**码，
     在无厂商码的后端上并不产生 ⇒ 改为从异常本身如实提取码（有则带出，无则明说）。

用法：
  python3 proto_error_recovery_loop.py --backend flagos      # 训练腿（torch_fl）
  python3 proto_error_recovery_loop.py --backend ascend      # 推理腿（torch_npu）
  python3 proto_error_recovery_loop.py --backend kunlun      # P800
  python3 proto_error_recovery_loop.py --backend cambricon   # 寒武纪 MLU
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))        # prototype/runtime/proto
_RUNTIME_DIR = os.path.dirname(_HERE)                     # prototype/runtime
_PKG_DIR = os.path.dirname(_RUNTIME_DIR)                  # prototype（import runtime 用）
sys.path.insert(0, _HERE)
sys.path.insert(0, _RUNTIME_DIR)
sys.path.insert(0, _PKG_DIR)

import runtime  # noqa: E402


def business_once(size: int = 512):
    """一段可重复的"业务"：在当前设备上做一次真实计算并返回校验值。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    x = torch.ones(size, size, device=dev)
    y = (x * 3).sum().item()
    return float(y)


def inject_shape_mismatch():
    """真实异常：形状不匹配的矩阵乘（参数类）。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    a = torch.randn(3, 4, device=dev)
    c = torch.randn(5, 6, device=dev)
    return a @ c                      # 必然抛出


def inject_oom():
    """真实异常：申请超出显存的张量（资源类）。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    try:
        stats = runtime.memory_stats(0)
        total = int(stats.get("total_mb", 0) or 0)
    except Exception:
        total = 0
    # ⚠️ 2026-09-22 修（证据卫生）：`total_mb` 为 0 或缺失时**不能照用**。
    #    旧写法只在本调用抛异常时才回退默认值；而后端如实降级（如拿不到设备总量）
    #    时返回的是 `total_mb=0`（**不抛异常**）⇒ `n = 0` ⇒ `torch.empty(0)`：
    #    既不报错也不占显存，**OOM 注入被静默跳过**，而记录里仍写着"已注入"。
    #    实测：910C 上 flagos 因 `acl.init rc=100002` 误降级 ⇒ total_mb=0 ⇒
    #    错误闭环出现「oom(L1_RESOURCE 期望) 未触发异常」——**看起来像后端缺陷，
    #    实际是测试工具的证据污染**（同"硬编码昇腾错误码"那次的同类问题）。
    if total <= 0:
        print(f"  [工具提示] memory_stats 未给出 total_mb（={total}）⇒ OOM 注入回退默认估值 "
              f"60000 MiB；若后端确实拿不到设备总量，请在后端侧修（勿在本工具里掩盖）")
        total = 60000
    # 2026-09-09 核查：一次性 3 倍显存申请**不会**拖死进程
    # （独立实验中正常抛 OutOfMemoryError 且后续业务正常，用时 9s；
    #   渐进式同样安全但耗时 38s）。故保持一次性。
    n = int((total * 1024 * 1024) * 3)          # 3 倍显存 → 必失败
    return torch.empty(n, dtype=torch.uint8, device=dev)


def inject_timeout(b):
    """真实异常：流同步超时（执行类）。

    2026-09-09 归因核查更正：真实触发**不会**终止进程。
    直调 pyACL（rc=507046）与统一封装路径（TimeoutError rc=507046）
    在干净进程与「OOM 之后」两种顺序下均被正常捕获、进程存活、后续业务正常。
    故此处改为**进程内真实触发**，交 handle 走 L3 处置。
    """
    import torch

    if not b.supports("bounded_sync"):
        return None, "该后端未声明有界同步能力，跳过（如实标注，不伪造）"

    dev = f"{b.device_type}:0"
    s = runtime.create_stream()
    with b.stream_context(s.native):
        a = torch.randn(8192, 8192, device=dev)
        for _ in range(5):
            a = a @ a
    try:
        b.synchronize_stream(s.native, timeout_ms=1)
    except TimeoutError as e:
        # ⚠️ 2026-09-22 修：原实现把「真实 507046」**硬编码在文案里** —— 507046 是**昇腾**错误码，
        #   在无厂商码的后端（昆仑芯 / 寒武纪）上会制造一条**假证据**（本实例根本不产生该码，
        #   只是恰好能触发 TimeoutError）。改为从异常本身如实提取：有码就带出，无码就明说。
        _m = re.search(r"(?:ret\s*=|error code is)\s*(\d+)", str(e))
        note = ("进程内捕获 TimeoutError，进程存活；"
                + (f"异常消息内厂商码={_m.group(1)}" if _m
                   else "异常消息内无厂商码（本后端为「超时上报」语义，不产生设备侧数字码）"))
        # 重放前置：等待流上剩余任务落地、设备归零
        try:
            b.synchronize(0)
            note += "；已等待设备归零"
        except Exception:
            note += "；设备归零等待失败"
        return e, note
    except Exception as e:
        return e, f"其他异常 {type(e).__name__}: {str(e)[:80]}"
    return None, "未触发超时（任务在 1ms 内完成）"


def l4_injection_plan(b):
    """决定 L4 注入用哪条消息，并给出**期望**（写进结果 JSON，使记录自描述）。

    之所以要分后端，是因为这条注入在不同后端上**验证的是完全不同的东西**：

    - 后端声明 `error_map`（有本厂商码表）→ 用**它自己的**码样例（`SAMPLE_CODED_ERROR`）
      触发码表路径，期望 `mapped=True` / `graded_by="code_map"`。
    - **未声明**（kunlun / cambricon：厂商码不透出为数字码）→ 注入一条携带**共享码表内数字码**
      的消息。共享码表（`conformance/errors.py` 的 `ACL_ERR_TO_CATEGORY`）是**昇腾 ACL 码表**，
      对这个后端而言该数字属**他厂码** ⇒ 这条注入实际是一次**诚实性负向测试**：
      期望 `mapped=False` / `graded_by≠code_map` / `error_code=None`，
      即**不得**把他厂数字码当成本厂商的码表命中。

    ⚠️ 2026-09-22 修（第 6 个跨后端缺陷）：原实现对所有后端统一标为「按码表触发」，
    对无码表后端是**错误描述**；而当时后端只降级了 `graded_by`、没降级 `mapped`，
    于是产出 `mapped=true` + `graded_by=message_hint_unexpected` 的自相矛盾记录
    （`mapped=True` 的契约含义是**确定分级**）。
    """
    sample = getattr(b, "SAMPLE_CODED_ERROR", None)
    if b.supports("error_map") and sample:
        return ("l4_by_code(本厂商码表触发)", sample,
                {"mapped": True, "graded_by": "code_map"},
                "期望：本厂商码表命中（mapped=True / graded_by=code_map）")
    return ("l4_by_code(码表内他厂码·诚实降级负向测试)",
            "AICORE exception, error code is 507015",
            {"mapped": False, "error_code": None},
            "期望：如实降级（mapped=False / error_code=None；不得声称码表命中）")


def handle(exc, backend, tag: str) -> dict:
    """设备侧三段：识别 → 处置 → 业务继续。"""
    rec: dict = {"inject": tag, "raw": str(exc)[:140]}
    fe = runtime.translate_error(exc, location=f"{backend.name}:0/{tag}")
    rec["category"] = getattr(fe.category, "value", str(fe.category))
    rec["disposition"] = getattr(fe, "disposition", None)
    rec["mapped"] = getattr(fe, "mapped", None)
    rec["graded_by"] = getattr(fe, "graded_by", None)
    # 2026-09-22 新增：把 error_code 也记进证据（否则"该码非本厂商码 ⇒ 必须为 None"
    # 这条期望在结果 JSON 里无从核对）
    rec["error_code"] = getattr(fe, "error_code", None)

    # ② 按 disposition 执行处置（设备侧能做的部分）
    action = "none"
    if rec["disposition"] == "retry":
        try:
            business_once(256)
            action = "retry_ok"
        except Exception as e2:
            action = f"retry_failed:{type(e2).__name__}"
    elif rec["disposition"] == "raise":
        action = "raise_to_caller（参数类，重试无意义）"
    elif rec["disposition"] in ("replay", "device_recovery"):
        r = runtime.recover_device(0, mode="probe")
        # 兼容：统一约定返回 dict；若后端仍返回 bool 则按布尔解读
        recovered = r.get("recovered") if isinstance(r, dict) else bool(r)
        rec["recover"] = r if isinstance(r, dict) else {"recovered": recovered}
        action = f"recover_probe={'ok' if recovered else 'fail'}"
        if rec["disposition"] == "replay":
            try:
                business_once(256)
                action += " + replay_ok"
            except Exception as e2:
                action += f" + replay_failed:{type(e2).__name__}"

    # ③ 业务继续验证（恢复后仍可正常计算）
    try:
        v = business_once(512)
        expect = 512 * 512 * 3
        rec["business_continues"] = abs(v - expect) < max(1.0, expect * 1e-4)
        rec["business_value"] = v
    except Exception as e2:
        rec["business_continues"] = False
        rec["business_error"] = f"{type(e2).__name__}: {str(e2)[:80]}"

    rec["action"] = action
    rec["ok"] = bool(rec.get("business_continues")) and rec["category"] is not None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"),
                    help="运行时后端名；也可用环境变量 DC_BACKEND（与训练腿/推理腿脚本一致）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-timeout", action="store_true",
                    help="跳过超时注入（默认触发；2026-09-09 核查已证明进程内安全）")
    args = ap.parse_args()

    runtime.use(args.backend)
    b = runtime.current()
    runtime.set_device(0)
    print(f"后端: {b.name} | 设备数: {runtime.device_count()}")

    results = []
    # 基线：注入前业务正常
    try:
        base = business_once(512)
        results.append({"inject": "baseline", "ok": True, "value": base})
        print(f"[基线] 业务正常: {base}")
    except Exception as e:
        results.append({"inject": "baseline", "ok": False, "error": str(e)[:100]})
        print(f"[基线] 异常: {e}")

    l4_tag, l4_msg, l4_expect, l4_expect_note = l4_injection_plan(b)

    def inject_l4():
        raise RuntimeError(l4_msg)

    injections = [
        ("shape_mismatch(L2_PARAM 期望)", inject_shape_mismatch, {}),
        ("oom(L1_RESOURCE 期望)", inject_oom, {}),
        ("stream_timeout(L3_EXECUTION 期望)", None, {"timeout": not args.no_timeout}),
        (l4_tag, inject_l4, {"expect": l4_expect, "expect_note": l4_expect_note}),
    ]

    for tag, fn, meta in injections:
        try:
            if meta.get("timeout"):
                exc, note = inject_timeout(b)
                if exc is None:
                    results.append({"inject": tag, "ok": None, "skipped": note})
                    print(f"[{tag}] 跳过：{note}")
                    continue
                rec_t = handle(exc, b, tag)
                rec_t["note"] = note or ""
                results.append(rec_t)
                print(f"[{tag}] {rec_t['category']} / {rec_t['disposition']} / "
                      f"业务继续={rec_t.get('business_continues')} / {note}")
                continue
            else:
                exc = None
                fn()
        except Exception as e:
            exc = e
        if exc is None:
            results.append({"inject": tag, "ok": False, "detail": "未触发异常（如实标注）"})
            print(f"[{tag}] 未触发异常")
            continue
        rec = handle(exc, b, tag)
        # 按后端的期望核对（2026-09-22 新增）：期望不符 ⇒ 本条**判失败**，
        # 不能只看"业务还在跑"就放过 —— 那正是"看起来通过"的来源。
        exp = meta.get("expect")
        if exp:
            rec["expectation"] = meta.get("expect_note", "")
            bad = sorted(k for k, v in exp.items() if rec.get(k) != v)
            rec["expect_matched"] = not bad
            if bad:
                rec["ok"] = False
                rec["expect_failed_on"] = bad
        results.append(rec)
        print(f"[{tag}] {rec['category']} / {rec['disposition']} / "
              f"action={rec['action']} / 业务继续={rec.get('business_continues')}"
              + (f" / 期望核对={'✅' if rec.get('expect_matched') else '❌'}"
                 f"{'' if rec.get('expect_matched') else ' 未达项=' + str(rec.get('expect_failed_on'))}"
                 if exp else ""))

    closed = [r for r in results if r.get("ok") is True]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if r.get("ok") is False]

    out = {
        "backend": b.name,
        "closed_loops": len(closed),
        "skipped": len(skipped),
        "failed": len(failed),
        "results": results,
        "verdict": "ERROR_RECOVERY_LOOP_PASS" if closed and not failed else "ERROR_RECOVERY_LOOP_PARTIAL",
        "note": ("设备侧职责：识别(translate_error) + 恢复执行(recover_device/重放/重试) + "
                 "业务继续验证。监控方向负责注入编排与恢复策略编排。"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = args.out or f"error_recovery_loop_{b.name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== {out['verdict']} | 闭环 {len(closed)} / 跳过 {len(skipped)} / 失败 {len(failed)} ===")
    print(f"结果: {path}")


if __name__ == "__main__":
    main()
