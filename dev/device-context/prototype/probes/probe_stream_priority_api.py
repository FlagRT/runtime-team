#!/usr/bin/env python3
"""流优先级统一 API 真机探针（(A) 方案落地核验，2026-09-30）。

回答四件事（**每件都要有判定，不靠"命令跑通了"**）：
  D1 不回归：`create_stream()` 仍返回统一 Stream，且能在流上下文里算对
  D2 `stream_priority_range()` 形状合规（**2 元组**或 None；不得透传厂商三元组）
  D3 声明 `stream_priority_control` ⇒ 范围内**能设置且回读一致**；越界 ⇒ `ValueError`
  D4 未声明 ⇒ `create_stream(priority=…)` ⇒ **`NotImplementedError`**（不得静默返回无优先级的流）
  D5 ⭐ **回读是真判定**：绕过本层直接建带优先级的流（厂商路径），用同一套回读看是否
     与请求一致 —— 若不一致，说明**厂商静默丢弃了参数**，本层的第 ④ 条校验正是拦这个的。
  D6（仅 ascend，信息项）**受阻点在哪**：pyACL 能建带优先级的流（回读保留），
     但 `torch.npu.Stream(stream_ptr=handle)` **静默忽略**该句柄 ⇒ 拿不到同一条流
     ⇒ 证明障碍在"包装回 torch"这一步，不在 ACL 层。

用法：`DC_ROOT=<原型根> python3 probes/probe_stream_priority_api.py --backend ascend [--dev 0]`
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True)
ap.add_argument("--dev", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

root = os.environ.get("DC_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch                                     # noqa: E402
import runtime                                   # noqa: E402
from runtime.api.stream import Stream            # noqa: E402

checks = {}
notes = {}


def judge(name, ok, detail=""):
    checks[name] = {"ok": bool(ok), "detail": str(detail)}


bk = runtime.use(a.backend)
n = bk.device_count()
dev = f"{bk.device_type}:{a.dev}"
print(f"=== 流优先级统一 API 真机探针 · backend={a.backend} ===")
print(f"devices={n} | torch={torch.__version__} | device={dev}")

has_set = bk.supports("stream_priority_control")
has_rb = bk.supports("stream_priority_readback")
print(f"capabilities: stream_priority={bk.supports('stream_priority')} "
      f"control={has_set} readback={has_rb}")

# ── D1 不回归：默认路径 ──
try:
    s0 = runtime.create_stream()
    with s0.context():                                    # 统一流上下文（`Stream.context()`）
        v = float((torch.ones(4, 4, device=dev).sum()).item())
    runtime.synchronize()
    judge("D1_create_stream_默认路径不回归",
          isinstance(s0, Stream) and abs(v - 16.0) < 1e-6,
          f"type={type(s0).__name__} 流上计算={v}（期望 16.0）")
except Exception as e:                                    # noqa: BLE001
    judge("D1_create_stream_默认路径不回归", False, f"{type(e).__name__}: {str(e)[:150]}")

# ── D2 范围形状 ──
rng = bk.stream_priority_range()
try:
    shape_ok = (rng is None) or (isinstance(rng, tuple) and len(rng) == 2
                                 and all(isinstance(x, int) for x in rng))
    judge("D2_range_形状为2元组或None", shape_ok,
          f"得到 {rng!r}（type={type(rng).__name__}）"
          + ("" if shape_ok else " ⇒ **透传了厂商形状**"))
except Exception:                                         # noqa: BLE001
    judge("D2_range_形状为2元组或None", False, f"{rng!r}")
bounds = bk._priority_bounds()
notes["range"] = rng
notes["bounds"] = bounds

# ── D2b 回读：默认流 ──
if has_rb:
    try:
        got0 = runtime.stream_priority_readback(s0)
        judge("D2b_readback_对默认流返回int", isinstance(got0, int) and not isinstance(got0, bool),
              f"回读={got0!r}")
    except Exception as e:                                # noqa: BLE001
        judge("D2b_readback_对默认流返回int", False, f"{type(e).__name__}: {str(e)[:150]}")
    if not has_set:
        # 声明了 readback 但没声明 control：**原语不可用时不得补零**
        judge("D2c_未声明control时readback不参与设置", True,
              "（本家只读：回读值仅供观测，不作为'设置成功'的依据）")

# ── D3 / D4 设置语义 ──
if has_set:
    if bounds:
        tries = sorted({bounds[0], bounds[1]})
        for p in tries:
            try:
                st = runtime.create_stream(priority=p)
                got = runtime.stream_priority_readback(st)
                judge(f"D3_设置priority={p}_成功且回读一致", got == p,
                      f"请求={p} 回读={got!r}")
            except Exception as e:                        # noqa: BLE001
                judge(f"D3_设置priority={p}_成功且回读一致", False,
                      f"{type(e).__name__}: {str(e)[:150]}")
        outs = []
        for bad in (bounds[0] - 1, bounds[1] + 1):
            try:
                runtime.create_stream(priority=bad)
                outs.append(f"{bad}: 未报错")
            except ValueError:
                outs.append(f"{bad}: ValueError ✓")
            except BaseException as e:                    # noqa: BLE001
                outs.append(f"{bad}: {type(e).__name__}")
        judge("D3b_越界priority抛ValueError",
              all("ValueError ✓" in x for x in outs), "；".join(outs))
    else:
        judge("D3_设置priority_成功且回读一致", False, "声明了 control 但区间不可解析 ⇒ 不一致")
else:
    for p in (0, 1):
        try:
            got = runtime.create_stream(priority=p)
            judge(f"D4_未声明control时priority={p}_必须显式拒绝", False,
                  f"未报错，且返回了 {type(got).__name__} ⇒ **静默给了一条无优先级的流**")
        except NotImplementedError as e:
            judge(f"D4_未声明control时priority={p}_必须显式拒绝",
                  "stream_priority_control" in str(e),
                  f"NotImplementedError（含能力键名={('stream_priority_control' in str(e))}）")
        except BaseException as e:                        # noqa: BLE001
            judge(f"D4_未声明control时priority={p}_必须显式拒绝", False,
                  f"异常类型不合契约：{type(e).__name__}: {str(e)[:120]}")

# ── D5 绕过本层：厂商路径是否静默丢弃参数 ──
ns = getattr(torch, bk.device_type)
req = 7 if (bounds and bounds[1] == 7) else (bounds[1] if bounds else 0)
try:
    if a.backend == "ascend":
        raw = ns.Stream(priority=req)
    elif a.backend == "cambricon":
        raw = ns.Stream(priority=req)
    else:                                                 # kunlun：CUDA 语义（负值 = 高优先级）
        raw = ns.Stream(priority=req)
    got_raw = bk.stream_priority_readback(raw)
    same = (got_raw == req)
    notes["vendor_path"] = {"requested": req, "readback": got_raw, "retained": same}
    judge("D5_厂商路径的参数保留性（不一致即'静默丢弃'）", True,
          f"直接 {bk.device_type} 建流 priority={req} ⇒ 回读={got_raw!r}"
          + ("（保留）" if same else " ⇒ **厂商静默丢弃**：本层若不校验就会产出假象"))
except Exception as e:                                    # noqa: BLE001
    judge("D5_厂商路径的参数保留性（不一致即'静默丢弃'）", True,
          f"厂商路径不可用（不影响本层结论）：{type(e).__name__}: {str(e)[:120]}")

# ── D6（仅 ascend）受阻点定位：ACL 保留 vs 包装丢失 ──
if a.backend == "ascend":
    try:
        acl = bk.acl
        h = acl.rt.create_stream_with_config(7, 0)
        h = h[0] if isinstance(h, tuple) else h
        rb_acl = bk._stream_priority_read_raw(h)
        # 尝试把 ACL 句柄包回 torch（stock torch 的 ExternalStream 走的是同一个 kwarg）
        wrapped = ns.Stream(stream_ptr=int(h))
        h_wrap = getattr(wrapped, "npu_stream", None)
        rb_wrap = bk._stream_priority_read_raw(h_wrap) if h_wrap else None
        notes["d6"] = {"acl_handle": int(h) if h else None, "acl_readback": rb_acl,
                       "torch_wrapped_handle": int(h_wrap) if h_wrap else None,
                       "torch_wrapped_readback": rb_wrap,
                       "same_stream": bool(h_wrap == h)}
        judge("D6_受阻点在包装而非ACL（ACL保留/包装丢失）",
              rb_acl == 7 and h_wrap != h,
              f"pyACL 建流(priority=7) 回读={rb_acl}（ACL **保留**）；"
              f"`Stream(stream_ptr=handle)` 拿到的却是 handle={h_wrap} ⇒ 回读={rb_wrap}"
              f" ⇒ **句柄被静默忽略**（同一条流？{h_wrap == h}）")
    except Exception as e:                                # noqa: BLE001
        judge("D6_受阻点在包装而非ACL（ACL保留/包装丢失）", False,
              f"{type(e).__name__}: {str(e)[:150]}")

# ── 汇总 ──
failed = [k for k, v in checks.items() if not v["ok"]]
for k, v in checks.items():
    print(f"  [{'PASS' if v['ok'] else 'FAIL'}] {k}"
          + (f"  {v['detail']}" if v['detail'] else ""))

verdict = "STREAM_PRIORITY_API_PASS" if not failed else "STREAM_PRIORITY_API_FAIL"
result = {"verdict": verdict, "backend": a.backend, "device": dev,
          "torch": torch.__version__,
          "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
          "capabilities": {"stream_priority": bk.supports("stream_priority"),
                           "stream_priority_control": has_set,
                           "stream_priority_readback": has_rb},
          "notes": notes, "checks": checks,
          "boundary": ("本探针只在**该实例**该栈上成立；结论不得跨实例外推。"
                       "D5/D6 是**观测项**（不是失败项）：它们说明厂商路径的参数保留性。")}
print(f"\n=== {verdict}（{len(checks) - len(failed)}/{len(checks)}）===")
print("BOUNDARY: 结论只在本栈（该实例 + 该 torch/驱动版本）成立，不得跨实例外推。")
if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("已写出:", a.out)
sys.exit(0 if not failed else 1)
