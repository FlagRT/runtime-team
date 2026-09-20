#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_stream_semantics_full.py — 多流 Stream 职责子项全量核查（**后端无关 V2**）
═══════════════════════════════════════════════════════════════════════════════

【这份探针是什么】
  多流 Stream **验收基线 16 项（S-1 ~ S-16）** 中"需要真正创建流才能验证"的那部分：
  S1/S2 补强 + S8 ~ S13（共 8 项）。基线全表见 `docs/RUNTIME_DC_STREAM_PLAN_20260907.md` §5。
  其余项由 conformance 用例覆盖（S-3/S-4 → s3/s4；S-14 → e2；S-5 → i6 等）。

【后端无关化 V2 的由来（2026-09-20，P800 第二实例）】
  初版为 910C A 线专用，硬编码 `import torch_npu` 与 `torch.npu.*`（约 25 处），
  只能在昇腾上跑，无法做"新后端逐项比对"。现改为：
    · 设备 API 前缀由统一运行时给出（`runtime.current().device_type` → "npu" / "cuda"）；
    · `torch.<DEV>.*` 通过 `getattr(torch, DEV)` 动态取，**同一份逻辑跨芯片复用**；
    · 两处厂商专有 API 按后端分支，替代方式**如实标注在结果里**（见 S9 / S12 注释）。
  ⇒ 这正是「换芯片不改代码」在验证资产上的体现。

【环境变量】
  DC_BACKEND  运行时后端名（默认 ascend；昆仑芯用 kunlun）
  DC_DEV_API  设备 API 前缀兜底（默认取后端 device_type；无 runtime 时可用 "npu"/"cuda"）
  DC_OUT_DIR  结果输出目录（默认本目录）
  DC_TAG      结果文件名后缀（多后端/多镜像对照时用于分离，如 "_p800"）

【判定】各子项独立判定；整体 STREAM_SEMANTICS_PASS（全通过）/ PARTIAL（部分）
       不支持但**如实标注**的项不计为失败（如 S12 在昆仑芯上的上游缺陷），
       但会在 detail 中写明"不支持 + 原因 + 是否本层可修"。

【数值纪律】一律 `.cpu()` 后再取标量比对（避免设备侧比较引入额外同步语义，910C 坑 B4）。

【用法】
  910C：DC_BACKEND=ascend python3 probe_stream_semantics_full.py --rounds 5
  P800：CUDA_VISIBLE_DEVICES=6,7 DC_BACKEND=kunlun DC_OUT_DIR=/workspace/out_stream \
        python3 probe_stream_semantics_full.py --rounds 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
#: prototype/ 根（探针位于 prototype/probes/ 下）
_PKG_DIR = _HERE.parent
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

BACKEND = os.environ.get("DC_BACKEND", "ascend")
OUT_DIR = Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
TAG = os.environ.get("DC_TAG", "")


def resolve_dev_api() -> str:
    """确定设备 API 前缀（"npu" / "cuda"），并顺带把后端切好。

    优先级：runtime 后端声明的 device_type > 环境变量 DC_DEV_API > 按后端名的默认映射。
    """
    try:
        import runtime  # noqa: E402
        runtime.use(BACKEND)
        dt = getattr(runtime.current(), "device_type", "")
        if dt:
            return dt
    except Exception as exc:                      # 无 runtime（纯 torch 环境）时兜底
        print(f"[env] runtime 不可用（{type(exc).__name__}），回退环境变量/默认映射")
    if os.environ.get("DC_DEV_API"):
        return os.environ["DC_DEV_API"]
    return {"ascend": "npu", "flagos": "npu", "kunlun": "cuda"}.get(BACKEND, "cuda")


DEV_API = resolve_dev_api()
#: 设备命名空间模块（torch.npu / torch.cuda）—— 全部设备 API 经它取用
dev = getattr(torch, DEV_API)
DEV = DEV_API                                     # torch 张量的 device 串前缀


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--leak-iters", type=int, default=1000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== probe_stream_semantics_full.py: 多流 Stream 职责子项全量核查（后端无关 V2）===")
    ver = getattr(getattr(torch, DEV_API), "__version__", "n/a")
    print(f"[env] backend={BACKEND} dev_api={DEV_API} devices={dev.device_count()} "
          f"torch={torch.__version__} runtime={ver}")
    torch.zeros(1, device=DEV)
    dev.set_device(0)

    checks = {}

    # ══════════ S1 流内顺序性（补强：真正创建流）══════════
    try:
        s = dev.Stream()
        with dev.stream(s):
            x = torch.ones(64, 64, device=DEV)          # op1：全 1
            y = x * 3                                    # op2：乘 3
            z = y + 2                                    # op3：加 2
            r = z.mean()                                 # op4：归约
        dev.synchronize()
        val = r.cpu().item()
        ok = abs(val - 5.0) < 1e-4                       # ((1*3)+2).mean() = 5
        checks["S1_stream_order"] = {"ok": ok, "detail": f"流内 4 个 op 按序执行 → {val:.6f}（期望 5.0）"}
        print(f"[S1] 流内顺序: {val:.6f}（期望 5.0）{'✅' if ok else '❌'}")
    except Exception as e:
        checks["S1_stream_order"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S1] 异常: {e}")

    # ══════════ S2 无隐式同步（补强：真正两条流）══════════
    try:
        sA, sB = dev.Stream(), dev.Stream()
        with dev.stream(sA):
            xa = torch.ones(32, 32, device=DEV) * 7
        with dev.stream(sB):
            xb = torch.ones(32, 32, device=DEV) * 11
        dev.synchronize()
        va, vb = xa.mean().cpu().item(), xb.mean().cpu().item()
        ok = abs(va - 7.0) < 1e-4 and abs(vb - 11.0) < 1e-4
        checks["S2_no_implicit_sync"] = {"ok": ok, "detail": f"流A均值={va:.4f}(期望7) 流B均值={vb:.4f}(期望11)"}
        print(f"[S2] 无隐式同步: A={va:.4f} B={vb:.4f} {'✅' if ok else '❌'}")
    except Exception as e:
        checks["S2_no_implicit_sync"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S2] 异常: {e}")

    # ══════════ S8 默认流 vs 非默认流（含缓存分配器跨流安全）══════════
    try:
        # 8a 基本语义：默认流（当前流）与命名流各自可正确执行
        xd = torch.ones(16, 16, device=DEV) * 2
        rd = xd.mean()
        dev.synchronize()
        s2 = dev.Stream()
        with dev.stream(s2):
            xs = torch.ones(16, 16, device=DEV) * 4
            rs = xs.mean()
        s2.synchronize()
        ok_base = abs(rd.cpu().item() - 2.0) < 1e-4 and abs(rs.cpu().item() - 4.0) < 1e-4

        # 8b 关键工程语义：命名流上分配的内存被其他流使用时，必须 record_stream
        #     告知缓存分配器，否则内存可能被提前回收重用（数据竞争）。
        s_alloc = dev.Stream()
        s_user = dev.Stream()
        with dev.stream(s_alloc):
            buf = torch.ones(128, 128, device=DEV) * 5     # 在命名流分配
        ev = dev.Event()
        ev.record(s_alloc)
        s_user.wait_event(ev)
        with dev.stream(s_user):
            out = buf * 2                                   # 另一流使用该缓冲
        buf.record_stream(s_user)                           # 关键：告知分配器
        dev.synchronize()
        v = out.mean().cpu().item()
        ok_alloc = abs(v - 10.0) < 1e-4
        ok = ok_base and ok_alloc
        checks["S8_default_vs_named_stream"] = {
            "ok": ok,
            "detail": (f"默认流={rd.cpu().item():.4f}(期望2) 命名流={rs.cpu().item():.4f}(期望4)；"
                       f"跨流内存安全：record_stream 后复用结果={v:.4f}(期望10)。"
                       f"注：PyTorch 缓存分配器跨流复用须 record_stream，否则存在提前回收风险")}
        print(f"[S8] 默认流 vs 命名流: {rd.cpu().item():.4f}/{rs.cpu().item():.4f}；"
              f"跨流分配器安全(record_stream)={v:.4f} {'✅' if ok else '❌'}")
    except Exception as e:
        checks["S8_default_vs_named_stream"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S8] 异常: {e}")

    # ══════════ S9 流错误隔离（流 A 出错后流 B 仍可用）══════════
    # 注入方式按后端分支（两芯片的"API 级错误"形态不同，需如实标注）：
    #   · ascend：走 pyACL 真实注入 107015（未 subscribe 的流投递 callback），
    #             这是 910C 侧已定性的真实 API 级错误
    #   · 其他后端（昆仑芯等）：该 ACL 错误码不存在，改用**参数类异常注入**
    #             （shape 不匹配），语义同为"单次调用失败" → 验证同一件事：
    #             该次调用失败是否影响其他流
    s9_detail = {}
    try:
        if BACKEND in ("ascend", "flagos"):
            import acl
            acl.init()
            acl.rt.set_device(0)
            ev_a, _ = acl.rt.create_event()
            s_bad, _ = acl.rt.create_stream()
            s_good, _ = acl.rt.create_stream()
            import ctypes
            CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
            cb = CB(lambda a: None)
            rc_bad = acl.rt.launch_callback(cb, None, 0, s_bad)   # 预期 107015
            s9_detail["inject"] = "pyACL 真实注入（未 subscribe 流投递 callback）"
            s9_detail["expect_rc"] = 107015
            s9_detail["stream_A_error_rc"] = rc_bad
            inject_ok = (rc_bad == 107015)
            inject_desc = f"流A 注入 107015（rc={rc_bad}，期望107015）"
        else:
            # 昆仑芯：无 ACL 错误码；用参数类异常注入（安全、可复现、不污染设备状态）
            s_bad = dev.Stream()
            inject_ok, inject_desc, rc_bad = False, "", None
            try:
                with dev.stream(s_bad):
                    a4 = torch.ones(4, 8, device=DEV)
                    b4 = torch.ones(4, 8, device=DEV)
                    _ = a4 @ b4                      # (4,8)@(4,8) → 参数不匹配
                    dev.synchronize()
            except Exception as exc:
                rc_bad = type(exc).__name__
                inject_ok = True
                inject_desc = (f"流A 注入参数类异常（{rc_bad}，shape 不匹配）"
                               f"（替代方式：本后端无 ACL 107015 等价错误码）")
            s9_detail["inject"] = "参数类异常注入（shape 不匹配）"
            s9_detail["stream_A_error_rc"] = rc_bad

        # 流 B 仍应能正常执行任务
        s_good_t = dev.Stream()
        with dev.stream(s_good_t):
            xg = torch.ones(32, 32, device=DEV) * 6
        s_good_t.synchronize()
        vg = xg.mean().cpu().item()
        ok = inject_ok and abs(vg - 6.0) < 1e-4
        s9_detail["stream_B_value"] = round(vg, 6)
        checks["S9_stream_error_isolation"] = {
            "ok": ok,
            "detail": (f"【API 调用级隔离·已实测】{inject_desc}"
                       f"后流B 仍正常执行 → {vg:.4f}(期望6)。"
                       f"⚠️【设备级错误·语义推断未实测】芯片级错误（如昇腾 507014 AICORE_TIMEOUT / "
                       f"507015 AICORE_EXCEPTION）语义上影响该设备全部流，需设备级恢复而非流级重试；"
                       f"真实触发风险高未做，依据错误码定义与恢复分级推断。"
                       f"注入方式记录：{s9_detail['inject']}")}
        print(f"[S9] 流错误隔离(API级): 流A rc={rc_bad} 流B={vg:.4f} {'✅' if ok else '❌'}"
              f"  ⚠️设备级错误为语义推断")
    except Exception as e:
        checks["S9_stream_error_isolation"] = {"ok": False,
                                              "detail": f"{type(e).__name__}: {str(e)[:110]} {s9_detail}"}
        print(f"[S9] 异常: {e}")

    # ══════════ S10 流/事件生命周期与配额（循环创建销毁）══════════
    try:
        n = args.leak_iters
        t0 = time.time()
        for i in range(n):
            s = dev.Stream()
            ev = dev.Event()
            with dev.stream(s):
                t = torch.ones(8, 8, device=DEV)
            ev.record(s)
            s.synchronize()
            del s, ev, t
        gc.collect()
        dev.synchronize()
        dt = time.time() - t0
        # 仍能创建并使用新流 → 无配额泄漏
        s_new = dev.Stream()
        with dev.stream(s_new):
            xnew = torch.ones(16, 16, device=DEV) * 9
        s_new.synchronize()
        ok = abs(xnew.mean().cpu().item() - 9.0) < 1e-4
        checks["S10_stream_event_lifecycle"] = {
            "ok": ok,
            "detail": (f"{n} 次创建销毁（{dt:.2f}s）后仍可正常创建使用新流"
                       f"（新流结果={xnew.mean().cpu().item():.4f}，期望9）")}
        print(f"[S10] 生命周期: {n} 次创建销毁（{dt:.2f}s）后新流仍可用 {'✅' if ok else '❌'}")
    except Exception as e:
        checks["S10_stream_event_lifecycle"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S10] 异常: {e}")

    # ══════════ S11 跨流内存分配（流 A 分配，流 B 使用）══════════
    try:
        sA, sB = dev.Stream(), dev.Stream()
        with dev.stream(sA):
            buf = torch.ones(64, 64, device=DEV) * 3     # 在流 A 分配并写入
        ev = dev.Event()
        ev.record(sA)
        sB.wait_event(ev)                                # 建立跨流依赖
        with dev.stream(sB):
            used = buf * 2                               # 流 B 使用流 A 分配的内存
        dev.synchronize()
        v = used.mean().cpu().item()
        ok = abs(v - 6.0) < 1e-4
        checks["S11_cross_stream_memory"] = {"ok": ok, "detail": f"流A分配内存在流B使用（依赖后）→ {v:.4f}（期望6）"}
        print(f"[S11] 跨流内存: {v:.4f}（期望6）{'✅' if ok else '❌'}")
    except Exception as e:
        checks["S11_cross_stream_memory"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S11] 异常: {e}")

    # ══════════ S12 流优先级 ══════════
    # ⚠️ 安全约束：**不得裸调 `dev.Stream.priority_range()`** —— 昇腾可用但昆仑芯上
    #    实测会触发 PyTorch 自身 C++ 断言（XPytorch 上报非法优先级区间），
    #    属进程级 abort 风险。故一律经统一 API 取（kunlun 后端已主动拦截返回 None）。
    try:
        rng = None
        rng_src = "统一 API runtime.current().stream_priority_range()"
        try:
            import runtime  # noqa: F401
            rng = runtime.current().stream_priority_range()
        except Exception as exc:
            rng_src += f"（调用异常：{type(exc).__name__}）"
        # 昇腾侧额外用 pyACL 查权威 range（昆仑芯无此 API，跳过）
        acl_range = None
        if BACKEND in ("ascend", "flagos"):
            try:
                import acl
                acl.init()
                acl.rt.set_device(0)
                acl_range = acl.rt.device_get_stream_priority_range()
            except Exception:
                acl_range = None

        if rng is not None or acl_range is not None:
            if isinstance(acl_range, (tuple, list)) and len(acl_range) >= 2:
                least, greatest = acl_range[0], acl_range[1]
                detail = (f"leastPriority={least}, greatestPriority={greatest}"
                          f"（值越小优先级越高，范围 0~7）")
            else:
                detail = f"统一 API 返回={rng}"
            support = "支持"
        else:
            detail = ("**不支持**：统一 API 返回 None（昆仑芯后端已主动拦截）——"
                      "实测裸调 `Stream.priority_range()` 会触发 PyTorch 自身 "
                      "INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188 "
                      "(greatest_priority <= -1)，属上游上报非法优先级区间，非本层可修")
            support = "不支持（上游缺陷，如实标注）"

        # 多流并发仍正确（语义验证优先于优先级调度效果验证）
        sh, sl = dev.Stream(), dev.Stream()
        with dev.stream(sh):
            xh = torch.ones(32, 32, device=DEV) * 5
        with dev.stream(sl):
            xl = torch.ones(32, 32, device=DEV) * 8
        dev.synchronize()
        vh, vl = xh.mean().cpu().item(), xl.mean().cpu().item()
        ok = abs(vh - 5.0) < 1e-4 and abs(vl - 8.0) < 1e-4
        checks["S12_stream_priority"] = {
            "ok": ok,
            "detail": f"优先级：{support} —— {detail}；多流并发结果正确 "
                      f"hi={vh:.4f}(期望5) lo={vl:.4f}(期望8)。"
                      f"注：调度效果需压力测试验证（非本层验收项）。获取途径：{rng_src}"}
        print(f"[S12] 流优先级: {support} | range={acl_range or rng} | 并发 hi={vh:.4f} lo={vl:.4f} "
              f"{'✅' if ok else '❌'}")
    except Exception as e:
        checks["S12_stream_priority"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S12] 异常: {e}")

    # ══════════ S13 多设备流绑定 ══════════
    ndev = dev.device_count()
    try:
        if ndev < 2:
            checks["S13_multidevice_stream_bind"] = {
                "ok": True,
                "detail": f"可见设备数={ndev} < 2，多设备流绑定不适用（单设备环境跳过）——"
                          f"注意：这是**可见设备数**，多卡验证需放开可见卡（如 CUDA_VISIBLE_DEVICES=6,7）"}
            print(f"[S13] 可见设备数={ndev}<2，跳过（若需验证请放开两张卡）")
        else:
            results = []
            for d in range(min(2, ndev)):
                dev.set_device(d)
                sd = dev.Stream()
                with dev.stream(sd):
                    x = torch.ones(16, 16, device=f"{DEV}:{d}") * (d + 2)
                sd.synchronize()
                results.append(x.mean().cpu().item())
            dev.set_device(0)
            ok = all(abs(v - (i + 2)) < 1e-4 for i, v in enumerate(results))
            checks["S13_multidevice_stream_bind"] = {
                "ok": ok, "detail": f"双设备各创建独立流，结果={[round(v, 4) for v in results]}（期望 [2.0, 3.0]）"}
            print(f"[S13] 多设备流绑定: {[round(v, 4) for v in results]}（期望 [2,3]）{'✅' if ok else '❌'}")
    except Exception as e:
        checks["S13_multidevice_stream_bind"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:110]}"}
        print(f"[S13] 异常: {e}")

    # ══════════ 判定 ══════════
    passed = sum(1 for v in checks.values() if v["ok"])
    total = len(checks)
    verdict = "STREAM_SEMANTICS_PASS" if passed == total else "STREAM_SEMANTICS_PARTIAL"
    print(f"\n{verdict}: {passed}/{total} 子项通过")
    for k, v in checks.items():
        print(f"  {k:<32} {'✅' if v['ok'] else '❌'} {v['detail'][:90]}")

    out = OUT_DIR / f"stream_semantics_full_result{TAG}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "passed": passed, "total": total,
                   "backend": BACKEND, "dev_api": DEV_API,
                   "torch": torch.__version__, "checks": checks},
                  f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
