"""P800（昆仑芯 kunlun）流优先级「退化为单点」独立审计探针（2026-10-08 · 用户要求自查）。

**要回答的问题**：后端说「`cuCtxGetStreamPriorityRange` 返回 `(0, 0)` ⇒ 优先级空间退化为单点
⇒ D2 不适用」—— 这是**真的设备属性**，还是**我只看了兼容层一层**？

**要打的四个新面**（旧证据都没覆盖）：
  Q1 **逐卡**查询（8 张卡全查）—— 排除「只在一张卡上测过就推广到全机」；
  Q2 **在显式创建上下文之后**查询 —— 排除「查询时机/无上下文导致返回默认值」；
  Q3 ⭐ **原生 XPU 接口面**：旧证据只说过「`libxpurt.so` 含 priority 的导出符号 = 0 个」。
     但 CUDA 兼容层（`cuCtxGetStreamPriorityRange`）**不等于**原生层 ——
     若原生有 `xpuStreamCreateWithPriority` / `xpuCtxGetStreamPriorityRange` 之类，
     那么「退化为单点」就只是**兼容层的属性**，而不是设备的属性 ⇒ 结论要改。
     本轮做**大小写不敏感 + 变体名**的全量符号扫描。
  Q4 **torch 侧与 C API 的交叉核对**：`torch.cuda.get_stream_priority_range()` /
     `Stream(priority=…)` / `ExternalStream` 与驱动 C API 是否一致。

判据（先写死）：
  R1 若任一卡的 range 非退化 ⇒ 「全机退化」**不成立**（只测一张卡是我的疏漏）。
  R2 若存在原生 XPU 优先级 API 且其 range 非退化 ⇒ 「设备属性」**不成立**（我只看了兼容层）。
  R3 若逐值建流 `cuStreamCreateWithPriority` 的回读恒 0 ⇒ 兼容层确实只有一个档位（佐证）。
"""
import argparse
import ctypes
import glob
import json
import os
import re
import subprocess
import sys

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

R = []


def note(k, v, why=""):
    R.append({"key": k, "value": v, "why": why})
    print(f"[P {k}] {v}" + (f"   ← {why}" if why else ""), flush=True)


def _maps_libs(pat):
    out = []
    try:
        for line in open("/proc/self/maps", encoding="utf-8", errors="ignore"):
            m = re.search(pat, line)
            if m and m.group(1) not in out:
                out.append(m.group(1))
    except BaseException:                                              # noqa: BLE001
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-devices", default=os.environ.get("AUDIT_SKIP_DEVICES", ""),
                    help="逗号分隔的卡号；**不去碰**它们（他人作业在用）")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip_devices.split(",") if x.strip() != ""}

    import torch
    print("=" * 70)
    note("env", f"torch={torch.__version__} cuda={torch.version.cuda} "
                f"dev_count={torch.cuda.device_count()}")
    note("visible", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    # ── 驱动库与映射 ──
    libcuda_paths = _maps_libs(r"(/\S*libcuda\.so\S*)")
    note("A1_maps_libcuda", libcuda_paths[:5] or "（未在 maps 中找到）")
    xpu_paths = _maps_libs(r"(/\S*libxpu\S*\.so\S*)")
    note("A2_maps_libxpu", xpu_paths[:8] or "（无）")

    # 通过 soname 打开（与后端 `_driver_handle` 同法）
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        note("A3_CDLL('libcuda.so.1')", "OK")
    except BaseException as e:                                          # noqa: BLE001
        note("A3_CDLL('libcuda.so.1')", f"{type(e).__name__}: {e}")

    # ── Q3：原生 XPU 接口面全量符号扫描（大小写不敏感 + 变体名）──
    print("\n===== Q3) 原生 XPU 接口面：全量符号扫描 =====")
    cand_libs = []
    for pat in ("/usr/local/xpu/**/*.so*", "/usr/local/xpu/**/lib/*.so*",
                "/usr/lib/x86_64-linux-gnu/libxpu*.so*", "/opt/**/libxpu*.so*",
                "/workspace/**/libxpu*.so*"):
        cand_libs += glob.glob(pat, recursive=True)
    # 点名找 native runtime 与 compat 层（旧证据提过 libxpurt.so —— 必须重新点名扫）
    for nm in ("libxpurt.so*", "libxpucuda.so*", "libxpu*.so*"):
        for root in ("/usr/local/xpu", "/usr/lib", "/usr/local/lib", "/opt"):
            cand_libs += glob.glob(os.path.join(root, "**", nm), recursive=True)[:40]
    cand_libs += [p for p in xpu_paths if re.search(r"\.so", p)]
    # 去重 + 排除 triton 目录（旧坑：从 triton 取同名库会版本错配）
    cand_libs = sorted({p for p in cand_libs if "triton" not in p})
    note("Q3_候选库数", len(cand_libs), str(cand_libs[:12]))

    found = {}
    for lp in cand_libs[:60]:
        try:
            r = subprocess.run(["nm", "-D", "--defined-only", lp],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                continue
            syms = set(re.findall(r"\b(\w*[Pp]riority\w*)\b", r.stdout))
            syms |= set(re.findall(r"\b(\w*StreamCreate\w*)\b", r.stdout))
            syms |= set(re.findall(r"\b(xpu\w*)\b", r.stdout))
            if syms:
                found[lp] = sorted(syms)[:40]
        except BaseException:                                          # noqa: BLE001
            continue
    note("Q3_命中的库与其符号", json.dumps(found, ensure_ascii=False)[:2500] if found else "（无命中）")

    # ── Q1/Q2：逐卡 + 显式上下文后查询 range ──
    print("\n===== Q1/Q2) 逐卡查询 range（含显式上下文）=====")
    fn_range = getattr(lib, "cuCtxGetStreamPriorityRange", None)
    note("Q1_有cuCtxGetStreamPriorityRange", fn_range is not None)
    fn_dev_get = getattr(lib, "cuDeviceGet", None)
    fn_ctx_create = getattr(lib, "cuCtxCreate_v2", None) or getattr(lib, "cuCtxCreate", None)
    fn_init = getattr(lib, "cuInit", None)
    note("Q1_有cuDeviceGet/cuCtxCreate", [fn_dev_get is not None, fn_ctx_create is not None])

    per_card = {}
    if fn_range is not None:
        fn_range.restype = ctypes.c_int
        fn_range.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        n = torch.cuda.device_count()
        for d in range(n):
            rec = {}
            if d in skip:
                # ⚠️ 纪律：**别人正在用的卡不去动它的上下文**（本轮卡 3 有他人 61 GB 作业）。
                #    代价是「这张卡的区间未测」——**如实标注**，而不是假装测过或悄悄跳过。
                per_card[str(d)] = {"skipped": f"该卡有他人作业在用 ⇒ 本轮不去创建上下文（未测）"}
                continue
            try:
                torch.cuda.set_device(d)
                _ = torch.zeros(1, device=f"cuda:{d}")          # 触发上下文创建
                torch.cuda.synchronize(d)
                least, greatest = ctypes.c_int(0), ctypes.c_int(0)
                rc = int(fn_range(ctypes.byref(least), ctypes.byref(greatest)))
                rec["rc"] = rc
                rec["range"] = [int(least.value), int(greatest.value)]
                rec["name"] = torch.cuda.get_device_name(d)
            except BaseException as e:                                  # noqa: BLE001
                rec["err"] = f"{type(e).__name__}: {str(e)[:90]}"
            per_card[str(d)] = rec
    note("Q1_Q2_逐卡range", json.dumps(per_card, ensure_ascii=False))
    deg = [k for k, v in per_card.items() if v.get("range") and v["range"][0] == v["range"][1]]
    nondeg = [k for k, v in per_card.items() if v.get("range") and v["range"][0] != v["range"][1]]
    note("Q1_退化卡数/非退化卡数", f"{len(deg)} / {len(nondeg)}",
         "非退化卡数 > 0 ⇒ 「全机退化」不成立")

    # ── Q4：torch 侧交叉核对 ──
    print("\n===== Q4) torch 侧交叉核对 =====")
    torch.cuda.set_device(0)
    try:
        note("Q4a_torch_get_stream_priority_range", torch.cuda.get_stream_priority_range())
    except BaseException as e:                                          # noqa: BLE001
        note("Q4a_torch_get_stream_priority_range", f"{type(e).__name__}: {str(e)[:110]}")
    try:
        s = torch.cuda.Stream(priority=-1)
        note("Q4b_torch_Stream(priority=-1).priority", getattr(s, "priority", "<无属性>"))
    except BaseException as e:                                          # noqa: BLE001
        note("Q4b_torch_Stream(priority=-1)", f"{type(e).__name__}: {str(e)[:110]}")
    try:
        note("Q4c_有ExternalStream", hasattr(torch.cuda, "ExternalStream"))
    except BaseException as e:                                          # noqa: BLE001
        note("Q4c_有ExternalStream", f"{type(e).__name__}: {str(e)[:80]}")

    # ── R3：逐值建流 + 回读 ──
    print("\n===== R3) cuStreamCreateWithPriority 逐值 + 回读 =====")
    f_create = getattr(lib, "cuStreamCreateWithPriority", None)
    f_getp = getattr(lib, "cuStreamGetPriority", None)
    note("R3_有create/getp", [f_create is not None, f_getp is not None])
    vals = {}
    if f_create is not None and f_getp is not None:
        f_create.restype = ctypes.c_int
        f_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        f_getp.restype = ctypes.c_int
        f_getp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        f_destroy = getattr(lib, "cuStreamDestroy_v2", None) or getattr(lib, "cuStreamDestroy", None)
        if f_destroy:
            f_destroy.restype = ctypes.c_int
            f_destroy.argtypes = [ctypes.c_void_p]
        for p in (-10, -5, -2, -1, 0, 1, 5, 10):
            h = ctypes.c_void_p(0)
            rc = int(f_create(ctypes.byref(h), ctypes.c_uint(0), ctypes.c_int(p)))
            got = None
            if rc == 0 and h.value:
                pv = ctypes.c_int(0x7FFFFFFF)
                rcg = int(f_getp(ctypes.c_void_p(h.value), ctypes.byref(pv)))
                got = int(pv.value) if rcg == 0 else f"rc={rcg}"
                if f_destroy:
                    f_destroy(ctypes.c_void_p(h.value))
            vals[str(p)] = {"create_rc": rc, "readback": got}
    note("R3_逐值建流回读", json.dumps(vals, ensure_ascii=False))

    # ══════════════ S) runtime API（`cuda*`）—— 与 driver API（`cu*`）是**两条实现路径** ══════════════
    # ⭐ 为什么必须补这一段：全量符号扫描显示**两个库**提供优先级 API ——
    #   `libxpucuda.so`（driver API `cu*`，本文件 Q1–R3 测的就是它）与
    #   `libcudart.so`（runtime API `cuda*`，**此前从未测过**）。
    #   只测其中一条就下「设备属性」结论 = **只看了半层**。判据：两条路径的区间是否一致。
    print("\n===== S) runtime API（cuda*）逐值 + 与 driver API 交叉 =====")
    try:
        rt = ctypes.CDLL("libcudart.so.12")
        f_rng = getattr(rt, "cudaDeviceGetStreamPriorityRange", None)
        note("S1_有cudaDeviceGetStreamPriorityRange", f_rng is not None)
        if f_rng is not None:
            f_rng.restype = ctypes.c_int
            f_rng.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            l3, g3 = ctypes.c_int(0), ctypes.c_int(0)
            rc3 = int(f_rng(ctypes.byref(l3), ctypes.byref(g3)))
            note("S1_runtime_range", {"rc": rc3, "least": int(l3.value), "greatest": int(g3.value)})
            drv = per_card.get("0", {}).get("range")
            note("S3_runtime_vs_driver_一致", (drv == [int(l3.value), int(g3.value)]),
                 f"driver(卡0)={drv} runtime={[int(l3.value), int(g3.value)]} "
                 f"⇒ 一致则「单点」不是某一条 API 的怪癖")
        f_c = getattr(rt, "cudaStreamCreateWithPriority", None)
        f_g = getattr(rt, "cudaStreamGetPriority", None)
        f_d = getattr(rt, "cudaStreamDestroy", None)
        note("S2_有建流/回读/销毁", [f_c is not None, f_g is not None, f_d is not None])
        if f_c and f_g:
            f_c.restype = ctypes.c_int
            f_c.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
            f_g.restype = ctypes.c_int
            f_g.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            if f_d:
                f_d.restype = ctypes.c_int
                f_d.argtypes = [ctypes.c_void_p]
            rtv = {}
            for p in (-2, -1, 0, 1, 2):
                h = ctypes.c_void_p(0)
                rc = int(f_c(ctypes.byref(h), ctypes.c_uint(0), ctypes.c_int(p)))
                got = None
                if rc == 0 and h.value:
                    pv = ctypes.c_int(0x7FFFFFFF)
                    rcg = int(f_g(ctypes.c_void_p(h.value), ctypes.byref(pv)))
                    got = int(pv.value) if rcg == 0 else f"rc={rcg}"
                    if f_d:
                        f_d(ctypes.c_void_p(h.value))
                rtv[str(p)] = {"create_rc": rc, "readback": got}
            note("S2_runtime逐值建流回读", json.dumps(rtv, ensure_ascii=False))
    except BaseException as e:                                          # noqa: BLE001
        note("S_失败", f"{type(e).__name__}: {e}")

    # ── N) 反向核对：有没有 `xpu*` 前缀的**原生**流 API（零命中才敢说"只有兼容层"）──
    print("\n===== N) 反向核对：原生 XPU 流 API 是否存在 =====")
    hit = []
    for lp in cand_libs:
        try:
            r = subprocess.run(["nm", "-D", "--defined-only", lp],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                continue
            h2 = sorted(set(re.findall(r"\b(?:xpu|XPU)(?:Stream|Queue)[A-Za-z]*\b", r.stdout)))
            if h2:
                hit.append({lp: h2[:20]})
        except BaseException:                                          # noqa: BLE001
            continue
    note("N_原生xpu流API", hit or "（零命中 ⇒ 不存在 xpu 前缀的原生流 API）",
         "零命中 ⇒ 优先级只有 CUDA 兼容层这一套")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"findings": R}, fh, ensure_ascii=False, indent=1)
        print(f"\n结果 -> {a.out}")
    print("\n===== P800 AUDIT DONE =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
