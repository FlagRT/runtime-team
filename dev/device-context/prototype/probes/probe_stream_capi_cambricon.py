#!/usr/bin/env python3
"""MLU590（寒武纪 `cambricon`）· **流原语三步判定 + 可包装性**（2026-10-08 第十二轮）。

【为什么要做】
后端注释里长期写着「本家**没有**独立的句柄式 C API（与 ascend 的 `aclrtStreamGetPriority` /
kunlun 的 `cuStreamGetPriority` 不同）」。**该断言是推测，从未按手册 §7.1 的三步判定验证过。**
而它直接决定契约 **§1.10 规则 4**（已释放对象必须明确报错）在本实例能否被真正行使：
只有存在「**厂商 C API 建流 + 包装成本层流**」的路径，才会产生「**本层拥有**」的流。

【三步判定（手册 §7.1：缺一不可）】
  1. **头文件入口表** —— `neuware/include/cnrt.h` 里的 queue（寒武纪把 stream 叫 queue）原语
  2. **库导出符号** —— `nm -D libcnrt.so` 里的同族符号
  3. **真调一次** —— ctypes 从 `/proc/self/maps` 取**真实**库（直接 `CDLL("libcnrt.so")` 可能抓到
     stub），真建队列 + 真回读 + 真销毁

【第 4 步（决定性，本脚本新增）】**能不能把厂商队列包回 torch**
  —— 这正是 910C 卡死的地方：pyACL 能建带优先级的流且可回读，但**没有任何入口包回 torch**
  （无 `torch.npu.ExternalStream`、`Stream(stream_ptr=…)` 静默忽略）⇒ 流进不了 torch 执行上下文，
  设置了也白设。**MLU590 必须实测同一件事，不能从 ascend 外推。**

用法（容器内，需设备）：
    DC_BACKEND=cambricon python3 probes/probe_stream_capi_cambricon.py --dev 0 --out <json>
"""
import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DC_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch_mlu  # noqa: E402

OUT: dict = {"steps": {}, "checks": {}}


def judge(name, ok, detail):
    OUT["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def note(step, detail):
    OUT["steps"][step] = detail
    print(f"[{step}] {detail}", flush=True)


def _real_lib_path(soname_substr="libcnrt"):
    """从 `/proc/self/maps` 取**已加载的真实**库路径（避开工具链 stub 库）。"""
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                if ".so" in line and soname_substr in line:
                    p = line.split()[-1]
                    if os.path.isabs(p):
                        return p
    except OSError:
        pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import runtime
    rt = runtime.use("cambricon")
    rt.set_device(a.dev)
    dev = f"mlu:{a.dev}"
    x = torch.ones(4, 4, device=dev)
    expect = float(x.sum().item() + x.numel())            # 16 + 16 = 32

    print("=" * 88)
    print("MLU590 流原语三步判定 + 可包装性")
    print("=" * 88)

    # ── 步骤 1：头文件入口表 ───────────────────────────────────────────
    print("\n[步骤 1] 头文件入口表")
    H = "/usr/local/neuware/include/cnrt.h"
    decls = []
    if os.path.exists(H):
        try:
            with open(H, encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.startswith("cnrtRet_t cnrtQueue") or ln.startswith("cnrtRet_t cnrtDeviceGetQueue"):
                        decls.append(ln.strip())
        except OSError:
            pass
    for d in decls:
        print("   ", d)
    note("1_headers", {"header": H, "declarations": decls})
    judge("S1 头文件里有 queue 建/销/回读原语",
          any("cnrtQueueCreate" in d for d in decls) and any("cnrtQueueDestroy" in d for d in decls),
          f"{len(decls)} 条声明（{H}）")

    # ── 步骤 2：库导出符号 ────────────────────────────────────────────
    print("\n[步骤 2] 库导出符号")
    libpath = _real_lib_path("libcnrt")
    syms = []
    if libpath:
        try:
            out = subprocess.run(["nm", "-D", "--defined-only", libpath],
                                 capture_output=True, text=True, timeout=60)
            syms = sorted({ln.split()[-1] for ln in out.stdout.splitlines()
                           if "cnrtQueue" in ln or "QueuePriorityRange" in ln})
        except (OSError, subprocess.SubprocessError):
            pass
    print(f"    realpath(from /proc/self/maps) = {libpath}")
    for s in syms:
        print("   ", s)
    note("2_symbols", {"lib": libpath, "symbols": syms})
    judge("S2 库导出符号里有 queue 建/销/回读",
          {"cnrtQueueCreateWithPriority", "cnrtQueueDestroy", "cnrtQueueGetPriority"} <= set(syms),
          f"{len(syms)} 个同族符号")

    # ── 步骤 3：真调一次（ctypes；从真实库取，不用 stub）──────────────────
    print("\n[步骤 3] 真调一次（ctypes + 真实库）")
    if not libpath:
        judge("S3 真调一次", False, "取不到真实 libcnrt 路径 ⇒ 无法真调（不猜）")
        return _finish(a)
    lib = ctypes.CDLL(libpath)

    lib.cnrtDeviceGetQueuePriorityRange.restype = ctypes.c_int
    lib.cnrtDeviceGetQueuePriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int),
                                                    ctypes.POINTER(ctypes.c_int)]
    pmin, pmax = ctypes.c_int(0), ctypes.c_int(0)
    rc_rng = int(lib.cnrtDeviceGetQueuePriorityRange(ctypes.byref(pmin), ctypes.byref(pmax)))
    print(f"    cnrtDeviceGetQueuePriorityRange rc={rc_rng} -> "
          f"least={pmin.value} greatest={pmax.value}")
    note("3_range", {"rc": rc_rng, "least": pmin.value, "greatest": pmax.value})

    lib.cnrtQueueCreateWithPriority.restype = ctypes.c_int
    lib.cnrtQueueCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                                ctypes.c_uint, ctypes.c_int]
    lib.cnrtQueueGetPriority.restype = ctypes.c_int
    lib.cnrtQueueGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.cnrtQueueDestroy.restype = ctypes.c_int
    lib.cnrtQueueDestroy.argtypes = [ctypes.c_void_p]

    # 逐个合法档位试：建 → 回读 → 销毁
    lo, hi = min(pmin.value, pmax.value), max(pmin.value, pmax.value)
    levels = list(range(lo, hi + 1)) if hi - lo <= 8 else [lo, hi]
    trials = []
    for lv in levels:
        q = ctypes.c_void_p(0)
        rc = int(lib.cnrtQueueCreateWithPriority(ctypes.byref(q), ctypes.c_uint(0), ctypes.c_int(lv)))
        got = None
        if rc == 0 and q.value:
            pv = ctypes.c_int(-999)
            rc_g = int(lib.cnrtQueueGetPriority(q, ctypes.byref(pv)))
            got = pv.value if rc_g == 0 else f"rc={rc_g}"
            rc_d = int(lib.cnrtQueueDestroy(q))
        else:
            rc_d = None
        trials.append({"request": lv, "create_rc": rc, "handle": hex(q.value or 0),
                       "readback": got, "destroy_rc": rc_d})
        print(f"    请求 {lv:>3} ⇒ create rc={rc} handle={hex(q.value or 0)} "
              f"回读={got} destroy rc={rc_d}")
    note("3_trials", trials)
    ok3 = all(t["create_rc"] == 0 and t["readback"] == t["request"] and t["destroy_rc"] == 0
              for t in trials) if trials else False
    judge("S3 C API 真建队列 + 回读一致 + 真销毁",
          ok3, f"{len(trials)} 个档位全部 create rc=0 / 回读==请求 / destroy rc=0")

    # ── 步骤 4（决定性）：能否包回 torch ──────────────────────────────
    print("\n[步骤 4] 决定性：厂商队列能否包成 torch 可用的流")
    probe = {
        "torch.mlu.ExternalStream": hasattr(torch.mlu, "ExternalStream"),
        "torch.mlu.Stream.__init__ params": None,
        "torch.mlu.Stream signature": None,
    }
    try:
        import inspect
        sig = inspect.signature(torch.mlu.Stream.__init__)
        probe["torch.mlu.Stream signature"] = str(sig)
        probe["torch.mlu.Stream.__init__ params"] = list(sig.parameters)
    except (TypeError, ValueError):
        pass
    for k, v in probe.items():
        print(f"    {k} = {v}")

    # 尝试 A：ExternalStream
    native = None
    wrapped_by = None
    q2 = ctypes.c_void_p(0)
    rc2 = int(lib.cnrtQueueCreateWithPriority(ctypes.byref(q2), ctypes.c_uint(0),
                                             ctypes.c_int(lo)))
    if rc2 == 0 and q2.value:
        if hasattr(torch.mlu, "ExternalStream"):
            try:
                native = torch.mlu.ExternalStream(q2.value)
                wrapped_by = "torch.mlu.ExternalStream"
            except BaseException as e:                                # noqa: BLE001
                print(f"    ExternalStream 失败：{type(e).__name__}: {str(e)[:100]}")
        if native is None:
            # 尝试 B：Stream(stream_ptr=...)
            try:
                native = torch.mlu.Stream(stream_ptr=q2.value)
                wrapped_by = "torch.mlu.Stream(stream_ptr=…)"
            except BaseException as e:                                # noqa: BLE001
                print(f"    Stream(stream_ptr=…) 失败：{type(e).__name__}: {str(e)[:100]}")
        if native is None:
            try:
                lib.cnrtQueueDestroy(q2)
            except BaseException:                                     # noqa: BLE001
                pass
    note("4_wrap", {"probe": probe, "wrapped_by": wrapped_by})
    judge("S4 存在把厂商队列包成 torch 流的入口",
          native is not None, f"wrapped_by={wrapped_by!r}")

    # ── 步骤 5：若可包 ⇒ 真跑算子 + 所有权判定 ─────────────────────────
    if native is not None:
        print("\n[步骤 5] 包装后的流：真跑算子 + 所有权判定")
        try:
            with runtime.Stream(rt, native).context():
                y = x + 1.0
            rt.synchronize(a.dev)
            ysum = float(y.sum().item())
            judge("S5a 包装后的流**真的能承载算子**",
                  abs(ysum - expect) < 1e-6, f"sum={ysum}（期望 {expect}）")
        except BaseException as e:                                    # noqa: BLE001
            judge("S5a 包装后的流**真的能承载算子**", False,
                  f"{type(e).__name__}: {str(e)[:110]}")
        # 所有权：本层未登记（因为不是走本层 C API 路径建的）⇒ release 应为 no-op
        try:
            judge("S5b 未经本层登记的流 ⇒ owns=False、release no-op（不得越权销毁）",
                  (not rt.owns_stream(native)) and (rt.release_stream(native) is False),
                  f"owns={rt.owns_stream(native)} release={rt.release_stream(native)!r}")
        except BaseException as e:                                    # noqa: BLE001
            judge("S5b 未经本层登记的流 ⇒ owns=False、release no-op", False,
                  f"{type(e).__name__}: {str(e)[:110]}")

    # ── 步骤 6：torch 侧流的生命周期（决定「该不该改走 C API」）───────────
    #
    # 为什么必须测：若 torch 侧的 `torch.mlu.Stream(priority=…)` **会**在建/弃之间泄漏设备队列，
    # 那我们现在每次调用都在漏（真缺陷）⇒ 必须改走 C API（可显式销毁）；
    # 若 torch 侧**池化复用**（不泄漏），那走 C API 反而引入「必须显式释放」的负担 ⇒ 保持现状。
    # ⚠️ 判据不能靠"应该会/应该不会"，必须实测。
    print("\n[步骤 6] torch 侧流的生命周期（池化 vs 每次新建）")
    N = 200
    seen = []
    for _ in range(N):
        st = torch.mlu.Stream(priority=lo)
        seen.append(int(getattr(st, "mlu_stream", 0)))
        del st
    gc.collect()
    distinct = len(set(seen))
    note("6_torch_stream_pool", {"created": N, "distinct_handles": distinct,
                                 "sample": [hex(v) for v in seen[:5]]})
    print(f"    建 {N} 条 torch.mlu.Stream(priority={lo}) 并逐条丢弃 ⇒ 不同句柄 {distinct} 个")
    print(f"    （池化复用 ⇒ 远小于 {N}；每次新建设备队列 ⇒ 接近 {N}）")

    # C API 路径对照：同一循环，不销毁
    capi = []
    keep = []
    for _ in range(N):
        q = ctypes.c_void_p(0)
        if int(lib.cnrtQueueCreateWithPriority(ctypes.byref(q), ctypes.c_uint(0),
                                              ctypes.c_int(lo))) == 0 and q.value:
            capi.append(int(q.value))
            keep.append(q)                       # 不销毁，稍后统一销毁
    capi_distinct = len(set(capi))
    for q in keep:
        try:
            lib.cnrtQueueDestroy(q)
        except BaseException:                                         # noqa: BLE001
            pass
    note("6_capi_pool", {"created": len(capi), "distinct_handles": capi_distinct})
    print(f"    对照：C API 建 {len(capi)} 条 ⇒ 不同句柄 {capi_distinct} 个（已全部销毁）")

    pooled = distinct < N // 2
    judge("S6 torch 侧流被**池化复用**（不泄漏设备队列）",
          pooled, f"建 {N} 条 ⇒ 不同句柄 {distinct} 个"
                  f"{'（池化）' if pooled else '（每次新建 ⇒ 疑似泄漏，应改走 C API）'}")

    # ── 步骤 7：丢弃包装对象后，底层队列是否仍可用（子进程，防段错误）──────
    print("\n[步骤 7] 丢弃 torch 流对象后，底层队列句柄是否仍可用（子进程隔离）")
    child = r"""
import ctypes, gc, json, sys, os
sys.path.insert(0, os.environ.get("DC_ROOT", "/work/dc_mlu_regen_20261008/prototype"))
import torch, torch_mlu
torch.mlu.init(); torch.mlu.set_device(0)
st = torch.mlu.Stream()
h = int(st.mlu_stream)
del st; gc.collect()
lib = ctypes.CDLL("/usr/local/neuware/lib64/libcnrt.so.7.4.0")
lib.cnrtQueueGetPriority.restype = ctypes.c_int
lib.cnrtQueueGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
pv = ctypes.c_int(-999)
rc = int(lib.cnrtQueueGetPriority(ctypes.c_void_p(h), ctypes.byref(pv)))
print(json.dumps({"handle": hex(h), "query_rc": rc, "readback": pv.value}))
"""
    try:
        r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                           timeout=300, env={**os.environ})
        tail = (r.stdout or "").strip().splitlines()
        res = tail[-1] if tail else ""
        alive = '"query_rc": 0' in res
        note("7_after_drop", {"rc": r.returncode, "stdout_tail": res,
                              "stderr_tail": (r.stderr or "").strip()[-160:]})
        print(f"    子进程 rc={r.returncode}；结果 = {res or '（无输出）'}")
        judge("S7 丢弃包装对象后底层队列**仍可用** ⇒ torch 不拥有该队列",
              alive, f"query rc 解读：{'队列仍存活（torch 未销毁）' if alive else '已不可用或子进程异常'}"
                     f"（子进程 rc={r.returncode}）")
    except BaseException as e:                                        # noqa: BLE001
        note("7_after_drop", {"error": f"{type(e).__name__}: {e}"})
        judge("S7 丢弃包装对象后底层队列仍可用", False, f"{type(e).__name__}: {str(e)[:100]}")

    # ── 步骤 8：三列对照（本层回读 / 设备 C API / torch 属性）─────────────────
    #
    # 这是**修复的验收判据**（2026-10-08 修「回读空转」后补的）：
    #   · 本层 `runtime.stream_priority_readback()` 必须 == **设备 C API 读数**（真回读）；
    #   · 而它与 **torch 属性**（`Stream.priority`）**不需要相等** —— 后者是构造参数的
    #     回显，**已被判定为不可用作回读**（旧实现用的正是它 ⇒ 判据空转）。
    #   ⭐ 一句话：**回读必须与"参数是否真的进了设备"因果相关**，而不是与"我们传了什么"。
    print("\n[步骤 8] 三列对照：本层回读 / 设备 C API / torch 属性")
    import runtime as _rt
    rows8 = []
    for req in (list(range(lo, hi + 1)) if hi - lo <= 8 else [lo, hi]):
        st = rt.create_stream(priority=req)
        try:
            layer = _rt.stream_priority_readback(st)
            h = int(getattr(st, "mlu_stream", 0) or 0)
            cac = lib.cnrtQueueGetPriority
            cac.restype = ctypes.c_int
            cac.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            pv = ctypes.c_int(-999)
            rc_c = int(cac(ctypes.c_void_p(h), ctypes.byref(pv)))
            dev = pv.value if rc_c == 0 else None
            tattr = getattr(st, "priority", None)
        finally:
            rt.release_stream(st)          # 本层拥有的流 ⇒ 必须释放（契约 §1.10 规则 3）
        rows8.append({"request": req, "layer_readback": layer, "device_capi": dev,
                      "torch_attr": tattr})
        print(f"    请求 {req} ⇒ 本层回读={layer!r}  设备C API={dev!r}  torch属性={tattr!r}"
              f"{'  ✅ 本层==设备' if layer == dev else '  ❌'}")
    note("8_three_way", rows8)
    judge("S8 本层回读 == 设备 C API 读数（回读是真回读，不是参数回显）",
          bool(rows8) and all(r["layer_readback"] == r["device_capi"] for r in rows8),
          f"{len(rows8)} 个档位全部一致"
          f"（torch 属性一列**允许不等**，它就是被替换掉的旧口径）")

    return _finish(a)


def _finish(a) -> int:
    failed = [k for k, v in OUT["checks"].items() if not v["ok"]]
    OUT["verdict"] = "CAMB_STREAM_CAPI_PASS" if not failed else "CAMB_STREAM_CAPI_FAIL"
    print("\n" + "=" * 88)
    print(f"通过 {len(OUT['checks']) - len(failed)} / 失败 {len(failed)}")
    for k in failed:
        print(f"  [FAIL] {k}: {OUT['checks'][k]['detail'][:140]}")
    print(f"verdict = {OUT['verdict']}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(OUT, f, ensure_ascii=False, indent=2)
        print(f"[out] {a.out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
