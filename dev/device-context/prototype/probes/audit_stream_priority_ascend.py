"""910C 流优先级「不适用」审计探针（2026-10-08 · 用户要求自查 · 合并版）。

**要回答的问题**：后端说 910C **无法**在「流可被 torch 执行上下文使用」的前提下设置优先级
⇒ 不声明 `stream_priority_control` ⇒ D2 出 `NOT_APPLICABLE`。
这是**真的接口/设备限制**，还是**我自己没找到路**？

**审计结论（本探针逐条取证）**：结论成立，且根因**比原表述更精确**。逐条证据：

| # | 检查 | 实测 |
|---|---|---|
| A | ACL 有没有 priority **setter** | ❌ 头文件 `aclrtSetStreamAttribute` 的枚举只有 `FAILURE_MODE / FLOAT_OVERFLOW_CHECK / USER_CUSTOM_TAG / CACHE_OP_INFO`；真实库里名字带 `Priority` 的导出符号**只有两个 Get** |
| B | ACL 层是否**保留**优先级 | ✅ `aclrtCreateStreamWithConfig(0/3/7)` 全部 `rc=0` 且**回读 == 请求**；回读是**真设备读数**（`aclrtStreamGetPriority` / `GetFlags` / `GetId` 三个独立口径都可用）⇒ **不是设备限制** |
| C | `torch.npu.Stream(priority=p)` 逐档位 | ⚠️ p=0..7 得到 **8 条互不相同的句柄**（stream_id 96→103）但**设备回读全部 = 0** ⇒ **参数确实没进设备**（旧证据"回读恒 0"到此才有逐档位原始读数） |
| D | kwarg 穷举 | 只有 `priority` / `stream_ptr` / `stream_id` 被接受；**`stream_ptr` 与 `stream_id` 都不改变底层优先** |
| E | `Stream(stream_id=<裸 ACL 句柄>)` | ⚠️ **构造不报错**（`stream_id` 逐位等于裸句柄），但**读 `.npu_stream` / `with torch.npu.stream(s)` / `s.synchronize()`** 任一步都抛 `INTERNAL ASSERT FAILED at NPUStream.cpp` / `Unrecognized stream …` |
| F | `_npu_setStream(stream_id=<裸 ACL 句柄>)` | ❌ 厂商原话：`RuntimeError: Unrecognized external NPU stream 187650865658800 on device npu:0` |
| G | C++ 面是否**有**正确入口 | ✅ **有** —— `libtorch_npu.so` 导出 `c10_npu::getStreamFromExternal(...)` 与 `setCurrentNPUStream(...)`；但 `torch_npu._C` **无任何 `*External*` 符号**，包内 Python **零处**引用 ⇒ **是「Python 绑定缺失」，不是「做不到」** |

⭐ **两个必须记住的仪器陷阱**（本探针自查出来并已修）：
  ① `hasattr(s, "priority")` 与 `getattr(s, "priority", 默认值)` **都会抛**
     `RuntimeError: NPU dose not support Stream.get_priority()` —— `hasattr` 只吞 `AttributeError`。
     ⇒ 首轮我用 `getattr` 兜默认值，把「读属性失败」**误记成「构造被拒绝」**，整段结论作废。
     **探「有没有这个能力」必须 `try/except BaseException` 并分类回报。**
  ② `Stream(stream_id=…)` **接受任何整数**（不校验）⇒ 「构造成功」**不能**当作「这条路可用」，
     真正的判据是**使用点**（读句柄 / 进上下文 / 同步）是否抛错。**接受 ≠ 生效。**

⚠️ 本探针**不做 pass/fail 判定** —— 它的产物是「路有/没有」的事实，裁定在报告里做。
危险动作一律**子进程隔离**（拿错句柄可能是未定义行为）。
"""
import argparse
import ctypes
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
    print(f"[A {k}] {v}" + (f"   ← {why}" if why else ""), flush=True)


def _real_libascendcl():
    """从 /proc/self/maps 取**真实已加载**的 libascendcl（避开工具链 stub 库 ⇒ rc=100039）。"""
    try:
        for line in open("/proc/self/maps", encoding="utf-8", errors="ignore"):
            m = re.search(r"(/\S*libascendcl\.so\S*)", line)
            if m and "stub" not in m.group(1).lower():
                return m.group(1)
    except BaseException:                                              # noqa: BLE001
        pass
    return None


CHILD = r'''
import ctypes, json, os, re, sys
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1"); sys.dont_write_bytecode = True
import torch, torch_npu
torch.npu.init(); torch.npu.set_device(0)
out = {}

lib_path = None
for line in open("/proc/self/maps", encoding="utf-8", errors="ignore"):
    m = re.search(r"(/\S*libascendcl\.so\S*)", line)
    if m and "stub" not in m.group(1).lower():
        lib_path = m.group(1); break
lib = ctypes.CDLL(lib_path)
for fn, rt, at in (("aclrtCreateStreamWithConfig", ctypes.c_int,
                    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint32]),
                   ("aclrtStreamGetPriority", ctypes.c_int,
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]),
                   ("aclrtStreamGetFlags", ctypes.c_int,
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]),
                   ("aclrtStreamGetId", ctypes.c_int,
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]),
                   ("aclrtDestroyStream", ctypes.c_int, [ctypes.c_void_p])):
    f = getattr(lib, fn, None)
    if f is not None:
        f.restype = rt; f.argtypes = at
out["libascendcl"] = lib_path

def read3(h):
    """回读三口径：priority / flags / stream_id（证明"回读真的看设备"而非看 Python 属性）"""
    if not h: return None
    pv = ctypes.c_uint32(0xFFFFFFFF); fv = ctypes.c_uint32(0xFFFFFFFF); iv = ctypes.c_int32(-1)
    rp = int(lib.aclrtStreamGetPriority(ctypes.c_void_p(int(h)), ctypes.byref(pv)))
    rf = int(lib.aclrtStreamGetFlags(ctypes.c_void_p(int(h)), ctypes.byref(fv)))
    ri = int(lib.aclrtStreamGetId(ctypes.c_void_p(int(h)), ctypes.byref(iv)))
    return {"priority": int(pv.value) if rp == 0 else f"rc={rp}",
            "flags": int(fv.value) if rf == 0 else f"rc={rf}",
            "stream_id": int(iv.value) if ri == 0 else f"rc={ri}", "rc": [rp, rf, ri]}

s0 = torch.npu.Stream()
dt, di = int(s0.device_type), 0

# ── B) ACL 保留性 + 回读自证 ──
out["B_default_stream_read3"] = read3(int(s0.npu_stream))
acl_keep = {}
for p in (0, 3, 7):
    h = ctypes.c_void_p(0)
    rc = int(lib.aclrtCreateStreamWithConfig(ctypes.byref(h), ctypes.c_uint32(p), ctypes.c_uint32(0)))
    acl_keep[str(p)] = {"create_rc": rc, "read3": read3(h.value) if h.value else None}
    if h.value:
        lib.aclrtDestroyStream(ctypes.c_void_p(h.value))
out["B_acl_preserves"] = acl_keep

# ── C) torch 侧逐档位 + 设备回读（旧证据缺的表）──
scan = {}
for p in range(0, 8):
    try:
        s = torch.npu.Stream(priority=p)
        h = int(s.npu_stream)
        # ⚠️ 绝不读 s.priority（会抛 RuntimeError；首轮就是被它坑的）
        scan[str(p)] = {"handle": hex(h), "stream_id": int(s.stream_id), "devread3": read3(h)}
    except BaseException as e:
        scan[str(p)] = {"err": f"{type(e).__name__}: {str(e)[:100]}"}
out["C_priority_scan"] = scan
out["C_distinct_device_priorities"] = sorted({str(v["devread3"]["priority"]) for v in scan.values()
                                              if isinstance(v, dict) and "devread3" in v})
out["C_distinct_handles"] = len({v.get("handle") for v in scan.values() if v.get("handle")})

# ── D) kwarg 穷举（**修正仪器**：把"构造接受"与"读属性抛异常"分开）──
kw = {"priority": 7, "stream_ptr": int(s0.npu_stream), "stream_id": int(s0.stream_id),
      "device_index": 0, "device_type": dt, "is_sync_launch": 1, "flags": 0}
kwres = {}
for k, v in kw.items():
    rec = {}
    try:
        s = torch.npu.Stream(**{k: v})
        rec["ctor"] = "接受"
        try:
            h = int(s.npu_stream)
            rec["handle"] = hex(h)
            rec["devprio"] = read3(h)["priority"]
        except BaseException as e:                                   # noqa: BLE001
            rec["handle_err"] = f"{type(e).__name__}: {str(e)[:90]}"
        try:
            rec["py_priority"] = s.priority                          # 会抛，故意试
        except BaseException as e:                                   # noqa: BLE001
            rec["py_priority"] = f"<抛 {type(e).__name__}: {str(e)[:60]}>"
    except BaseException as e:                                       # noqa: BLE001
        rec["ctor"] = f"拒绝 {type(e).__name__}: {str(e)[:90]}"
    kwres[k] = rec
out["D_kwarg_scan"] = kwres

# ── E) 把裸 ACL 流塞进 torch 对象：构造 vs 使用点 ──
raw = ctypes.c_void_p(0)
lib.aclrtCreateStreamWithConfig(ctypes.byref(raw), ctypes.c_uint32(7), ctypes.c_uint32(0))
rawv = int(raw.value or 0)
E = {"raw_handle": hex(rawv), "raw_devpriority": read3(rawv)["priority"]}
try:
    s = torch.npu.Stream(stream_id=rawv, device_index=di, device_type=dt)
    E["ctor_ok"] = True
    E["stream_id_equals_raw"] = (int(s.stream_id) == rawv)
    for label, fn in (("读 .npu_stream", lambda: int(s.npu_stream)),
                      ("repr(s)", lambda: repr(s)),
                      ("s.synchronize()", lambda: s.synchronize())):
        try:
            E[label] = f"OK -> {fn()!r}"[:130]
        except BaseException as e:                                   # noqa: BLE001
            E[label] = f"**抛 {type(e).__name__}**: {str(e)[:150]}"
    try:
        getraw = torch_npu._C._npu_getCurrentRawStream
        with torch.npu.stream(s):
            inside = int(getraw(di))
            x = torch.ones(8, 8, device="npu:0"); y = x @ x
            torch.npu.synchronize()
        E["with_torch_npu_stream"] = {"ok": True, "inside_raw": hex(inside), "sum": float(y.sum().item())}
    except BaseException as e:                                       # noqa: BLE001
        E["with_torch_npu_stream"] = f"**抛 {type(e).__name__}**: {str(e)[:170]}"
except BaseException as e:                                           # noqa: BLE001
    E["ctor_ok"] = False
    E["ctor_err"] = f"{type(e).__name__}: {str(e)[:170]}"
out["E_external_handle_into_torch"] = E

# ── F) _npu_setStream 用裸句柄（厂商原话最有力）──
try:
    torch_npu._C._npu_setStream(stream_id=rawv, device_index=di, device_type=dt)
    out["F_setStream_raw"] = "**居然成功**（需复查）"
except BaseException as e:                                           # noqa: BLE001
    out["F_setStream_raw"] = f"{type(e).__name__}: {str(e)[:220]}"

# ── F2) 对照：用「池内 id」走同一条路（证明链路本身没坏）──
try:
    getraw = torch_npu._C._npu_getCurrentRawStream
    other = torch.npu.Stream()
    torch_npu._C._npu_setStream(stream_id=int(other.stream_id), device_index=int(other.device_index),
                                device_type=int(other.device_type))
    out["F2_setStream_poolid"] = {"ok": True,
                                  "became_other": int(getraw(di)) == int(other.npu_stream)}
except BaseException as e:                                           # noqa: BLE001
    out["F2_setStream_poolid"] = f"{type(e).__name__}: {str(e)[:160]}"

# ── G) Python 侧接口面：有没有 External 入口 ──
out["G_torch.npu_ExternalStream"] = hasattr(torch.npu, "ExternalStream")
out["G_C_External_符号"] = sorted(a for a in dir(torch_npu._C) if "xternal" in a)
out["G_C_setStream族"] = sorted(a for a in dir(torch_npu._C)
                                if re.search(r"setStream|getCurrent.*Stream|RawStream", a))
print("CHILD_JSON " + json.dumps(out, default=str))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import torch
    import torch_npu                                                # noqa: E402

    torch.npu.init()
    torch.npu.set_device(0)
    note("env", f"torch={torch.__version__} torch_npu={torch_npu.__version__} "
                f"npu={torch.npu.get_device_name(0)}")

    # ── A) 头文件面：除 GetPriority 外还有没有 priority setter ──
    print("\n===== A) 头文件面 =====")
    import glob as _glob
    hdrs = []
    for pat in ("/usr/local/Ascend/cann-*/aarch64-linux/include/acl/acl_rt.h",
                "/usr/local/Ascend/ascend-toolkit/*/aarch64-linux/include/acl/acl_rt.h",
                "/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h"):
        hdrs += sorted(_glob.glob(pat))
    hdrs = sorted(set(hdrs))
    note("A0_acl_rt.h", hdrs[:3] or "（未找到）")
    if hdrs:
        txt = open(hdrs[0], encoding="utf-8", errors="ignore").read()
        note("A1_名字带Priority的函数", sorted(set(re.findall(r"\baclrt\w*[Pp]riority\w*\s*\(", txt))))
        note("A2_aclrtStreamAttr枚举值", sorted(set(re.findall(r"ACL_STREAM_ATTR_\w+", txt))),
             "枚举里没有 priority ⇒ aclrtSetStreamAttribute 不是优先级 setter")

    # ── B0) 真实库的 priority 符号 ──
    print("\n===== B0) 真实 libascendcl 的 priority 符号 =====")
    lp = _real_libascendcl()
    note("B0_lib", lp or "（未找到）")
    if lp:
        for tool in (["nm", "-D", "--defined-only", lp], ["readelf", "--dyn-syms", "-W", lp]):
            try:
                r = subprocess.run(tool, capture_output=True, text=True, timeout=120)
                if r.returncode == 0:
                    note("B0_priority符号",
                         sorted(set(re.findall(r"\b(aclrt\w*[Pp]riority\w*)\b", r.stdout))) or "（无）")
                    break
            except BaseException as e:                                  # noqa: BLE001
                note("B0_失败", f"{tool[0]}: {type(e).__name__}")

    # ── 子进程：B/C/D/E/F/G ──
    print("\n===== 子进程：B/C/D/E/F/G =====")
    r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True,
                       timeout=900, env={**os.environ})
    lines = [l for l in (r.stdout or "").splitlines() if l.startswith("CHILD_JSON")]
    if lines:
        for k, v in json.loads(lines[-1][len("CHILD_JSON "):]).items():
            note(f"CH_{k}", json.dumps(v, ensure_ascii=False)[:1000])
    else:
        note("CH_失败", {"rc": r.returncode, "stdout尾": (r.stdout or "")[-300:],
                         "stderr尾": (r.stderr or "")[-300:]})

    # ── G2) 包内 Python 侧引用 ──
    print("\n===== G2) torch_npu 包内 Python 侧引用 =====")
    pkg = os.path.dirname(torch_npu.__file__)
    hits = []
    for dp, _dd, ff in os.walk(pkg):
        for f in ff:
            if f.endswith(".py"):
                fp = os.path.join(dp, f)
                txt = open(fp, encoding="utf-8", errors="ignore").read()
                for pat in ("getStreamFromExternal", "ExternalStream", "_npu_setStream"):
                    if pat in txt:
                        for i, ln in enumerate(txt.splitlines(), 1):
                            if pat in ln:
                                hits.append(f"{os.path.relpath(fp, pkg)}:{i}: {ln.strip()[:100]}")
    note("G2_包内引用", json.dumps(hits[:15], ensure_ascii=False) if hits else "（无）")

    # ── G3) C++ 侧是否**有**入口（有则结论是"绑定缺失"而非"做不到"）──
    print("\n===== G3) libtorch_npu.so 的 C++ 入口 =====")
    libnpu = None
    for line in open("/proc/self/maps", encoding="utf-8", errors="ignore"):
        m = re.search(r"(/\S*libtorch_npu\.so\S*)", line)
        if m:
            libnpu = m.group(1)
            break
    note("G3_libtorch_npu", libnpu or "（未找到）")
    if libnpu:
        r2 = subprocess.run(["nm", "-D", "--defined-only", "-C", libnpu],
                            capture_output=True, text=True, timeout=240)
        for pat, label in ((r"getStreamFromExternal", "getStreamFromExternal"),
                           (r"setCurrentNPUStream", "setCurrentNPUStream"),
                           (r"getStreamFromPool", "getStreamFromPool")):
            note(f"G3_{label}",
                 sorted(set(re.findall(rf"[\w:]*{pat}[\w:<>,\s\*&()]*", r2.stdout)))[:3] or "（未找到）")

    # ── H) 统一面门禁行为 ──
    print("\n===== H) 统一面门禁 =====")
    sys.path.insert(0, os.environ.get("DC_ROOT", os.getcwd()))
    import runtime                                                   # noqa: E402
    bk = runtime.use("ascend")
    note("H1_supports", {k: bk.supports(k) for k in
                         ("stream_priority", "stream_priority_control", "stream_priority_readback")})
    note("H2_range", runtime.stream_priority_range())
    for p in (0, 7):
        try:
            runtime.create_stream(priority=p)
            note(f"H3_create_stream(priority={p})", "**成功了**（与主张矛盾，需复查）")
        except BaseException as e:                                   # noqa: BLE001
            note(f"H3_create_stream(priority={p})", f"{type(e).__name__}: {str(e)[:90]}",
                 "显式拒绝（非静默给无优先级流）")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"findings": R}, fh, ensure_ascii=False, indent=1)
        print(f"\n结果 -> {a.out}")
    print("\n===== 910C AUDIT DONE =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
