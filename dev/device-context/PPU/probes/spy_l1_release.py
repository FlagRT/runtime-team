"""自核验：给 duty_response_audit 的 release_stream 打点，定位 L1 为何返回 True。

做法：按审计的真实顺序在**同一进程**里跑各项（跳过需要子进程隔离的侵入项），
spy 包装 `backend.release_stream`，记录每次调用的 id / 是否在册 / 返回值。
"""
import importlib.util
import os
import sys

AUDIT = "/workspace/runtime-team/dev/device-context/prototype/scripts/duty_response_audit.py"
spec = importlib.util.spec_from_file_location("dc_audit", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_audit"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
orig = bk.release_stream


def spy(native):
    pre_owned = id(native) in bk._reg("_owned_streams")
    ent = bk._reg("_owned_streams").get(id(native))
    holder_ok = None
    if ent is not None:
        holder_ok = ent[1]() is native
    r = orig(native)
    if r is True or (ent is not None):
        print(f"[SPY] id={id(native)} type={type(native).__name__} "
              f"in_owned_reg={pre_owned} holder_is_same={holder_ok} "
              f"entry={ent if ent is None else (ent[0], 'holder')} -> ret={r!r}", flush=True)
    return r


bk.release_stream = spy

print("=== 按序跑非侵入项（与审计同序）===", flush=True)
for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        continue
    try:
        status, detail = r["fn"](env)
    except BaseException as e:                                  # noqa: BLE001
        status, detail = False, f"{type(e).__name__}: {e}"
    label = {True: "OK", False: "FAIL", None: "SKIP"}[status]
    if sid.startswith(("K", "L")):
        print(f"  [{label}] {sid} {str(detail)[:150]}", flush=True)

print("=== 结束时在册情况 ===", flush=True)
print("  _owned_streams keys:", list(bk._reg("_owned_streams").keys()), flush=True)
print("  _released_streams keys:", list(bk._reg("_released_streams").keys()), flush=True)
