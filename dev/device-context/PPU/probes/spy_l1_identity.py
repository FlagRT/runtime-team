"""自核验（终）：在 L1 那一次调用前，把「在册对象」与「本次对象」都打出来。

目的：判定 `_owned_streams` 里被命中的条目到底是不是**同一个对象**
（若 holder 解析出的对象与 native 类型不同却仍 `is` 成立/判为一致，说明守卫失效）。
"""
import importlib.util
import os
import sys

AUDIT = "/workspace/runtime-team/dev/device-context/prototype/scripts/duty_response_audit.py"
spec = importlib.util.spec_from_file_location("dc_audit2", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_audit2"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
orig = bk.release_stream
depth = {"n": 0}


def spy(native):
    depth["n"] += 1
    reg = bk._reg("_owned_streams")
    ent = reg.get(id(native))
    print(f"\n[SPY#{depth['n']}] id(native)={id(native)} type={type(native).__name__} "
          f"repr={native!r}", flush=True)
    print(f"          in_owned={ent is not None} in_released={id(native) in bk._reg('_released_streams')}", flush=True)
    if ent is not None:
        h, holder = ent
        held = holder()
        print(f"          在册句柄={h} 持有器解析出={type(held).__name__ if held is not None else None} "
              f"same_object={held is native} id(held)={id(held) if held is not None else None}", flush=True)
    r = orig(native)
    print(f"          -> ret={r!r}", flush=True)
    return r


bk.release_stream = spy

for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        continue
    try:
        status, detail = r["fn"](env)
    except BaseException as e:                                  # noqa: BLE001
        status, detail = False, f"{type(e).__name__}: {e}"
    if sid in ("K6", "K7", "L1"):
        label = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"  [{label}] {sid} {str(detail)[:120]}", flush=True)
    if sid == "L1":
        break
