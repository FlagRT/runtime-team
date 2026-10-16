"""终极判定：命中条目的 holder 到底是「活的 weakref」还是「已死 weakref」。

矛盾点（不可两立）：
  · sys.setprofile 显示 `_register_owned_stream` 只被调用 2 次，**都是 ExternalStream**；
  · 但命中条目里 `holder()` 返回的是一个**存活的 `Stream`**（id 与 native 相同）。

本探针打印 weakref 的 `repr`（CPython 会显示 `to 'X' at 0x…` 或 `dead`），
并另建一条指向 native 的对照 weakref，以判定「同一条」还是「两条」。
"""
import importlib.util
import sys
import weakref

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_final", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_final"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
orig_rel = bk.release_stream
done = []


def spy(native):
    ent = bk._reg("_owned_streams").get(id(native))
    if ent is not None and not done:
        done.append(1)
        w = ent[1]
        owned = bk._reg("_owned_streams")
        ctx = bk._reg("_stream_ctx")
        print(f"[现场] native id={id(native)} type={type(native).__name__} repr={native!r}", flush=True)
        print(f"       条目 key={id(native)}", flush=True)
        print(f"       ent[0](handle)={ent[0]!r}", flush=True)
        print(f"       holder 类型={type(w).__name__}  repr={w!r}", flush=True)
        try:
            got = w()
            print(f"       w() -> {got!r} id={id(got) if got is not None else None} "
                  f"is_native={got is native}", flush=True)
        except BaseException as e:                            # noqa: BLE001
            print(f"       w() 抛异常 {type(e).__name__}: {e}", flush=True)
        w2 = weakref.ref(native)
        print(f"       对照 w2=weakref.ref(native) repr={w2!r} w2() is native={w2() is native} "
              f"w is w2={w is w2}", flush=True)
        print(f"       owned is ctx = {owned is ctx}", flush=True)

        def dump(nm, tbl):
            rows = []
            for k, v in tbl.items():
                try:
                    h = v[1]()
                except BaseException:                          # noqa: BLE001
                    h = "<err>"
                rows.append((k, v[0], type(v[1]).__name__,
                             type(h).__name__ if h is not None else None,
                             id(h) if h is not None else None))
            print(f"       [{nm}] {len(rows)} 条: {rows}", flush=True)

        dump("owned", owned)
        dump("ctx", ctx)
        dump("released", bk._reg("_released_streams"))
    return orig_rel(native)


bk.release_stream = spy

for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        status, detail = mod._run_in_child(sid, "ppu")
    else:
        try:
            status, detail = r["fn"](env)
        except BaseException as e:                            # noqa: BLE001
            status, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
    if sid == "L1":
        lab = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"[L1] {lab} :: {str(detail)[:120]}", flush=True)
        break
print("[done]", flush=True)
