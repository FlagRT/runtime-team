"""命中那一刻的完整现场（弱引用版复现）

已确证（sys.setprofile，不替换函数）：
  · `_register_owned_stream` 只被调用 2 次，**都是 ExternalStream**；
  · `note_stream_created` 被调用 19 次，**多数是 Stream（默认流）**，且**多个 Stream 复用同一地址**；
  · **第一次 register 的对象地址 == 那些 Stream 的地址**（地址复用确凿）。

本探针回答最后一个问题：命中的条目**到底来自哪张表**、两表是否被指向同一 dict、
以及 `_stream_ctx` / `_released_streams` 里是否也有同一 native。
"""
import importlib.util
import sys

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_spot", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_spot"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
print(f"[env] backend={bk.name} cls={type(bk).__name__}", flush=True)

orig_rel = bk.release_stream
seen = []


def snap(tag, native):
    owned = bk._reg("_owned_streams")
    ctx = bk._reg("_stream_ctx")
    rel = bk._reg("_released_streams")
    print(f"--- {tag} ---", flush=True)
    print(f"  native id={id(native)} type={type(native).__name__}", flush=True)
    print(f"  两表同一 dict? owned is ctx = {owned is ctx}", flush=True)
    for nm, tbl in (("owned", owned), ("ctx", ctx), ("released", rel)):
        rows = []
        for k, v in tbl.items():
            try:
                held = v[1]()
            except BaseException as e:                       # noqa: BLE001
                held = f"<err {e}>"
            rows.append((k, v[0], type(v[1]).__name__,
                         type(held).__name__ if held is not None else None,
                         id(held) if held is not None else None))
        print(f"  [{nm}] {len(rows)} 条: {rows}", flush=True)


def spy(native):
    ent = bk._reg("_owned_streams").get(id(native))
    if ent is not None:
        held = None
        try:
            held = ent[1]()
        except BaseException:                                # noqa: BLE001
            pass
        seen.append((id(native), type(native).__name__, ent[0],
                     type(ent[1]).__name__,
                     type(held).__name__ if held is not None else None,
                     held is native))
        snap(f"命中现场 #{len(seen)}", native)
    return orig_rel(native)


bk.release_stream = spy

for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        status, detail = mod._run_in_child(sid, "ppu")
    else:
        try:
            status, detail = r["fn"](env)
        except BaseException as e:                           # noqa: BLE001
            status, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
    if sid in ("K6", "K7", "L1"):
        lab = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"[{lab}] {sid} {str(detail)[:120]}", flush=True)
    if sid == "L1":
        break

print("\n[汇总] 命中记录（id, type, handle, holder_kind, held_type, held_is_native）:", flush=True)
for s in seen:
    print("   ", s, flush=True)
