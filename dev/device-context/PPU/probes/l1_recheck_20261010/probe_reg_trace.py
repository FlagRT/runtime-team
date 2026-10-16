"""抓「谁把默认流登记进 _owned_streams」—— 极轻量插桩（只 append 类型名，不抓栈）

背景（2026-10-10 复核）：弱引用版探针复现 L1 FAIL 时，hits 显示
  {type: "Stream", holder_kind: "ReferenceType", held_type: "Stream", held_is_native: true}
⇒ 命中的条目里存的对象**就是 L1 的默认流本身**，且**弱引用仍有效**（对象存活）。
但按代码，`_register_owned_stream` 的唯一调用点是 `_create_stream_raw(priority=<int>)`，
那里的 native 一律是 `torch.cuda.ExternalStream` ⇒ **类型对不上**。
本探针只做一件事：记录每次登记的**对象类型**（+ 调用者函数名），看是否真有 `Stream` 被登记。
"""
import importlib.util
import sys

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_trace_reg", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_trace_reg"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]

REG_LOG = []
NOTE_LOG = []
orig_reg = bk._register_owned_stream
orig_note = bk.note_stream_created


def reg_spy(native, handle):
    caller = sys._getframe(1).f_code.co_name            # 极轻量：只要函数名
    REG_LOG.append((caller, type(native).__name__, id(native)))
    return orig_reg(native, handle)


def note_spy(native):
    NOTE_LOG.append((type(native).__name__, id(native)))
    return orig_note(native)


bk._register_owned_stream = reg_spy
bk.note_stream_created = note_spy

l1_native_id = None
for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        status, detail = mod._run_in_child(sid, "ppu")
    else:
        try:
            status, detail = r["fn"](env)
        except BaseException as e:                      # noqa: BLE001
            status, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
    if sid == "L1":
        lab = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"[L1] {lab} :: {str(detail)[:150]}", flush=True)
        break

print("[REG ] register 调用（caller, 类型, id）:", flush=True)
for e in REG_LOG:
    print("       ", e, flush=True)
print("[NOTE] note_stream_created 调用数 =", len(NOTE_LOG),
      "类型集合 =", sorted({t for t, _ in NOTE_LOG}), flush=True)
reg = bk._reg("_owned_streams")
print("[表 ] _owned_streams 现存条目：", flush=True)
for k, v in reg.items():
    held = v[1]()
    print(f"       key={k} handle={v[0]} holder={type(v[1]).__name__} "
          f"held_type={type(held).__name__ if held is not None else None}", flush=True)
print("[表 ] 两表是否同一 dict：",
      bk._reg("_owned_streams") is bk._reg("_stream_ctx"), flush=True)
print("[表 ] _stream_ctx 条目数 =", len(bk._reg("_stream_ctx")), flush=True)
print("[表 ] _released_streams 条目数 =", len(bk._reg("_released_streams")), flush=True)
