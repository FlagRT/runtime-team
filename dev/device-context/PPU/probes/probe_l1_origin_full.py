"""L1 根因定位（终）：登记侧来源标记 + 不跳过任何项 + 在 L1 现场 dump 登记表。

与上一版差别：① 不再跳过 INVASIVE_SIDS（那会改变分配序列、把缺陷掩盖掉）；
② 对 `_register_owned_stream` 打来源标记；③ L1 失败时立刻 dump 登记表全貌。
"""
import importlib.util
import inspect
import sys
import weakref

AUDIT = "/workspace/runtime-team/dev/device-context/prototype/scripts/duty_response_audit.py"
spec = importlib.util.spec_from_file_location("dc_audit4", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_audit4"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]

EVENTS = []


def wr(o):
    try:
        weakref.ref(o)
        return "wr-OK"
    except TypeError:
        return "NO-wr"


orig_reg = bk._register_owned_stream


def reg_spy(native_stream, handle):
    caller = "?"
    for fr in inspect.stack()[1:8]:
        if fr.function not in ("reg_spy", "_create_stream_raw"):
            caller = f"{fr.function}:{fr.lineno}"
            break
    EVENTS.append(("REGISTER", f"{type(native_stream).__name__}@{id(native_stream)} "
                               f"handle={handle} {wr(native_stream)} caller={caller}"))
    return orig_reg(native_stream, handle)


bk._register_owned_stream = reg_spy

orig_rel = bk.release_stream


def rel_spy(native_stream):
    reg = bk._reg("_owned_streams")
    ent = reg.get(id(native_stream))
    detail = f"in_owned={ent is not None}"
    if ent is not None:
        held = ent[1]()
        detail += (f" handle={ent[0]} held={type(held).__name__ if held is not None else 'DEAD'}"
                   f" same={held is native_stream}")
    r = orig_rel(native_stream)
    EVENTS.append(("RELEASE", f"{type(native_stream).__name__}@{id(native_stream)} {detail} -> {r!r}"))
    # 发现「非本层拥有的流却返回 True」→ 立刻 dump
    if r is True and "in_owned=False" in detail:
        EVENTS.append(("!!BUG", "非拥有的流返回 True"))
    if r is True:
        EVENTS.append(("DUMP@releaseTrue", " ; ".join(
            f"{k}->({h},{type(v[1]()).__name__ if v[1]() is not None else 'DEAD'}@{id(v[1]()) if v[1]() is not None else '-'})"
            for k, (h, v) in bk._reg("_owned_streams").items())))
    return r


bk.release_stream = rel_spy

for r in mod.RESULTS:
    sid = r["sid"]
    EVENTS.append(("ITEM", f"enter {sid}"))
    try:
        status, detail = r["fn"](env)
    except BaseException as e:                                      # noqa: BLE001
        status, detail = False, f"{type(e).__name__}: {e}"
    label = {True: "OK", False: "FAIL", None: "SKIP"}[status]
    EVENTS.append(("ITEM", f"{sid} => {label} : {str(detail)[:90]}"))

print("=== 事件流（只看 REGISTER / RELEASE / L1 / L2 / 异常）===", flush=True)
for kind, msg in EVENTS:
    if kind in ("REGISTER",) or kind.startswith("DUMP") or kind == "!!BUG" or "L1" in msg or "L2" in msg:
        print(f"[{kind}] {msg}", flush=True)
