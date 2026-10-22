"""同进程内对齐「登记调用」与「命中现场」——一次定案。

已确证（各自独立运行）：
  · 命中条目的 holder 是一条**未失效**的 weakref，目标就是 native（`w()` 返回 native）；
  · `sys.setprofile` 记到 register 只调用 2 次、都是 ExternalStream（但那次未 dump 命中现场）。

本探针把两者放进**同一进程**：
  · setprofile 记录每次 register/note 的 (kind, id, type, 调用栈)
  · release_stream 命中时 dump 现场，并**标出该 key 在 register 记录里的登记类型**
这样「key 的登记类型」与「命中时的对象类型」可以直接对齐，无需跨运行推断。
"""
import importlib.util
import sys
import traceback

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_h", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_h"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
CLS = type(bk)
REG_CODE = CLS._register_owned_stream.__code__
NOTE_CODE = CLS.note_stream_created.__code__

REG = []
NOTE = []


def prof(frame, event, arg):
    if event != "call":
        return prof
    code = frame.f_code
    if code is REG_CODE:
        nat = frame.f_locals.get("native_stream")
        REG.append((id(nat), type(nat).__name__,
                    " | ".join(l.strip() for l in traceback.format_stack(frame)[-4:-1])))
    elif code is NOTE_CODE:
        nat = frame.f_locals.get("native_stream")
        NOTE.append((id(nat), type(nat).__name__))
    return prof


orig_rel = bk.release_stream
done = []


def spy(native):
    ent = bk._reg("_owned_streams").get(id(native))
    if ent is not None and not done:
        done.append(1)
        w = ent[1]
        print(f"[命中] key={id(native)} native 类型={type(native).__name__} repr={native!r}", flush=True)
        print(f"       holder repr={w!r}  w() 类型={type(w()).__name__ if w() is not None else None}", flush=True)
        print(f"       该 key 在 register 记录里的登记类型："
              f"{[t for (i, t, _) in REG if i == id(native)] or '（无 ⇒ 该条目不是 register 写的）'}", flush=True)
        print(f"       register 全部记录（共 {len(REG)} 次）：{[(i, t) for (i, t, _) in REG]}", flush=True)
        print(f"       note 记录（共 {len(NOTE)} 次）类型分布："
              f"{ {t: sum(1 for _, x in NOTE if x == t) for _, t in NOTE} }", flush=True)
        print(f"       owned 键={list(bk._reg('_owned_streams').keys())}", flush=True)
        print(f"       ctx 键  ={list(bk._reg('_stream_ctx').keys())}", flush=True)
    return orig_rel(native)


sys.setprofile(prof)
bk.release_stream = spy
try:
    for r in mod.RESULTS:
        sid = r["sid"]
        if sid in mod.INVASIVE_SIDS:
            status, detail = mod._run_in_child(sid, "ppu")
        else:
            try:
                status, detail = r["fn"](env)
            except BaseException as e:                    # noqa: BLE001
                status, detail = False, f"{type(e).__name__}: {str(e)[:90]}"
        if sid == "L1":
            lab = {True: "OK", False: "FAIL", None: "SKIP"}[status]
            print(f"[L1] {lab} :: {str(detail)[:110]}", flush=True)
            break
finally:
    sys.setprofile(None)

print(f"\n[register 调用栈全量] 共 {len(REG)} 次", flush=True)
for i, t, st in REG:
    print(f"  id={i} type={t}\n      {st}", flush=True)
