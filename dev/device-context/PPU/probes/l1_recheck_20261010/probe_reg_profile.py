"""抓「登记」的真实调用现场 —— 用 sys.setprofile，**不替换任何函数对象**。

为什么换手段：
  · 上一版 `probe_reg_trace.py` 用 `bk._register_owned_stream = reg_spy`（**替换实例属性**）
    ⇒ 加了这层后登记记录只剩 ExternalStream，而 L1 仍 FAIL ⇒ **插桩改变了触发形态**。
  · 本版用 `sys.setprofile`：**不改动被测对象的任何属性**，只在函数被调用时读 frame，
    扰动仅剩"时间开销 + 读 f_locals"（且只在目标函数被调用时读），语义不变。

观测目标（回答一个问题）：
  是否存在「把 `torch.cuda.Stream`（默认路径的流）登记进 `_owned_streams`」的调用？
  若有 ⇒ 打印**调用栈**定位来源。
  若没有 ⇒ 说明 FAIL 的机制不是"登记了 Stream"，需另找。
"""
import importlib.util
import sys
import traceback

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_prof", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_prof"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]

CLS = type(bk)
REG_CODE = CLS._register_owned_stream.__code__
NOTE_CODE = CLS.note_stream_created.__code__
OWNED_CODE = CLS._owned_handle.__code__

REG = []          # (kind, id, type, 栈)
HANDLE_Q = []     # _owned_handle 查询：被查对象的 id/type


def prof(frame, event, arg):
    if event != "call":
        return prof
    code = frame.f_code
    if code is REG_CODE:
        nat = frame.f_locals.get("native_stream")
        REG.append(("register", id(nat), type(nat).__name__,
                    "".join(traceback.format_stack(frame)[-4:-1])))
    elif code is NOTE_CODE:
        nat = frame.f_locals.get("native_stream")
        REG.append(("note", id(nat), type(nat).__name__, ""))
    return prof


print("[probe] 开始（弱引用版；setprofile 观察，不替换函数）", flush=True)
sys.setprofile(prof)
try:
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
            print(f"[L1] {lab} :: {str(detail)[:140]}", flush=True)
            break
finally:
    sys.setprofile(None)

print(f"\n[观察] register/note 调用共 {len(REG)} 次：", flush=True)
for kind, oid, tname, stack in REG:
    print(f"  - {kind:8s} id={oid} type={tname}", flush=True)
    if stack:
        for line in stack.strip().splitlines():
            print(f"        {line.strip()}", flush=True)

reg = bk._reg("_owned_streams")
print(f"\n[表] _owned_streams 条目 {len(reg)} 条：", flush=True)
for k, v in reg.items():
    held = v[1]()
    print(f"    key={k} handle={v[0]} holder={type(v[1]).__name__} "
          f"held={type(held).__name__ if held is not None else None} "
          f"held_id={id(held) if held is not None else None}", flush=True)
