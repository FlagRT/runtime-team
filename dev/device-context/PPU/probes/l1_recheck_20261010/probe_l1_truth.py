"""L1 真相探针（2026-10-10 复核轮）

目的：在**与真实审计完全相同的序列**下，观察 `_owned_streams` 被命中的那一刻到底是什么。

与上一轮 spy 的两个关键修正：
  ① **不跳过侵入项**：真实审计对 `INVASIVE_SIDS` 走 `_run_in_child()`（子进程隔离），
     而上一轮 spy 用 `continue` 直接跳过 ⇒ 分配序列被改变 ⇒ 观测不可信。
  ② **只在命中时记录**：非命中路径零分配，尽量避免 heisenbug。

判读要点：
  · holder_kind = 'ref'      ⇒ 走弱引用（`weakref.ref` 可用）
  · holder_kind = 'function' ⇒ 走 `_id_holder` 的 TypeError 回退（强持有）
  · same=True 且 held_type='ExternalStream' ⇒ 该流**真是本层创建的**（不是 id 复用）
  · same=True 且 held_type='Stream'         ⇒ 一条厂商流被登记过（不该发生）
"""
import importlib.util
import json
import sys

PROTO = "/workspace/runtime-team/dev/device-context/prototype"
sys.path.insert(0, PROTO)
AUDIT = PROTO + "/scripts/duty_response_audit.py"

spec = importlib.util.spec_from_file_location("dc_audit_truth", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_audit_truth"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]
print(f"[env] backend={bk.name} device_type={bk.device_type}", flush=True)

hits = []
orig_rel = bk.release_stream


def spy(native):
    ent = bk._reg("_owned_streams").get(id(native))
    if ent is not None:                       # 只在命中时记录（罕见路径）
        held = ent[1]()
        hits.append({
            "id": id(native),
            "type": type(native).__name__,
            "handle": ent[0],
            "holder_kind": type(ent[1]).__name__,
            "held_type": type(held).__name__ if held is not None else None,
            "held_is_native": held is native,
        })
    return orig_rel(native)


bk.release_stream = spy

for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:                       # 与真实审计一致
        status, detail = mod._run_in_child(sid, "ppu")
    else:
        try:
            status, detail = r["fn"](env)
        except BaseException as e:                     # noqa: BLE001
            status, detail = False, f"{type(e).__name__}: {str(e)[:110]}"
    if sid in ("J2", "J3", "K5", "K6", "K7", "L1", "L2", "L3"):
        lab = {True: "OK", False: "FAIL", None: "SKIP"}[status]
        print(f"[{lab}] {sid} {str(detail)[:110]}", flush=True)
    if sid == "L1":
        print("[hits] " + json.dumps(hits, ensure_ascii=False), flush=True)
        reg = bk._reg("_owned_streams")
        print("[reg] " + json.dumps(
            [[k, type(v[1]).__name__,
              type(v[1]()).__name__ if v[1]() is not None else None]
             for k, v in reg.items()], ensure_ascii=False), flush=True)
        break

# ── 序列之后再问：`_id_holder` 对本栈的流到底走哪条分支 ──
try:
    s = bk.create_stream(-1)
    h = bk._id_holder(s)
    print(f"[holder] 优先级流类型={type(s).__name__} "
          f"_id_holder 返回={type(h).__name__} 弱引用={'是' if type(h).__name__ == 'ref' else '否'}",
          flush=True)
    bk.release_stream(s)
except BaseException as e:                             # noqa: BLE001
    print(f"[holder] 探测异常 {type(e).__name__}: {str(e)[:120]}", flush=True)

try:
    d = bk.create_stream()
    h2 = bk._id_holder(d)
    print(f"[holder] 默认流类型={type(d).__name__} _id_holder 返回={type(h2).__name__}", flush=True)
except BaseException as e:                             # noqa: BLE001
    print(f"[holder] 默认流探测异常 {type(e).__name__}: {str(e)[:120]}", flush=True)
