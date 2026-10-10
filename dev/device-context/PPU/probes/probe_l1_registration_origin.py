"""L1 根因定位（登记侧来源标记探针）。

做法：不修改原型代码，在进程内包装三个点，按审计真实顺序跑到 L1：
  ① `_create_stream_raw`  —— 每次建流记 id / 类型 / 是否可弱引用 / 句柄
  ② `_register_owned_stream` —— 每次**登记**记 id / 类型 / 句柄 / **调用点**
  ③ `release_stream` —— 每次释放记命中与否、身份复核结果、返回值
目的：看清「L1 那一次命中的在册条目」到底是**哪一次登记**写进去的。
"""
import importlib.util
import inspect
import os
import sys
import weakref

AUDIT = "/workspace/runtime-team/dev/device-context/prototype/scripts/duty_response_audit.py"
spec = importlib.util.spec_from_file_location("dc_audit3", AUDIT)
mod = importlib.util.module_from_spec(spec)
sys.modules["dc_audit3"] = mod
spec.loader.exec_module(mod)

env = mod.build_env("ppu")
bk = env["backend"]

LOG = []


def wr_support(o):
    try:
        weakref.ref(o)
        return "weakref-OK"
    except TypeError:
        return "NO-weakref(强引用回退)"


# ① 建流打点
orig_raw = bk._create_stream_raw


def raw_spy(priority=None):
    native = orig_raw(priority)
    ref = "None" if native is None else f"{type(native).__name__}@{id(native)}"
    LOG.append(f"[BUILD] priority={priority!r} -> {ref} {wr_support(native) if native is not None else ''}")
    return native


bk._create_stream_raw = raw_spy

# ② 登记打点（带调用点）
orig_reg = bk._register_owned_stream


def reg_spy(native_stream, handle):
    caller = "?"
    try:
        st = inspect.stack()
        for fr in st[1:6]:
            if fr.function not in ("reg_spy", "_create_stream_raw", "raw_spy"):
                caller = f"{fr.function}():{fr.lineno}"
                break
    except Exception:
        pass
    LOG.append(f"[REGISTER] {type(native_stream).__name__}@{id(native_stream)} handle={handle} "
               f"{wr_support(native_stream)} caller={caller}")
    return orig_reg(native_stream, handle)


bk._register_owned_stream = reg_spy

# ③ 释放打点
orig_rel = bk.release_stream


def rel_spy(native_stream):
    reg = bk._reg("_owned_streams")
    ent = reg.get(id(native_stream))
    info = f"in_owned={ent is not None}"
    if ent is not None:
        held = ent[1]()
        info += (f" handle={ent[0]} holder_type={type(held).__name__ if held is not None else 'DEAD'}"
                 f" same={held is native_stream}")
    r = orig_rel(native_stream)
    LOG.append(f"[RELEASE] {type(native_stream).__name__}@{id(native_stream)} {info} -> {r!r}")
    return r


bk.release_stream = rel_spy

print("=== 按审计真实顺序跑到 L1 ===", flush=True)
for r in mod.RESULTS:
    sid = r["sid"]
    if sid in mod.INVASIVE_SIDS:
        # ⚠️ **本行有缺陷（2026-10-10 复核发现）**：真实审计对侵入项走
        #    `mod._run_in_child()`（**子进程隔离**），这里直接 `continue` 会**改变分配序列**
        #    ⇒ 观测不可信。**请改用 `probe_l1_truth.py`**。
        continue
    LOG.append(f"--- enter {sid} ---")
    try:
        status, detail = r["fn"](env)
    except BaseException as e:                                      # noqa: BLE001
        status, detail = False, f"{type(e).__name__}: {e}"
    label = {True: "OK", False: "FAIL", None: "SKIP"}[status]
    LOG.append(f"--- {sid} => {label} ---")
    if sid == "L1":
        break

print("\n".join(LOG), flush=True)
print("\n=== 结束在册 ===", flush=True)
for k, (h, holder) in list(bk._reg("_owned_streams").items()):
    o = holder()
    print(f"  id={k} handle={h} holder={type(o).__name__ if o is not None else 'DEAD'}@{id(o) if o is not None else '-'}", flush=True)
