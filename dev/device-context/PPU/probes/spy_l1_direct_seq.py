"""自核验（终局）：直接复刻 K6 → K7 → L1 的序列，打印对象类型与 id 复用实况。"""
import os
import sys

sys.path.insert(0, "/workspace/runtime-team/dev/device-context/prototype")

import torch  # noqa: E402

import runtime  # noqa: E402

runtime.use("ppu")
b = runtime.current()
print(f"[env] backend={b.name} dev={b.device_count()}", flush=True)
print(f"[env] 两张登记表是否同一 dict：{b._reg('_owned_streams') is b._reg('_released_streams')}", flush=True)


def dump(tag):
    reg = b._reg("_owned_streams")
    items = []
    for k, (h, holder) in reg.items():
        o = holder()
        items.append((k, hex(h), type(o).__name__ if o is not None else "DEAD"))
    print(f"[{tag}] _owned_streams = {items}", flush=True)


print("\n===== 复刻 K6：区间端点建流（不释放）=====", flush=True)
for v in (-3, 0):
    st = runtime.create_stream(priority=v)
    nat = st._native_obj
    print(f"  K6-like v={v}: type={type(nat).__name__} id={id(nat)} repr={nat!r}", flush=True)
    got = runtime.stream_priority_readback(st)
    print(f"              回读={got!r}", flush=True)
dump("K6 之后")

print("\n===== 复刻 K7：默认路径建流 + 回读 =====", flush=True)
st7 = runtime.create_stream()
print(f"  K7-like: type={type(st7._native_obj).__name__} id={id(st7._native_obj)} "
      f"in_owned={id(st7._native_obj) in b._reg('_owned_streams')}", flush=True)
dump("K7 之后")

print("\n===== 复刻 L1：默认路径建流 + release =====", flush=True)
st1 = runtime.create_stream()
nat1 = st1._native_obj
print(f"  L1-like: type={type(nat1).__name__} id={id(nat1)} repr={nat1!r}", flush=True)
print(f"          in_owned={id(nat1) in b._reg('_owned_streams')} "
      f"owns_stream={b.owns_stream(nat1)}", flush=True)
dump("L1 调用前")
rel = runtime.release_stream(st1)
print(f"          release_stream -> {rel!r}", flush=True)
dump("L1 之后")

print("\n===== 对照：torch.cuda.ExternalStream 的类型名 =====", flush=True)
h = ctypes_handle = None
try:
    import ctypes
    lib = b._driver_handle()
    hp = ctypes.c_void_p(0)
    lib.cuStreamCreateWithPriority.restype = ctypes.c_int
    lib.cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    rc = int(lib.cuStreamCreateWithPriority(ctypes.byref(hp), ctypes.c_uint(1), ctypes.c_int(0)))
    h = hp.value
    es = torch.cuda.ExternalStream(h, device=torch.cuda.current_device())
    print(f"  rc={rc} ExternalStream 实例 type={type(es).__name__} repr={es!r}", flush=True)
    lib.cuStreamDestroy_v2.restype = ctypes.c_int
    lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    print(f"  清理 destroy rc={int(lib.cuStreamDestroy_v2(ctypes.c_void_p(int(h))))}", flush=True)
except BaseException as e:                                        # noqa: BLE001
    print(f"  对照失败：{type(e).__name__}: {e}", flush=True)
