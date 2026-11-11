#!/usr/bin/env python3
"""PPU 优先级流「释放后再建」的对象身份实验（T6 失败根因定位）。

问题现象（T6 D2 探针，2026-10-14）：run_group 每组自建自释放，但后续组的
`runtime.create_stream(priority=…)` 返回的流在**使用点**被判为
「该流已被 release_stream() 释放」⇒ T6 主判据全 FAIL。

要判定的是：新建的流是否**就是**那条已释放的对象（同一 Python 对象 / 同一句柄）。

用法：docker exec <容器> bash -lc "cd <原型根> && CUDA_VISIBLE_DEVICES=1 python3 -u this.py"
"""
import os
import sys

_ROOT = os.environ.get("DC_ROOT", "/workspace/runtime-team/dev/device-context/prototype")
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

runtime.use(os.environ.get("DC_BACKEND", "ppu"))
runtime.set_device(0)

PRIO = -3
recs = []

print("=== 第 1 轮：建 → 释放 → 再建 ===")
s1 = runtime.create_stream(priority=PRIO)
n1 = s1._native_obj
h1 = n1.cuda_stream
print(f"  s1: native id={id(n1)} type={type(n1).__name__} handle={h1:#x}")
runtime.release_stream(s1)
print("  已 release_stream(s1)")

s2 = runtime.create_stream(priority=PRIO)
n2 = s2._native_obj
h2 = n2.cuda_stream
print(f"  s2: native id={id(n2)} type={type(n2).__name__} handle={h2:#x}")
print(f"  ▶ 同一对象 n1 is n2 = {n1 is n2}")
print(f"  ▶ 同一 id       = {id(n1) == id(n2)}")
print(f"  ▶ 同一句柄      = {h1 == h2}（{h1:#x} vs {h2:#x}）")

try:
    s2.synchronize()
    print("  ▶ 使用 s2：✅ 可用")
    use_ok = True
except BaseException as e:
    print(f"  ▶ 使用 s2：❌ {type(e).__name__}: {str(e)[:110]}")
    use_ok = False

# 第三轮：不释放 s2，直接建第三条，看句柄是否继续复用
print("=== 第 2 轮：不释放 s2，再建第三条 ===")
s3 = runtime.create_stream(priority=PRIO)
n3 = s3._native_obj
print(f"  s3: native id={id(n3)} handle={n3.cuda_stream:#x}  （s2 仍在使用中，应得新句柄）")
try:
    s3.synchronize()
    print("  ▶ 使用 s3：✅ 可用")
except BaseException as e:
    print(f"  ▶ 使用 s3：❌ {type(e).__name__}: {str(e)[:110]}")

# 收尾释放（只释放本层拥有的）
for tag, s in (("s2", s2), ("s3", s3)):
    try:
        r = runtime.release_stream(s)
        print(f"  release_stream({tag}) -> {r}")
    except BaseException as e:
        print(f"  release_stream({tag}) 抛 {type(e).__name__}")

print("VERDICT:", "SAME_OBJECT_REUSE" if (n1 is n2) else ("SAME_HANDLE_DIFF_OBJECT" if h1 == h2 else "NEW_OBJECT_NEW_HANDLE"))
if not use_ok:
    print("（现象复现：新建的流在使用点被判为已释放）")
