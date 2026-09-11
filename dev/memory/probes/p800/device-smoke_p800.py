#!/usr/bin/env python
"""S2.1 设备层可用性验证（Route A / P800）

覆盖: cuda.is_available / device_count / 设备名 / 基础 tensor 运算 / D2H-H2D
在 flagos-fl-dev-p800 容器内运行:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=1 python /workspace/dev/memory/probes/routeA_s2_1_device.py
"""
import os
import time

import torch

print("=" * 70)
print("S2.1 设备层验证")
print("=" * 70)

print(f"torch version      : {torch.__version__}")
print(f"torch.version.cuda : {torch.version.cuda}")
print(f"cuda.is_available  : {torch.cuda.is_available()}")
print(f"cuda.device_count  : {torch.cuda.device_count()}")
print(f"torch.xpu 存在      : {hasattr(torch, 'xpu')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(未设置)')}")

dev = "cuda:0"  # 容器内 CUDA_VISIBLE_DEVICES=1 已把物理卡1映射为 cuda:0
print(f"\n当前设备           : {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info(0)
print(f"HBM free/total     : {free/1e9:.1f} / {total/1e9:.1f} GB")

# --- 基础运算 ---
print("\n--- 基础运算 ---")
a = torch.randn(4096, 4096, device=dev)
b = torch.randn(4096, 4096, device=dev)
t0 = time.time()
c = a @ b
torch.cuda.synchronize()
t1 = time.time()
print(f"4096x4096 matmul   : ok ({t1-t0:.3f}s, {2*4096**3/(t1-t0)/1e12:.2f} TFLOPS)")

x = torch.randn(1024, 1024, device=dev)
y = (x * 2.0 + 1.0).relu().sum()
print(f"elementwise+relu   : ok (sum={y.item():.1f})")

# --- D2H / H2D ---
print("\n--- D2H / H2D 传输 (1GB) ---")
n = 256 * 1024 * 1024  # 1GiB float32
g = torch.randn(n, device=dev)
t0 = time.time()
h = g.cpu()
torch.cuda.synchronize()
t1 = time.time()
dt = t1 - t0
print(f"D2H 1GiB           : {dt:.3f}s -> {1/dt:.2f} GB/s")

t0 = time.time()
g2 = h.to(dev)
torch.cuda.synchronize()
t1 = time.time()
dt = t1 - t0
print(f"H2D 1GiB           : {dt:.3f}s -> {1/dt:.2f} GB/s")

# --- 多卡可见性 ---
print("\n--- 多卡可见性 (物理卡1 视角) ---")
for i in range(torch.cuda.device_count()):
    f, t = torch.cuda.mem_get_info(i)
    print(f"  cuda:{i} free={f/1e9:.1f}GB total={t/1e9:.1f}GB")

print("\n设备层验证通过" if torch.cuda.is_available() else "设备层验证失败")
