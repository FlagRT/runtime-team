#!/usr/bin/env python
"""P800 环境验证探针 (memory 子方向 / 重建版)

在 flagos-fl-dev-p800 容器内运行:
    source /root/miniconda/bin/activate python310_torch29_cuda
    python /workspace/dev/memory/probes/p800_env_check.py

验证: torch/flag_gems/vllm/vllm-plugin-fl 版本、XPU 可见性、vendor 识别、HBM 容量
对应 910c 流程的"环境验证"环节。
"""
import os
import sys

# ---- 必须在 import flag_gems/vllm 之前设置 ----
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")

import torch  # noqa: E402
import flag_gems  # noqa: E402

print("=" * 60)
print("P800 环境验证探针")
print("=" * 60)

# 1. torch / 设备
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
print(f"cuda device_count: {torch.cuda.device_count()}")
print(f"torch.xpu attr   : {hasattr(torch, 'xpu')}")

# 2. flag_gems vendor 识别
print(f"flag_gems        : {flag_gems.__version__}")
try:
    from flag_gems.runtime.backend.device_finder import DeviceDetector
except ImportError:
    from flag_gems.runtime.backend.device import DeviceDetector
d = DeviceDetector()
print(f"detected vendor  : {d.vendor_name} | device: {d.name}")

# 3. 每卡 HBM
print("per-card HBM (free/total GiB):")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  card {i}: {free/1e9:.1f} / {total/1e9:.1f}")

# 4. 一个小 matmul 冒烟
a = torch.randn(1024, 1024, device="cuda:0")
b = (a @ a).sum().item()
print(f"matmul smoke     : ok (sum={b:.2f})")

# 5. vllm / 插件
import vllm  # noqa: E402
print(f"vllm             : {vllm.__version__}")
try:
    import importlib.metadata as md
    print(f"vllm-plugin-fl   : {md.version('vllm-plugin-fl')}")
except Exception as e:
    print(f"vllm-plugin-fl   : 未识别 ({e})")

print("=" * 60)
print("环境验证通过" if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else "环境验证失败")
