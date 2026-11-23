"""判别：计算是否成功 vs 仅清理崩溃。结果写文件后 os._exit(0) 跳过析构。"""
import os
import time

import torch

t0 = time.time()
x = torch.randn(512, 512, device="cuda")
y = float((x @ x).sum())
msg = f"计算OK={y != 0} 值={y:.1f} 耗时={time.time() - t0:.1f}s 设备={torch.cuda.get_device_name(0)}"
with open("/work/ctools/torch_filetest.txt", "w") as f:
    f.write(msg + "\n")
    f.flush()
    os.fsync(f.fileno())
os._exit(0)  # 跳过所有析构（xblas destroy 崩溃点）
