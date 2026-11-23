"""宿主机直跑版：判别 docker 层还是内核驱动层。结果写宿主机路径。"""
import os
import time

import torch

t0 = time.time()
x = torch.randn(512, 512, device="cuda")
y = float((x @ x).sum())
msg = f"宿主机计算OK={y != 0} 值={y:.1f} 耗时={time.time() - t0:.1f}s 设备数={torch.cuda.device_count()}"
with open("/home/xianyiyuan/fgw/ctools/torch_filetest.txt", "w") as f:
    f.write(msg + "\n")
    f.flush()
    os.fsync(f.fileno())
os._exit(0)
