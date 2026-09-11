#!/usr/bin/env python3
"""纯 ctypes HCCL 直测:2 进程 HcclGetRootInfo + HcclCommInitRootInfo + AllReduce。

用法: ASCEND_RT_VISIBLE_DEVICES=0,1 python hccl_direct.py 0 2  (两进程同时)
"""
import ctypes
import os
import sys
import time

HCCL = "/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64/libhccl.so"
ACL = "/usr/local/Ascend/cann-9.0.0/lib64/libascendcl.so"

rank = int(sys.argv[1])
nranks = int(sys.argv[2])

acl = ctypes.CDLL(ACL)
h = ctypes.CDLL(HCCL)
h.HcclGetRootInfo.restype = ctypes.c_int
h.HcclCommInitRootInfo.restype = ctypes.c_int
h.HcclAllReduce.restype = ctypes.c_int
h.HcclCommDestroy.restype = ctypes.c_int

# 设备上下文
rc = acl.aclInit(ctypes.c_void_p(0))
rc = acl.aclrtSetDevice(ctypes.c_int32(rank))  # 逻辑设备 = rank(ASCEND_RT_VISIBLE_DEVICES=0,1)
print(f"[r{rank}] aclrtSetDevice rc={rc}")

class RootInfo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 512)]

root = RootInfo()
if rank == 0:
    rc = h.HcclGetRootInfo(ctypes.byref(root))
    print(f"[r0] HcclGetRootInfo rc={rc}")
    # 写到文件给 rank1(简单粗暴)
    with open("/tmp/hccl_root.bin", "wb") as f:
        f.write(bytes(root.data))
else:
    for _ in range(100):
        if os.path.exists("/tmp/hccl_root.bin"):
            break
        time.sleep(0.1)
    with open("/tmp/hccl_root.bin", "rb") as f:
        root.data = (ctypes.c_ubyte * 512)(*f.read())

comm = ctypes.c_void_p(0)
rc = h.HcclCommInitRootInfo(ctypes.c_uint32(nranks), ctypes.byref(root),
                            ctypes.c_uint32(rank), ctypes.byref(comm))
print(f"[r{rank}] HcclCommInitRootInfo rc={rc}")
if rc != 0:
    sys.exit(1)

# 每 rank 一个 4 元素 int32 缓冲: allreduce 加和
N = 4
buf = (ctypes.c_int32 * N)(*([rank + 1] * N))
count = ctypes.c_uint64(N)
dtype = ctypes.c_int32  # HCCL_INT32 = 0
red = ctypes.c_int32(0)  # HCCL_REDUCE_SUM = 0
aclrtstream = ctypes.c_void_p(0)
acl.aclrtCreateStream(ctypes.byref(aclrtstream))

rc = h.HcclAllReduce(ctypes.cast(buf, ctypes.c_void_p), ctypes.cast(buf, ctypes.c_void_p),
                     count, ctypes.c_int32(0), ctypes.c_int32(0), comm, aclrtstream)
acl.aclrtSynchronizeStream(aclrtstream)
print(f"[r{rank}] HcclAllReduce rc={rc} -> {list(buf)} (期望 {[nranks*(nranks+1)//2]*N})")
rc = h.HcclCommDestroy(comm)
print(f"[r{rank}] HcclCommDestroy rc={rc}")
