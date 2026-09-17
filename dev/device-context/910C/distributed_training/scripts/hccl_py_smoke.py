#!/usr/bin/env python3
"""
torch_fl 环境下的 HCCL 直连测试（隔离 flagcx，验证 torch_fl 是否破坏 HCCL P2P）
用法: torchrun --nproc_per_node=2 hccl_py_smoke.py
"""
import os
import sys
import ctypes

def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 2))
    master = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29599"))

    # 1. 先初始化 torch_fl（模拟 flagcx 的真实环境）
    import torch
    import torch_fl
    from torch_fl import flagos
    print(f"[py] rank{rank}: torch_fl init, devices={flagos.device_count()}")
    x = torch.randn(4, 4, device="flagos")
    print(f"[py] rank{rank}: flagos tensor ok {x.device}")

    # 2. ctypes 加载 HCCL
    libacl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so")
    libacl.aclrtSetDevice.argtypes = [ctypes.c_int32]
    libacl.aclrtSetDevice.restype = ctypes.c_int
    sret = libacl.aclrtSetDevice(rank)
    print(f"[py] rank{rank}: aclrtSetDevice({rank}) ret={sret}")
    libhccl = ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/latest/lib64/libhccl.so")
    HcclRootInfo = ctypes.c_ubyte * 4108
    HcclComm = ctypes.c_void_p
    libhccl.HcclGetRootInfo.argtypes = [ctypes.POINTER(HcclRootInfo)]
    libhccl.HcclGetRootInfo.restype = ctypes.c_int
    libhccl.HcclCommInitRootInfo.argtypes = [ctypes.c_uint32, ctypes.POINTER(HcclRootInfo), ctypes.c_uint32, ctypes.POINTER(HcclComm)]
    libhccl.HcclCommInitRootInfo.restype = ctypes.c_int
    libhccl.HcclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, HcclComm, ctypes.c_void_p]
    libhccl.HcclAllGather.restype = ctypes.c_int

    # 3. TCPStore 交换 rootinfo
    import torch.distributed as dist
    store = dist.TCPStore(master, port, world, rank == 0, timeout=dist.default_pg_timeout)
    root = HcclRootInfo()
    if rank == 0:
        ret = libhccl.HcclGetRootInfo(ctypes.byref(root))
        print(f"[py] rank{rank}: HcclGetRootInfo ret={ret}")
        store.set("root", bytes(root))
    else:
        data = store.get("root")
        for i, b in enumerate(data):
            root[i] = b
    comm = HcclComm()
    ret = libhccl.HcclCommInitRootInfo(world, ctypes.byref(root), rank, ctypes.byref(comm))
    print(f"[py] rank{rank}: HcclCommInitRootInfo ret={ret} comm={comm.value}")

    # 4. allgather（device 内存，直接 ACL 分配）
    libacl.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    libacl.aclrtMalloc.restype = ctypes.c_int
    libacl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    libacl.aclrtCreateStream.restype = ctypes.c_int
    libacl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    libacl.aclrtSynchronizeStream.restype = ctypes.c_int
    libacl.aclrtFree.argtypes = [ctypes.c_void_p]
    libacl.aclrtFree.restype = ctypes.c_int

    d_send = ctypes.c_void_p()
    d_recv = ctypes.c_void_p()
    ret = libacl.aclrtMalloc(ctypes.byref(d_send), 8, 0)
    ret |= libacl.aclrtMalloc(ctypes.byref(d_recv), 16, 0)
    print(f"[py] rank{rank}: aclrtMalloc ret={ret} send={hex(d_send.value)} recv={hex(d_recv.value)}")
    stream = ctypes.c_void_p()
    libacl.aclrtCreateStream(ctypes.byref(stream))
    HCCL_DATA_TYPE_INT64 = 5
    ret = libhccl.HcclAllGather(d_send, d_recv, 1, HCCL_DATA_TYPE_INT64, comm, stream)
    print(f"[py] rank{rank}: HcclAllGather(aclrtMalloc buf) ret={ret}")

    # 5. 关键测试：用 torch_fl flagos tensor 的 data_ptr 做 buffer
    t_in = torch.zeros(1, dtype=torch.int64, device="flagos")
    t_out = torch.zeros(2, dtype=torch.int64, device="flagos")
    in_ptr = t_in.data_ptr()
    out_ptr = t_out.data_ptr()
    print(f"[py] rank{rank}: flagos tensor ptr in={hex(in_ptr)} out={hex(out_ptr)}")
    ret2 = libhccl.HcclAllGather(
        ctypes.c_void_p(in_ptr), ctypes.c_void_p(out_ptr), 1, HCCL_DATA_TYPE_INT64,
        comm, stream,
    )
    print(f"[py] rank{rank}: HcclAllGather(flagos tensor) ret={ret2}")
    libacl.aclrtSynchronizeStream(stream)
    libacl.aclrtFree(d_send)
    libacl.aclrtFree(d_recv)
    print(f"[py] rank{rank}: DONE")

if __name__ == "__main__":
    main()
