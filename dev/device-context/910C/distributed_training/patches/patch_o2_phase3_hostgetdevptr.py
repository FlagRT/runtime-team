#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""O2 阶段三 #6: 昇腾 hostGetDevicePointer / hostRegister 接线（DAG 断点①② + launch_kernel 保护）

改动（cann_adaptor.cc + launch_kernel.cc）：
1. cannAdaptorHostGetDevicePointer 真实现（aclrtHostGetDevicePointer，判 NULL）
2. cannAdaptorDeviceMalloc(flagcxMemHost) 在 aclrtMallocHost 后追加 aclrtHostRegisterV2
   （实测：不 register 则 GetDevicePointer 返回成功但别名=NULL，t6 T1）
3. cannAdaptorDeviceFree(flagcxMemHost) 配套 unregister（未注册时 507911 忽略）
4. cannAdaptorHostRegister 用 aclrtHostRegisterV2（旧 4 参接口在 8.5.0 报 507899）
5. cannAdaptorHostUnregister 真实现
6. cannAdaptor 结构体接线（GetVendor 后一个槽位从 NULL 改为 HostGetDevicePointer）
7. launch_kernel.cc:48 判错 + WARN（void 上下文不能用 FLAGCXCHECK）

用法：python3 patch_o2_phase3_hostgetdevptr.py <FlagCX根>
验证：t6_hostgetdevptr.cpp（scripts/p2_atomic_probe/，910C 8.5.0 实测 T4 PASS）

注意（2026-09-01 patch 验证发现）：
- cann_adaptor.cc 为 CRLF 行尾、launch_kernel.cc 为 LF 行尾，两文件行尾不同！
- 必须用二进制模式 + 行尾感知处理，否则文本模式读写会把 CRLF 转 LF，
  导致整个文件 diff 被污染（374 行全变）。
"""
import sys, os

ROOT = sys.argv[1].rstrip("/")

def load(p):
    """读 bytes，归一化行尾为 LF，返回 (data, was_crlf)"""
    data = open(p, "rb").read()
    crlf = b"\r\n" in data
    if crlf:
        data = data.replace(b"\r\n", b"\n")
    return data, crlf

def store(p, data, crlf):
    if crlf:
        data = data.replace(b"\n", b"\r\n")
    open(p, "wb").write(data)

def B(s):
    """str -> utf-8 bytes（bytes 字面量只接受 ASCII，中文注释必须经 encode）"""
    return s.encode("utf-8")

# ============ cann_adaptor.cc ============
p = os.path.join(ROOT, "flagcx/adaptor/device/cann_adaptor.cc")
s, crlf = load(p)

def rep(old, new, label):
    global s
    n = s.count(old)
    if n == 0:
        print(f"[FAIL] {label}: pattern not found")
        sys.exit(1)
    if n > 1:
        print(f"[FAIL] {label}: ambiguous ({n})")
        sys.exit(1)
    s = s.replace(old, new, 1)
    print(f"[ok] {label}")

# 1) hostGetDevicePointer 真实现（加在 getVendor 之后）
rep(B("""flagcxResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return flagcxSuccess;
}
// TODO:unsupport"""),
    B("""flagcxResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  // 注意：CANN 的 aclrtHostGetDevicePointer 参数顺序是 (pHost, pDevice, flag)，
  // 与 CUDA 的 cudaHostGetDevicePointer(pDevice, pHost, 0) 相反，勿写反。
  // 且对未 register 的内存会"返回成功但别名=NULL"（t6 T1 实测），必须判 NULL。
  aclError e = aclrtHostGetDevicePointer(pHost, pDevice, 0);
  if (e != ACL_SUCCESS) {
    return flagcxUnhandledDeviceError;
  }
  if (*pDevice == NULL) {
    return flagcxInternalError; // 内存未 register，别名不可用
  }
  return flagcxSuccess;
}
// TODO:unsupport"""),
    "1. cannAdaptorHostGetDevicePointer")

# 2) deviceMalloc(flagcxMemHost) 追加 RegisterV2
rep(B("""  if (type == flagcxMemHost) {
    DEVCHECK(aclrtMallocHost(ptr, size));
  } else {"""),
    B("""  if (type == flagcxMemHost) {
    // 注意：aclrtMallocHost 后必须显式 aclrtHostRegisterV2，否则
    // aclrtHostGetDevicePointer 返回成功但设备别名=NULL（t6 T1 实测）。
    // 用 V2（P0 探针验证组合）；旧 4 参 aclrtHostRegister 在 8.5.0 报
    // ACL_ERROR_RT_DRV_INTERNAL_ERROR(507899)（t6 T2 实测）。
    DEVCHECK(aclrtMallocHost(ptr, size));
    DEVCHECK(aclrtHostRegisterV2(*ptr, size,
                                  ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED));
  } else {"""),
    "2. deviceMalloc host+RegisterV2")

# 3) deviceFree(flagcxMemHost) 配套 unregister
rep(B("""  if (type == flagcxMemHost) {
    DEVCHECK(aclrtFreeHost(ptr));
  } else {"""),
    B("""  if (type == flagcxMemHost) {
    // 对称于 deviceMalloc 里的 aclrtHostRegisterV2；未注册时 unregister
    // 返回 ACL_ERROR_HOST_MEMORY_NOT_REGISTERED(507911)，忽略即可
    aclrtHostUnregister(ptr);
    DEVCHECK(aclrtFreeHost(ptr));
  } else {"""),
    "3. deviceFree host+Unregister")

# 4) hostRegister 用 V2
rep(B("""flagcxResult_t cannAdaptorHostRegister(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorHostUnregister(void *) { return flagcxNotSupported; }"""),
    B("""flagcxResult_t cannAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  // 对标 cudaHostRegister(ptr, size, cudaHostRegisterMapped)。
  // 必须用 aclrtHostRegisterV2：旧 4 参 aclrtHostRegister 在 8.5.0 上报
  // ACL_ERROR_RT_DRV_INTERNAL_ERROR(507899)（t6 T2 实测）；V2 为 P0 探针验证组合。
  DEVCHECK(aclrtHostRegisterV2(ptr, size,
                               ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED));
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(aclrtHostUnregister(ptr));
  return flagcxSuccess;
}"""),
    "4. hostRegister V2 + hostUnregister")

# 5) 结构体接线（GetVendor 后一槽：NULL -> HostGetDevicePointer）
rep(B("""cannAdaptorGetVendor, NULL,
      // GDR functions"""),
    B("""cannAdaptorGetVendor, cannAdaptorHostGetDevicePointer,
      // GDR functions"""),
    "5. cannAdaptor struct wiring")

store(p, s, crlf)

# ============ launch_kernel.cc ============
p2 = os.path.join(ROOT, "flagcx/core/launch_kernel.cc")
t, crlf2 = load(p2)

old2 = B("""  // Get device pointer alias
  deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);""")
new2 = B("""  // Get device pointer alias
  flagcxResult_t res = deviceAdaptor->hostGetDevicePointer(&dSignalsPool,
                                                           (void *)signalsPool);
  if (res != flagcxSuccess) {
    // initialize() 由构造函数调用（void 上下文），错误只能在此观测；
    // dSignalsPool 保持 NULL，后续 getDevicePtr 解引用会暴露问题
    WARN("hostGetDevicePointer failed for semaphore pool: %d", (int)res);
  }""")
if t.count(old2) != 1:
    print("[FAIL] 6. launch_kernel.cc: pattern not found/ambiguous")
    sys.exit(1)
t = t.replace(old2, new2, 1)
store(p2, t, crlf2)
print("[ok] 6. launch_kernel.cc WARN guard")

print("=== O2-#6 PATCH DONE ===")
