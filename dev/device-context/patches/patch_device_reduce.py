#!/usr/bin/env python3
"""落地「设备侧 reduce（Sum）」到 FlagCX，消除异构 allreduce 的 D2H+CPU reduce+H2D。

用法: python3 patch_device_reduce.py <flagcx_root> <ascend|nvidia>
- ascend: 改 cann_adaptor.cc（aclnnInplaceAdd）+ ascend.mk 链接 aclnn
- nvidia: 改 cuda_adaptor.cc + 写 adaptor/kernel/nvidia/flagcx_device_reduce.cu
- 通用: flagcx_device_adaptor.h 加 reduceSum 字段 + uni_runner.cc 加设备侧 reduce 分支

幂等：每个替换都检查 marker，已应用则跳过。
"""
import sys, os

def patch(path, old, new, marker):
    with open(path) as f:
        s = f.read()
    if marker in s:
        print(f"[skip] {path}: marker already present")
        return
    if old not in s:
        print(f"[ERROR] {path}: old string NOT FOUND")
        sys.exit(1)
    if s.count(old) != 1:
        print(f"[ERROR] {path}: old string not unique (count={s.count(old)})")
        sys.exit(1)
    with open(path, "w") as f:
        f.write(s.replace(old, new, 1))
    print(f"[ok] {path}")

def main():
    root = sys.argv[1]
    side = sys.argv[2]

    # ---------- 1. adaptor 头：加 reduceSum 字段（两侧通用） ----------
    # 注意：头文件有两个结构体（flagcxDeviceAdaptor_v1 与 flagcxDeviceAdaptor_latest），
    # latest 的 gdrPtrMunmap 后紧跟 "// Stream functions"（无空行），v1 有空行。
    # 用这个差异精确锚定 latest（cann/cuda 实际初始化的结构体）。
    hdr = os.path.join(root, "flagcx/adaptor/include/flagcx_device_adaptor.h")
    patch(hdr,
        "  flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n"
        "  // Stream functions\n",
        "  flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n"
        "  // Device-side reduce (Sum) for heterogeneous allreduce: dst += src on\n"
        "  // device, avoiding D2H + host reduce + H2D. CANN: aclnnInplaceAdd;\n"
        "  // NVIDIA: CUDA kernel in adaptor/kernel/nvidia/flagcx_device_reduce.cu.\n"
        "  flagcxResult_t (*reduceSum)(void *dst, const void *src, size_t count,\n"
        "                              flagcxDataType_t datatype, flagcxStream_t stream);\n"
        "  // Stream functions\n",
        "flagcxResult_t (*reduceSum)")

    # ---------- 2. uni_runner.cc：加设备侧 reduce 分支（两侧通用） ----------
    ur = os.path.join(root, "flagcx/runner/uni_runner.cc")
    patch(ur,
        "    // 2) D2H the gathered slice, then reduce on host\n"
        "    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(\n"
        "        hostBuf, tmpDev, n * esize * nranks, flagcxMemcpyDeviceToHost, stream,\n"
        "        nullptr));\n",
        "    // Kistich(device-reduce-sum): for Sum on fp32/fp16/bf16, reduce directly\n"
        "    // on device to avoid D2H + host reduce + H2D. tmpDev holds nranks\n"
        "    // sub-slices [rank0..rank{nranks-1}]; copy rank0 into recvbuff then\n"
        "    // accumulate the rest on device (aclnnInplaceAdd / CUDA kernel).\n"
        "    if (op == flagcxSum &&\n"
        "        (datatype == flagcxFloat32 || datatype == flagcxFloat16 ||\n"
        "         datatype == flagcxBfloat16)) {\n"
        "      // 同步：allgather 的 proxy H2D 在独立 cpStream，与主线程 commStream\n"
        "      // 跨流竞态；全设备同步确保 tmpDev 就绪后再 D2D/kernel 读。\n"
        "      FLAGCXCHECK(deviceAdaptor->deviceSynchronize());\n"
        "      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(\n"
        "          (char *)recvbuff + off * esize, tmpDev, n * esize,\n"
        "          flagcxMemcpyDeviceToDevice, stream, nullptr));\n"
        "      for (int r = 1; r < nranks; r++) {\n"
        "        FLAGCXCHECK(deviceAdaptor->reduceSum(\n"
        "            (char *)recvbuff + off * esize,\n"
        "            (const char *)tmpDev + r * n * esize, n, datatype, stream));\n"
        "      }\n"
        "      FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));\n"
        "      continue;\n"
        "    }\n"
        "    // 2) D2H the gathered slice, then reduce on host\n"
        "    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(\n"
        "        hostBuf, tmpDev, n * esize * nranks, flagcxMemcpyDeviceToHost, stream,\n"
        "        nullptr));\n",
        "Kistich(device-reduce-sum)")

    if side == "ascend":
        # ---------- 3. cann_adaptor.cc ----------
        ca = os.path.join(root, "flagcx/adaptor/device/cann_adaptor.cc")
        patch(ca,
            '#include "alloc.h"\n',
            '#include "alloc.h"\n'
            '// aclnn 单算子：设备侧 reduce（dst += src）\n'
            '#include "aclnn/aclnn_base.h"\n'
            '#include "aclnn/acl_meta.h"\n'
            '#include "aclnnop/aclnn_add.h"\n',
            '#include "aclnnop/aclnn_add.h"')

        # 实现函数：插在结构体初始化之前（cannAdaptorStreamWriteValue64 之后）
        impl = '''flagcxResult_t cannAdaptorReduceSum(void *dst, const void *src, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxStream_t stream) {
  aclDataType aclDt;
  switch (datatype) {
  case flagcxFloat32: aclDt = ACL_FLOAT; break;
  case flagcxFloat16: aclDt = ACL_FLOAT16; break;
  case flagcxBfloat16: aclDt = ACL_BF16; break;
  default: return flagcxInvalidArgument;
  }
  int64_t dims[1] = {(int64_t)count};
  aclTensor *self = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, dst);
  aclTensor *other = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, const_cast<void *>(src));
  float one = 1.0f;
  aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
  uint64_t wsSize = 0;
  aclOpExecutor *executor = nullptr;
  aclnnStatus st = aclnnInplaceAddGetWorkspaceSize(self, other, alpha, &wsSize, &executor);
  if (st == 0) {
    void *ws = nullptr;
    aclrtMalloc(&ws, wsSize ? wsSize : 1, ACL_MEM_MALLOC_HUGE_FIRST);
    st = aclnnInplaceAdd(ws, wsSize, executor, stream->base);
    aclrtFree(ws);
  }
  aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyScalar(alpha);
  return st == 0 ? flagcxSuccess : flagcxUnhandledDeviceError;
}

'''
        anchor = '''flagcxResult_t cannAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}
'''
        patch(ca, anchor, anchor + "\n" + impl, "cannAdaptorReduceSum")

        # 结构体字段
        patch(ca,
            "      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n",
            "      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n"
            "      cannAdaptorReduceSum,\n",
            "cannAdaptorReduceSum,\n")

        # ---------- 4. ascend.mk：链接 aclnn ----------
        amk = os.path.join(root, "makefiles/ascend.mk")
        patch(amk,
            "DEVICE_LINK  := -lascendcl\n",
            "DEVICE_LINK  := -lascendcl -lopapi -lnnopbase\n",
            "-lopapi")

    elif side == "nvidia":
        # ---------- 3. cuda_adaptor.cc ----------
        ca = os.path.join(root, "flagcx/adaptor/device/cuda_adaptor.cc")
        patch(ca,
            '#include "param.h"\n',
            '#include "param.h"\n'
            '// 设备侧 reduce host 封装（实现于 adaptor/kernel/nvidia/flagcx_device_reduce.cu）\n'
            'extern "C" void flagcxLaunchReduceSum(void *dst, const void *src, size_t count,\n'
            '                                       int datatype, void *stream);\n',
            'flagcxLaunchReduceSum')

        impl = '''flagcxResult_t cudaAdaptorReduceSum(void *dst, const void *src, size_t count,
                                    flagcxDataType_t datatype,
                                    flagcxStream_t stream) {
  flagcxLaunchReduceSum(dst, src, count, (int)datatype, (void *)stream->base);
  return flagcxSuccess;
}

'''
        anchor = '''flagcxResult_t cudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}
'''
        patch(ca, anchor, anchor + "\n" + impl, "cudaAdaptorReduceSum")

        # 结构体字段
        patch(ca,
            "      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n",
            "      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);\n"
            "      cudaAdaptorReduceSum,\n",
            "cudaAdaptorReduceSum,\n")

        # ---------- 4. 写 .cu kernel ----------
        cu_path = os.path.join(root, "flagcx/adaptor/kernel/nvidia/flagcx_device_reduce.cu")
        if os.path.exists(cu_path):
            print("[skip] .cu already exists")
        else:
            cu = r'''/* 设备侧 reduce（Sum）kernel：dst += src，供 uniRunner 异构 allreduce 使用。
 * 由 cuda_adaptor.cc 的 cudaAdaptorReduceSum 通过 flagcxLaunchReduceSum 调用。
 * flagcxDataType_t: flagcxFloat16=6, flagcxFloat32=7, flagcxBfloat16=9
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__global__ void flagcxReduceSumKernelF32(const float *a, const float *b, float *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
__global__ void flagcxReduceSumKernelF16(const __half *a, const __half *b, __half *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __hadd(a[i], b[i]);
}
__global__ void flagcxReduceSumKernelBF16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, __nv_bfloat16 *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) out[i] = __hadd(a[i], b[i]);
}

extern "C" void flagcxLaunchReduceSum(void *dst, const void *src, size_t count,
                                      int datatype, void *stream) {
  cudaStream_t s = (cudaStream_t)stream;
  int threads = 256;
  size_t blocks = (count + threads - 1) / threads;
  switch (datatype) {
    case 7: /* flagcxFloat32 */
      flagcxReduceSumKernelF32<<<blocks, threads, 0, s>>>(
          (const float *)dst, (const float *)src, (float *)dst, count);
      break;
    case 6: /* flagcxFloat16 */
      flagcxReduceSumKernelF16<<<blocks, threads, 0, s>>>(
          (const __half *)dst, (const __half *)src, (__half *)dst, count);
      break;
    case 9: /* flagcxBfloat16 */
      flagcxReduceSumKernelBF16<<<blocks, threads, 0, s>>>(
          (const __nv_bfloat16 *)dst, (const __nv_bfloat16 *)src, (__nv_bfloat16 *)dst, count);
      break;
    default:
      break;
  }
}
'''
            with open(cu_path, "w") as f:
                f.write(cu)
            print(f"[ok] wrote {cu_path}")

        # ---------- 5. 编译链修复（nvidia 侧必需） ----------
        # 5a. CUDA 13 的 libcu++ 需要 C++17（否则 .cu 编译报错）
        nvmk = os.path.join(root, "makefiles/nvidia.mk")
        patch(nvmk,
            "DEVICE_COMPILE_FLAG := -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g ",
            "DEVICE_COMPILE_FLAG := -std=c++17 -c --cudart=shared -Xcompiler -fPIC -MMD -MP -rdc=true -g ",
            "-std=c++17 -c")
        # 5b. gencode 补 sm_89（4090 Ada，否则 kernel 静默不执行）
        genmk = os.path.join(root, "makefiles/nvidia_gencode.mk")
        patch(genmk,
            "CUDA12_GENCODE  = -gencode=arch=compute_90,code=sm_90",
            "CUDA12_GENCODE  = -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90",
            "compute_89")
        # 5c. Makefile 拆分 COMPILE_KERNEL_HOST：COMPILE_KERNEL=1 只编 .cu，
        #     COMPILE_KERNEL_HOST=1（默认 0）才启用 DAG kernel proxy（否则干扰 socket proxy）
        mk = os.path.join(root, "Makefile")
        patch(mk,
            "ifeq ($(COMPILE_KERNEL), 1)\n\tCOMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL\n\tCOMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST\nendif\n",
            "ifeq ($(COMPILE_KERNEL), 1)\n\tCOMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL\nendif\nifeq ($(COMPILE_KERNEL_HOST), 1)\n\tCOMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST\nendif\n",
            "ifeq ($(COMPILE_KERNEL_HOST), 1)")

    else:
        print("side must be ascend or nvidia")
        sys.exit(1)

    print("DONE")

if __name__ == "__main__":
    main()
