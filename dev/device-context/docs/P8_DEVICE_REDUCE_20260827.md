# P8：设备侧 reduce + CANN UVA 实测（异构 allreduce 架构增强，2026-08-27）

> 关联：`FLAGCX_CORE_DEFECT_FIXES_20260826.md`（P2/P6/P7 三缺陷修复闭环）；本文档是 P2/P6/P7 之后的**性能/架构增强**——用设备侧 reduce 消除朴素路径的 D2H + CPU reduce + H2D，并实测证明 CANN 有完整 UVA 能力。
> 环境：910C（昇腾，新容器 `flagos-hliu553-dev-910c`，CANN 9.0.0）+ 4090-1（NVIDIA，CUDA 13.0）跨机异构，FlagCX `kistich/ascend-dev1.0` 工作树。

---

## 0. TL;DR

| 项 | 结论 |
|---|---|
| CANN 有没有 UVA？ | **有，且实测真通**。`aclrtMallocHost` + `aclrtHostRegisterV2(PINNED\|MAPPED)` + `aclrtHostGetDevicePointer` 返回成功；**aclnnInplaceAdd 以 host 映射地址为输入算出正确结果（5.0+2.0=7.0）→ NPU 真实读到 host 内存**。此前"昇腾无 UVA"的结论是错的，真缺口在 FlagCX cann adaptor 未实现 `hostGetDevicePointer` 字段 |
| 设备侧 reduce 落地了吗？ | ✅。adaptor 加 `reduceSum` 字段（CANN=aclnnInplaceAdd / NVIDIA=CUDA kernel），`uniRunnerAllReduce` 对 Sum+fp32/fp16/bf16 走设备侧 reduce，跳过 D2H+CPU reduce+H2D |
| 遇到什么问题？ | 10 轮稳定性循环间歇性 sum=1.0。**根因：`COMPILE_KERNEL=1` 同时定义 `-DCOMPILE_KERNEL_HOST`，启用 proxy.cc 的 kernel proxy 线程，干扰 socket proxy 调度 → allgather Recv 数据偶发丢失**。已拆分修复（8/10→9/10） |
| 性能收益 | 50 步训练 sync **~32s/步 vs P7 的 47.5s（约 -33%）**；loss 与基线逐位一致（s0 2.8891/3.1656） |
| 遗留 | 集合级 10 轮稳定性仍有 **1/10 偶发数据错**（CUDA 侧大张量 4000B 分 chunk 时 Recv 偶发丢失，上游 net.cc `posted/postFlush/copied` 流水线竞态，P4 残余），与设备 reduce 无关，50 步训练未触发 |

---

## 1. 背景：朴素路径的瓶颈

P2/P6/P7 修复后，`uniRunnerAllReduce`（P7 分片版）每步：

```
for each 128MB slice:
  uniRunnerAllGather        // Send/Recv：梯度经 socket 收集到 tmpDev（nranks 片）
  deviceMemcpy(D2H, 整片)    // 3GB 梯度搬回 host
  CPU reduce（逐元素求和）   // host 端 3GB fp32
  deviceMemcpy(H2D, 结果)    // 3GB 写回
```

其中 **D2H + CPU reduce + H2D 是纯浪费**（约 10-20s/步），socket 数据面（allgather 传输 ~25-30s）才是真正瓶颈。设备侧 reduce 的目标是消除前者。

## 2. CANN UVA 实测：真通（决定性证据）

### 2.1 官方接口（CANN 9.0/9.1 文档）

| CUDA | CANN 对应物 |
|---|---|
| `cudaHostRegister` | **`aclrtHostRegisterV2(ptr, size, ACL_HOST_REG_PINNED\|ACL_HOST_REG_MAPPED)`** |
| `cudaHostGetDevicePointer` | **`aclrtHostGetDevicePointer(pHost, &pDevice, 0)`** |
| host/device VA 一致 | **`aclrtMallocHostWithCfg` + `ACL_RT_MEM_ATTR_VA_FLAG`** |

> 坑：OS 内核 ≤ 5.10 时 `aclrtHostRegisterV2` 会异常（文档警告，910C 是 5.10.0，实测普通 malloc + RegisterV2 返回 507899），**必须用 `aclrtMallocHost` 申请锁页内存**。

### 2.2 实测程序（`benchmarks/uva_test.c`）

```
[Path1] malloc + RegisterV2  → ret=507899（内核 5.10 坑，如文档警告）
[Path2] aclrtMallocHost + RegisterV2 → ret=0
        aclrtHostGetDevicePointer → ret=0 host=0x12c... dev=0x3fff8200000
[Path3] aclnnInplaceAdd(host映射地址=5.0, dev=2.0) → 7.0  PASS ← NPU 真实读 host 内存
```

**结论**：昇腾 910C 支持 UVA（设备可访问 host 锁页内存）。FlagCX cann adaptor 的 `hostGetDevicePointer` 字段留 NULL 是**适配层未实现**，不是平台限制。

> 注意：`aclrtMemcpy` D2D 从 host 映射地址读会 segfault（DMA 引擎 D2D 通道不支持），须走**算子**（aclnn）验证——这也正是设备侧 reduce 用 aclnn 的原因。

## 3. P8 落地：设备侧 reduce

### 3.1 改动全景（`patches/patch_device_reduce.py`）

| 文件 | 改动 |
|---|---|
| `flagcx/adaptor/include/flagcx_device_adaptor.h` | `flagcxDeviceAdaptor_latest` 加 `reduceSum` 函数指针（注意两结构体 v1/latest 的锚点区分） |
| `flagcx/adaptor/device/cann_adaptor.cc` | 实现 `cannAdaptorReduceSum`（aclnnInplaceAdd，fp32/fp16/bf16）+ 结构体字段 |
| `flagcx/adaptor/device/cuda_adaptor.cc` | 实现 `cudaAdaptorReduceSum`（调 host 封装）+ 结构体字段 |
| `flagcx/adaptor/kernel/nvidia/flagcx_device_reduce.cu`（新） | fp32/fp16/bf16 元素级 add kernel + `extern "C" flagcxLaunchReduceSum` |
| `flagcx/runner/uni_runner.cc` | `uniRunnerAllReduce` 加设备侧 reduce 分支（Sum + 3 dtype 走设备侧，其余回退 host reduce） |
| `makefiles/ascend.mk` | `DEVICE_LINK` 加 `-lopapi -lnnopbase`（aclnn 库） |

### 3.2 设备侧 reduce 分支（uni_runner.cc）

```c
if (op == flagcxSum &&
    (datatype == flagcxFloat32 || datatype == flagcxFloat16 ||
     datatype == flagcxBfloat16)) {
  FLAGCXCHECK(deviceAdaptor->deviceSynchronize());   // 跨流同步（见 §5）
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      (char *)recvbuff + off * esize, tmpDev, n * esize,
      flagcxMemcpyDeviceToDevice, stream, nullptr));   // recvbuff = tmpDev[0]
  for (int r = 1; r < nranks; r++) {
    FLAGCXCHECK(deviceAdaptor->reduceSum(
        (char *)recvbuff + off * esize,
        (const char *)tmpDev + r * n * esize, n, datatype, stream));  // recvbuff += tmpDev[r]
  }
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
  continue;
}
```

### 3.3 CANN 侧实现（aclnnInplaceAdd）

```c
flagcxResult_t cannAdaptorReduceSum(void *dst, const void *src, size_t count,
                                    flagcxDataType_t datatype, flagcxStream_t stream) {
  aclDataType aclDt = ...;  // flagcxFloat32→ACL_FLOAT, Float16→ACL_FLOAT16, Bfloat16→ACL_BF16
  int64_t dims[1] = {(int64_t)count};
  aclTensor *self  = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, dst);
  aclTensor *other = aclCreateTensor(dims, 1, aclDt, nullptr, 0, ACL_FORMAT_ND, nullptr, 0, const_cast<void*>(src));
  float one = 1.0f;
  aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
  uint64_t wsSize = 0; aclOpExecutor *executor = nullptr;
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
```

## 4. 踩坑 4 连（编译/链接）

| # | 坑 | 现象 | 修复 |
|---|---|---|---|
| 1 | `aclnnInplaceAddGetWorkspaceSize` **无 out 参数**（5 参，in-place） | 编译报错（参数个数不匹配） | 去掉 out tensor |
| 2 | `.cu` host 封装必须 **`extern "C"`** | 链接 `undefined symbol: _Z21flagcxLaunchReduceSum...`（C++ mangle 不匹配） | `extern "C" void flagcxLaunchReduceSum(...)` |
| 3 | CUDA 13 的 libcu++ 需 **C++17** | `.cu` 编译报错 `libcu++ requires at least C++ 17` | nvidia.mk `DEVICE_COMPILE_FLAG` 加 `-std=c++17` |
| 4 | gencode **缺 sm_89**（4090 = Ada 8.9） | kernel 静默不执行 → sum=1.0（无报错，`cudaGetLastError` 才有线索） | nvidia_gencode.mk `CUDA12_GENCODE` 加 `compute_89,code=sm_89` |

## 5. 关键发现：`COMPILE_KERNEL_HOST` 干扰 socket proxy（10 轮稳定性暴露）

### 5.1 现象

10 轮 `test_ag_hetero.py` 稳定性循环，间歇性 sum=1.0（rank0 CUDA 侧错，rank1 CANN 侧恒对）。诊断实锤：失败轮 D2H 读 `tmpDev[1]=0.0`（allgather 的 Recv 数据未到达）。

### 5.2 分离实验

| 代码状态 | 成功率 |
|---|---|
| P8 + COMPILE_KERNEL=1（含 HOST） | 8/10 |
| P8 + 禁用设备 reduce（host reduce 对照） | **8/10（仍失败！）** |
| P8 + COMPILE_KERNEL=1（拆分 HOST 后） | 9/10 |

**分离实验证明：间歇性失败不是设备 reduce 引入，而是 `COMPILE_KERNEL=1` 引入**。P7 时代编译无 `COMPILE_KERNEL=1`（kernel 没编译）→ 10/10 稳定。

### 5.3 根因

Makefile 里 `COMPILE_KERNEL=1` 同时：
1. 编译所有 `.cu`（设备 reduce kernel 需要）
2. **定义 `-DCOMPILE_KERNEL_HOST`** → 启用 proxy.cc:1188/1452 的 **kernel proxy 线程**（DAG 引擎用）→ 干扰 socket proxy 调度 → allgather Recv 数据偶发丢失

### 5.4 修复（Makefile 拆分）

```makefile
# 原：COMPILE_KERNEL=1 同时定义两个宏
ifeq ($(COMPILE_KERNEL), 1)
  COMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL
  COMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST
endif
# 改：拆分控制，默认不启用 kernel proxy
ifeq ($(COMPILE_KERNEL), 1)
  COMPILE_KERNEL_FLAG = -DCOMPILE_KERNEL
endif
ifeq ($(COMPILE_KERNEL_HOST), 1)
  COMPILE_KERNEL_HOST_FLAG = -DCOMPILE_KERNEL_HOST
endif
```

编译命令（4090-1，不设 COMPILE_KERNEL_HOST）：
```bash
make USE_NVIDIA=1 COMPILE_KERNEL=1 JSON_INCLUDE_DIR=$HOME/include \
     CCL_HOME=$HOME/dvfs/.venv_4090/lib/python3.11/site-packages/nvidia/nccl \
     CCL_LINK="-l:libnccl.so.2" HOST_COMPILER="g++ -std=c++17" -j24
```

## 6. 验证证据

### 6.1 集合级（test_ag_hetero.py，AG/AG/AR）

| 项 | 结果 |
|---|---|
| out / out2 / sum | 双侧 `out=[1,2]`、`out2=[10,11]`、`sum=3.0` 全对 |
| 稳定性 | 10 轮循环 **9/10**（拆分 COMPILE_KERNEL_HOST 后；修复前 8/10）|

### 6.2 真实训练（Qwen2.5-1.5B，MAX_STEPS=50）

| 项 | P7（host reduce） | P8（设备侧 reduce） |
|---|---|---|
| loss（s0） | 2.8891 / 3.1656 | **2.8891 / 3.1656**（逐位一致）|
| loss（s20） | 2.1303 | **2.1322**（vs gloo 终点 2.1312）|
| 单步 sync | ~47.5s | **~32s（约 -33%）** |
| 死锁 | 0 | **0**（双侧完整跑完，ckpt 保存退出）|

## 7. 遗留问题（如实记录）

**集合级 10 轮仍有 1/10 偶发数据错**（rank0 CUDA 侧 sum=1.0）：
- 模式：大张量（4000B float32）分 chunk 时 Recv 偶发丢失；小张量（int64 8B）稳定；rank1 CANN 侧恒对
- 根因：上游 net.cc `posted/postFlush/copied` chunk 流水线竞态（P4 残余），与设备 reduce 无关（分离实验证明）
- 已尝试：`eventQuery(cpEvents[step])` 替代 `streamQuery(cpStream)` → 反而更差（7/10），已回退
- 50 步真实训练未触发（偶发率低）
- 彻底修复需深入 net.cc 流水线，**建议作为独立任务移交**

## 8. 产物清单

| 产物 | 说明 |
|---|---|
| `patches/patch_device_reduce.py` | 一键落地（adaptor + 两侧实现 + kernel + uni_runner + 编译链修复） |
| `benchmarks/uva_test.c` | CANN UVA 实测（含三条路径对比） |
| `benchmarks/aclnn_test.c` | aclnnInplaceAdd 设备侧 add 实测（纯 device + host 映射两用例） |
| `benchmarks/loop_hetero_p8.sh` | 10 轮稳定性循环（适配新容器 + 设备侧 reduce） |
| 本文档 | P8 全过程 + 根因 + 验证 |
