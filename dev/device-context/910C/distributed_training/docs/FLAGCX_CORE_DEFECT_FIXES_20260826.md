# 缺陷⑩ 修复全过程：FlagCX 核心库原生 allreduce（死锁 P2 + 数据错乱 P6 + 显存 OOM P7）

> 更新：2026-08-26 ｜ 作者：Kistich ｜ 范围：FlagCX 原生 allreduce 路径（多后端 CUDA/昇腾），模型 Qwen2.5-1.5B
> 关联补丁：`patches/patch_p2_fix_deadlock.py`、`patches/patch_p6_sync_before_cb.py`、`patches/patch_p7_sliced_allreduce.py`

## 0. TL;DR

缺陷⑩（原生 allreduce 在多 rank 训练中间歇死锁 + 数据错乱）经根因定位，实为 **两个相互独立的 FlagCX 实现缺陷**，均已修复；真实训练验证时又暴露 **第三个缺陷（P7 显存 OOM）**，一并修复：

| 子缺陷 | 现象 | 根因文件 | 修复 |
|---|---|---|---|
| **P2 死锁** | 第 3 个 collective / 多轮必卡死 | `flagcx/core/launch_kernel.cc` + `flagcx/core/group.cc` | `cpuAsyncKernel` 回调内不再 `wait()`；改由 `groupLaunch` 主线程 `wait()` |
| **P6 数据错乱** | 昇腾侧发出"慢一拍"旧数据（out2 错、sum 错） | `flagcx/adaptor/device/cann_adaptor.cc` | `aclrtLaunchCallback` 前先 `aclrtSynchronizeStream(stream->base)` |
| **P7 显存 OOM**（真实训练验证发现） | step0 成功、step≥1 崩溃 `flagcxUnhandledDeviceError: Call to Device function failed` | `flagcx/runner/uni_runner.cc` | 分片 allreduce：临时设备缓冲从 `bytes×nranks`（6GB）降为 `slice×nranks`（默认 256MB） |

修复后集合级验证 **10/10 轮** 全对（`out=[1,2]`、`out2=[10,11]`、`sum=3.0`），**0 死锁**；真实训练冒烟 5/5 + 完整 50/50 步通过（详见 §5）。

---

## 1. 缺陷现象（修复前）

集合级验证脚本（双 rank，跑 AG→AG→AR 三个 collective）：
- **死锁**：前两次侥幸通过，**第 3 个 collective（AR）必卡死**；多轮循环第 3 轮必挂。
- **数据错乱**：即使不死锁的轮次，昇腾侧 `out2` 出现"慢一拍"——应发出 `[10,11]` 却发 `[0,2]`/`[2,11]`，`sum` 时而对不上。

> 注：此前因死锁与错乱共存，该路径长期被 Python 层朴素 `all_gather+sum` 绕开。修复后原生路径为默认验收路径。

---

## 2. 根因 A：原生 allreduce 竞态死锁（P2）

### 2.1 源码级根因

`flagcx/core/launch_kernel.cc` 中，`cpuAsyncKernel` 被注册为 `cudaLaunchHostFunc`（CUDA）/ `aclrtLaunchCallback`（昇腾）的回调，其原始实现：

```cpp
void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  semaphore->wait();   // ← BUG：在 host-func 回调内自旋等待所有 op 完成
}
```

`semaphore->wait()` 会自旋直到所有 collective op 完成。但 op 的完成依赖 **proxy 线程** 执行 `cudaMemcpyAsync`（net.cc 的 D2H copy）。在 **CUDA 侧**，`cudaLaunchHostFunc` 的回调执行期间，CUDA driver 会占用其回调/内部锁；`cpuAsyncKernel` 在回调内 `wait()` 自旋 → 永久占住该锁 → proxy 线程的 `cudaMemcpyAsync` 永远拿不到执行机会 → op 永不完成 → `wait()` 永不返回 → **三方死锁**（双 rank 与 proxy 线程互相等待）。

- 纯**时序竞态**：若 proxy 的 memcpy 在 host func 开始自旋之前就已入队并完成，则 collective 侥幸过关——这正是"前两次能过、第 3 次必死"的原因。
- **Ascend 侧为何不卡在 driver 锁**：`aclrtLaunchCallback` 走独立线程 + SubscribeReport 订阅机制，无 CUDA 式全局回调锁语义，故昇腾侧不会在此处死锁；但死锁修复对两侧 core 代码一致，统一应用。

### 2.2 修复

把"等待完成"从 host-func 回调内移出，改由调用方**主线程**在 `flagcxGroupLaunch` 末尾执行：

- `launch_kernel.cc`：`cpuAsyncKernel` 只 `signalStart()` 后立即返回（host func 永不阻塞）。
- `group.cc`：`flagcxGroupLaunch` 在 `launchHostFunc` 之后由主线程 `semaphore->wait()`。主线程不持有 driver 回调锁，数据可见性安全（recv op 在 H2D copy 经 `eventQuery` 确认完成后才 `subCounter`；collective 返回 Python 前不会入队后续 GPU 工作）。

### 2.3 干净 diff

```diff
--- a/flagcx/core/launch_kernel.cc
+++ b/flagcx/core/launch_kernel.cc
@@ void cpuAsyncKernel(void *args) {
   flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
   semaphore->signalStart();
-  semaphore->wait();
 }
```

```diff
--- a/flagcx/core/group.cc
+++ b/flagcx/core/group.cc
@@ } else {
       FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                 (void *)semaphore.get()));
+      semaphore->wait();   // 主线程强制完成，勿在 host func 回调内 wait()
     }
```

> 复现补丁：`patches/patch_p2_fix_deadlock.py <launch_kernel.cc> <group.cc>`（已剥离诊断打印，仅保留上述修复逻辑）。

---

## 3. 根因 B：昇腾发送路径数据错乱（P6）

### 3.1 源码级根因

死锁修掉后，约 60% 轮次出现数据错乱。逐字节打点证明：**Ascend 侧 `aclrtLaunchCallback` 的回调执行并不等待 stream 上的前置任务完成**。

`cannAdaptorLaunchHostFunc` 的主路径（此前为修 ACL 107015 实现的 SubscribeReport 版本）直接：

```cpp
aclError err = aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base);
```

其中 `fn` 即 `signalStart()`——它触发 proxy 线程去执行 tensor 的 D2H 拷贝。由于 ACL 回调**不等 stream 前置任务**，signalStart 抢跑，proxy 的 D2H 在 NPU tensor 的写入 kernel 完成**之前**就读走缓冲 → 昇腾侧发出的是"上一拍"的旧数据（发 0 应发 2、发 2 应发 11）。

- **CUDA 侧为何幸免**：`cudaLaunchHostFunc` 的 `stream=NULL` 语义为 legacy default stream，**天然对所有流保序**，前置 tensor 写入先于回调执行，故 CUDA 侧无此问题。这导致缺陷只在含昇腾后端的场景暴露。

### 3.2 修复

在 `aclrtLaunchCallback` **之前**显式同步该 stream：

- `group.cc` 已入队 `eventRecord(op->event, op->stream)` + `streamWaitEvent(launchStream, op->event)`，故同步 `launchStream` 会**传递性地**等待 `op->stream` 上全部 tensor 写入工作完成，再让 signalStart 触发 D2H。
- 主路径加 `aclrtSynchronizeStream(stream->base);`（**正确 API 名**，注意不是 `aclrtStreamSynchronize`）。
- NULL 分支（base==nullptr，即 torch_npu 默认流）补防御性 `aclrtSynchronizeDevice()`——本场景该分支未命中，但可作为安全网。

### 3.3 干净 diff

```diff
--- a/flagcx/adaptor/device/cann_adaptor.cc
+++ b/flagcx/adaptor/device/cann_adaptor.cc
@@ void cannAdaptorLaunchHostFunc(...) {
   if (stream == NULL || stream->base == nullptr) {
-    // No stream: run the host func directly.
-    fn(args);
+    // Kistich(fix-stale-data): base==nullptr 是 torch_npu 默认流（ACL 中
+    // nullptr 即默认流），并非“无流”；直接执行会跳过流序，故先同步设备。
+    aclrtSynchronizeDevice();
+    fn(args);
     return flagcxSuccess;
   }
   ...
-  aclError err =
+  // Kistich(fix-stale-data): ACL 回调不等待 stream 前置任务；先同步，保证
+  // proxy 的 D2H 在 tensor 写入完成后才发生（恢复 CUDA legacy-stream 等价 happens-before）。
+  aclrtSynchronizeStream(stream->base);
+  aclError err =
       aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base);
```

> 复现补丁：`patches/patch_p6_sync_before_cb.py <cann_adaptor.cc>`（主路径修复；已修正为正确 API `aclrtSynchronizeStream`，并剔除诊断打印）。

---

## 4. 根因 C：真实训练 step≥1 崩溃（P7，2026-08-26 真实训练验证发现）

### 4.1 现象

真实训练（原生 allreduce 路径）：

- **step 0 双 rank 成功**：rank0 loss=2.8891、rank1 loss=3.1656——与参考基线首步**逐位一致**，证明 P2+P6 修复在大张量（3GB flat 梯度）上也正确。
- **step 1 立即同步崩溃**（无任何新 op 入队打点，[s0] 与 Traceback 紧邻）：
  ```
  torch.distributed.DistBackendError: FLAGCX error in: flagcx/src/backend_flagcx.cpp:879,
  flagcxUnhandledDeviceError: Call to Device function failed.
  Last error:
  Undefined: flagcxComm is not fully initialized.
  ```
- 大显存侧（64GB HBM）不崩，卡在等待对端 rank。

### 4.2 源码级根因

`flagcx/runner/uni_runner.cc` 的 `uniRunnerAllReduce`（朴素路径）**每次调用都分配 `bytes × nranks` 的临时设备缓冲**：

```cpp
FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, bytes * nranks, flagcxMemDevice, stream));
```

3GB flat 梯度 × 2 rank = **6GB/步**。在小显存卡（24GB）上的时序：

| 时刻 | 显存占用 | 6GB 分配 |
|---|---|---|
| step0 allreduce（`optimizer.step()` **之前**） | 模型 3.1G + 梯度 3.1G + flat 3.1G + 激活/杂项 ≈ 11G | ✅ 成功（空闲 ~13G） |
| step0 结束后 | `optimizer.step()` 建立 AdamW fp32 状态 ×2 ≈ **+12.4G** | — |
| **step1 allreduce** | ≈ 23.7G / 24G，空闲 <1G | ❌ `cudaMallocAsync` OOM |

关键放大因素：`DEVCHECK` 宏**失败时静默返回** `flagcxUnhandledDeviceError`（无任何日志），因此日志中零 WARN；而报错尾巴 "flagcxComm is not fully initialized" 是 `getFlagcxErrorDetailStr()` **无 comm 句柄**时 `flagcxGetLastError(NULL)` 的固定字符串——**纯红鲱鱼**，与 comm 状态无关。

此前的集合级测试（张量仅 1000 float）从未触发大缓冲分配，故 P2/P6 修复验证时未暴露。

### 4.3 修复

**分片（sliced）allreduce**：按片处理（默认 128MB/片，`FLAGCX_AR_SLICE_MB` 可调），每片执行 allgather → D2H → host reduce → H2D。临时设备缓冲降为 `slice × nranks`（默认 256MB）。原有的 per-dtype reduce switch 原样保留（分片作用域内遮蔽 `count`/`bytes` 复用）。

```diff
--- a/flagcx/runner/uni_runner.cc
+++ b/flagcx/runner/uni_runner.cc
@@ uniRunnerAllReduce(...)
-  // 1) gather all ranks into a temporary device buffer (nranks slices)
-  void *tmpDev = nullptr;
-  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, bytes * nranks,
-                                          flagcxMemDevice, stream));
-  FLAGCXCHECK(
-      uniRunnerAllGather(sendbuff, tmpDev, count, datatype, comm, stream));
-
-  // 2) D2H the gathered buffer, then reduce on host
-  char *hostBuf = (char *)malloc(bytes * nranks);
-  if (hostBuf == nullptr)
-    return flagcxSystemError;
-  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
-      hostBuf, tmpDev, bytes * nranks, flagcxMemcpyDeviceToHost, stream,
-      nullptr));
-  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
+  // Kistich(fix-oom): slice-bounded temp buffers（见 patches/patch_p7_sliced_allreduce.py 全文）
+  size_t sliceBytes = 128 << 20;              // FLAGCX_AR_SLICE_MB 可调
+  ...
+  for (size_t off = 0; off < count; off += sliceCount) {
+    size_t n = min(sliceCount, count - off);
+    FLAGCXCHECK(uniRunnerAllGather((const char *)sendbuff + off * esize,
+                                   tmpDev, n, datatype, comm, stream));
+    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
+        hostBuf, tmpDev, n * esize * nranks, flagcxMemcpyDeviceToHost, stream,
+        nullptr));
+    FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
+    {  // 遮蔽 count/bytes，复用下方原 reduce switch
+      size_t count = n;
+      size_t bytes = n * esize;
       for (int r = 1; r < nranks; r++) { ... 原 per-dtype reduce 原样保留 ... }
-  // 3) H2D the reduced slice
-  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
-      recvbuff, hostBuf, bytes, flagcxMemcpyHostToDevice, stream, nullptr));
-  free(hostBuf);
-  FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
-  return flagcxSuccess;
+      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
+          (char *)recvbuff + off * esize, hostBuf, bytes,
+          flagcxMemcpyHostToDevice, stream, nullptr));
+    }
+  }
+  free(hostBuf);
+  FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
+  return flagcxSuccess;
```

> 复现补丁：`patches/patch_p7_sliced_allreduce.py <uni_runner.cc>`（幂等，含完整 before/after）。

---

## 5. 验证证据

### 5.1 集合级

| 项目 | 修复前 | 修复后 |
|---|---|---|
| 死锁 | 第 3 collective 必卡 | **0/10 轮死锁** |
| 数据正确 | out2 错 ~60%、sum 错 ~30% | **10/10 轮全对**（`out=[1,2]`、`out2=[10,11]`、`sum=3.0`） |
| allreduce 耗时（1KB 张量） | 死锁（无结果） | **~0.01s/次**（10 轮均值） |

- AG→AG→AR 三 collective，双 rank；通过标准：`out=[1,2]`、`out2=[10,11]`、`sum=3.0` 三者齐全，进程正常退出，无卡死。
- 稳定性：循环连跑 10 轮（端口递增，避免 TIME_WAIT 串扰），`PASS=10 FAIL=0`。

### 5.2 真实训练（Qwen2.5-1.5B，3GB flat 梯度）

| 项目 | 修复前 | 修复后 |
|---|---|---|
| step0（P2+P6 修复后） | — | rank0 loss=2.8891 / rank1 loss=3.1656，**与参考基线首步逐位一致** |
| step≥1（P7 修复前） | **崩溃**（`Call to Device function failed`，6GB 临时缓冲 OOM） | — |
| step≥1（P7 修复后） | — | **冒烟 5/5 步通过**，双侧 `[done]` 正常退出，ckpt 保存 |
| **完整 50 步** | — | **✅ 通过**：rank0 loss 2.8891→2.1 区间（s20=2.1303），`[done] total=2394s sync_total=2374s`；rank1 loss 3.1656→1.8-2.6 区间，`[done] exited`；两 rank 同步收敛，**全程零死锁零数据异常** |

**验收判定**：缺陷⑩ 视为修复闭环，当且仅当：① 集合级三 collective 全过无卡死；② 稳定性 10 轮 PASS；③ 真实训练全程无死锁/数据异常、loss 收敛趋势与同构基线方向一致。

---

## 6. 性能（诚实口径）

| 指标 | 数值 |
|---|---|
| 集合级小张量 | ~0.01s/次 |
| 真实训练 3GB 梯度 | **~49s/步**（网络数据面物理传输 ~25-30s + 工作树残留诊断打印每步 ~60MB 的开销） |
| 优化空间 | 剥离诊断打点、切换 RDMA/RoCE 数据面后可大幅提升 |

---

## 7. 复现补丁用法

```bash
# 在昇腾侧容器（已 bind mount FlagCX 源码）与 CUDA 侧分别打补丁
# P2（核心，两侧都要）：
python patches/patch_p2_fix_deadlock.py \
    /home/.../FlagCX/flagcx/core/launch_kernel.cc \
    /home/.../FlagCX/flagcx/core/group.cc
# P6（仅昇腾侧 cann adaptor）：
python patches/patch_p6_sync_before_cb.py \
    /home/.../FlagCX/flagcx/adaptor/device/cann_adaptor.cc
# P7（核心，两侧都要）：
python patches/patch_p7_sliced_allreduce.py \
    /home/.../FlagCX/flagcx/runner/uni_runner.cc
# 重新编译 FlagCX（昇腾侧容器内）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
make USE_ASCEND=1 -j24
# CUDA 侧：
make USE_NVIDIA=1 JSON_INCLUDE_DIR=$HOME/include \
     CCL_HOME=<nccl-python-package路径> \
     CCL_LINK='-l:libnccl.so.2' HOST_COMPILER='g++ -std=c++17' -j24
```

---

## 8. 后续

1. **提交 FlagCX 源码修复**：将上述干净 diff 提交到 `FlagRT/FlagCX` 分支 `kistich/ascend-dev1.0`（当前工作树改动尚未 commit，待本轮 PR merge 后执行）。
2. **剥离诊断打印**：提交上游前剥离 `net.cc`/`proxy.cc`/`transport.cc`/`launch_kernel.h` 中诊断打点（`[P1-STATE]`/`[P1-COPY]`/`[P2-COPY]`/`[P4-SEND-DATA]`）——它们不影响正确性（`pollStart` 为纯原子读）但每步产生 ~60MB stderr，拖慢单步 ~20s。
3. **吞吐优化**：网络数据面下 3GB 梯度同步 ~49s/步（其中物理传输 ~25-30s）；切换 RDMA 数据面后可大幅压缩。
