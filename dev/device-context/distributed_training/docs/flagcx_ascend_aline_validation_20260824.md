# 昇腾 Ascend A 线验证报告（2026-08-24）

> 分支：`kistich/ascend-dev1.0`（基于 dev-1.0 @ 4e0e0cb）
> 环境：910C（4 NPU / 8 chip, HBM 64GB, CANN 9.0.0, 驱动 25.5.0）+
> `flagrt/ascend-operator-runtime:0.1.0-cann9.0-py311-torch2.10-arm64` 官方镜像 +
> torch_npu 2.10.0
> 定位：把 910C 双卡 HCCL 实测的四层根因修复带到 dev-1.0 基线（A 线主线，
> torch_npu 路线）；原 `kistich/ascend-flagcx-adapt`（torch_fl/B 线基线）降级为
> 预研支线，不再作为开发主线。

## 一、分支内容（3 个 commit）

| commit | 内容 |
|---|---|
| 5f7ad78 | 四层根因修复 rebase 到 dev-1.0：HcclRootInfo(4108B)>flagcxUniqueId(256B) 溢出 → thread_local + bootstrap 广播完整 RootInfo；设备绑定注释与处理；**保留** dev-1.0 已合入的 broadcast 字节数修复（c0494fe） |
| 60e0695 | 移除对 torch_fl `GetCurrentStream` 的硬链接依赖（A 线无 torch_fl/libflagos.so，`undefined symbol`）；stream 0 统一走 `GetFlagcxCurrentAclStream()` 的 dlsym 动态解析 + 自建流 fallback |
| 0686805 | **A 线 stream 语义修复**：`getStreamByIndex(0)` 在 ASCEND 分支优先解析 torch_npu 当前流——`c10::impl::getDeviceGuardImpl(PrivateUse1)->getStream()` 取 c10::Stream，再 `dlopen("libtorch_npu.so", RTLD_NOLOAD)` + `dlsym("_ZNK7c10_npu9NPUStream6streamEv")` 转为 aclrtStream（NPUStream 内存布局以 c10::Stream 开头，this 指针可直接复用）。**注意：torch_npu 对默认流返回 nullptr（ACL null=默认流语义），nullptr 是有效流**，不得 fallback 到自建流，否则与 tensor op 无 happens-before → allgather 数据全 0。不链接 libtorch_npu.so，保持 torch 插件与 host runtime 解耦（延续 8ebdeba 的设计） |

## 二、已验证通过项（torch_npu + `backend="flagcx"`）

- **双卡 allgather 数据正确性**：两 rank 两轮 `out=[1,2]` / `out2=[10,11]`
  全对，多轮稳定复现；修复前的 `free(): invalid pointer` 退出崩溃随 stream
  修复一并消失（测试：runtime-team `dev/device-context/benchmarks/test_ag_npu.py`）
- **allreduce 压测**：1000 次 × 200MB bf16，中位 2.1ms（首帧 1.74s 为通信建立，正常）
- **小模型 DDP**：50M 参数 MLP × 3000 iter，稳定 3.2ms/iter 无退化
- **通信链路**：bootstrap 广播 RootInfo、HCCS 同节点通信、E_PARA/网卡根因均未复现

## 三、遗留问题（已解决，2026-08-24 下午）

**现象（原问题）**：Qwen2.5-1.5B（1.54B params, bf16）+ DDP + flagcx backend 训练，
**step ≈ 765±20 起**每步耗时 0.1s → ~2.4s，rank0 loss 在固定两个值间精确交替
（模型参数停止更新的表象），吞吐单调下滑。原生 `backend="hccl"` 同脚本训练
2481 步全程健康（loss 1.9472、5428 tok/s）。

**根因（源码核查 + 分层实验确认）**：`flagcxCannEvent` 构造时 `aclrtCreateEvent`
但没有析构函数 → **每次 collective 泄漏一个 aclrtEvent**（DDP 下每步 ~bucket 数
≈120 个），累积到 ~765 步时 ACL event 资源耗尽：`aclrtCreateEvent` 变慢（每步变慢）
+ event 状态污染（loss 交替）。CUDA 版用 `at::cuda::CUDAEvent`（RAII 自动销毁），
CANN 版漏了析构。

**修复链（commit d296824，逐步收敛）**：
1. collective 跑在 torch_npu 当前流（guardImpl + dlsym NPUStream::stream()，0686805）
2. `flagcxWork::wait()` 的 event block 改用 collective 流（不再走自建流 fallback）
3. future 完成语义：`markCompleted` 前 `event->synchronize()`（PyTorch 2.10 Reducer
   用 `bucket.future_work->wait()`，必须让 future 完成 == 通信完成）
4. work 构造 / fn / event record 三处统一为同一次 stream 解析（避免解析间切流错位）
5. **`~flagcxCannEvent()` 补析构销毁 aclrtEvent（根因修复）**

**验证结果（修复后完整复跑）**：Qwen2.5-1.5B DDP + flagcx backend
**2481 步全程稳定**：最终 loss **1.9501**（与原生 hccl 1.9472 / B 线 1.9436 相当）、
吞吐 **4157 tok/s**（原生 hccl 的 ~77%，同步等待语义损失可后续优化）、
checkpoint 正常保存、进程干净退出。

**诊断资产**（runtime-team `dev/device-context/benchmarks/`）：
`test_ag_npu.py`（allgather）、`test_work_sem.py`（work/future 完成语义实测）、
`train_qwen_1_5b_npu.py`（训练，hccl/flagcx 双后端可切换，一行 sed 复现分界实验）。

## 四、路线状态说明

- **A 线（主线）**：厂商插件（torch_npu）+ FlagGems + FlagCX —— 本分支即 A 线
  基线；torch_npu 原生 hccl 训练闭环已达成（见上），FlagCX 适配以本分支推进。
- **B 线（预研支线）**：torch_fl 路线降级，原分支 `kistich/ascend-flagcx-adapt`
  保留不删、不再承担交付；其中 stream 语义等结论已吸收进本分支。
