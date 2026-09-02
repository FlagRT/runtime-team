# flagcx vs hccl 吞吐差距分析（910C 同构）

> 日期：2026-08-26 · 环境：910C 新容器 flagos-hliu553-dev-910c（torch_npu 2.10.0，2×NPU）
> 结论先行：**差距主因 = FlagCX backend 每次 collective 的 CPU launch 开销是原生 hccl 的 36 倍**；
> 总通信 CPU 耗时 flagcx 是 hccl 的 18 倍。修复点在 FlagCX torch 插件/adaptor 的 per-call 开销。

## 一、背景

- 同脚本同模型（Qwen2.5-1.5B 双卡 DDP），仅 `dist.init_process_group(backend=...)` 不同：
  - `hccl`：torch_npu 官方 backend，直调 HCCL —— 5428 tok/s
  - `flagcx`：FlagCX torch 插件（HCCL adaptor）—— 4157 tok/s（本次验收 4245）
- 差距稳定在 ~77%（flagcx/hccl），即 flagcx 慢 ~23-30%

## 二、实验设计

- 训练脚本参数化：`BACKEND`（hccl/flagcx）、`MAX_STEPS`、`PROFILE`（torch.profiler，CPU activity）
- 60 步 × 2 backend，profiler schedule：wait=10 / warmup=5 / active=20（35 步有效记录）
- 统计：allreduce 类事件次数、CPU self 耗时、单次平均

## 三、结果

| 指标 | flagcx | hccl | 差异 |
|---|---|---|---|
| allreduce 事件类型 | 1 | 2 | — |
| **调用次数（active 20 步）** | **850** | **1700** | flagcx 为 hccl 的 1/2 |
| **CPU 总耗时（self）** | **615.8 ms** | **34.1 ms** | **flagcx 18×** |
| **单次平均 CPU 开销** | **0.72 ms** | **0.02 ms** | **flagcx 36×** |
| tok/s（s55，含 profiler 开销） | ~3150 | ~3570 | ~12% |

无 profiler 时（8/24 全量验证）：4245 vs 5428 tok/s（77%）。

## 四、结论

1. **主因：per-collective CPU launch 开销**。flagcx 每次 allreduce 的 CPU self 时间 0.72ms vs hccl 0.02ms（36 倍）。按 850 次/35 步 ≈ 24 次/步折算，每步光 CPU launch ≈ 17ms；而 hccl 全程仅 34ms/35 步 ≈ 1ms/步。**flagcx 的 CPU 通信开销是 hccl 的 18 倍**，直接拖慢每步（通信无法与计算充分重叠）。
2. **排除嫌疑：coalesced 不融合**。flagcx 调用次数（850）反而少于 hccl（1700），说明调用次数不是瓶颈（hccl 的 2× 次数可能是 bucket 处理/内部拆分差异，但每次极快，无碍）。
3. **开销来源推断**：FlagCX 调用链（ProcessGroup → flagcx C++ 插件 → HCCL adaptor）比 torch_npu 原生（ProcessGroup → HCCL）多一层封装；每 collective 的事件/流管理、参数校验、可能经过 uniRunner/DAG 调度路径，累积成 0.72ms/次。
4. **修复方向（FlagCX 上游可贡献）**：
   - 精简 adaptor per-call 路径（事件/流处理、参数校验热路径）
   - 优化 collective 与当前流的绑定（避免额外同步点）
   - 对照 xliu969 host-runtime stream 方案（B 线已验证 torch_npu 当前流语义）

## 五、可复现

```bash
# 新容器（venv + LD_LIBRARY_PATH 就绪）
cd /workspace/dev/device-context/benchmarks
BACKEND=flagcx PROFILE=1 MAX_STEPS=60 torchrun --nproc_per_node=2 --master_port=29513 train_qwen_1_5b_npu.py
BACKEND=hccl   PROFILE=1 MAX_STEPS=60 torchrun --nproc_per_node=2 --master_port=29514 train_qwen_1_5b_npu.py
# 观察 [profiler] backend=... allreduce calls / total_self_cpu_ms / avg_self_cpu_ms
```

训练脚本 `train_qwen_1_5b_npu.py` 已参数化（BACKEND/MAX_STEPS/PROFILE），作为 A 线验收资产保留。
