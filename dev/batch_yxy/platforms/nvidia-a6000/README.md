# NVIDIA A6000 数据（归档）

本目录只保存**从验证记录中提取的结构化数据**，原始产物不在本仓。

| 项 | 值 |
| --- | --- |
| 平台 | 4 × RTX A6000（49140 MiB），驱动 595.71.05 |
| 测试卡 | GPU3（`CUDA_VISIBLE_DEVICES=3`） |
| Conda | `vllm` 0.21.0；`vllm_add` Python 3.12.10 / torch 2.13.0+cu130 |
| 模型 | Qwen3-Embedding-0.6B（本地权重） |
| 数据来源 | `docs/flagscheduler_验证记录.md`：2026-10-21、2026-10-25（两节）、2026-11-04（两节） |
| 原始产物 | `artifacts/nvidia_gpu3_20261025/`、`artifacts/nvidia_gpu3_perf_20261025/`、`artifacts/torch_gpu3_20261104/`（在 GPU 机器上，未随本仓提交） |

## 1. Embedding A/B/C 性能对照（2026-10-25，192 条 mixed）

同机同卡闭环离线回放，每路径独立进程、同 warmup。

| 路径 | 吞吐 req/s | p50 / p99 ms | 相对 A |
| --- | --- | --- | --- |
| A 原路径直连（一次 `LLM.embed`） | 467.6 | 410.5 / 410.5 | 基线 |
| B 统一入口 batch=8 | 411.4 | 242 / 429 | -12% |
| B32 统一入口 batch=32 | **561.3** | **206 / 315** | **+20%** |
| C 自定义 scheduler（与 B 同策略，插件继承原生） | 417.4 | 233 / 421 | ≈B（1.5% 噪声） |

- 决策层微基准：`BatchCoordinator` submit 2.3–2.6 µs/请求、派发 1.9–3.5 µs/请求，占比 <0.3%。
- 限制：每配置单次运行、闭环回放、未测在线流式到达；<5% 按噪声处理。
- **注意**：此表的 C 是“自定义 scheduler 插件”，不是长度分桶；分桶策略的 NVIDIA 数据见第 4 节（torch-predict 后端，不可直接横比）。

## 2. Embedding 数值一致性（2026-10-25，36 条）

| 检查 | 结果 |
| --- | --- |
| 原生 scheduler | 36/36 completed，1024 维、有限、L2 范数 1.0 |
| 自定义 scheduler | 36/36 completed（启动参数注入 `scheduler_cls=FlagScheduler`） |
| 两路径数值对照 | max abs diff **2.83e-3**，max cos 偏差 **1.28e-4** |
| ID 映射 | 两轮各自与输入 36 ID 严格一一对应 |
| 资源回收 | 退出后显存恢复 1 MiB |

## 3. torch-predict CPU → GPU（2026-11-04，300 条）

执行段 p50：DeepFM 4.01→2.17 ms（-46%）、DIN 4.65→1.89 ms（-59%）、MMoE 0.96→0.81 ms（-15%）。
DeepFM 的 GPU e2e 反升 266 ms，归因于单批 ~258 ms 执行尖刺在单批串行下放大为排队。

## 4. 组批策略扫描（2026-11-04，torch-predict，300 条 mixed）

| batch 策略 | 吞吐 req/s | e2e p50 ms |
| --- | --- | --- |
| 1 | 76.5 | 110.6 |
| 8 | 103.1 | — |
| 32 | 109.1 | — |
| 32 + padding ≤ 0.3 | **123.4（+61%）** | **40.8（-63%）** |

批不变性：fp32 max 3.9e-6；fp16 max ~6e-2 / mean ~2e-4。
**该表是 torch-predict 工作负载**，与 embedding A/B/C 不是同一后端，只能作为“NVIDIA 上组批/分桶方向”的参考。
