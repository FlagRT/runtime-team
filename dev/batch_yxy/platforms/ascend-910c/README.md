# 昇腾 910C 数据（归档）

| 项 | 值 |
| --- | --- |
| 平台 | 16 × Ascend910（本机 `npu-smi` 26.1.1，驱动 26.1.1），测试使用单卡 `davinci6`（容器内索引 0，`Ascend910_9382`） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| 运行时 | vLLM 0.20.2 / vllm-ascend v0.20.2rc1-a3，Python 3.11.15 |
| 模型 | Qwen3-Embedding-0.6B（本地无副本，ModelScope 下载，1.19 GB / 311 tensors，只读挂载） |
| 关键开关 | `enforce_eager=True`；基准默认**未开** prefix caching；单卡、`ASCEND_RT_VISIBLE_DEVICES=0` |
| 分支 | `xianyiyuan/fix-vllm-embed-and-compat`（含 `Already borrowed` 修复、vLLM 版本兼容、910C 脚本数据生成修复） |

## 1. 组批矩阵（n=192 mixed，`benchmarks/bench_matrix_compare.py`）

A = 一次直连（单次 `LLM.embed` 192 条）；B = 按 batch 档位只封口不分组；C = 同档位 + 长度分桶。
每配置预热 1 轮，3 轮取中位数；A 自身跨轮数值差为 0（同组成重复调用确定）。

### 1.1 `max_total_tokens=32768`

| 配置 | req/s | e2e p50 | p95 | p99 | 批次 | bs p50 | padding | vs A max_abs / cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | **110.5** | 1737.7 | 1737.7 | 1737.7 | 1 次调用 | n/a | n/a | 基线 |
| B32 | 89.9（-19%） | 1522.3 | 2134.4 | 2134.4 | 6 | 32 | 0.562 | 3.8e-3 / 0.99980 |
| B64 | 91.4（-17%） | 1309.9 | 1999.2 | 2100.5 | 6 | 34 | 0.479 | 3.2e-3 / 0.99980 |
| B128 | 91.9（-17%） | 1296.3 | 1999.5 | 2089.6 | 6 | 34 | 0.479 | 3.0e-3 / 0.99980 |
| C32 | 79.7（-28%） | 555.4 | 2408.0 | 2408.0 | 12 | 17 | 0.115 | 3.2e-3 / 0.99974 |
| C64 | 82.3（-26%） | 528.2 | 2332.3 | 2332.3 | 10 | 17 | 0.085 | 4.1e-3 / 0.99975 |
| C128 | 85.3（-23%） | 530.4 | 2250.2 | 2250.2 | 9 | 17 | 0.093 | 4.1e-3 / 0.99975 |

> 该轮 B64/B128 被 32768 token 预算钳成与 B32 相同的 6 批（bs≈34），batch 档位未真正拉开。

### 1.2 `max_total_tokens=131072`（只跑 B/C，2 轮）

| 配置 | req/s | e2e p50 | p95 | 批次 | bs p50 | padding |
| --- | --- | --- | --- | --- | --- | --- |
| B32 | 89.7 | 1517.7 | 2139.5 | 6 | 32 | 0.562 |
| B128 | **96.1（-13% vs A）** | 1411.2 | 1997.1 | 2 | 64 | 0.568 |
| C32 | 78.4 | 1764.2 | 2449.8 | 10 | 16 | 0.126 |
| C128 | 86.2 | 551.7 | 2226.3 | 6 | 12 | 0.111 |

## 2. A/B/C 基准（n=64，修复后代码）

`benchmarks/bench_embedding_batching.py --executor vllm --scenario mixed --num 64 --enforce-eager`

| 路径 | req/s | e2e p50 | 正确性 vs A |
| --- | --- | --- | --- |
| A | 98.5 | 649.6 | 基线 |
| B（batch=32） | 81.9（-17%） | 781.6 | max_abs 2.5e-3 / cos 0.99979 |
| C（分桶） | 70.3（-29%） | **241.5** | max_abs 3.5e-3 / cos 0.99974 |

## 3. 功能与调度验收

| 检查 | 结果 |
| --- | --- |
| `examples/npu_functional_check.py`（24 条） | CHECK PASS：全返回、ID 一一对应、1024 维；6 批 `[8,3,1,4,4,4]`；跨桶 0；baseline vs dynamic cos min 0.9997 |
| `examples/sched_engine_910c_vllm.py`（32 条 × 3 到达序） | 各 6 批、32 结果、1024 维；封口 `batch_limit 3 / wait_limit 3`；legacy vs scheduled cos min/mean 1.0000（逐元素一致 0/32，批组成不同所致） |
| FlagScheduler `vllm-embed`（32 条，2 档优先级） | **32/32 completed**，ID 全匹配、1024 维；execution_ms p50 82.3（首批含预热 6.86 s，故 p95/max 同值）；queue_ms p50 7041.9（一次性灌入） |
| 并发提交实验（`--pipelined-workers 2`） | **挂起**：650 线程停在 `S`、CPU 2.6%、AICore 0%；确认 vLLM 同步 `LLM.embed` 非线程安全 |
| 修复前对照 | 未修复时 `count_cost` 与 `LLM.embed` 共用 tokenizer，首批后即 `RuntimeError: Already borrowed`（4/32 后中断） |

## 4. 原始数据文件

| 文件 | 内容 |
| --- | --- |
| `data/matrix-tokenbudget-32768.json` | 1.1 节全部逐轮记录、内嵌向量对比 |
| `data/matrix-tokenbudget-131072.json` | 1.2 节逐轮记录 |
| `data/bench-vllm-mixed-n64.json` | 第 2 节 A/B/C 原始输出 |
| `data/flagscheduler-32req-summary.json` | 第 3 节 FlagScheduler 32 请求聚合（状态/维度/队列与执行分位） |

限制：单卡单实例；matrix 第一轮 3 轮、第二轮 2 轮；未开 prefix caching；绝对吞吐不可与其他平台直接横向比较。
