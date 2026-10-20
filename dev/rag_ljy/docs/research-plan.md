# 研究计划：pre-generation RAG serving 的阶段调度与准入

更新：2026-10-10。本文档定义研究方向、实验矩阵、基线与论文骨架；
实跑证据与结果索引见 [artifacts/README.md](../artifacts/README.md)。

## 1. 范围界定

研究对象是生成阶段之前的 RAG 服务管线：

```text
vLLM embedding(NPU) → BM25+dense 双库检索(CPU) → RRF → Transformers 精排(NPU)
```

- 单机单卡：物理 NPU 7（容器内 `npu:0`）+ Milvus/Elasticsearch 容器。
- 不做：多厂商加速器抽象（device-context 仅用其内存/错误接口）、生成阶段、
  多节点分布式、训练。
- 论文 claim 明确限定为 pre-generation serving，生成接入列为 future work。

## 2. 研究问题

核心问题（RQ0）：在单加速器 + CPU 检索库的 RAG 管线中，阶段级队列、
跨请求合批与准入控制如何联合影响吞吐与尾延迟？

- RQ1（合批收益边界）：各阶段的 batch size × 延迟 × 吞吐曲线是什么形状？
  精排 batch-fill 等待何时值得、何时不值得？现有观察——双请求并行提交
  排队延迟增加而吞吐不变——的成因与边界条件。
- RQ2（准入策略）：请求数准入 vs 基于 NPU HBM/主机内存水位的动态准入，
  在过载下的尾延迟、503 率与恢复行为差异。
- RQ3（质量-延迟权衡）：qrels 指标下，top-k/RRF 常数/精排深度/引擎/dtype
  配置空间的质量-延迟 Pareto 前沿；引擎与数值精度变化引起的排名漂移幅度。

## 3. 与现有工作的差异

| 工作 | 侧重点 | 本项目差异 |
|---|---|---|
| RAGCache (ASPLOS'25)、CacheBlend (EuroSys'25) | KV cache 复用、检索/prefill 流水线 | 面向生成阶段；不覆盖检索管线调度 |
| DistServe (OSDI'24)、SplitWise | prefill/decode 解耦 | 单模型，不是多阶段异构资源 DAG |
| Parrot (OSDI'24)、SGLang、Teola | LLM 程序 serving | 检索库不是一等调度资源 |
| HeteroQuery (SIGMOD'25) | LLM 应用 CPU/GPU 资源分配 | 无阶段队列准入，无 NPU 实测 |
| ServerlessRAG、RAGServe (arXiv) | 扩展性行为、配置建模 | 无调度机制与水位准入 |

差异化定位：把 pre-generation 检索管线作为一等调度对象，在非 CUDA
加速器上给出首批公开表征数据。

## 4. 基线定义

| 基线 | 配置 | 状态 |
|---|---|---|
| B0 串行参考 | 逐请求直通，无并发；作为正确性与延迟下界参照 | 已有（200 例脚本中的串行参考） |
| B1 共享线程池 | 全部阶段共用一个 executor，无独立阶段队列 | 需实现（加配置开关） |
| B2 阶段队列（现状） | 独立有界 FIFO + 固定 worker（2/4/1），仅计数准入 | 已实现 |
| B3 B2+精排合批 | 跨请求 rerank batch dispatcher，可调等待窗口 | 需实现 |
| B4 B2+水位准入 | device-context 内存遥测驱动准入，替代纯计数 | 需实现 |

B3/B4 组合为完整方案 B5。所有基线共用同一固定查询集与同一服务端代码路径，
仅切换配置，避免实现差异污染对比。

## 5. 实验矩阵

### 5.1 变量轴

| 轴 | 取值 |
|---|---|
| 负载到达 | 闭环 1/2/4/8 并发；开环泊松 λ=0.5–8 req/s |
| 检索库 | milvus / elasticsearch |
| 检索模式 | sequential / concurrent（同请求内 BM25/dense 并行） |
| embedding 提交并发 | 1 / 2 / 4（`--embedding-workers`） |
| 阶段队列容量 | 1 / 4 / 8 / 16 |
| 精排合批窗口 | off / 5 / 10 / 20 ms（B3） |
| 准入策略 | 计数 / 水位（B4） |
| 检索配置 | top-k 25/50/100、RRF k 20/60/100、精排深度 10/30/50 |
| 数值/引擎 | bf16/fp32；vLLM vs Transformers embedding |

### 5.2 实验项

| 编号 | 对应 RQ | 内容 | 指标 | 产物 |
|---|---|---|---|---|
| E1 | 动机 | 现状 B2 在 1/2/4/8 并发下的阶段分解 | 各阶段排队/执行 p50/p95/p99 | `artifacts/research/e1-load/` |
| E2 | RQ1 | 精排与 embedding 的 batch 扫描（B3 窗口×并发） | 延迟-吞吐曲线、合批率 | `artifacts/research/e2-batch/` |
| E3 | RQ1 | vLLM 引擎侧合批取证（engine 日志+API 侧重叠对齐） | forward 合批证据 | `artifacts/research/e3-vllm-batching/` |
| E4 | RQ2 | 过载场景：计数 vs 水位准入（B2 vs B4） | 503 率、p99、恢复时间、水位曲线 | `artifacts/research/e4-admission/` |
| E5 | RQ3 | 配置空间扫描（8 的检索配置轴） | nDCG@10/MRR@10/Recall@50 vs 延迟 Pareto | `artifacts/research/e5-quality/` |
| E6 | RQ3 | 跨引擎/dtype 排名漂移：固定查询重复运行 | 排名重叠率、nDCG 差值 | `artifacts/research/e6-consistency/` |
| E7 | 汇总 | B0/B1/B2/B3/B4/B5 同负载横评 | goodput@SLO、尾延迟 | `artifacts/research/e7-comparison/` |

E1–E4 为系统实验（同质量配置），E5–E6 为质量实验（同性能配置），两者解耦。
SLO 在 E1 数据上标定（如 e2e p99 ≤ 串行参考的 2 倍），E7 报告 goodput@SLO。

### 5.3 公平性措施

- 客户端并发与库内检索并发解耦（当前 200 例脚本两者同时变化，不能沿用）。
- 共享主机噪声：每组 A/B 交错重复 ≥3 轮，报告均值与离散度。
- 每次运行保存 health 采样、stage 日志与源码快照，沿用现有 evidence 格式。

## 6. 需要补齐的机制与工具

1. qrels 质量评测脚本：NFCorpus qrels 按 query ID 关联，nDCG@10/MRR@10/Recall@50。
2. 开环负载发生器：泊松到达、固定种子、p50/p95/p99 分阶段报告。
3. B1 共享线程池开关；B3 跨请求精排 dispatcher（注意 asyncio.shield 语义）。
4. device-context 内存遥测接入 `/health` 与准入路径。
5. 查询集扩充：50 条固定集用于回归；质量与负载实验扩至 NFCorpus test 全部
   318 条有 qrels 的查询；语料规模不足时评估补充 MS MARCO 子集。

## 7. 论文骨架

- 题目候选：StageQueue: SLO-Aware Stage Scheduling and Admission for
  Pre-Generation RAG Serving（非 CUDA 加速器为测试环境而非 claim）。
- Claim 列表：
  1. 首个单加速器上混合检索 RAG 管线的阶段级行为表征（含合批无效的反直觉观察）；
  2. 阶段队列 + 跨请求精排合批 + 水位准入的调度设计；
  3. 合批收益边界的排队模型分析与实验验证；
  4. 质量感知的配置 Pareto 与跨引擎排名漂移测量。
- 章节：Intro / Background & Motivation(E1) / Design(B1–B5) / Model(RQ1 分析) /
  Evaluation(E2–E7) / Related Work / Discussion & Limitations。
- 计划图表：管线与队列结构图；并发-排队延迟-吞吐动机图；batch 扫描曲线；
  过载下两种准入对比；质量-延迟 Pareto；基线横评表。

## 8. 里程碑

| 里程碑 | 内容 | 依赖 |
|---|---|---|
| M1 | qrels 评测脚本 + 50 条基线质量报告 | qrels 数据 |
| M2 | 开环负载发生器 + E1 完成 | 无 |
| M3 | B1 实现，B0/B1/B2 横评 | M2 |
| M4 | B3 dispatcher + E2/E3 | M2 |
| M5 | B4 水位准入 + E4 | device-context 内存接口 |
| M6 | E5/E6 质量实验 | M1 + 查询扩充 |
| M7 | E7 汇总 + 论文初稿 | M3–M6 |

## 9. 风险与未决问题

- 吞吐提升可能有限：若合批收益小，论文重心转向收益边界的表征与模型
  （负结果 + 解释本身可发表，但需 B1 对照支撑）。
- vLLM 在昇腾上的 forward 合批行为不可控：E3 取证失败则精排 dispatcher
  仍是独立贡献，embedding 侧改为报告性发现。
- device-context 内存反馈接口未接入：B4 有工程风险，需先做接口可行性验证。
- 共享主机测量噪声：已见的缓存/负载影响需靠交错重复与轮次报告控制。
- 单机单卡外部效度有限：claim 收窄为单加速器节点内的调度问题。
