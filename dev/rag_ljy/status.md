# RAG 工作流进展

更新日期：2026-09-10

## 已完成进展

**已跑通 Elasticsearch + Qwen3 Embedding + Qwen3 Reranker 的完整检索精排链路，基于 5 条样本文档返回按相关性排序的 Top 5 片段。常驻查询 API 已完成实现。**

| 成果 | 当前进展 |
|---|---|
| NPU 模型运行环境 | 已启动单一 vLLM Ascend 容器，加载 embedding、reranker 两个模型并完成查询 |
| Elasticsearch 数据链路 | 已完成服务启动、1024 维索引创建、样本文档向量化入库及检索 |
| 混合检索与精排 | 已完成 BM25 + dense 双路检索、RRF 融合、Qwen3 精排，返回片段、来源及各阶段分数 |
| 常驻查询服务 | 已实现模型启动时加载一次、后续请求复用、无空闲卸载；目标 NPU 连续运行待验证 |
| 双数据库适配 | 已实现 Elasticsearch / Milvus 统一接口；Elasticsearch 已实跑，Milvus 实跑待完成 |
| 环境依赖修正 | 已修正项目依赖范围，并在安装脚本中按镜像版本约束推理栈，避免覆盖镜像要求的 Transformers 版本 |

## 已取得的查询结果

查询：`How does hybrid search work?`。样本规模：**5 条文档**。以下为实际终端返回结果，分数保留 8 位小数。

| 最终排名 | 文档 | BM25 排名 | Dense 排名 | RRF score | Rerank score |
|---|---|---:|---:|---:|---:|
| 1 | `hybrid-search` | 1 | 1 | 0.03278689 | 0.99944717 |
| 2 | `ranking` | — | 3 | 0.01587302 | 0.63703084 |
| 3 | `rag-overview` | — | 4 | 0.01562500 | 0.03114383 |
| 4 | `generation` | — | 2 | 0.01612903 | 0.01590639 |
| 5 | `ascend-runtime` | — | 5 | 0.01538462 | 0.00555492 |

直接解释 hybrid search 的文档位于第 1，BM25 和 dense 均将其排在首位。精排将 `ranking` 从 dense 第 3 提升至最终第 2，将 `generation` 从 dense 第 2 降至最终第 4，完成了候选顺序调整。首条 RRF 分数与两路首位一致：`2 / 61 ≈ 0.03278689`。

`—` 表示该片段未出现在返回的 BM25 排名中。Rerank score 是相关性信号，不作为准确率。当前结果验证了小样本链路；尚未形成整体检索质量或性能指标。

## 实现概要

### 入库与查询

```text
入库：文档 → 切块 → Qwen3 embedding → 片段、元数据及向量写入数据库
查询：问题 → query embedding → BM25 + dense → RRF → Qwen3 reranker → 排序片段 JSON
```

- **文档处理**：[documents.py](src/rag_engine/documents.py) 按字符切块，通过文档 ID、版本、片段序号和内容生成 SHA-256 `chunk_id`，支持同一数据重复写入时更新相同 ID。
- **Embedding**：[embedding.py](src/rag_engine/embedding.py) 使用 `Qwen/Qwen3-Embedding-0.6B`，执行最后有效 token pooling 和 L2 归一化；query 添加检索 instruction。
- **检索融合**：[retrieval.py](src/rag_engine/retrieval.py) 合并 BM25 与 dense 结果，按 `sum(1 / (60 + rank))` 计算 RRF 分数；[pipeline.py](src/rag_engine/pipeline.py) 编排完整查询。
- **精排**：[reranker.py](src/rag_engine/reranker.py) 使用 `Qwen/Qwen3-Reranker-0.6B`，依据 query/document 提示的 `yes` / `no` logits 计算分数并降序排序。
- **数据库适配**：[stores/](src/rag_engine/stores/) 提供统一接口。Elasticsearch 使用 BM25 + cosine kNN；Milvus 使用 BM25 sparse index + cosine dense AUTOINDEX。一次查询使用一个后端，复用相同的融合与精排逻辑。

### 常驻查询

[scripts/serve_queries.py](scripts/serve_queries.py) 与 [server.py](src/rag_engine/server.py) 实现 FastAPI 常驻服务；[query_runtime.py](src/rag_engine/query_runtime.py) 统一 CLI / API 的模型初始化与输出格式。

服务采用 **1 个进程、1 个专用推理线程、1 组 embedding / reranker 模型**。启动时加载模型，后续请求复用，不按查询释放，也不进行空闲卸载；进程停止后结束模型生命周期。已实现健康检查、请求校验和并发保护，同一时刻处理 1 个查询，重叠请求返回 503 供重试。目标 NPU 上的连续查询、显存常驻及持续运行时长待记录。

### 当前配置规模

以下为实现参数，不是性能实测值。

| 参数 | 默认值 |
|---|---:|
| Embedding 输出维度 | 1024 |
| 切块长度 / 重叠 | 1200 / 150 字符 |
| 每路检索候选上限 | 50 |
| RRF 常数 / 融合后候选上限 | 60 / 30 |
| 精排返回上限 | 5 |
| Embedding / reranker batch size | 8 / 4 |
| 两模型输入长度上限 | 各 8192 tokens |
| NPU 推理 dtype | BF16 |

## 运行镜像

RAG 模型执行统一使用 **一个 NPU 容器**：

| 项目 | 当前环境 |
|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` |
| 容器 | `rag-ljy-vllm-910c` |
| Python | 镜像提供的 Python 3.11.15 |
| 模型执行 | Transformers + PyTorch / `torch_npu` |
| 模型存储 | 宿主机 RAID：`/mnt/raid/jliu171/models` |
| 数据服务 | 独立 CPU Elasticsearch / Milvus 服务 |

当前 embedding、reranking 在该镜像内通过 Transformers 直接推理；常驻查询 API 由本项目实现。尚未接入 vLLM generation server。

## 下一阶段

1. 完成常驻 API 的 NPU 连续查询验证，记录加载耗时、后续请求耗时和空闲显存占用。
2. 完成 Milvus 同数据集入库与查询，保存与 Elasticsearch 可对比的结果。
3. 扩充带相关文档标注的数据集，评估 Recall@K、MRR / nDCG；此前讨论的约 100 条 mock 数据尚未生成。
4. 采集 P50 / P95 延迟、QPS、显存峰值和错误率；目前尚无生产负载测量数据。
5. 接入 generation 模型，将精排片段作为上下文生成带来源的最终回答。当前输出仍止于排序片段。
