# RAG 工作流进展

更新日期：2026-09-16

## 已完成进展

**已跑通 Elasticsearch + Qwen3 Embedding + Qwen3 Reranker 的样本检索精排链路，并将 NFCorpus 的 3,633 条原始文档切为 6,543 个 chunks，写入 Elasticsearch 和 Milvus。已通过两个数据库的实时计数核对，每个后端均有 6,543 条 chunk 记录。常驻查询 API 已完成实现；NFCorpus 上的检索质量及性能指标尚待记录。**

| 成果 | 当前进展 |
|---|---|
| NPU 模型运行环境 | 已启动单一 vLLM Ascend 容器，加载 embedding、reranker 两个模型并完成查询 |
| Elasticsearch 数据链路 | 已完成服务启动、1024 维索引创建、样本文档检索及 NFCorpus 的 6,543 个 chunks 入库 |
| NFCorpus 数据集 | 已完成 corpus 转换、切块、真实向量生成及双后端入库；保留原始 document ID，queries / qrels 用于后续评测 |
| 混合检索与精排 | 已完成 BM25 + dense 双路检索、RRF 融合、Qwen3 精排，返回片段、来源及各阶段分数 |
| 常驻查询服务 | 已实现模型启动时加载一次、后续请求复用、无空闲卸载；目标 NPU 连续运行待验证 |
| 检索耗时测量 | 已实现可选阶段计时及 benchmark runner，支持 sequential / concurrent、retrieval-only / full-query、warmup、重复测试和 raw CSV / 统计 JSON |
| 两路搜索并发 | 已实现同一数据库的 BM25 / dense 并发请求，使用常驻两个工作线程；保留顺序模式，RRF / rerank 等待两路完成 |
| 双数据库适配 | 已实现 Elasticsearch / Milvus 统一接口，并核对 NFCorpus 在两个后端的入库计数；NFCorpus 查询、精排与结果对比待记录 |
| 环境依赖修正 | 已修正项目依赖范围，并在安装脚本中按镜像版本约束推理栈，避免覆盖镜像要求的 Transformers 版本 |

## 新数据集：NFCorpus 双后端入库

NFCorpus 来自 Hugging Face 的 `BeIR/nfcorpus`，相关性标注来自
`BeIR/nfcorpus-qrels`。本次只入库 **corpus**；**queries** 保存评测问题，
**qrels** 保存问题与相关文档的对应关系，不作为知识文档写入数据库。

| 项目 | 当前结果 / 位置 |
|---|---|
| 原始文档数 | 3,633 条 |
| 入库输入文件 | 1 个转换后的 JSONL，包含 3,633 条原始文档记录 |
| 切块参数 | 最大 1200 字符，重叠 150 字符 |
| 切块后规模 | 6,543 个 chunks，不是 6,543 个文件 |
| 向量 | Qwen3-Embedding-0.6B，1024 维真实 embedding，运行于 `npu:3` |
| Elasticsearch | `nfcorpus-chunks-v1`，实时计数核对为 6,543 条记录 |
| Milvus | `default` 数据库中的 `nfcorpus_chunks_v1`，实时计数核对为 6,543 条记录 |
| 原始语料 | `/mnt/raid/jliu171/data/nfcorpus/corpus/` |
| 转换输出 | `/mnt/raid/jliu171/data/nfcorpus/processed/nfcorpus_documents.jsonl` |
| 容器内入库输入副本 | `/tmp/nfcorpus_documents.jsonl` |
| 相关性标注 | `/mnt/raid/jliu171/data/nfcorpus/qrels/` 下的 train / dev / test TSV |

[convert_beir_corpus.py](scripts/convert_beir_corpus.py) 在宿主机 `sol` 环境中
读取 Parquet 并生成项目 JSONL，保留原始 `document_id`，便于对照 qrels。
[ingest_corpus_both.py](scripts/ingest_corpus_both.py) 使用同一份文档和切块参数，
加载一次 embedding 模型，每个 chunk 的向量仅计算一次，复用于两个后端；
embedding batch size 为 8，数据库写入 batch size 为 64。

本次使用独立的 NFCorpus index / collection，没有替换原有 5 条样本文档资源。
一次查询仍只访问所选的一个数据库，由该后端提供 BM25 和 dense 两路检索，支持顺序和并发两种模式。
两个数据库的记录数已核对一致，但这不等于已完成检索质量、完整精排或性能评测。
原始及转换数据保存在 RAID 上；容器内 `/tmp` 副本只是入库输入，入库后不再依赖它。

## 已取得的查询结果（原有 5 条样本文档）

查询：`How does hybrid search work?`。样本规模：**5 条文档**。以下为原有样本的实际终端返回结果，**不是 NFCorpus 的评测结果**，分数保留 8 位小数。

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

- **文档处理**：[corpus.py](src/rag_engine/corpus.py) 转换 BEIR corpus 并保留原始文档 ID；[documents.py](src/rag_engine/documents.py) 按字符切块，通过文档 ID、版本、片段序号和内容生成 SHA-256 `chunk_id`，支持同一数据重复写入时更新相同 ID。
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

### 顺序检索 baseline 测量

[timing.py](src/rag_engine/timing.py) 为公共检索与 pipeline 添加可选 wall-clock
计时，不改变默认返回结果。记录 `embedding_ms`、`bm25_ms`、
`dense_ms`、`search_ms`、`rrf_ms`、`rerank_ms`、`total_ms`。其中 `search_ms`
直接测量两路搜索的整体区间：顺序模式为 BM25 开始到 dense 返回；并发模式
为提交前到两路都完成。不包含 RRF，不是估算的两路 `max()`。

[benchmark_search.py](scripts/benchmark_search.py) 加载模型一次并复用，保留旧
`benchmark_sequential.py` 入口，默认 sequential，模式需显式选择 concurrent：
retrieval-only 预计算 query vectors 后测量搜索 / RRF；full-query 测量完整
embedding → 搜索 → RRF → rerank。模型初始化、预计算及 warmup 与正式样本
分开，不纳入稳态查询延迟。逐条请求，固定 seed 和参数，支持重复测试。

[prepare_benchmark_queries.py](scripts/prepare_benchmark_queries.py) 已通过实际
Parquet 预览验证：从 NFCorpus 的 3,237 条 queries 中，按 `test.tsv` 选择
323 条 test queries。结果保存 raw CSV、包含 mean / P50 / P95 / P99 的
summary JSON 及本次 queries；错误单独统计，未执行的阶段留空。
正式测量产物复制到 `/mnt/raid/jliu171/data/benchmarks/`，不保存在 home 项目目录。
完整命令见 [README.md](README.md#顺序检索性能测量先建立-baseline)。

当前尚未完成全量 test queries 的性能 baseline，不能据短跑推断数据集性能或
并发收益。计时为 pipeline Python 调用边界，不是 HTTP 延迟或
NPU kernel profiler，也不是多请求负载下的最大 QPS。

顺序测量实现阶段通过 **53 个硬件无关测试**，并在 `npu:3` 完成单问题 live smoke test：
`What are the effects of statins on breast cancer survival?`。每个 run 为 1 次
warmup + 2 次正式记录，均无失败；下列数字仅用于验证计时链路，**不是全量
NFCorpus baseline，也不是两个数据库的性能对比结论**。

| Smoke 路径 | BM25 mean (ms) | Dense mean (ms) | Search mean (ms) | Total mean (ms) |
|---|---:|---:|---:|---:|
| Elasticsearch / full-query | 17.162 | 18.130 | 35.306 | 363.087 |
| Milvus / retrieval-only | 4.589 | 4.523 | 9.121 | 9.257 |

验证产物位于容器内 `/tmp/rag-ljy-measurement-validation/` 的独立 run 子目录；
正式数据集 benchmark 使用 `/tmp/rag-ljy-benchmarks/`，再复制到自己的 RAID 目录。

### BM25 / dense 并发实现

`execution_mode=concurrent` 使用常驻 `ThreadPoolExecutor(max_workers=2)`，
先提交两个数据库请求，再等待两路都完成，最后执行原有 RRF / rerank。
仅数据库请求线程并发；embedding 和 rerank 仍在同一个 NPU 推理线程中运行。
Branch 使用线程内局部计时，完成后合并；`search_ms` 包含实际调度和等待开销。
任一路失败均等待另一请求结束后传播异常，不静默返回部分结果。

查询 CLI / API 的默认模式为 concurrent，支持 `--execution-mode sequential`
回到原顺序路径；benchmark 的默认模式保持 sequential，保存结果时标记实际
模式。Search pool 跨请求复用并在退出时关闭，原有 API 单 query 并发保护不变。
已有数据库无需重新入库；旧常驻 API 需要重启进程才能采用新实现。

并发改动已通过 **63 个硬件无关测试**，覆盖两路重叠执行、RRF 结果不变、
模型线程不变、pool 复用 / 关闭、失败后的另一分支排空及 benchmark 模式标记。
全量 323 条 test queries 的顺序 / 并发性能对比仍待采集，不据单问题短跑
宣称性能提升比例。

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
2. 基于已入库的 NFCorpus 完成 Elasticsearch / Milvus 的混合检索和精排，保存同一组问题的可对比结果。
3. 使用 NFCorpus queries / qrels 评估 Recall@K、MRR / nDCG；按原始 `document_id` 去重后进行文档级评测，避免将同一文档的多个 chunks 重复计入。此前讨论的约 100 条 mock 数据尚未生成，目前已改用真实且带相关性标注的 NFCorpus。
4. 使用已实现的 sequential / concurrent 路径采集全量 NFCorpus 性能对比，记录两路检索、合并检索区间及完整查询的 mean / P50 / P95；固定相同语料、向量和检索参数，预热后重复测试。后续另做成功 QPS、CPU / NPU 内存峰值及错误率的负载测量。当前已实现并发，尚未取得全量性能 baseline 或并发收益数据。
5. 接入 generation 模型，将精排片段作为上下文生成带来源的最终回答。当前输出仍止于排序片段。
