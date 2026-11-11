# rag_ljy 项目总结与预期交付

更新日期：2026-09-16

## 项目目标

在 `dev/rag_ljy` 下构建面向连续查询的 Ascend NPU RAG 工作流：文档入库后，依次完成问题向量化、混合检索、粗排、精排和答案生成，并返回可追溯的来源。Elasticsearch 与 Milvus 作为可切换的完整检索后端，在相同数据和模型条件下比较结果与性能。

模型执行统一使用一个 NPU 容器 `rag-ljy-vllm-910c`，镜像为 **`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`**。Embedding、reranking 和后续 generation 均计划在该容器内运行；Elasticsearch、Milvus 为独立 CPU 数据库容器。模型文件保存在 `/mnt/raid/jliu171/models`。

## 当前进展

- **Elasticsearch 检索精排已实际跑通。** 已下载并运行 `Qwen/Qwen3-Embedding-0.6B` 和 `Qwen/Qwen3-Reranker-0.6B`，完成 1024 维向量入库、BM25 + dense 检索、RRF 融合与精排。现有 5 条样本文档返回 Top 5；问题 `How does hybrid search work?` 的目标文档排名第 1，rerank score 为 0.99944717。
- **常驻查询 API 已实现。** 模型在服务启动时加载一次，后续请求复用，无空闲卸载；进程停止时释放。当前采用单进程、单推理线程，同一时刻处理一个查询，提供健康检查和并发保护；目标 NPU 上的连续查询与显存常驻待验证。
- **双数据库接口已实现，Milvus 启动问题已自行修复。** 两个后端共用文档切块、模型、RRF 和精排逻辑；Milvus 的同数据集入库与检索结果仍需记录。当前查询输出止于排序片段，生成模型尚未选择和接入。

## 工作流实现

入库由 `scripts/ingest_test_data.py` 完成：读取文档、切块、生成稳定 `chunk_id`、调用 embedding 模型，再写入所选数据库。重复写入相同数据时更新相同 ID。

查询由 `src/rag_engine/pipeline.py` 编排：问题 embedding → 所选数据库的 BM25 和 dense 双路检索 → RRF 粗排 → Qwen3 reranker 精排。默认每路召回上限 50，RRF 后保留最多 30 条，最终返回最多 5 条片段，包含文本、来源和各阶段排名及分数。一次查询仅使用一个数据库。

`scripts/query.py` 用于单次查询，执行完成后退出；`scripts/serve_queries.py` 和 `src/rag_engine/server.py` 用于持续接收 HTTP 请求并保持模型常驻。当前两个模型通过 Transformers + `torch_npu` 直接推理，尚未调用 vLLM generation server。

## 预期完成的交付

| 交付项 | 完成标准 |
|---|---|
| 持续查询服务 | 连续处理多个不同问题，复用同一组模型；正常空闲时不卸载，显式停止后释放资源，并记录实际显存变化 |
| 双后端完整验证 | 同一数据集分别写入 Elasticsearch / Milvus，两个后端均完成混合检索和精排，保存可比较的输出 |
| 可评估的数据集 | 从现有 5 条样本扩充到约 100 条 mock 文档，配套问题及预期相关文档标注 |
| 完整 RAG 回答 | 选择 generation 模型并在现有 NPU 容器内接入，将问题与精排上下文交给模型，返回回答及来源；无相关证据时说明不足 |
| 量化评估 | 记录 Recall@K、MRR / nDCG、P50 / P95 延迟、QPS、显存峰值和错误率，区分首次加载与后续查询开销 |

下一步优先完成 Milvus 入库查询和模型常驻实跑验证，再扩充评估数据、接入答案生成。当前没有生产负载下的性能数据，也尚未设定具体 QPS 或延迟验收阈值。
