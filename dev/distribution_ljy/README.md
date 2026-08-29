# Elasticsearch + Ascend NPU RAG（distribution_ljy）

> **状态：🔄 方案已确定，待实现** ｜ 本文档记录实现顺序和验收标准

## 目标

构建一条可运行的 RAG 链路：Elasticsearch 同时完成 BM25 稀疏检索和向量检索，`torch_npu` 在 Ascend NPU 上完成 embedding、rerank 和大模型生成。

第一版只使用 Elasticsearch，不同时部署 Milvus 或 FAISS。

## 整体工作流

### 文档入库

```text
原始文档
  -> 清洗和切块
  -> torch_npu embedding
  -> Elasticsearch 保存文本、元数据和向量
```

### 在线问答

```text
用户问题
  -> torch_npu 生成 query embedding
  -> Elasticsearch BM25 + dense vector 两路召回
  -> RRF 合并并完成 coarse ranking（粗排），取 Top 20~50
  -> torch_npu reranker 完成 fine ranking（精排）
  -> 取 Top 3~5 构造 prompt
  -> NPU 上的大模型生成答案
  -> 返回答案和引用来源
```

## 容器划分

第一版使用两个容器，先减少部署复杂度：

| 容器 | 设备 | 职责 |
|---|---|---|
| `rag-elasticsearch` | CPU | 保存 chunk；执行 BM25、向量检索和 RRF |
| `flagos-distribution_ljy-dev-910c` | Ascend NPU | RAG API、embedding、rerank、prompt 构造和生成 |

- Elasticsearch 不映射任何 `/dev/davinci*` 设备，也不占用 NPU/DrvMng 客户端名额。
- NPU 容器中以 `torch_npu` 作为设备后端，代码使用 `device="npu"`。
- 同一 Python 进程不同时加载 `torch_npu` 和 `torch_fl`。
- 链路稳定后，再按吞吐和故障隔离需要把生成服务拆成独立 NPU 容器。

## Elasticsearch 数据约定

索引暂定为 `rag-chunks-v1`，每条记录代表一个 chunk：

| 字段 | 类型 | 用途 |
|---|---|---|
| `chunk_id` | `keyword` | chunk 唯一标识 |
| `document_id` | `keyword` | 原文档标识 |
| `document_version` | `integer` | 文档版本 |
| `title` | `text` | 标题检索 |
| `text` | `text` | BM25 检索和 prompt 上下文 |
| `source_uri` | `keyword` | 答案引用来源 |
| `metadata` | `object` | 业务过滤字段 |
| `embedding` | `dense_vector` | dense vector 检索 |

`embedding.dims` 必须等选定 embedding 模型后，按模型真实输出维度填写，不能先随意写死。

## 实现顺序

1. **部署 Elasticsearch**
   - 在 `docker-compose.yml` 增加单节点、固定版本的 Elasticsearch 服务。
   - 仅绑定宿主机 `127.0.0.1:9200`，使用 named volume 持久化数据。
   - 增加健康检查；开发环境验证通过后再讨论生产集群和安全配置。
2. **建立索引**
   - 添加创建 `rag-chunks-v1` mapping 的脚本。
   - 写入少量测试 chunk，先验证 BM25 查询和元数据过滤。
3. **实现文档入库**
   - 加载文档、清洗、切块并生成稳定的 `chunk_id`。
   - 用 `torch_npu` embedding 模型批量生成向量并 bulk 写入 Elasticsearch。
4. **实现混合召回和 coarse ranking（粗排）**
   - 同时执行 BM25 和 dense vector 检索。
   - 使用 RRF 合并两路结果并进行 coarse ranking，输出 Top 20~50 候选。
5. **实现 NPU fine ranking（精排）**
   - 用 `torch_npu` reranker 对 query/chunk 对打分。
   - 选出 Top 3~5，并保留 `source_uri` 和检索分数。
6. **实现生成**
   - 将精选 chunk 注入 prompt。
   - 调用 Ascend NPU 上的生成模型，返回答案、引用和耗时。
7. **端到端验收**
   - 固定一组问题和期望来源，记录召回率、排序结果、答案引用和端到端延迟。
   - 验证无相关文档时不会伪造来源。

## 第一阶段出口标准

- `docker compose` 可以启动 Elasticsearch 和 NPU 开发容器。
- 容器重启后 Elasticsearch 数据仍然存在。
- 可以完成测试文档的切块、向量化和入库。
- 一个查询可以完成 BM25 + dense vector + RRF + rerank + generation。
- 返回结果包含答案和可追踪的 `document_id`、`chunk_id`、`source_uri`。
- `torch_npu` 与 Elasticsearch 的故障能够分别定位，不依赖 Milvus 或 FAISS。

## 任务看板

| # | 任务 | 状态 | 依赖 | 出口标准 |
|---|---|---|---|---|
| 1 | Elasticsearch Compose 服务 | ⬜ | — | 健康检查通过，数据持久化 |
| 2 | Index mapping 和测试数据 | ⬜ | 1 | BM25 查询返回预期 chunk |
| 3 | 文档切块与 NPU embedding | ⬜ | 2 | 批量入库并可向量检索 |
| 4 | BM25 + vector + RRF coarse ranking（粗排） | ⬜ | 3 | 返回 Top 20~50 可解释候选 |
| 5 | NPU reranker fine ranking（精排） | ⬜ | 4 | 输出排序后的 Top 3~5 |
| 6 | Prompt 与 NPU generation | ⬜ | 5 | 返回答案和引用 |
| 7 | 端到端评测 | ⬜ | 6 | 记录质量和延迟基线 |

> 状态图例：⬜ 待开始 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消

## 目录约定

```text
dev/distribution_ljy/
├── README.md           # 方案、任务看板和运行说明
├── docker-compose.yml  # NPU 开发容器及后续 Elasticsearch 服务
├── .env.example        # 环境变量模板
├── docs/               # 设计、调研和验证记录
├── scripts/            # 建索引、导入数据等运维脚本
├── src/                # ingestion、retrieval、rerank、generation 代码
└── benchmarks/         # 检索质量和端到端延迟测试
```

## 当前开发容器启动方式

当前 `docker-compose.yml` 只定义了 NPU 开发容器；Elasticsearch 服务将在任务 1 中加入。

```bash
cd /home/jliu171/runtime-team/dev/distribution_ljy
cp .env.example .env
docker compose -f ../compose.base.yml -f docker-compose.yml up -d
docker ps | grep flagos-distribution_ljy-dev-910c
```

进入开发容器：

```bash
docker exec -it flagos-distribution_ljy-dev-910c bash
```

## 工作原则

- 先让最小端到端链路跑通，再拆分服务或做性能优化。
- embedding、reranker 和生成模型及其版本必须固定并记录。
- 索引 mapping 变更使用新索引版本，不直接破坏已有数据。
- 不修改宿主机驱动或系统配置。
- 多卡测试前先用 `npu-smi` 确认卡空闲。
- 公共红线和 DrvMng 限制以仓库主 README 为准。
