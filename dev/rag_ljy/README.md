# RAG 检索与阶段调度（rag_ljy）

更新：2026-11-11。入口供新同学查看；详细操作、设计和结果按用途分开。

项目最新状态统一记录在根目录 [status.md](status.md)，该文件必须保留在此处并持续更新。

当前实现：NFCorpus → vLLM 文档/查询向量化 → BM25 + dense → RRF →
Qwen3 精排，支持多个 HTTP 请求通过独立有界队列推进各阶段，vLLM embedding默认2路提交并发。文档与查询默认使用
Qwen3-Embedding-0.6B 的 vLLM 引擎；精排使用 Qwen3-Reranker-0.6B 的
Transformers 实现。默认接口返回排序片段；新增可选OneRec生成阶段、上下文token预算和生成显存预算，
开启与验证范围见[OneRec接入](docs/generation.md)；真实单请求与双库三模式功能已通过，含生成的85%专项未执行。
查询API默认通过device-context后台采集设备内存，在 `/health.device_memory` 暴露缓存读数。
可开启内存准入原型：余量不足返回503、低水位恢复接收，并保留每请求全链路预算；
启动配置及验证范围见[运行指南](docs/operations.md)。

## 从这里开始

| 想做什么 | 文件 |
|---|---|
| 启动三个容器、embedding 引擎和查询 API | [运行指南](docs/operations.md) |
| 接入OneRec生成与单die NPU配置 | [OneRec接入](docs/generation.md) |
| 理解源码、检索模式与多请求调度 | [实现结构](docs/architecture.md) |
| 复用固定50条查询、运行测试、找保存结果 | [评测指南](docs/evaluation.md) |
| 查看已完成内容、证据与后续工作 | [status.md](status.md) |
| 查看研究方向、实验矩阵与论文骨架 | [研究计划](docs/research-plan.md) |
| 了解 device-context 接入 | [接入说明](docs/device-context.md) |
| 找文件移动后的路径 | [路径变更表](docs/paths.md) |
| 查看实验记录与原始证据 | [结果索引](artifacts/README.md) |

## 目录

```text
rag_ljy/
├── src/rag_engine/      # 主实现：模型适配、阶段调度、数据库接口
├── scripts/            # runtime / data / benchmarks / checks / diagnostics
│   └── check_multi_request/  # 多请求HTTP校验入口
├── tests/              # 硬件无关测试
├── docker_compose/     # 数据库与NPU容器配置
├── data/               # 样本文档、固定查询集
├── docs/               # 当前指南；history/ 保存此前记录
└── artifacts/          # 具日期的结果、日志与复现快照
```

基础NPU Compose保留`/dev/davinci1`；OneRec overlay通过`RAG_NPU_DIE`选择单die，
当前改试物理卡0的`/dev/davinci0`，容器内仍为`npu:0`。单独`docker start`不会更新设备映射。
`rag-ljy-milvus` 为内嵌 etcd 的单容器；`rag-ljy-elasticsearch` 为单容器。
NFCorpus 原始3,633文档分为6,543块，两个数据库保存相同块和向量。

代码入口见 [scripts/README.md](scripts/README.md)。完整语料、模型和数据库
持久化目录位于宿主机 RAID；`.venv`、缓存和包元数据为本地生成文件。
