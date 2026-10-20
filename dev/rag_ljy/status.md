# RAG 当前状态

文档更新：2026-10-20（北京时间）。最近完整回归与设备内存实测：2026-10-14。

**本文件必须保留在 `dev/rag_ljy/status.md`，作为项目当前状态的固定入口，随开发持续更新。**

目前交付形态是可运行的 **RAG检索与阶段调度原型**：支持在线多请求、独立有界阶段队列、
混合检索、精排及设备内存探测。内存感知准入和答案生成尚未实现，尚未达到生产部署标准。
本文根据当前源码及已保存的实测证据整理；以下验证日期是实际运行日期。

## 已实现功能

| 模块 | 当前实现 |
|---|---|
| 数据处理与入库 | NFCorpus原始3,633篇文档按1200字符、重叠150字符切为6,543块，保留文档与块ID；同一批文档向量复用于两个数据库。 |
| 文本向量化 | 文档和查询默认使用Qwen3-Embedding-0.6B的vLLM服务，输出1024维向量；CLI、API、入库与基准使用同一embedding工厂。 |
| 混合检索 | 通过统一数据库接口选择Milvus或Elasticsearch，在所选数据库内执行BM25和dense检索，支持顺序／并发两种模式，再进行RRF融合。 |
| 结果精排 | Qwen3-Reranker-0.6B通过Transformers执行，返回排序片段、来源及分数；当前精排后端仍是Transformers。 |
| 在线多请求 | FastAPI接受同时及错峰到达的HTTP请求；每个请求拥有独立asyncio任务、请求ID及中间结果，按自身依赖推进。 |
| 独立阶段队列 | 向量化、检索和精排各使用一个有界`asyncio.Queue`；每个任务通过自己的Future接收结果，完成后即可进入下一阶段，无需整组等待。 |
| 数量准入与背压 | 默认最多8个在途请求，包含执行与等待；超限返回503及`Retry-After: 1`；已接收请求遇阶段队列满时等待空间。 |
| 取消、异常与关闭 | 请求异常反馈到对应请求；HTTP等待取消后，已接收工作继续执行并占用容量；服务关闭前排空请求和阶段队列。 |
| 设备适配 | 设备选择、可见性检查与初始化通过`dev/device-context` runtime；底层Torch、厂商扩展、驱动及运行库由平台环境提供。 |
| 设备内存探测 | 通过`runtime.memory_stats()`后台读取总量／已用／可用内存，默认每秒采样；`/health.device_memory`暴露缓存读数、时间、年龄、过期标记及错误状态。 |
| 可观察性 | `/health`展示真实数据库、检索模式、在途数量、阶段队列状态及设备内存；服务端日志记录请求ID和阶段排队／执行时间。 |
| 数据库部署 | Milvus采用内嵌etcd的单容器部署；Elasticsearch单容器部署；数据通过RAID目录持久化。 |

阶段worker默认为：向量化2个HTTP worker、检索4个worker、精排1个worker；每个阶段8个等待槽。
本地模型在其初始化线程执行，内存探测使用独立采样线程。向量化worker并发指向vLLM提交请求，
不等同于已验证这些请求进入同一次模型forward。

每个请求的查询、向量、候选与结果目前通过参数和局部变量传递，采用固定阶段依赖；
尚未抽象为通用流程上下文或动态工作流引擎。

## 已保存的验证证据

| 日期 | 验证范围 | 结果与证据 |
|---|---|---|
| 2026-10-09 | 双库全量重建与持久化 | 两个后端各6,543块，全量ID及向量核对通过；Milvus重启后数据保留。[重建记录](artifacts/rebuilds/vllm-rebuild-2026-10-09/manifest.json)、[重启后验证](artifacts/rebuilds/vllm-rebuild-2026-10-09/post_restart_verification.json)。 |
| 2026-10-09 | 检索阶段顺序／并发比较 | 固定50条查询×两数据库×两模式，共200次检索成功，两库各50/50排序一致；测量前预计算向量，不包含模型阶段。[检索报告](artifacts/rebuilds/vllm-rebuild-2026-10-09/comparison-20261009T045201Z-2a44ec/report.md)。 |
| 2026-10-09 | 独立队列完整API链路 | 同一50条查询×两库×两种配置，共200次完整向量化→检索→精排调用全部成功；两库各50/50有序结果、内容及容差内分数一致。[完整链路报告](artifacts/benchmarks/queued-pipeline-2026-10-09_134251-78a48d/report.md)。 |
| 2026-10-09 | 同时／错峰、超限及恢复 | 同时8请求、错峰8请求均成功且匹配串行参考；9请求突发接受8个、预期拒绝1个，之后恢复成功；本轮34次调用中33次成功、1次预期503。[简要报告](artifacts/checks/queue_evidence/2026-10-09-summary/report.md)、[详细证据](artifacts/checks/queue_evidence/2026-10-09_140744-milvus-30d953/report.md)。 |
| 2026-10-09 | 跨请求阶段推进 | 33个成功请求均有完整三阶段记录且自身依赖顺序正确；观察到60对跨请求阶段区间重叠。[服务端时间线](artifacts/checks/queue_evidence/2026-10-09_140744-milvus-30d953/stage_timeline.csv)。 |
| 2026-10-14 | 设备内存及跨进程占用 | NPU总量62,748MiB，初始化后单次读取约0.7–1.2ms；独立进程分配128MiB时已用158→348MiB，退出后回到158MiB，增量包含运行环境开销。[单文件实测报告](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |
| 2026-10-14 | 真实API后台内存采样 | 173次health检查均得到正常且未过期的缓存读数，采样时间戳持续刷新；2条串行及同样2条并发查询全部成功，结果一致。这173次是HTTP检查次数，并非173次独立设备采样。[实测报告](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |
| 2026-10-14 | 完整回归 | `python -m pytest tests -q`：120项通过，覆盖设备路由、队列、API、内存指标校验、采样失败恢复及线程隔离等。[回归记录](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |

固定输入是[同一组50条NFCorpus查询](data/queries/nfcorpus-50.jsonl)，200次调用不代表200条不同查询。
有序结果比较采用分数相对容差1e-4、绝对容差1e-6；一致性验证不等于基于qrels的检索质量评估。
完整API的两种配置同时改变了检索模式与客户端提交并发度，不能单独据此计算新增队列的开销。
该轮未观察到明显吞吐提升，不能将阶段重叠表述为已取得吞吐加速。

vLLM曾出现相同输入的向量不稳定问题，当前启动器为embedding进程设置
`TRITON_ENABLE_TASKQUEUE=true`；固定输入及pipeline控制实验已保存。
定位记录见[诊断结果](artifacts/diagnostics/vllm-mismatch-diagnosis-2026-10-09/summary.json)。

## 当前运行配置与依赖

以下是部署配置；容器和服务的实时在线状态通过health检查确认。

| 项目 | 当前配置 |
|---|---|
| 开发容器 | `rag-ljy-vllm-910c`；物理NPU7，容器内逻辑`npu:0`。 |
| vLLM embedding服务 | `http://127.0.0.1:8089`；独立模型进程。 |
| RAG查询API | `http://127.0.0.1:8088`；入口`POST /query`，状态`GET /health`。 |
| Elasticsearch | `rag-ljy-elasticsearch`，宿主机`127.0.0.1:9201`映射到容器9200；索引`nfcorpus-chunks-v1`。 |
| Milvus | `rag-ljy-milvus`，`127.0.0.1:19530`；集合`nfcorpus_chunks_v1`。 |
| 计算分工 | 两个数据库的BM25／dense检索均在CPU执行；向量化与精排模型在NPU执行。 |
| device-context | 目前通过`PYTHONPATH`加载`dev/device-context/prototype`，用于设备初始化与内存探测。 |
| 推理环境 | 当前昇腾环境依赖镜像中的PyTorch／torch_npu、vLLM-Ascend及匹配驱动／运行库；跨平台完整验证尚未完成。 |
| Python打包 | 已有`pyproject.toml`，包名`flagos-rag-ljy`；独立安装交付的命令入口与依赖验收仍待完善。 |

启动时须使用现有NFCorpus索引／集合名称；操作命令、环境变量和端口说明见[运行指南](docs/operations.md)。

## 当前限制与下一步

- **内存感知准入尚未实现。** 当前内存读数只用于观察，请求数量上限仍控制准入；主机内存监测、请求内存预算及基于内存的接收／拒绝策略待补充。
- 内存接口统一返回值的统计范围尚未由后端元数据明确，故标记`backend_reported_unverified`；Ascend兼容回退可能只反映当前进程分配量。跨进程测试仅验证了本次环境，后续准入需确认读数来源、范围和时效。
- 下一步先尝试可配置的内存阈值准入原型，初步验证超限限制与回落恢复，记录测试结果；水位值及更短采样间隔尚未确定。
- 尚未验证vLLM实际forward合批，尚未实现跨请求精排合批或vLLM精排后端；当前单请求候选batch不等同于跨请求batch。
- 阶段队列为进程内存对象，暂无重启后的请求恢复或跨进程任务分发；请求截止时间、持续负载及更多故障场景仍需完善。
- 尚未完成NFCorpus qrels的nDCG／MRR／Recall质量评测；中文analyzer配置及其重建验证待按需求开展。
- 生成模型尚未接入，目前API输出为精排后的检索片段。
- 独立pip安装包的交付验证待完成，设备运行环境、模型与数据库仍需分别准备。

## 实现与文档入口

| 内容 | 入口 |
|---|---|
| 阶段队列与请求调度 | [scheduler.py](src/rag_engine/scheduler.py)、[server.py](src/rag_engine/server.py) |
| 设备适配与内存采样 | [device.py](src/rag_engine/device.py)、[memory_probe.py](src/rag_engine/memory_probe.py) |
| 内存探测复跑 | [check_memory.py](scripts/checks/check_memory.py) |
| 服务启动 | [serve_embeddings.py](scripts/runtime/serve_embeddings.py)、[serve_queries.py](scripts/runtime/serve_queries.py) |
| 结构、操作与评测 | [实现结构](docs/architecture.md)、[运行指南](docs/operations.md)、[评测指南](docs/evaluation.md)、[device-context接入](docs/device-context.md) |
| 测试与原始证据 | [结果索引](artifacts/README.md)、[机器可读索引](artifacts/index.json) |

后续功能、配置或验证结果发生变化时，直接更新本文件，并链接对应的实现与证据。
