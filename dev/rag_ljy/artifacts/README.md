# 实验结果与证据

更新：2026-10-09。所有结果按用途保存，原始记录保留；新运行写入具日期和唯一ID的目录。

## 当前有效结果

| 内容 | 文件 |
|---|---|
| device-context设备内存探测：跨进程分配/释放验证、120项回归通过 | [单文件内存探测报告](checks/memory_probe/2026-10-14_132823-f0ad80/report.md) |
| 在线同时/错峰各8请求、超限拒绝与恢复、跨阶段重叠及22项调度测试 | [在线队列实测证据](checks/queue_evidence/2026-10-09_140744-milvus-30d953/report.md) |
| 独立有界阶段队列：200次完整API调用，双库各50/50结果一致 | [队列测试报告](benchmarks/queued-pipeline-2026-10-09_134251-78a48d/report.md) |
| 当前双库全量重建参数与验证 | [current_rebuild](rebuilds/vllm-rebuild-2026-10-09/manifest.json) |
| 单容器Milvus重启、全量ID和向量验证 | [current_store_verification](rebuilds/vllm-rebuild-2026-10-09/post_restart_verification.json) |
| 重建后vLLM文档/查询：200例成功，两库各50/50排名一致 | [current_vllm_retrieval](rebuilds/vllm-rebuild-2026-10-09/comparison-20261009T045201Z-2a44ec/report.md) |
| 当前vLLM API：串行/同时/错峰6例通过 | [current_multi_request](checks/multi_request/2026-10-09_134804-milvus-174bdb/summary.json) |
| 固定50条NFCorpus查询 | [frozen_queries](../data/queries/nfcorpus-50.jsonl) |

## 历史比较与诊断

| 内容 | 文件 |
|---|---|
| 旧PyTorch向量：双库×双模式200例 | [pytorch_retrieval_baseline](benchmarks/concurrent_vs_sequential_2026-09-23/comparison-20260923T111642Z-231050/report.md) |
| 旧文档向量下的跨引擎对比；vLLM稳定但与PyTorch排名不同 | [vllm_vs_pytorch](benchmarks/vllm-baseline-200-2026-10-09_044059Z-afcc8d/summary.json) |
| Triton taskqueue控制实验与首个层级差异证据 | [vllm_taskqueue_diagnosis](diagnostics/vllm-mismatch-diagnosis-2026-10-09/summary.json) |
| taskqueue=true：30次固定token输入 | [vllm_fixed_inputs](diagnostics/runs/fixed-input-2026-10-09_113719-taskqueue-enabled-60f0e3.json) |
| taskqueue=true：18次pipeline复现 | [vllm_fixed_pipeline](diagnostics/runs/diagnosis-2026-10-09_113725-bfd164/summary.json) |

## 目录与引用

- `benchmarks/`：检索/跨引擎对比，保留原始耗时和排名。
- `rebuilds/`：数据库重建、计数、重启与新基线。
- `checks/embeddings/`：向量相似度冒烟检查。
- `checks/memory_probe/`：设备内存读数与验证，每次运行仅保存一个小型报告。
- `checks/multi_request/`：HTTP检查的完整结果，包括历史失败。
- `checks/queue_evidence/`：同时/错峰、超限与恢复验证；完整响应、服务端阶段时间线及health队列采样。
- `diagnostics/`：差异定位、控制实验；`runs/`保存raw复现。

下游程序读取 [index.json](index.json) 的稳定键，路径相对 `dev/rag_ljy`。
迁移表 [relocations.json](relocations.json) 可翻译原路径：选择最长匹配旧前缀，替换为新前缀。
历史JSON内的运行路径及快照保持原样；不是当前可运行入口。
旧运行的 `SHA256SUMS` 仍可在其目录使用，新输入校验见 [固定查询](../data/queries/README.md)。

如何复跑见 [评测指南](../docs/evaluation.md)，文件迁移清单见 [路径表](../docs/paths.md)。
