# P800历史证据与材料发布核验

硬件实验日期：2026-10-29。材料发布整理：2026-11-03。本目录只含精简结果、审核与原始文件哈希，不含模型权重、全量张量或dev2运行时制品。

| 材料 | 说明 |
|---|---|
| [报告](../P800接入与安全回退验证-20261029.md) | 环境、模型、准入和失败边界 |
| [summary](summary.json) | 首轮SiLU资格与三路径 |
| [extensions](extensions-summary.json) | sigmoid/tanh及DeepFM |
| [model](model-summary.json) | Qwen首步目标梯度超差，原生及控制路径 |
| [workload](workload-summary.json) | 模型审核：DeepFM限定批准、Qwen拒绝 |
| [onerec](onerec-summary.json) | OneRec候选失败、保留原生和严格阻断 |
| [11-03发布核验](publication-audit-20261103.json) | 离线构建安装94项CPU控制、case 34项控制、3条模型审核证据哈希、10个运行时文件与实测wheel一致 |
| [集成入口](../../cases/kunlun-unary/README.md) | baseline/demo、脚本及审核配置；依赖尚未发布的精确dev2制品 |

原始summary中的publication/local uncommitted等字段保留10-29生成时含义，不改写历史证据。11-03发布材料不是新增硬件实验，也不表示数值失败已修复。dev0不支持全部dev2接口；缺少匹配制品时，本目录可用于审查但不能宣称一键复现。

CPU控制使用模拟/原生路径，不证明目标内核正确。新构建wheel哈希与历史实测wheel不同，只有所列10个运行时源码逐字节一致；不把两件制品说成同一个包。未开启P800 GEMM兼容补丁的旧资格不能直接用于env/v1或新机。
