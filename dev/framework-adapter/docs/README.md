# 接入与回退报告索引

整理：2026-11-11。当前完成情况与发布状态看[STATUS](../STATUS.md)，运行路径看[HANDOFF](../HANDOFF.md)。这里按主题定位材料，不重复维护进度或把旧“未推送”带入今天。

## 优先阅读

本期对外统一入口：[阶段进展与验证入口-20261111](阶段进展与验证入口-20261111.md)，含三模型、算子扩展、基座修复、数值定位及dev4发布证据。

| 主题 | 报告 | 证据与执行入口 |
|---|---|---|
| PPU Qwen embedding，11-11 | [独立资格、模型三路与复现](../cases/embedding-config/README.md#qwen-embedding独立资格及模型三路) | [核验摘要](evidence-qwen-embedding-20261111/summary.json)、[实测baseline/demo](../cases/embedding-config/demo/README.md)；原始张量位置及审核输入看HANDOFF |
| embedding统一配置，11-11 | [实现、模型回归与差异记录](../cases/embedding-config/README.md) | [核验摘要](evidence-embedding-config-20261111/summary.json)；承接独立接口，新代码资格按新身份处理，原件位置见HANDOFF |
| PPU DeepFM embedding，11-10 | [两输入接入与模型验证](PPU-DeepFM-embedding接入验证-20261110.md) | [9轮核验摘要](evidence-embedding-20261110/summary.json)、[case与审核](../cases/embedding/README.md)；原件、构建和冻结输入见HANDOFF |
| 三模型清单与ReLU扩展，11-10 | [清单、选择及验证](三模型算子清单与ReLU接入验证-20261110.md) | [完整逐项表](evidence-operator-inventory-20261110/算子清单-20261110.md)、[case](../cases/operator-inventory/README.md)；实际调用与源码可用性分列，原始证据路径见HANDOFF |
| PPU Qwen/OneRec模型验证，11-10 | [两模型接入与安全准入](PPU两模型接入与安全准入验证-20261110.md) | [证据索引](evidence-ppu-models-20261110/README.md)、[case](../cases/ppu-models/README.md)、[实际baseline/demo](../cases/ppu-models/demo/README.md)；成功与数值拒绝分别记录 |
| PPU DeepFM训练与准入，11-10 | [模型训练与安全准入](PPU-DeepFM训练与安全准入验证-20261110.md) | [摘要](evidence-ppu-deepfm-20261110/summary.json)、[case](../cases/ppu-deepfm/README.md)；原生保留成功，直接候选参数超差，失败审核保留 |
| PPU算子资格与三路验证，11-10 | [接入与安全回退验证](PPU算子接入与安全回退验证-20261110.md) | [资格与审核](../cases/ppu-unary/review-20261110/)、[三路结果](evidence-ppu-20261110/routes-result.json)、[case](../cases/ppu-unary/README.md) |
| 训练v3补包审核与模型三路，11-06 | [Qwen回归报告](训练v3补包审核与Qwen回归-20261106.md) | [摘要与SHA](evidence-overlay-regression-20261106/summary.json)、[baseline/demo](../cases/cann-shmem-validation/demo/20261106-overlay/README.md)；补包身份及限定模型审核，承接同日最小修复验证 |
| 训练v3补依赖，11-06 | [cann-shmem修复验证](cann-shmem修复验证-20261106.md) | [摘要](evidence-cann-shmem-20261106/summary.json)、[case与baseline/demo](../cases/cann-shmem-validation/README.md)；补充原缺失问题的修复组合与1.7.0源校验问题 |
| 公共训练回调候选，11-04 | [接口对接与本地验证](公共训练回调接口对接-20261104.md) | [摘要](evidence-training-callback-20261104/summary.json)、[CPU输出](evidence-training-callback-20261104/cpu-tests.txt)；本地dev4提案，旧CLI保留 |
| 公共入口安全检查，11-04 | [旧审核失效与严格拒绝](公共训练入口安全检查-20261104.md) | [摘要](evidence-public-safety-20261104/summary.json)、[baseline/demo](../cases/ascend-official-unary/demo/20261104-public-safety/README.md)；后续增量，本地未推 |
| dev3源码发布，11-04 | [发布审查](dev3发布审查-20261104.md) | [核验摘要及脚本](evidence-dev3-release-20261104/)、[依赖锁](../package.lock.json) |
| 公共训练入口模型审核，11-04 | [接入与三路回归](公共训练入口模型审核接入-20261104.md) | [摘要](evidence-public-entry-20261104/summary.json)、[baseline/demo](../cases/ascend-official-unary/demo/20261104-public-entry/README.md)；替代同日勘察的“未实施”状态，对应dev3源码候选 |
| 新P800 DeepFM三路回归，11-04 | [新机环境与三路训练](P800新机三路回归-20261104.md) | [摘要及7轮结果](evidence-p800-regression-20261104/summary.json)、[实际baseline/demo](../cases/kunlun-unary/demo/20261104-deepfm-on/README.md)；GEMM补丁关闭 |
| dev2源码候选发布，11-03 | [发布审查](dev2发布审查-20261103.md) | [摘要与核验脚本](evidence-dev2-release-20261103/)、[精确依赖锁](../package.lock.json)；构建/CPU审查，不是新硬件成绩 |
| P800三算子及三模型，10-29 | [接入与安全回退验证](P800接入与安全回退验证-20261029.md) | [P800 case](../cases/kunlun-unary/README.md)、[单算子](evidence-p800-20261029/summary.json)、[扩展/DeepFM](evidence-p800-20261029/extensions-summary.json)、[Qwen](evidence-p800-20261029/model-summary.json)、[模型审核](evidence-p800-20261029/workload-summary.json)、[OneRec](evidence-p800-20261029/onerec-summary.json) |
| 910C双rank更新前阻断，10-28 | [步骤门控验证](双rank步骤门控验证-20261028.md) | [精简证据](evidence-step-guard-20261028/summary.json)、[配置](../demo/training-configs/README.md) |
| 910C环境差异、扩展及真实模型，10-23 | [环境准入与算子扩展](环境准入与算子扩展验证-20261023.md) | [证据索引](evidence-expansion-20261023/README.md)、[官方环境case](../cases/ascend-official-unary/README.md) |
| 历史dev0候选，10-28 | [候选交付记录](候选交付记录-20261028.md) | 该日期commit与实测身份保留；现行版本看依赖锁及发布审查 |

## 问题与设计

| 材料 | 日期及用途 |
|---|---|
| [三通道接入方案](https://github.com/FlagRT/framework-adapter/blob/6ee91294e7ab028f2e804225482f58d24c8ccd5a/docs/三通道接入方案.md)、[接口约定](https://github.com/FlagRT/framework-adapter/blob/6ee91294e7ab028f2e804225482f58d24c8ccd5a/docs/接入接口约定.md) | 11-03独立仓0.1评审稿，现行方案入口；三路线及实际/拟议API分开，不是功能全部验收 |
| [算子组对接Issue](handoff/算子组对接Issue.md) | 10-09原稿、10-23 SDPA增量；清单/接口/替换边界，不等于跨组事项已确认 |
| [三级回退方案](handoff/原型与三级回退方案.md) | 10-09 v0.2设计快照；其中实现状态已过时，现状看STATUS |
| [09-22实验复现说明](handoff/实验复现说明.md) | 10-09整理的历史推理实验，不是当前安装包使用手册 |
| [资格审核说明](资格验证与审核流程.md) | 协调仓历史说明；现行包配置以独立功能仓docs/资格验证与审核流程.md为准 |
| [训练v3基线与回归](训练v3基线与回归准备-20261016.md) | 目标编译失败、模型未运行；与旧官方成功环境分开 |

## 验证沿革

| 日期 | 文档 | 与后续的关系 |
|---|---|---|
| 10-28 | [门控修补前失败](双rank失败联动验证-20261028.md)、[候选发布准备](候选发布准备-20261028.md) | 修补结果/发布结果分别看上表；旧失败保留 |
| 10-24 | [双rank设备初始化阻塞](双rank失败联动验证-20261024.md) | 当时未进入故障测试，不能写成通过 |
| 10-20 | [训练入口配置接入](训练入口配置接入验证-20261020.md)、[拆仓证据](evidence-package-20261020/README.md) | 旧官方环境安装包验证；功能代码随后在独立仓维护 |
| 10-18 | [公共适配器实测](910C公共适配器实测-20261018.md)、[开发记录](公共适配包开发记录-20261018.md) | 包初期/真实NPU与sigmoid小模型；默认模式失败仍保留 |
| 10-14 | [设备v0.3训推与SiLU](910C-v03推理与训练验证-20261014.md)、[证据](evidence-v03-20261014/) | v0.3为设备原型版本，不是训练镜像v3 |
| 09-22 | [模型接入与RMSNorm差异](910C模型接入与安全回退-20260922.md)、[证据](evidence-model-20260922/README.md) | 早期推理原型，不能替代后续训练结论 |
| 09-16 | [CPU模型基线](0.6B模型来源算子基线-20260916.md)、[证据](evidence-model-20260916/) | CPU与当时NPU阻塞分开 |
| 09-09及更早 | [多芯片探索](多芯片环境与算子验证-20260909.md)、[PyTorch原型](PyTorch独立接入-20260907.md)、[vLLM单卡](910C单卡联调-20260907.md)、[最小验证](安全回退最小验证-20260903.md) | 历史路线，不是当前平台验收 |

## 内部归档

- [11-03整理前入口](archive/session-20261103/README.md)：状态、完整运行交接、导航和报告索引。
- [10-18状态快照](archive/session-20261018/STATUS.md)、[10-18运行快照](archive/session-20261018/HANDOFF.md)、[09-22入口快照](历史入口快照-20260922.md)。
- [内部草稿](archive/internal-drafts/)、[09-16周报](周报-20260916.md)、[09-09任务草案](框架接入与安全回退-9月任务规划-20260909.md)：不作为当前计划。

## 命名和保存

日期报告用“主题-YYYYMMDD.md”；索引用日期+用途称呼材料，不用“最新实验”“本周复现”作唯一定位。原始probe.log、runner.log、summary.json保留在日期/运行ID目录内，它们是程序输出，不是人读的总结。

已发布报告、案例、证据路径保持稳定；原始记录不为改名而改字节。发现错误追加带日期勘误，不能删除失败记录。完整本地日志与构建目录从[产物索引](/Users/empty/PhD/Project/DomesticRuntime/outputs/README.md)定位。
