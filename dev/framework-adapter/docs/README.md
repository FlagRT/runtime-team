# framework-adapter 文档目录

整理：2026-10-20。当前进度看[STATUS](../STATUS.md)，环境/复现看[HANDOFF](../HANDOFF.md)，代码地图看[方向README](../README.md)。

## 运行报告（按证据日期，不按整理日期）

| 日期 | 报告与证据 | 结论 / 发布 |
|---|---|---|
| 10-20 | [训练入口配置接入](训练入口配置接入验证-20261020.md)、[配置说明](../demo/training-configs/README.md) | 源码及独立wheel安装两轮，均双进程0.6B训练关闭/开启/无资格各3步通过；旧官方非训练v3，未发布 |
| 10-18 | [公共适配器NPU实测](910C公共适配器实测-20261018.md)、[标准demo](../cases/ascend-official-unary/README.md) | 两算子12目标＋12回退、0.6B推理/3步训练；后续新增sigmoid小模型4×5步确定性回归及baseline严格检查，保留默认模式失败；未发布 |
| 10-16 | [训练v3回归](训练v3基线与回归准备-20261016.md)、[case](../cases/ascend-train-v3/README.md)、[本地持久结果备份](/Users/empty/PhD/Project/DomesticRuntime/outputs/framework-runs/20261016-v3-063504/README.md) | strict及5类原生回退通过，目标SiLU缺shmem，模型未跑；新增代码/材料未发布 |
| 10-14 | [v0.3训推与SiLU验证](910C-v03推理与训练验证-20261014.md)、[证据](evidence-v03-20261014/) | 旧环境局部成功及数值限制，含raw注册第4步失败、同权重诊断、小样本训练、新适配器三步通过；发布8ecf3ae |
| 09-22 | [模型接入与安全回退](910C模型接入与安全回退-20260922.md)、[证据](evidence-model-20260922/README.md) | 历史SiLU推理、RMSNorm失败；发布2aad264，不是最近实测 |
| 09-16 | [CPU模型基线](0.6B模型来源算子基线-20260916.md)、[证据](evidence-model-20260916/) | 历史CPU验证及当时NPU阻塞 |
| 09-09 | [多芯片探索](多芯片环境与算子验证-20260909.md)、[证据](evidence-multichip-20260909/) | 历史探索，不代表当前平台验收 |
| 09-07 | [PyTorch接入](PyTorch独立接入-20260907.md)、[证据](evidence-pytorch-20260907/)；[910C单卡](910C单卡联调-20260907.md)、[证据](evidence-20260907/) | 历史原型与vLLM路线 |
| 09-03 | [最小验证](安全回退最小验证-20260903.md) | 最早原型，不替代模型验证 |

## 对外交接（保留已发布路径）

三份发布于2c73f48；不代表接口已全部确认。10-13会议分工及后续训练成果以STATUS/日期报告为准，复现说明中的实验仍为09-22。

- [算子组对接Issue](handoff/算子组对接Issue.md)：接口/清单/替换与融合原则、RMSNorm问题。
- [实验复现说明](handoff/实验复现说明.md)：旧实验入口、材料和固定链接。
- [原型与三级回退方案](handoff/原型与三级回退方案.md)：设计和缺口，不是完整三级已实现证明。

## 内部历史（不是当前交付）

- [四份旧内部草稿](archive/internal-drafts/)：09-23/09-26版本，10-18从handoff归档；对外用上面三份。
- [10-18整理前STATUS](archive/session-20261018/STATUS.md)、[HANDOFF](archive/session-20261018/HANDOFF.md)：完整保留覆盖式更新之前的流水，只用于追溯。
- [更早入口快照](历史入口快照-20260922.md)、[0916周报](周报-20260916.md)、[0909规划](框架接入与安全回退-9月任务规划-20260909.md)：均非当前要求。
- [本次整理清单](/Users/empty/PhD/Workspace/DomesticRuntime/协作/整理记录-20261018.md)。

## 维护

日期报告放实验结论，evidence目录放证据，cases放可运行demo，handoff放对外交接，archive放历史内部材料。
已发布报告/证据/代码不为美观迁移；失败记录不删。完整运行日志留执行端，本地持久备份与仓内截选分别注明来源，不把/tmp当唯一存储。
新agent只读STATUS/HANDOFF及本次任务涉及的报告，不必重读全部归档。

## 当前本地开发入口

[10-18公共适配包开发记录](公共适配包开发记录-20261018.md)：准入、SiLU/sigmoid公共机制、首次本地构建与未完成项。CPU控制不是新增NPU成绩。
