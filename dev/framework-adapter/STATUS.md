# framework-adapter · 当前状态

更新：2026-11-03｜顾宬 / cgu135。本页是公共协调仓的状态与交付物指针；功能包在独立仓，详细测试材料在本人证据分支。硬件成绩截至10-29，11-03为方案、材料核验及数据保全，不计新增硬件实验。

## 本期交付

已形成“三种接入方式”的方案与接口约定评审稿：算子注册、库替换、版本补丁，分别说明接入位置、接口、版本检查、撤销/重建和安全退出。现有API、未发布候选和拟议接口分开标注；尚未跨组确认，不等于三条路线都已实现。

- [三通道接入方案](https://github.com/FlagRT/framework-adapter/blob/376f8d3a74dca4cfa973127fb8c7ab2695078eb6/docs/三通道接入方案.md)
- [接入接口约定](https://github.com/FlagRT/framework-adapter/blob/376f8d3a74dca4cfa973127fb8c7ab2695078eb6/docs/接入接口约定.md)
- [文档入口](https://github.com/FlagRT/framework-adapter/tree/376f8d3a74dca4cfa973127fb8c7ab2695078eb6/docs)

## 已完成、问题和边界

| 功能点 | 当前完成情况 | 问题/未完成 |
|---|---|---|
| 接入开关与原生保留 | PyTorch eager下SiLU、sigmoid、tanh已有前反向接入；关闭、不满足条件时保留合法原生；退出检查恢复 | 非所有ATen/自定义/融合入口；不能等同于三个框架均已接入 |
| 910C训练接入 | 旧官方环境Qwen有限训推及固定双rank入口验证；正常各3步，前向/反向故障注入均双方0次优化器调用 | 训练v3目标编译仍有TLE/shmem问题；门控不含更新后回滚或跨芯片恢复 |
| P800接入与审核 | 三种单算子通过；DeepFM FP32 batch16/32限定短训练通过；失败/无资格走原生，严格目标可阻断 | Qwen FP32 seq32、OneRec BF16 seq16候选梯度超差，未准入、不改算子；新env/v1资格需另验 |
| OneRec拒绝验证 | 原生、原生重复、关闭、已知失败保留原生四条轨迹各3步；后三条7191项对照一致；候选失败0次更新 | 不是OneRec目标实现训练通过，也不是业务效果验收 |
| 三路线交付 | 方案和接口约定已提交评审文本 | 通用注册所有权、独立库替换生命周期、版本补丁管理与验收尚待落地；完整三级回退未完成 |

本方向只负责调用、准入、选择、恢复边界与证据，不修改算子内核或放宽精度阈值。现有模型测试是有限配置的合成输入短训练。数值资格是离线验证，不是在线自动精度检测。

## 证据与代码版本

| 材料 | 固定版本/链接 |
|---|---|
| 功能仓及本次方案文档 | [FlagRT/framework-adapter](https://github.com/FlagRT/framework-adapter)，文档提交376f8d3；已发布运行时代码仍为0.3.0.dev0，基准85a9896 |
| 910C接入实测 | [10-18报告](https://github.com/FlagRT/runtime-team/blob/8963353653ff8610d44b1f6a89be43ea680b77cf/dev/framework-adapter/docs/910C公共适配器实测-20261018.md) |
| 双rank更新前阻断 | [10-28报告](https://github.com/FlagRT/runtime-team/blob/8963353653ff8610d44b1f6a89be43ea680b77cf/dev/framework-adapter/docs/双rank步骤门控验证-20261028.md) |
| P800三模型、通过与失败 | [10-29报告](https://github.com/FlagRT/runtime-team/blob/c9f32bf7edf80db2d74100cebd5c8acea0686ee6/dev/framework-adapter/docs/P800接入与安全回退验证-20261029.md)、[结构化证据](https://github.com/FlagRT/runtime-team/tree/c9f32bf7edf80db2d74100cebd5c8acea0686ee6/dev/framework-adapter/docs/evidence-p800-20261029) |
| 本次材料核验 | [11-03审查记录](https://github.com/FlagRT/runtime-team/blob/c9f32bf7edf80db2d74100cebd5c8acea0686ee6/dev/framework-adapter/docs/evidence-p800-20261029/publication-audit-20261103.json)：94项包CPU控制、34项集成控制通过；10个运行时文件与实测wheel一致；不是新硬件成绩 |
| P800集成入口 | [case与baseline/demo](https://github.com/FlagRT/runtime-team/tree/c9f32bf7edf80db2d74100cebd5c8acea0686ee6/dev/framework-adapter/cases/kunlun-unary)：依赖尚未公开发布的精确dev2候选制品，不能用dev0替代 |

本次只发布方案文档和集成材料，没有发布本地dev2运行时代码/wheel。个人分支证据已提供固定提交链接，不表示其全部正文合入公共dev-1.0。共享分支不复制另一套交付包。

## 下一阶段

1. 对照功能拆分表确认合同节点，评审三路线的接口、所有权、版本约束和安全状态；不将11-04状态更新期限误作全部功能完成期限。
2. 收口候选审查和可复现制品，再补公共训练入口的模型身份/审核采集；库替换和补丁按方案逐项验收。
3. 新P800环境按统一基座复建，先做一个已有通过模型的关闭/开启/不满足条件三路回归，再扩平台。本人实验资料9395文件、约120.5GiB已迁移并校验，公共模型另协调。
4. 消费设备方向兼容表，数值问题交算子/编译方向；不重复建设全组设备矩阵或接管通信验收。

本页至少每两周更新一次，重要交付同步更新证据链接。11月排期以正式功能表为准，合同具体日期待核；尚无自动定期更新机制。

## 交付物指针（总组约定保留）

- 独立仓：FlagRT/framework-adapter（私有，main为唯一工作线；查看链接需相应权限）。
- 历史v0.2.0验收锚93daa5e保留，不替代本页新版本和限定证据。
- 布局收录于[dev/repos.yaml](../repos.yaml)，由dev/clone_all.sh收拢。
- 公共dev-1.0本方向只维护状态与指针，不将交付物正文和个人分支全部历史材料并入。
