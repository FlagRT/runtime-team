# P800 / PPU 实测材料

更新：2026-11-05。本目录收录人工整理的结果摘要与证据索引，不是完整原始日志包。结论限定于各报告中的源码、镜像、模型与设备范围；本轮为已有产物复核，未新增硬件负载。

## 从哪里看

| 方向 | 报告 | 当前结论 |
|---|---|---|
| P800 Toolkit | [10-09 实测](P800-Toolkit实测结果-20261009.md) | 12 Case / 485 目标测量、监控、诊断通过；20 个重复组不稳定 |
| PPU Toolkit | [10-20 实测](PPU-Toolkit实测结果-20261020.md) | 已登记 12 Case 通过；INT8 官方工具失败不计入通过范围 |
| P800 Operation | [10-21 实测](P800-Operation实测结果-20261021.md) | Docker/local 各 308 适用项尝试，原始失败和缺口保留 |
| PPU Operation | [10-23 实测](PPU-Operation实测结果-20261023.md) | 两入口各 299 partial / 9 failed，另有 9 个缓存路由缺口 |
| P800 Inference v2 | [截至 11-05 实测及整合复核](P800-Inference-v2实测结果-20261105.md) | 所测范围未见新增回归；OneRec 分歧、klx1 异常、整合新双卡待验 |
| PPU Inference v2 | [截至 11-04 实测](PPU-Inference-v2实测结果-20261104.md) | Qwen 单卡/TP2 链路可用；OneRec 未设备验证、性能未取得资格 |

Base 历史验收见 [P800](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/kunlunxin/validation.md)、[PPU](https://github.com/FlagRT/FlagPerf/blob/41ab2dce5a94906919905437ebc8dec97a3d36c5/docs/thead/validation.md)。没有单独复制 Base 报告；它与 Toolkit 是不同测量路径，不能混用性能或资格结论。

## 版本与发布状态

| 范围 | 状态（2026-11-05 查询） |
|---|---|
| Base / Toolkit | P800 PR #1、PPU #4 已合并至 dev-1.0；P800 BF16 快路径 #3 已合并 |
| Operation | P800 #5、PPU #6 已合并；公共文档链接固定到 `41ab2dce` |
| Inference v2 核心 | PR #7/#8/#9 已合并；厂商适配与旧 Inference V1–V4 不混称 |
| Inference v2 厂商适配 | [整合 PR #12](https://github.com/FlagRT/FlagPerf/pull/12) 待评审，固定头 `315a2c32`；独立 #10/#11 已关闭未合并 |

## 如何理解这些结果

| 状态 / 证据 | 含义 |
|---|---|
| passed / partial / failed | 按对应层判定；partial 表示部分条件或证据未满足，不等于正确性失败，也不等于全部通过 |
| not-qualified / not_qualified | 尚未取得相应资格；不能把观测性能写成认证基线 |
| A/A 重复一致 | 同栈确定性检查，不是独立数学参考，也不证明单卡与双卡一致 |
| 哈希一致 | 证明文件与保存索引一致，不证明测量方法或根因判断必然正确 |
| 历史与新镜像 | 不跨源码、镜像或批次继承完整验收；补测与原失败分别保留 |

## 本轮核验与原始材料范围

| 材料 | 本轮核对方式 | 未随文提供的内容 |
|---|---|---|
| P800 Toolkit | 覆盖/分层状态与稳定性汇总；下载清单 6,315 个文件字节数及 SHA-256 全部匹配 | 原始样本、stdout/stderr 和监控 JSONL 不全在本地 |
| PPU Toolkit | 最终 summary、12 Case manifest、封存 runner 核对；索引中 4,850 个文件字节数及 SHA-256 全部匹配 | 未复制全套样本、工具和报告依赖；容器退出码 2 如实保留 |
| P800 Operation | 两套 728 项任务计数复算、历史独立审计 79/81 与失败项复读 | 原始张量及大 trace 不随摘要提供 |
| PPU Operation | 两入口 audit 308 适用项、correctness 299、routing 290/9 与归因勘误交叉核对 | 这里未重新逐文件核验全套原始矩阵 |
| P800 Inference | 两机整合 summary、对照审计、代表性新单卡结果与开放缺陷复读 | 未新跑设备；未重验全部历史 capture |
| PPU Inference | 组会摘要、健康检查与 PR 固定提交适配指南交叉核对 | 本轮未直接复读 PPU 原始 capture；不能宣称已重新逐文件验真 |

### 来源指纹

下面是编写摘要时实际读取文件的 SHA-256，供维护者与本地留档对照。它们不是可从本目录下载的证据附件，不代表完整原始实验已公开；需要原件时按报告运行 ID 联系性能方向维护者。

| 来源 | SHA-256 |
|---|---|
| P800 Toolkit 覆盖与稳定性 | `11f9c75aff5909939a1a81fcac32ded4160d1c74b19ca2d7b77ec149ac7de402` |
| PPU Toolkit 最终状态 | `dea74c5686ef65fef4e020e31dbd229f7d31e864fe8e7233458a4393978974d5` |
| PPU Toolkit 分项证据 | `e271ea02dccc8ff9a78a8274d59e8b669241f36eb23ae387874f409db6e0ddfe` |
| P800 Operation Docker | `46898f4496777ee0631e28dd9791c580dc7e5e4c22e1141705058004cc984241` |
| P800 Operation local | `f0a25f35446b3f2b5c9c3df81a36527c5da745f295cc8bab29bd1fef7e15d191` |
| P800 Operation 历史审计 | `763787048487f45be6ed25530f52da175e6f4da7cc4caba3cb2fc78653a18ac2` |
| PPU Operation 两入口审计 | `fc21c99f87a0bd440c6a47e0041dff7bf55f0a684f7dd8740e54dd2d0472e2d7` |
| P800 整合复核说明 | `e94472a362b432f0c10586673c2f82a8d0a31e122ad3de039eb7c977f20b388c` |
| PPU Inference 固定提交指南 | `4c4d497d9a79f9a104ae2252f468a39774c2b9eda283f2f0a04c86c1256c13c2` |

原始组会材料、失败、启动异常和后续重跑均保持不变。本目录仅保留必要摘要，未包含账号、内部服务器地址、模型权重或大 trace。
