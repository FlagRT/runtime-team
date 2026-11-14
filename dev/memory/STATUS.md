# memory · STATUS

最近更新：2026-11-14｜开发主力：刘鑫；既有画像/卸载负责人：陈沅昕（请假中，返回后分工待议）。

本页按 [战略目标 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md) 维护当前阶段、关键阻塞和更新日期；详细历史看板仍见 [README](README.md) / [PROGRESS](PROGRESS.md)。组件功能、接口和单测在 [FlagMemory](https://github.com/FlagRT/FlagMemory)，本仓负责组内同步和验收资料。

## 当前阶段

- 对应战略目标 §3 第4条（显存池定义与占用峰值画像）；旧画像与三栈卸载成果保留。当前已推进分级块池、租约与流有序归还、统一桥接及单模型连续池输入验证。
- 池核心/主机合同已验收；新统一桥接三生命周期真实3通过0跳过；最终源主机715通过/1旧跳过/11真卡未选；OneRec-8B单加载16对/32 forward、maxdiff0、1alloc8MiB/1release/live0已独立接受。精确源码身份与限制见 [阶段验收摘要](https://github.com/FlagRT/FlagMemory/blob/xliu969/pool-impl/docs/validation/2026-11-05-pool-stage.md)。
- 招标显存组件18条整体尚未完成；11月12日设计评审已通过（2026-11-05提前：用户主审，设备方向对接口章确认无异议）；11月25日接口冻结已实质达成（2026-11-05双方认可，冻结文件v1落device-context目录、已随2026-11-05公共合入进入dev-1.0树内）；余11月28日阶段公共交付复核与11月30日分层设计出口待组织，不能以本次固定模型输入链替代。

## 关键阻塞与需总组同步

- 接口语言面已裁定并落地（2026-11-05，ADR-0015 Accepted）：「接口头文件」=类型存根+接口文档，C头触发式（非Python消费方或验收方明确要求）；定稿措辞已在组件仓公共 dev-1.0（f3c7125）。
- 下游交界已联合冻结（2026-11-05，双方认可）：统一内存句柄加「用途标签+所属流」两字段、强持有与能力缺失语义，冻结文件v1 = [HANDLE_AND_POOL_INTERFACE_FREEZE_20261105](../device-context/docs/HANDLE_AND_POOL_INTERFACE_FREEZE_20261105.md)（归属device-context目录，已在公共dev-1.0树内）；两件开放项设备方向已于2026-11-10枚举审核回执（[HANDLE_FIELDS_ENUM_REVIEW_20261110](../device-context/docs/HANDLE_FIELDS_ENUM_REVIEW_20261110.md)）答复：寒武纪按我方2026-11-10真机取证（[分层存储设计第十节补记二](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)）接受改判方向、统一层代码声明待其探测窗口；两字段落地形态已定（HANDLE_EXT_KEYS独立扩展键+allocate()扩签名纯透传，与寒武纪声明更新同窗口）。枚举登记v2（[机读](../device-context/docs/HANDLE_FIELDS_ENUM_REGISTRY_20261110.yaml)）五处修订我方2026-11-13无异议、规范名定SCRATCH，Accepted由设备方向写入。通用分层管理器及池回调未接线，OOM防护真机、画像工具归位仍待推进。
- 当前设备结果限单芯片固定模型输入缓冲，不含权重/KV/激活接管、多rank/训练/generate、业务质量或性能收益；HBM仅离散观察，完整logits未落盘，不能独立重算全部向量。
- 两仓唯一个人汇总线分别为组件仓 `xliu969/pool-impl`、本仓 `xliu969/memory-stage-closeout`；历史300份脱敏资料及旧报告脱敏修正已进入本仓个人线，组件仓旧reports改为索引；两仓公共 `dev-1.0` 均已由用户合入（本仓494cc017；组件仓f3c7125含接口章定稿措辞）。新桥接原始证据私有保留，不随历史批次公开。

## 组内同步与月度待办

- 两仓公共合入均已完成（用户执行）；后续公共合入沿用同一流程：个人分支提交后由用户经 sync_to_dev 执行，两仓是独立Git仓、分别操作。
- 本仓标准动作见 [根README标准提交流程](../../README.md#标准提交流程) 及 [同步脚本](../sync_to_dev.sh)：在已提交且干净的个人分支运行 `bash dev/sync_to_dev.sh`，无参数；会合入并推送本仓公共 `dev-1.0`，结束停在公共分支。该动作由用户最后执行。组件仓在公共协作仓的 `FlagMemory/` 子目录时，可从该组件仓个人分支运行 `bash ../dev/sync_to_dev.sh`，脚本按当前Git仓根操作；组件仓自身没有同名脚本。
- 组件仓布局与协作线见 [仓登记](../repos.yaml)；环境启动、精确版本登记与核验遵循 [环境规范](../ENV-SPEC.md)，基座以现行锁定声明为准。
- 后续按未完成18条范围组织正式评审、接口与下游确认、分层/防护/画像工具交付及跨场景联合验收，分别报告质量与性能。

## 功能×芯片进度矩阵

状态四档：**待探查 / 待接入 / 已接入 / 已验收**（已验收 = 真卡实测通过且证据入档）。功能说明文档锚定组件仓 [FlagMemory](https://github.com/FlagRT/FlagMemory)：[显存池概念与设计基础](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/显存池概念与设计基础.md)、[分层存储概念与设计基础](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)。

| 功能 | 昇腾 910C | 昆仑芯 P800 | 平头哥 PPU | 寒武纪 MLU | 壁仞 BR_0 |
|---|---|---|---|---|---|
| [池本体（分级块池）](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/显存池概念与设计基础.md) | 已验收：[三生命周期真卡证据](evidence/flagmemory-pool/ascend-pool-lifecycle-3de6d5a/)、[OneRec 单输入](evidence/flagmemory-pool/onerec-input-consume-4967c6a/)、[验证叙事](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/validation/2026-11-04-ascend-pool-onerec.md) | 已验收（公开仓暂无独立归档件；池阶段验收摘要见 [2026-11-05-pool-stage](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/validation/2026-11-05-pool-stage.md)） | 已验收（公开仓暂无独立归档件；真卡全套收尾见 [ppu-tier-closeout-536bf3b](evidence/flagmemory-pool/ppu-tier-closeout-536bf3b/)） | 已验收（[池三生命周期真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/pool/test_pool_realcard.py)；句柄能力已实测具备、统一层声明与接线待设备方向窗口，见 [接口冻结文件](../device-context/docs/HANDLE_AND_POOL_INTERFACE_FREEZE_20261105.md)） | 待接入（机器权限未通） |
| [芯片接线（backends）](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 已验收：[昇腾接线真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/tiering/test_npu_backend_realcard.py) | 已验收：[昆仑芯接线真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/tiering/test_xpu_backend_realcard.py) | 已验收：[平头哥接线真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_ppu_tier_manager_realcard.py) | 已验收：[寒武纪接线真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/tiering/test_mlu_backend_realcard.py) | 待接入 |
| [分层存储（管理器端到端+真模型）](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 已验收：[昇腾管理器端到端真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_npu_tier_manager_realcard.py) | 已验收：[昆仑芯管理器端到端真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_xpu_tier_manager_realcard.py) | 已验收（含真模型 A/B）：[ppu-tier-closeout-536bf3b](evidence/flagmemory-pool/ppu-tier-closeout-536bf3b/) | 已验收：[寒武纪管理器端到端真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/tiering/test_mlu_tier_manager_realcard.py) | — |
| [异步搬运（真卡 E2E）](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 已验收：[昇腾异步轨 E2E 真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_npu_tier_manager_realcard.py) | 已验收：[昆仑芯异步轨 E2E 真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_xpu_tier_manager_realcard.py) | 已验收：[平头哥异步轨 E2E 真卡用例](https://github.com/FlagRT/FlagMemory/blob/xliu969/tier-impl/tests/tiering/test_ppu_tier_manager_realcard.py) | 已验收：[寒武纪异步轨 E2E 真卡用例](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/tests/tiering/test_mlu_tier_manager_realcard.py) | — |
| [KV 卸载插件（connector）](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/flagmemory/tiering/offload/README.md) | 已接入（运行环境=推理腿镜像已定档）：[npu RUNBOOK](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/flagmemory/tiering/offload/npu/RUNBOOK.md) | 已接入（运行环境已定档）：[xpu RUNBOOK](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/flagmemory/tiering/offload/xpu/RUNBOOK.md) | 先例未入本线（[PPU 移植闭环结果报告](docs/goals/proto-ppu-202610/)） | 已接入（运行环境=推理腿镜像已定档）：[mlu RUNBOOK](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/flagmemory/tiering/offload/mlu/RUNBOOK.md) | — |
| [原生优化路径](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 可立项待拍板（VMM 全链条候选，见[分层存储设计第十节](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)） | 可立项待拍板（流有序候选须按 eager 语义设计并走 ctypes 直调通路，见[分层存储设计第十节](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)） | 待探查 | 可立项待拍板（流序候选事实面全好且约束优于昆仑芯——eager+按序回收+提交期报错+free 无滞留，torch 层无门禁三通路；大VA预留机制齐备但物理粒度 32 MiB，见[分层存储设计第十节补记三](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)） | 待探查 |
| [能力/粒度取证](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 已验收：[SDK 原语取证补记](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md) | 已验收：[SDK 原语取证补记](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)（粒度实例另见[显存池设计第八节第 4 条](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/显存池概念与设计基础.md)） | 已验收（四家显存利用率实测同向，见[显存池设计第五节](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/显存池概念与设计基础.md)） | 已验收（2026-11-10；证据见[分层存储设计第十节补记二](https://github.com/FlagRT/FlagMemory/blob/dev-1.0/docs/design/分层存储概念与设计基础.md)） | 部分探明（驱动栈）；公开仓暂无登记件，待落点入档 |

> 注记：上表 FlagMemory 链接中锚定分支 （已推送公开）的文件，在该分支合入公共  后链接随之固化。runtime-team 仓内链接为相对路径；「待接入 / 待探查」端元的链接指向最接近的解释文档。

历史结果见 [批次报告](docs/goals/) 与 [画像/卸载证据](evidence/flagmemory-offload/)；新池 910C 真卡证据（映射 / 健康 / 三生命周期 + OneRec 单输入消费，2026-11-04，脱敏副本）见 [池真卡证据](evidence/flagmemory-pool/)。历史材料不代表新池跨平台能力。
