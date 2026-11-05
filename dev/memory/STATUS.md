# memory · STATUS

更新：2026-11-05。开发主力：刘鑫；既有画像/卸载负责人：陈沅昕（请假中，返回后分工待议）。

## 当前组件与阶段

组件代码、接口、设计与单测位于 [FlagMemory](https://github.com/FlagRT/FlagMemory)。个人汇总线为 `xliu969/pool-impl`；协作仓个人同步线为 `xliu969/memory-stage-closeout`，均待用户手动合入各仓公共 `dev-1.0`，尚未公共合入。

| 能力 | 当前状态 |
|---|---|
| 分级块池、租约/底层身份、流有序归还与统一设备桥接 | 池核心和主机合同阶段已验收；原生优化关闭 |
| 昇腾 910C 新桥接 | 固定新源码三生命周期真实 3 通过、0 跳过，pending 保护观察接受 |
| 连续模型池输入 | 最终源码 OneRec-8B 单加载16对/32 forward，池同存储实际消费、maxdiff=0，1alloc8MiB/1release/live0，独立接受 |
| 三栈 KV→Host 卸载与画像 | 历史成果保留，按批次阅读，不外推为新池验证 |
| 通用分层存储与 guard/profile 收拢 | 分层管理器及池回调未接线；OOM 防护真机与画像工具归位未完成 |
| 包结构、主机门禁 | 已进入组件个人实现线；正式环境登记与发布仍待月度出口核定 |

## 验收与限制

固定源身份、715主机通过/1旧跳过/11真卡未选，以及三生命周期和连续模型的精确边界见 [组件阶段公开验收摘要](https://github.com/FlagRT/FlagMemory/blob/xliu969/pool-impl/docs/validation/2026-11-05-pool-stage.md)。最终源为 `eef16077fb9d5982cfd136e54570ac17888b7757`，新桥接设备成绩源为 `a51ec5c00505eb6a816fcdeb57b8ccdb8c3357c5`。本轮只分支收尾和文档同步，无新增测试或设备执行。

不覆盖权重/KV/激活池化、多芯片/多rank、训练/generate、性能收益或业务质量；接口语言面仍待正式冻结。固定模型输入链通过不等于显存完整交付或月度验收完成。

## 月度待办

- 完成剩余显存组件交付项，确定通用分层管理器、OOM防护与画像工具的交付边界。
- 完成正式接口、下游联合确认与环境登记，按现行公共锁定声明接入；设备层沿用厂商官方插件与统一使用接口。
- 扩展尚未覆盖的芯片/场景与联合模型验收，分别登记质量和性能结论。
- 用户审查后手动合入两仓公共开发线，公共合入前按个人线状态使用。

## 历史资料入口

组件设计和使用入口见 [组件文档](https://github.com/FlagRT/FlagMemory/tree/xliu969/pool-impl/docs)。协作仓 [显存历史报告](https://github.com/FlagRT/runtime-team/tree/xliu969/memory-stage-closeout/dev/memory/docs/goals) 与 [历史画像/卸载证据](https://github.com/FlagRT/runtime-team/tree/xliu969/memory-stage-closeout/dev/memory/evidence/flagmemory-offload) 为对应批次可公开验收资料；新桥接原始日志仍私有保留，不混入历史迁移集。
