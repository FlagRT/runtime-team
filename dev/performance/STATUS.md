# performance · STATUS

> 总组速览入口（[战略文档 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md)格式）｜详细进展、边界与命令见 [README.md](README.md)
> 最近更新：2026-09-16

## 当前阶段

- 对应战略文档 §5 performance 任务 2/4、§3 验收标准 2/5：Base/Operation 的基础规格评测、证据和报告工具已经形成；两段 Demo 的性能数据对齐与统一验收报告尚未完成。
- `FlagPerf_advance/dev-zkm@ad326754` 已形成 Operation V3：统一 52 Case 入口、公共失败诊断与离线/复放检查、结论优先报告、实时进度，以及日常统计/显存/Ascend Profiling。
- 最新历史全矩阵为 308 个适用组合：277 passed、24 blocked、5 failed、2 partial，另有 420 个类型组合不适用；52/52 Case 至少有一个实测组合通过，但不代表完整矩阵通过。
- 独立 FlagPerf 远端当前为 `dev-zkm@3e7c558b`；其中 Operation 公开基线来自 `8ded0d74`，尚不包含上述 V3 增量。`main@66eb17e4` 保持上游基线。

## 关键阻塞

1. Operation V3 尚未选择性同步、审查并发布到独立 FlagPerf；当前必须从 `FlagPerf_advance` 使用 V3 命令。
2. 最新公共诊断、报告和进度变更主要完成锁定 CANN 9 镜像内离线回归，尚无新增 NPU replay、多服务器或第二厂商实机验收。
3. Ascend Profiling 仅在 Device14 上验证代表算子和模式，尚未覆盖全部 52 Case，也不是图节点—Backend—运行时的完整跨层 Profiling。
4. 两段 Demo 的训练吞吐、推理吞吐/时延及其他方向证据尚未按统一口径收拢，因此不能把 Operation 工具进展写成战略统一验收报告已完成。

## 下一步

- 先将 Operation V3 以最小增量同步到 FlagPerf 个人分支并完成审查，再补代表性 NPU replay/Profiling 验收。
- 按锁定训练/推理镜像收拢两段 Demo 证据，统一预热、并发、计时和状态口径，生成战略 §3 第 5 条要求的验收报告。
