# performance · STATUS

> 总组速览入口（[战略文档 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md)格式）｜详细进展、边界与命令见 [README.md](README.md)
> 最近更新：2026-10-09

## 当前阶段

- 对应战略文档 §5 performance 任务 2/4、§3 验收标准 2/5：Base/Operation 基础规格评测与 Inference 工具链持续扩展；A100 参考比较（Oracle—Device 差分第一段）已实现；两段 Demo 的统一验收报告尚未完成。
- 独立 FlagPerf 公开迁移（2026-09-27/29）：Operation V3 与 Inference V1–V1.6 已选择性同步至 `FlagPerf/dev-zkm@a5f26fe9`（252 项离线回归＋实机功能验收）；`main@66eb17e4` 保持上游基线。
- Inference V1.4–V1.6（09-26 → 09-29）：FlagGems/FlagTree/FlagCX 显式开关与单轴比较、层级性能（`performance --level layer`）、preview 续探＋轻量证据＋分组搜索；首次完整 preview 墙时单卡 −53.8%、TP −55.4%（搜索周转观察值，非推理加速）。
- Inference V2/V2.1（09-29 → 09-30）：A100 `oracle` 精度参考包与 `performance-baseline` 同规模基线包，`--oracle`/`--baseline` 只读导入比较；CPU FP64 逐样本/层指标，倍率＝被测/A100，不设达标阈值。
- Inference V3/V3.1（10-01 → 10-05）：公共 ModelAdapter 契约；DeepFM（TP hidden 差分 max_abs 5.59e-8）、DLRM（与 DeepFM 共享 RecommendationAdapter，398 项回归）、OneRec-8B 有状态生成（12 条 token 轨迹一致，每份 profile 2,304 AllReduce＋32 AllGather 全归因）。
- Inference V3.2–V4（10-06 → 10-07）：CUDA/Ascend 大 trace 分析独立时限与恢复、`undefined_metrics` 与变异性筛查、A100 验收收口；统一监控旁路 `TelemetryProvider`（A3/910C DCMI，默认 1 秒，`monitor.html` 离线筛选），小规模开销对照 +0.0658%（95% 上限 +0.69%）。
- 成本优化与续验（10-07 → 10-09）：schema 4 比较核心包（OneRec 18.78 GB→52.12 MB）、`--dry-run`/`analyze`/`repack-reference`；2.62 GB trace 解析 165.70 s→111.91 s（−32.5%）；Ascend 续验 21 项通过，10-09 Qwen 4 项复验通过；DeepFM FlagTree/FlagCX TP2 剩余与 A100 12 项待资源。

## 关键阻塞

1. Inference V2–V4（A100 参考、推荐模型、监控旁路、成本优化）尚未选择性同步到独立 FlagPerf；当前必须从 `FlagPerf_advance` 使用。
2. Inference on 路径显著变慢仍未归因到具体算子（需算子级 profiling）；DeepFM FlagTree/FlagCX TP2 剩余项与 A100 12 项短测被资源阻塞（A100 SSH 拒绝连接、本地 16 个 Ascend 芯片被现有服务占用，未停止他人服务）。
3. Oracle—Device 差分只有第一段（A100 参考比较）；误差传播、敏感算子归因与唯一根因闭环未实现；公共诊断、通信归因与监控均无多服务器或第二厂商实机验收，TP=4/8 无实机证据。
4. 两段 Demo 的训练吞吐、推理吞吐/时延及其他方向证据尚未按统一口径收拢；Inference 证据为 PyTorch/Transformers 路径，不等于 vLLM 服务口径验收，不能写成战略统一验收报告已完成。

## 下一步

- 资源空闲后补齐待补设备矩阵（DeepFM FlagTree/FlagCX TP2、A100 12 项）并收口 10-09 in_progress 快照。
- 将 Inference V2–V4 以最小增量同步到 FlagPerf 个人分支并完成审查 PR；优先算子级归因 on 路径变慢并在无共享负载环境复测性能基线。
- 按锁定训练/推理镜像收拢两段 Demo 证据（含 vLLM 服务口径），统一预热、并发、计时和状态口径，生成战略 §3 第 5 条要求的验收报告；随后深化 Oracle—Device 误差归因闭环，推进故障恢复与长稳/热加载能力。
