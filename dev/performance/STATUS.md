# performance · STATUS

> 总组速览入口（[战略文档 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md)格式）｜详细进展、边界与命令见 [README.md](README.md)
> 最近更新：2026-11-05（P800/PPU；核心历史阶段截至 10-28）

## 当前阶段

- 对应战略文档 §5 performance 任务 2/4、§3 验收标准 2/5：Base/Operation 基础规格评测与 Inference 工具链持续扩展；A100 参考比较已实现，inference_v2 P4 首批已推进到算子级 `introduced` 判定；两段 Demo 的统一验收报告尚未完成。
- 独立 FlagPerf 公开迁移（2026-09-27/29）：Operation V3 与 Inference V1–V1.6 已选择性同步至 `FlagPerf/dev-zkm@a5f26fe9`（252 项离线回归＋实机功能验收）；`main@66eb17e4` 保持上游基线。
- Inference V1.4–V1.6（09-26 → 09-29）：FlagGems/FlagTree/FlagCX 显式开关与单轴比较、层级性能（`performance --level layer`）、preview 续探＋轻量证据＋分组搜索；首次完整 preview 墙时单卡 −53.8%、TP −55.4%（搜索周转观察值，非推理加速）。
- Inference V2/V2.1（09-29 → 09-30）：A100 `oracle` 精度参考包与 `performance-baseline` 同规模基线包，`--oracle`/`--baseline` 只读导入比较；CPU FP64 逐样本/层指标，倍率＝被测/A100，不设达标阈值。
- Inference V3/V3.1（10-01 → 10-05）：公共 ModelAdapter 契约；DeepFM（TP hidden 差分 max_abs 5.59e-8）、DLRM（与 DeepFM 共享 RecommendationAdapter，398 项回归）、OneRec-8B 有状态生成（12 条 token 轨迹一致，每份 profile 2,304 AllReduce＋32 AllGather 全归因）。
- Inference V3.2–V4（10-06 → 10-07）：CUDA/Ascend 大 trace 分析独立时限与恢复、`undefined_metrics` 与变异性筛查、A100 验收收口；统一监控旁路 `TelemetryProvider`（A3/910C DCMI，默认 1 秒，`monitor.html` 离线筛选），小规模开销对照 +0.0658%（95% 上限 +0.69%）。
- 成本优化与续验（10-07 → 10-09）：schema 4 比较核心包（OneRec 18.78 GB→52.12 MB）、`--dry-run`/`analyze`/`repack-reference`；2.62 GB trace 解析 165.70 s→111.91 s（−32.5%）；Ascend 续验 21 项通过，10-09 Qwen 4 项复验通过；DeepFM FlagTree/FlagCX TP2 剩余与 A100 12 项待资源。
- inference_v2 / flagprof（10-09 → 10-21）：与旧 inference 的“V2 A100 参考”不同，是与模型解耦的观测基座。S1 显式 ID 关联风险验证、T1–T8.3 账本与格式契约、T9.0–T9.3 官方采集/原生归属/record/attach/inspect 已形成；T9.4 生产入口收口通过限定范围 Gate，生产导入 100,000 个设备事件零丢弃。研发快照 `FlagPerf_advance/dev-zkm@a66356b5`；本次仅同步进展，P0 未退出、L-字段保持 `1.0-rc`，见 [README §11](README.md#inference-v2-progress)。
- inference_v2 容量取证（T9.5/T9.5b）：Stage 0 原 A/B 均未满足全部条件，5% A/A 校准未成立；后续 B2X 主矩阵 15/15 零 drop，op/span 最高三次无损登记档 70k/s、满队列采样内存在预算内。但严格 100ms 消费者死亡检测有 **102.312728ms** 有效失败，最终 store **243/244**、correlate 79/79、conformance 21/21。**证据已提交、待设计者 Gate，候选未选定、Stage 1 未开始，生产 Writer 未替换**；均非真实模型性能验收，详情见 [README §11](README.md#inference-v2-progress)。
- inference_v2 本周主线（10-22 → 10-28）：T9.5b Gate（D39：Writer 选 B2X、插 T9.5c/T9.6）→ D33 方案 B 单进程生产候选（10-25，目标无需显式 `step()`）→ T9.7 Writer Stage 1 接入生产＋T9.8 真实负载校准（轻量记录约 70 条/s、扰动测不出）→ **D45：P0 退出、L-字段升至 `1.0`**（10-27，退出矩阵 13/13）→ T10.1–T10.3 → **D49：P1 退出**（10-28；主机→设备桥接生产路径成立、FlagGems 命中清单与 v1 一致 23 函数、`perf-lite` 吞吐比 1.0032、回退清单收窄为 5 主表键）→ **D50：P4 首批当天完成**（T10.4/T11.1/T11.2：固定输入采集 372 键/侧；三组跨芯片差分首差异均 `layers.0`、首个 `introduced` 为 `mean.dim`/`pow.Tensor_Scalar`；独立 NP 82/82 连接性能证据；v2 逐字节复现 v1 A100 参考 360/360）。研发快照演进至 `FlagPerf_advance/dev-zkm@35f94819`；设计者复审提议 D51（T11.3 报告口径修补、T12.1 P2a 导入分析）待批准，详见 [README §11](README.md#inference-v2-progress)。

### P800 / PPU 当前适配状态

| 方向 | 做到了什么 | 还缺什么 / 结果入口 |
|---|---|---|
| Base / Toolkit | P800 PR #1、BF16 快路径 #3、PPU #4 已合并；P800 Toolkit 12 Case / 485 目标、PPU Toolkit 12 Case 测量与诊断通过 | P800 传输不稳定与健康限制、PPU 16-rank 资格保留；runtime candidate，不自动晋级 |
| Operation | P800 #5、PPU #6 已合并；每厂商每入口各尝试 308 适用项 | P800 数值/编译/运行失败与计数器等缺口；PPU 9 项失败、9 个缓存路由缺口未关闭 |
| Inference v2 | 整合 PR #12 待评审；P800 Qwen/OneRec、PPU Qwen 已测核心链路；P800 整合版 klx2 新单卡可用 | OneRec 单卡/TP2 分歧、性能资格、klx1 异常和整合新双卡待验；PPU OneRec 未实测 |
| 证据复核 | 11-05 复读结果、版本与失败记录；历史数据操作 321 成功 / 21 原有失败，未发现所测范围新增回归 | 不将离线操作数当成模型覆盖数；[实测报告与来源索引](evidence/README.md) |

上述通过均限定于报告中的模型、镜像与测试批次，不等于所有设备、全部能力或完整性能资格通过。

## 关键阻塞

以下旧 Inference 与核心研发待办保留 10-28 的历史范围；P800/PPU 的新增适配状态以上表及 evidence 为准，不据此代替核心设计者 Gate。

1. Inference V2–V4（A100 参考、推荐模型、监控旁路、成本优化）尚未选择性同步到独立 FlagPerf；当前必须从 `FlagPerf_advance` 使用。
2. Inference on 路径显著变慢仍未归因到具体算子（需算子级 profiling）；DeepFM FlagTree/FlagCX TP2 剩余项与 A100 12 项短测被资源阻塞（A100 SSH 拒绝连接、本地 16 个 Ascend 芯片被现有服务占用，未停止他人服务）。
3. Oracle—Device 差分只有第一段（A100 参考比较）；inference_v2 P4 首批已把差分推进到算子级 `introduced` 判定与 ULP 分布（v1 未实现），但隔离上游误差的 replay 与唯一根因闭环未实现；旧 Inference 的诊断、通信归因与监控无多服务器或第二厂商对应能力的实机验收，TP=4/8 无实机证据。inference_v2 的能力范围见 [README §11](README.md#inference-v2-progress)，不替代上述能力验收。
4. 两段 Demo 的训练吞吐、推理吞吐/时延及其他方向证据尚未按统一口径收拢；Inference 证据为 PyTorch/Transformers 路径，不等于 vLLM 服务口径验收，不能写成战略统一验收报告已完成。
5. P800/PPU Base 与 Toolkit 仍有资格边界：P800 传输不稳定与健康限制；PPU 16-rank 独占资格、16-target 监控和 rank 上限审计待补。详见[实测索引](evidence/README.md)。
6. 厂商适配未解决问题：P800 的 XBLAS/XRE/P2P 路径与 Operation 缺口；PPU INT8 官方工具失败、九项 Operation 失败及路由缺口。问题分流不是全部根因已闭合；历史 harness 修复与撤回记录保留。详见 README §10。
7. inference_v2：D51（P4 首批复审＋T11.3/T12.1 下发）待用户批准；P2a 导入速度挡在 v1 已有的 OneRec generate 与设备视图前面（2,401,518 条导入曾于 900 秒上限退出）；P4 第二批 replay 未实施；多卡（P2b）、设备侧回退率与 generate 场景未开始。oracle 报告头条的掩码哨兵/相对误差/ULP 口径问题由 T11.3 修补。原生 owned、项目 linked、格式适用性与采集完整性仍分别判断。

## 下一步

- inference_v2：先裁决 D51，随后 T11.3（报告口径修补＋重出六份报告＋`pow(x,2)` 发现正式版交 FlagGems 组）与 T12.1（P2a 导入 prepare 定点分析，三份封存采集、无需设备）并行执行；按 T12.1 规模曲线定 P2a 门限，再推进导入提速、设备视图与 OneRec generate；P4 第二批（replay、`layers.27` 下钻）排在 T12.1 之后。本次文档核对不重跑设备实验或完整研发 CI，失败及未知边界保留，见 [README §11](README.md#inference-v2-progress)。
- 资源空闲后补齐待补设备矩阵（DeepFM FlagTree/FlagCX TP2、A100 12 项）并收口 10-09 in_progress 快照。
- 将 Inference V2–V4 以最小增量同步到 FlagPerf 个人分支并完成审查 PR；优先算子级归因 on 路径变慢并在无共享负载环境复测性能基线。
- P800：跟进 PR #12，健康设备上补 klx1 与整合新 TP2；继续定位 OneRec 分歧和性能资格缺口，不改变标准模型计算策略。
- PPU：补独占资格和监控；按原协议复核算子失败与路由缺口，取得 OneRec 权重后单独验证；已合并的 Base/Toolkit/Operation 不再列为待提交。
- 按锁定训练/推理镜像收拢两段 Demo 证据（含 vLLM 服务口径），统一预热、并发、计时和状态口径，生成战略 §3 第 5 条要求的验收报告；随后深化 Oracle—Device 误差归因闭环，推进故障恢复与长稳/热加载能力。
