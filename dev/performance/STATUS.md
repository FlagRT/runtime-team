# performance · STATUS

最近更新：2026-11-25。团队公共分支为 `FlagRT/FlagPerf dev-1.0@46450a2d`，推理包为 0.2.0。使用入口、各版本差异与实测记录见 [README](README.md#flagperf-生产进展)。

## 当前阶段

- **推理性能与精度诊断已进入公共分支。** 支持单卡和单机 TP2 的逐层精度对照、算子重放、设备耗时及通信分析。P800/PPU 适配已随 PR #12 合入。PR #15 增加昇腾/NVIDIA 资源监控、OOM 与停滞等故障诊断、计算图导出及离线报告。[组件复验](https://github.com/FlagRT/FlagPerf/blob/46450a2dbe3618815faa0ff63460cee5104d6903/inference/docs/validation-components-20261110.md) · [PR #15](https://github.com/FlagRT/FlagPerf/pull/15)
- **训练适配已于 11-25 随 PR #16 合入。** 910C、P800、PPU 有单卡及 DDP/FSDP2 实测，报告按 rank 分列通信与计算。双卡 rerank-0.6B 已完成三组跨厂商逐层对照。P800 能核对通信次数和字节数，设备内核事件缺失，等待时间与带宽待补。[训练验证记录](https://github.com/FlagRT/FlagPerf/blob/46450a2dbe3618815faa0ff63460cee5104d6903/docs/training/v2-validation.md) · [FlagScale 接入](docs/training/FlagScale使用FlagPerf最佳实践.md)
- **生产仓本地已提交 0.3.0。** 用户可指定结果盘，报告支持从概览下钻到证据。其上的配置化工作流尚未提交，11-25 已在 CPU 和昇腾完成首用实测。单卡 Qwen3 两次相同输入的 31 个张量边界全部一致。[本地版本与实测范围](README.md#生产仓本地开发进展)
- **战略验收对齐 §3 第 2、5 条及 §5 performance 四项任务。** 两段环境已登记，Base/Toolkit/Operation 已有分平台测量与监控证据。两段 Demo 的同口径性能数据仍待收齐，统一验收报告待生成。[战略目标](../../docs/运行时层原型验证-战略目标-910C.v1.md) · [910C 环境锁](../stack.lock.910c.v4.yaml) · [基础规格与算子实测](evidence/README.md)

## 关键阻塞与需总组同步

- **两段 Demo 联合验收**需要锁定镜像下的双卡训练吞吐、单卡 vLLM 服务的向量正确性及吞吐/时延，并对齐预热、并发和计时窗口。已有 PyTorch/Transformers 推理记录可作为组件证据。
- **跨厂商覆盖**仍有明确缺口。P800 OneRec 单卡与 TP2 第 14 个生成 token 有分歧，PPU 缺 OneRec 设备端生成记录，两平台推理性能资格待验。训练另有多模态模型与引擎级流水并行待验。[P800 实测](evidence/P800-Inference-v2实测结果-20261105.md) · [PPU 实测](evidence/PPU-Inference-v2实测结果-20261104.md) · [训练完成报告](docs/training/FlagPerf训练适配完成报告.md)
- **推理诊断交付**仍需补齐共享目录下的离线分析和模型层归属证据。当前在线能力到故障检测与取证，任务重试、节点隔离和模型热加载待实现。[首用实测](README.md#生产仓本地开发进展) · [生产路线图](https://github.com/FlagRT/FlagPerf/blob/46450a2dbe3618815faa0ff63460cee5104d6903/inference/docs/10-roadmap.md)
