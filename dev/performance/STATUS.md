# performance · STATUS

> 总组速览入口（[战略文档 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md)格式）｜详细进展、边界与命令见 [README.md](README.md)
> 最近更新：2026-09-23

## 当前阶段

- 对应战略文档 §5 performance 任务 2/4、§3 验收标准 2/5：Base/Operation 基础规格评测与 Inference 精度/性能工具均已形成；两段 Demo 的统一验收报告尚未完成。
- Operation V3（`FlagPerf_advance/dev-zkm@ad326754`）：统一 52 Case 入口、公共失败诊断与离线/复放检查、结论优先报告、实时进度，以及日常统计/显存/Ascend Profiling。最新历史全矩阵为 308 个适用组合：277 passed、24 blocked、5 failed、2 partial；52/52 Case 至少有一个实测组合通过。
- Inference V1–V1.3（`FlagPerf_advance/dev-zkm@d49d3910`，本周交付）：Qwen3-Embedding-0.6B 向量精度工具重建完成——单卡 off/on 差分（模型级＋28 层，策略 28 候选 26 接受）、单卡总级性能（设备 1，off 32.242 ms vs on 4010.026 ms/批）、单机 TP=2 HCCL 性能与通信归因（设备 2、4，2016/2016 事件归因，off 48.257 ms vs on 2214.774 ms）、TP 逐 rank 精度（设备 4、6，12,200 项独立校验通过）。
- Inference on/off 差异（单卡 124.37 倍、TP 45.895 倍）是当前镜像/策略/固定输入下的观察值；同板卡存在外部负载，属功能与测量链路验收，非独占性能基线；on 变慢尚未归因到具体算子。
- 独立 FlagPerf 远端当前为 `dev-zkm@3e7c558b`；其中 Operation 公开基线来自 `8ded0d74`，尚不包含 V3 与 Inference 增量。`main@66eb17e4` 保持上游基线。

## 关键阻塞

1. Operation V3 与 Inference V1–V1.3 均未选择性同步、审查并发布到独立 FlagPerf；当前必须从 `FlagPerf_advance` 使用。
2. Inference 的 on 路径显著变慢未归因到具体算子：调用级路由取证不是硬件 kernel fallback 率，需算子级 profiling；A100 参考导入未实现，尚无 NVIDIA—国产设备差分闭环。
3. Operation 公共诊断、Inference 通信归因均无多服务器或第二厂商实机验收；Inference 的 NVIDIA 路径与 TP=4/8 仅有接口合同/配置，无实机证据。
4. 两段 Demo 的训练吞吐、推理吞吐/时延及其他方向证据尚未按统一口径收拢；Inference 证据为 PyTorch/Transformers 路径，不等于 vLLM 服务口径验收，不能写成战略统一验收报告已完成。

## 下一步

- 将 Operation V3 与 Inference V1–V1.3 以最小增量同步到 FlagPerf 个人分支并完成审查，再补代表性 NPU replay/Profiling 验收（Operation）与算子级归因、无共享负载复测（Inference）。
- 按锁定训练/推理镜像收拢两段 Demo 证据（含 vLLM 服务口径），统一预热、并发、计时和状态口径，生成战略 §3 第 5 条要求的验收报告。
