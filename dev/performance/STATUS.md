# performance · STATUS

> 总组速览入口（[战略文档 §8.2](../../docs/运行时层原型验证-战略目标-910C.v1.md)格式）｜详细进展、边界与命令见 [README.md](README.md)
> 最近更新：2026-10-21

## 当前阶段

- 对应战略文档 §5 performance 任务 2/4、§3 验收标准 2/5：Base/Operation 基础规格评测与 Inference 工具链持续扩展；A100 参考比较（Oracle—Device 差分第一段）已实现；两段 Demo 的统一验收报告尚未完成。
- 独立 FlagPerf 公开迁移（2026-09-27/29）：Operation V3 与 Inference V1–V1.6 已选择性同步至 `FlagPerf/dev-zkm@a5f26fe9`（252 项离线回归＋实机功能验收）；`main@66eb17e4` 保持上游基线。
- Inference V1.4–V1.6（09-26 → 09-29）：FlagGems/FlagTree/FlagCX 显式开关与单轴比较、层级性能（`performance --level layer`）、preview 续探＋轻量证据＋分组搜索；首次完整 preview 墙时单卡 −53.8%、TP −55.4%（搜索周转观察值，非推理加速）。
- Inference V2/V2.1（09-29 → 09-30）：A100 `oracle` 精度参考包与 `performance-baseline` 同规模基线包，`--oracle`/`--baseline` 只读导入比较；CPU FP64 逐样本/层指标，倍率＝被测/A100，不设达标阈值。
- Inference V3/V3.1（10-01 → 10-05）：公共 ModelAdapter 契约；DeepFM（TP hidden 差分 max_abs 5.59e-8）、DLRM（与 DeepFM 共享 RecommendationAdapter，398 项回归）、OneRec-8B 有状态生成（12 条 token 轨迹一致，每份 profile 2,304 AllReduce＋32 AllGather 全归因）。
- Inference V3.2–V4（10-06 → 10-07）：CUDA/Ascend 大 trace 分析独立时限与恢复、`undefined_metrics` 与变异性筛查、A100 验收收口；统一监控旁路 `TelemetryProvider`（A3/910C DCMI，默认 1 秒，`monitor.html` 离线筛选），小规模开销对照 +0.0658%（95% 上限 +0.69%）。
- 成本优化与续验（10-07 → 10-09）：schema 4 比较核心包（OneRec 18.78 GB→52.12 MB）、`--dry-run`/`analyze`/`repack-reference`；2.62 GB trace 解析 165.70 s→111.91 s（−32.5%）；Ascend 续验 21 项通过，10-09 Qwen 4 项复验通过；DeepFM FlagTree/FlagCX TP2 剩余与 A100 12 项待资源。
- inference_v2 / flagprof（10-09 → 10-21）：与旧 inference 的“V2 A100 参考”不同，是与模型解耦的观测基座。S1 显式 ID 关联风险验证、T1–T8.3 账本与格式契约、T9.0–T9.3 官方采集/原生归属/record/attach/inspect 已形成；T9.4 生产入口收口通过限定范围 Gate，生产导入 100,000 个设备事件零丢弃。研发快照 `FlagPerf_advance/dev-zkm@a66356b5`；本次仅同步进展，P0 未退出、L-字段保持 `1.0-rc`，见 [README §11](README.md#inference-v2-progress)。
- inference_v2 容量取证（T9.5/T9.5b）：Stage 0 原 A/B 均未满足全部条件，5% A/A 校准未成立；后续 B2X 主矩阵 15/15 零 drop，op/span 最高三次无损登记档 70k/s、满队列采样内存在预算内。但严格 100ms 消费者死亡检测有 **102.312728ms** 有效失败，最终 store **243/244**、correlate 79/79、conformance 21/21。**证据已提交、待设计者 Gate，候选未选定、Stage 1 未开始，生产 Writer 未替换**；均非真实模型性能验收，详情见 [README §11](README.md#inference-v2-progress)。

- P800 BF16 `torch.mm` 快路径（2026-10-20）：默认 XMLIR 慢路径吞吐约 118.9 TFLOPS；`XMLIR_MATMUL_FAST_MODE=1` 后实测约 223.9 TFLOPS，BF16/FP16 比值约 0.45→0.87。该环境变量已固化到 Base runtime 合同，无算子代码改动；精度校验通过，误差与原路径同量级。开关只作用于 XMLIR/Triton JIT，不作用于 Toolkit 的 XBLAS 预编译内核（仍约 118.85 TFLOPS），该厂商库路径缺口未解决。
- P800 Base/Toolkit 基础规格评测收口（10-08 → 10-09）：昆仑芯 P800 Base 15 case（10 可执行＋5 skip）114 run 测量与正确性通过、严格五次组 15 passed、8 卡五次稳定测量（FP32 118.47 / FP16 259.17 TFLOPS、INT8 504.53 TOPS、带宽 2235.95 GB/s）；Toolkit 12 case 全量复测通过并固化口径（8192³、repeat 5、samples 50、D2D 双行分列），与 Base 差距四类归因定案（口径/内核/厂商路径/共享卡），同机制处 1.01× 重合。冻结 tag `p800-base-frozen-20261008` 已随分支 `zhiyu/kunlunxin-p800` 推 origin。
- P800 Operation（10-13 → 10-21）：原生 PyTorch 与 FlagGems 两条路径均已接通，支持 Docker/本地执行，含正确性、性能、显存、时间线、失败诊断和报告。52 case；Docker 与本地模式各有 308 个适用组合（420 not-applicable），精度/编译/运行失败原样保留。PR 已提交待评审，`validated:false` 不变，完整能力对齐未完成；六组硬件计数器无可信采集，另有已登记的 trace 及元素计数记录缺口。
- 平头哥 PPU Base 冻结（10-15）：17 组五次资格（85 run）＋容量查询；FP32 34.5 / FP16 135.6 / BF16 135.0 TFLOPS、INT8 269.2 TOPS、带宽 2052.4 GB/s、容量 103.04 GB、MPI 8-rank AllReduce 25.91 GB/s；P2P 结果限于已测代表性设备对。分支 `zhiyu/thead-ppu`＋tag `ppu-base-frozen-20261015` 已推，PR 已提交；16-rank 资格与 11-rank 审计仍未完成。
- 平头哥 PPU Toolkit 调研定案（10-19）：官方测量工具全部内置于镜像 `flagtree-ppu-3.6-dev:202608`（sailbandwidth 35 testcase、p2pBandwidthLatencyTest、pccl-tests 六件套、comm_tools、libacblas acblasGemmEx、ppu-smi v1.28），T1–T7 小测试通过，无需自研或第三方工具。
- 平头哥 PPU Toolkit 收口（10-20）：官方工具 12 case 最终验收 run `20261020T140938Z` 为 12/12 通过（测量与诊断）；FP32 34.504 / FP64 5.515 / FP16 135.064 / BF16 134.686 / TF32 67.555 TFLOPS，H2D CE/SM 29.0/26.9 GB/s，P2P 单向约 46.5、双向约 88–90 GB/s。修复两处 harness 问题：P2P rank stdout 交错（`14f6b356`）及 BF16 correctness 小缓冲 layout 越界（`41d329a1`）；修复后最终全量通过，旧的厂商归因已撤回并保留勘误链。33 个提交已推、PR 已提交；回归 390 passed、1 skipped。PPU runtime 仍为 candidate / `validated:false`。
- P800 P2P/AllReduce 偏小收口（10-20）：单机 XL 直连对 28.85 GB/s（BKCL 构建 6 条 XL ring）反而低于 PCIe 对 37.88 GB/s（12 rings）；官方文档候选共 11 个环境变量变体全部无效（带宽与 ring 数均不变）——非用户侧可解，BKCL 对 XL 拓扑的 ring 构造属厂商行为，待厂商反馈。P800/PPU 未解决问题全清单见 README §10。

## 关键阻塞

1. Inference V2–V4（A100 参考、推荐模型、监控旁路、成本优化）尚未选择性同步到独立 FlagPerf；当前必须从 `FlagPerf_advance` 使用。
2. Inference on 路径显著变慢仍未归因到具体算子（需算子级 profiling）；DeepFM FlagTree/FlagCX TP2 剩余项与 A100 12 项短测被资源阻塞（A100 SSH 拒绝连接、本地 16 个 Ascend 芯片被现有服务占用，未停止他人服务）。
3. Oracle—Device 差分只有第一段（A100 参考比较）；误差传播、敏感算子归因与唯一根因闭环未实现；旧 Inference 的诊断、通信归因与监控无多服务器或第二厂商对应能力的实机验收，TP=4/8 无实机证据。inference_v2 已有 A100/Ascend 官方采集与原生归属验证，范围见 [README §11](README.md#inference-v2-progress)，不替代上述能力验收。
4. 两段 Demo 的训练吞吐、推理吞吐/时延及其他方向证据尚未按统一口径收拢；Inference 证据为 PyTorch/Transformers 路径，不等于 vLLM 服务口径验收，不能写成战略统一验收报告已完成。
5. 昆仑芯 P800：BF16 快路径已进入 Base runtime 合同；Base 统一入口端到端复测仍需单独窗口。平头哥 PPU：16-rank 资格待独占窗口，11-rank collect 审计仍受 qualification.py `ranks<=8` 限制。
6. 厂商侧未解决问题：P800 P2P/AllReduce 带宽偏小、XBLAS BF16 快内核缺口、XRE pinned 整块异步返回 -999 与同步 memcpy 约 10 GB/s；PPU Toolkit INT8 `acblasGemmEx` 输出错误、`acblasLtMatmul` SIGSEGV。PPU BF16 correctness 和 P2P stdout 的旧失败已定位为 harness 缺陷并修复，不再归因厂商。详见 README §10。
7. inference_v2：T9.5b 的严格死亡时限反例、纯 Python SPSC 跨进程内存序和在途 C 口径仍待 Gate 处理；D31 已以热路径成本/繁忙时无损落盘/突发吸收三项并报替代单一速率，500k accepted/s 只作历史基准。项目 module/route 探针与正式分析器尚未交付；D33 自动步进尚待任务单与实现，当前窗口仍需显式 `step()`。原生 owned、项目 linked、格式适用性与采集完整性分别判断。

## 下一步

- inference_v2 先完成 T9.5b 设计者 Gate，按裁决修订实现/契约与验收，获授权后推进 Stage 1 生产化，再做 P0 退出复审；自动步进及导入规模成本另按任务推进。本次文档核对不重跑设备实验或完整研发 CI，失败及未知边界保留，见 [README §11](README.md#inference-v2-progress)。
- 资源空闲后补齐待补设备矩阵（DeepFM FlagTree/FlagCX TP2、A100 12 项）并收口 10-09 in_progress 快照。
- 将 Inference V2–V4 以最小增量同步到 FlagPerf 个人分支并完成审查 PR；优先算子级归因 on 路径变慢并在无共享负载环境复测性能基线。
- 昆仑芯 P800：继续跟进 Operation PR 评审、六组计数器与记录缺口；Base 统一入口端到端复测按独占窗口安排。XBLAS BF16、XRE 传输及 P2P/AllReduce 问题保留并反馈厂商。
- 平头哥 PPU：跟进已提交的 Base＋Toolkit PR；申请 16-rank 独占窗口，并按维护流程评估 qualification.py rank 上限及 11-rank 审计。保持 runtime candidate 状态，等待厂商正式软件栈后再复评。
- 按锁定训练/推理镜像收拢两段 Demo 证据（含 vLLM 服务口径），统一预热、并发、计时和状态口径，生成战略 §3 第 5 条要求的验收报告；随后深化 Oracle—Device 误差归因闭环，推进故障恢复与长稳/热加载能力。
