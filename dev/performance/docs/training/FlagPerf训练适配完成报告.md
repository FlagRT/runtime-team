# 三平台训练适配交付报告（2026-11-17）

> 交付物：FlagPerf 训练 v2（`zhiyu/perf-train` @ `3d2f8adf`）
> 平台：平头哥 PPU（thead1/thead2）、昆仑芯 P800（klx1）、昇腾 910C（npu1-11/27）
> 配套：使用文档《使用文档.md》、证据索引《证据索引.md》

## 一、交付范围

1. **标准训练执行**：分类 / 因果语言模型 / CTR（14 架构）/ embedding / rerank
   五类任务；确定性合成语料与密封包（逐文件 sha 校验，包不改一字）。
2. **训练协议能力**：SGD/AdamW、梯度累积、检查点与恢复（含 RNG 状态）、AMP
   （bfloat16/float16）、DDP、FSDP2 分片（per-block+root 自动发现）。
3. **观察态 profiling**（模型层 §5.5 需求）：官方 profiler 会话（thead/PPU、
   Ascend、NVIDIA 三套厂商会话）、19 入口集合通信探针（含 p2p 家族）、训练相
   段区间、等待/带宽分解报表、零漂移对照工具。
4. **厂商平台适配**：PPU（XDNN/XPU 栈）、昆仑芯 P800（XPU/torch 2.9）、
   Ascend 910C（torch_npu/HCCL）。同一套代码、同一协议。
5. **算子与通信 profiling 的完整闭环**：设备内核时间线（算子级，含相区间
   归因）→ 通信画像报表（分列/字节账/等待带宽/相段）→ 训练数值对比
   （`compare`，0 容差逐张量）→ 单算子微基准（op_micro，三平台实测）。
6. **真实框架集成**：已在 FlagScale 自身训练入口实跑通过（见第五部分），
   集成工件随仓库交付。

## 二、模型层（检索组）需求完成情况（重点）

> 对照源：检索组方法论文档（FlagScale `feat/retrieval-flagos`，三层归因方法 +
> §5.5 对运行时层的四条需求 + §5.6 移交边界）。逐条映射到我方交付与证据。
> **证据在哪**：每条需求的实测数字与证据文件见 2.1 表格的"证据"列；
> 归档证据在 `证据索引.md` 指向的目录（`证据-最佳实践实跑/` 为本轮实跑输出，
> `交付材料/20261114-…/evidence-v8/` 为 PPU 观察态全链证据），全部可逐文件
> 校验（SHA256SUMS）。

### 2.1 §5.5 运行时层四条需求（他们明确点名要运行时层做的）

| # | 需求（原文要点） | 状态 | 我方交付 | 证据 |
|---|---|---|---|---|
| 1 | **通信与计算分列**：all-gather / reduce-scatter / p2p 单列，不与算子混排 | ✅ 完成 | 独立通信事件流（t=comm），op 分类+role（model/control/p2p）+字节+peer；19 入口探针（含 p2p 家族与 object 变体） | rerank06b 每卡 172 comm（ag 114/4.14GB + rs 58/4.77GB）双 rank 对称；p2p 车逐项字节账本断言；`evidence-v8/observe-summary.json` |
| 2 | **按 rank 归集**：PP 下每 stage 只持部分层，只看 rank0 会漏瓶颈 | ✅ 完成 | 每 rank 独立 writer store + sealed trace；报表逐 rank 分列 | 双 rank 逐卡数据；`comm-report.json`（comm_by_op/device/segments 均 per rank） |
| 3 | **区分"等待"与"带宽"**：Ring 通信测到的是墙钟停滞，不是带宽；两者分开报 | ✅ 完成（分级） | wait = 跨 rank 内核起点最大差（到达偏斜）；带宽 = payload/传输段。时钟域硬守卫；PPU 厂商分解（控制内核+DMA）逐行标 `vendor_decomposition_detected`，带宽派生量声明不可作线上值 | wait 中位 5.1–523.7 ns 有效（PPU）；P800 因厂商 kineto 无内核事件，wait/带宽不可得（如实标注）；Ascend 平台非确定（行为级） |
| 4 | **同并行度对比**：kernel ms 随卡数变化，跨并行度不可比 | ✅ 完成 | 协议记录 rank/world/backend/sharding/seed；相段按 step 墙钟占比；跨并行度运行一律以协议字段区分 | rerank06b data 相 2.363–2.379s 占 step 38.5–39.8%（2 rank）；P800/Ascend 的 2/4/8/16 卡均记录 world_size |

### 2.2 方法的其余边界（§5.1/5.2/5.3/5.6）

| 条目 | 现状 |
|---|---|
| §5.1 只发现不定位根因 | 同边界：本层给 kernel/通信的可测量身份与占比；kernel 内根因移交算子/编译层 |
| §5.2 每芯片 profiler 调通 | **PPU ✅**、**910C ✅（补上其文档中 910C"只有微基准"的缺口——我方观察链在 910C 实测打通，torch_npu.profiler store 88MB complete）**、**P800 部分**：CPU 侧+xccl 调用层完整，设备内核事件在驱动 5.0.21.47 上两种配方均失败（其文档配方 `XPU_CUPTI_ENABLE_DEVICE=0` 实测报 `etiEventSamplingSetMode error 17`；默认配方 attach 成功但 trace 无 kernel 事件）→ 交厂商联合升级 |
| §5.3 kernel 名映射 | 类别归一已内置（pccl/nccl/xccl/aten 前缀族）；厂商语义→微基准映射仍需人工/厂商提供（与其文档结论一致） |
| §5.6 移交清单 | 其"建议移交运行时层"的**多卡通信与计算时间分解已交付**（§2.1 四条）；"为什么 kernel 慢"保持移交算子/编译层 |

### 2.3 与他们单卡方法的关系

他们三层脚本（`profile_step_segments.py` / `profile_op_micro.py` / 差距归因）
解决单卡口径；本层（多卡）恰是其 §5.5 指名的实现方。两者口径互补：
单卡算子耗时表由他们出，多卡通信分解由我方出，仓库分别维护。

## 三、三平台验证矩阵（实机实测）

### 3.1 回归

| 平台 | 机器 | 结果 |
|---|---|---|
| PPU | thead1 / thead2 | **230 passed / 0 failed**（各 40s） |
| P800 | klx1 | **230 passed / 0 failed**（54s） |
| 910C | npu1-11 | **230 passed / 0 failed**（33s） |

### 3.2 训练

| 项 | PPU | P800 | 910C |
|---|---|---|---|
| CTR 14 架构单卡 | ✅（11-12 全量） | ✅ 13/14（dien 见限制） | ✅ 13/14（dien 厂商 FP16 限制） |
| DDP 多卡 | ✅ 2 卡（11-13） | ✅ 2/4/8 卡（+100 步长跑） | ✅ **8 卡与 16 卡**（20 步） |
| FSDP 0.6B | ✅ 2 卡 | ✅ 2/4 卡 | ✅ 2 卡 |
| **8B FSDP** | ✅ 2 卡首通（11-13） | ✅ 2/4/16 卡（16 卡 47s） | ✅ 4 卡（稳定性轮） |
| embedding last/cls、bge-m3 | ✅ 双机 | ✅ | ✅ |
| rerank 两语义 | ✅ 双机 | ✅ | ✅ |
| checkpoint/resume | ✅ | ✅（单卡+4 卡） | ✅（修复后单卡） |
| AMP bf16 | ✅ | ✅ | —（未单测，套件外） |
| **稳定性压力** | — | 3 组长跑 | **10/10 轮**（5×DDP16-100 步≈30s/轮、3×DDP8-150 步、2×8B FSDP r4） |

### 3.3 观察态 profiling

| 平台 | 状态 |
|---|---|
| PPU | ✅ 全链（19 入口探针、172 comm/卡、六相区间、comm-report、零漂移 16 sha 逐位一致） |
| P800 | ✅ 全链（修复 ext4 门/xfs、identity、厂商 profiler 开关后）——store 114M、trace 192k 事件、comm 171/87 与 store 交叉一致；**设备内核事件缺失为厂商 kineto 限制** |
| 910C | ✅ 全链（vendor=ascend 自动识别，torch_npu.profiler）——store 88M、manifest complete |

### 3.4 运行镜像（三平台，名称 + ID）

| 平台 | 镜像名称 | 镜像 ID |
|---|---|---|
| 平头哥 PPU | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64` | `1ba7ead15378` |
| 昆仑芯 P800 | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` | `cac5ddae5f62` |
| 昇腾 910C | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64` | `62bcaae3f839` |

（910C comm 变体：`flagrt/ascend-operator-runtime-comm:…flagcx0.13.0…`，ID `e13a15d0d02b`，未使用。）

## 四、已知限制（如实，非本轮缺陷）

1. **910C dien**：GRU 融合算子 `DynamicGRUV2` 仅支持 FP16（CANN 9.0/9.1/9.2 实测
   一致），FP32 训练不被支持 → 厂商算子规格缺口，交算子团队；其余 13 架构正常。
   框架不通过静默换 dtype 规避。
2. **910C 位精确性**：平台逐跑非确定（控制实验：同一命令两次 base 运行 final sha
   均不同），sha 级"零漂移"判据在 Ascend 不适用；观察不扰协议以行为级证据论证。
3. **P800 设备内核事件**：厂商 kineto 设备侧采集未接通（trace 仅 CPU 侧 +
   xccl 调用层标注），带宽分解不可做；comm 计数与字节账完整可用。
4. **DDP 传统 Reducer**：DDP 的 C++ Reducer 内部不可见（FSDP2 funcol 已覆盖）。
5. **ascend2 环境**：租户容器上机后 HCCL NPU 网卡绑定失败（`EI0020`，环境级），
   该机 FSDP/观察态冒烟待租户退场复测；能力已由 ascend1 覆盖。

## 五、与 FlagScale（检索组训练框架）的集成建议

### 5.1 定位判断

- FlagScale `feat/retrieval-flagos` = **模型层**（训练配置 `conf/train/bge_m3.yaml`、
  会话式训练脚本、单卡分析脚本）；flagprof = **运行时层**（attach、通信探针、
  per-rank store、多卡报表）。按其三层架构，正确关系是**依赖注入，不是合仓**。
- 合仓不推荐：两层生命周期与维护方不同（模型层随 FlagScale/检索实验演进；
  运行时层随 FlagPerf/厂商适配演进），且我方已按包形态组织（`inference/pyproject.toml`）。

### 5.2 推荐集成路径（按成本从低到高）

1. **零侵入（立即可用）**：训练脚本原样跑，加环境变量
   `FLAGPROF_TRAIN_OBSERVE=<store>`；vendor 自动识别（cuda→thead/nvidia、
   npu→ascend）。前提：脚本用标准 `torch.distributed` 进程组（Megatron/FSDP2 均满足）。
2. **包依赖（推荐落点，已交付适配器并实跑验证）**：**2026-11-18 已在 FlagScale 自身训练入口
   （`flagscale/train/train_retrieval.py`，bge-m3、DDP 双卡、PPU）实跑通过**：
   train_loss 0.116918 / validation 0.000107；store 1.6 GB complete；
   `comm={'model/allreduce': 2}`（评测归约；DDP 梯度 Reducer 按设计不可见）、
   `phases={data:28, forward:27, backward:27, optimizer:27, step:27}`；
   开/关采集两跑 `model.safetensors` sha256 逐字节相同（`4b346bb7…`）。
   集成工件（patch+配置+数据脚本+复现命令）见
   `inference/tools/flagscale_integration/`；实跑证据见
   `证据-最佳实践实跑/fs-real/`。
   适配器本体：`inference/tools/flagscale_adapter.py`
   （`ObservationSession`，约 90 行）——训练入口只加一个 `with obs:` 与五个
   `obs.phase(...)` 标记即可。**已用外部 FSDP2 两卡循环端到端验证**（PPU）：
   双 rank 每步 `allgather 2 + reducescatter 2`、五相 ×2、manifest complete。
   传统 DDP 的 C++ Reducer 不可见（已知边界）；FSDP2 funcol 与 Megatron p2p
   （send/recv/batch_isend_irecv）均覆盖。验证 demo：`inference/tools/flagscale_demo.py`。
3. **脚本级替换（可选）**：其 `profile_step_segments.py` 的 profile 阶段可换成
   flagprof attach——同样"计时与 profile 分阶段"，且直接获得多卡 §5.5 报表，
   避免维护两套采集。
