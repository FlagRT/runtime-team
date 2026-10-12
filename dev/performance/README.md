# performance — 性能评测与诊断

> 分支：`dev-zkm` ｜ PR 目标：`dev-1.0` ｜ 更新：2026-10-09
> 总组速览：[STATUS.md](STATUS.md) ｜ 本文档：详细进展、证据边界和使用入口
> 职责：统一国产设备的算子、推理与基础规格评测入口，保留可归因的正确性、性能、路由、监控和运行环境证据。

本目录是 runtime-team 的性能方向协作入口。正式代码在独立的
[FlagRT/FlagPerf](https://github.com/FlagRT/FlagPerf) 仓库维护；本仓只同步阶段进展、环境约束、验证边界和后续任务，
不复制 FlagPerf 源码或实验产物。

## 1. 当前进展

| 能力 | 当前状态 | 已验证范围 | 交付状态 |
| --- | --- | --- | --- |
| Base Benchmark | Ascend CANN 9/Torch-FL 适配和统一宿主入口已形成 | 保留原 Case 配置、warmup、计时和结果语义；现有实机证据以具体 Case、设备和报告为准，不外推为全 Case 稳定基线 | 已发布至 `FlagPerf/dev-zkm@3e7c558b`，未进入 `main` |
| Base Toolkit | 厂商工具执行、选卡、证据链和人类可读报告已形成 | MindCluster ToolBox、DMI、`npu-smi`、HCCL；测量、诊断、监控和健康状态独立记录 | 同上 |
| FlagCX P2P | 声明范围内已完成资格验证 | 单机双 rank、NPU6/NPU7、SIO/HCCS_SW、4/16/64/256 MiB；C0–C5 与 30-run compact formal 矩阵 | 公开内容已进入 `FlagPerf/dev-zkm`；不外推到反向、多机或长稳 |
| Operation 公开基线 | 52 Case 已接入厂商中立 CLI | 最新历史矩阵：308 个适用组合中 277 passed、24 blocked、5 failed、2 partial；420 个不适用项单列 | 公开基线来自 `FlagPerf/dev-zkm@8ded0d74`；V3 增量已随 2026-09-27 迁移进入 `FlagPerf/dev-zkm@1eba1052` |
| Operation V3 | 公共诊断、报告、进度、日常指标和可选 Profiling 已实现 | 代表算子硬件验证 + 锁定镜像离线回归；范围见下文 | diagnose、失败补采、日常指标、独立 profiling、进度与报告已于 2026-09-27 选择性公开迁移；研发仓演进至 `FlagPerf_advance/dev-zkm@76f1f733` |
| Inference 精度（单卡） | 向量精度工具已形成：off/on 差分、preview 策略、FX/ONNX 导出 | Qwen3-Embedding-0.6B、BF16、物理设备 0、12 条固定输入；28 候选 26 接受、2 排除；模型＋全部 28 层 360 组记录配对有效、异常 0 | V1 已随 2026-09-27 迁移公开；后续增量仅在 `FlagPerf_advance/dev-zkm` |
| Inference 性能（单卡 total） | 时延、吞吐、显存、传输与独立路由取证链路已实测 | 物理设备 1、每侧 270 批次、无执行失败；off 32.242 ms vs on 4010.026 ms/批（124.37 倍，观察值）；51/51 独立复核 | 同上 |
| Inference TP（性能/通信/精度） | 单机 TP=2 HCCL：全局窗口计时、通信归因、逐 rank 精度已实测 | 性能与通信为物理设备 2、4（218/218 复核、2016/2016 事件归因）；精度为物理设备 4、6（12,200 项独立校验） | 同上 |
| Inference 组件开关与层级性能（V1.4–V1.6） | FlagGems/FlagTree/FlagCX 显式 off/on/both、联合 preview、`performance --level layer`、报告筛选/portable/重分析、preview 续探与分组搜索已实现 | 三组件单轴实机对照（99 项离线＋9 项实机＋2,044 项复算）；层级性能 496,476 项一致性检查；首次完整 preview 墙时单卡 −53.8%/−54.5%、TP −55.4%（周转观察值，非推理加速） | V1.4/V1.5 随 09-27、层级性能与 preview 优化随 09-29 公开迁移至 `FlagPerf/dev-zkm@a5f26fe9` |
| Inference A100 参考（V2/V2.1） | `oracle` 精度参考包、`performance-baseline` 基线包生成与 `--oracle`/`--baseline` 只读导入比较、入口收敛与统一结果精简已实现 | Qwen3 BF16 单卡/TP=2：A100 模型级＋全部 28 层精度参考与同规模性能比较；CPU FP64 逐样本/层指标，倍率＝被测/A100；不设达标阈值 | 仅在 `FlagPerf_advance/dev-zkm`，未同步独立 FlagPerf |
| Inference 推荐模型（V3/V3.1） | DeepFM、DLRM（共享 RecommendationAdapter）与 OneRec-8B 有状态生成契约已接入 | DeepFM TP hidden 差分 max_abs≈5.59e-8；DLRM 398 项回归；OneRec 12 条 token 轨迹一致、每份 profile 2,304 AllReduce＋32 AllGather 全归因 | 同上 |
| Inference 监控旁路与成本优化（V4） | `TelemetryProvider`（A3/910C DCMI）默认秒级采集与 `monitor.html`；schema 4 比较核心包、`--dry-run`/`analyze`/`repack-reference` 离线恢复已实现 | Qwen3 TP=2 监控验收 895 次采集/29,117 条记录；小规模开销对照 +0.0658%（95% 上限 +0.69%）；OneRec 比较核心 18.78 GB→52.12 MB；2.62 GB trace 解析 165.70 s→111.91 s | 同上 |
| 两段 Demo 统一验收报告 | 未完成 | Operation/Base/Inference 是支撑能力；Inference 证据为 PyTorch/Transformers 路径与专用固定输入，不等于 vLLM 服务口径验收；训练吞吐与跨方向证据尚未统一收拢 | 对应战略文档 §3 第 5 条，仍为下一阶段交付 |

当前最重要的边界是：**“研发目录已实现”“某组离线测试通过”“代表算子实机通过”“已发布到独立仓库”是四种不同状态。**
Operation V3 与 Inference V1–V1.6 已于 2026-09-27/29 选择性公开迁移（`FlagPerf/dev-zkm@a5f26fe9`），但 Inference V2–V4（A100 参考、推荐模型、监控旁路、成本优化）仍仅在 `FlagPerf_advance/dev-zkm@76f1f733`；任何一者都不能仅凭提交存在被视为已发布能力，也不能替代两段 Demo 的最终验收。
Inference 的性能数字在采集期间存在同板卡外部负载，属于功能与测量链路验收，不是独占环境基线。

## 2. Operation V2/V3 新进展

### 2.1 从一次运行到可复核结论

当前工具不是简单执行算子后打印耗时，而是把原任务、补充检查和事后阅读分开：

```text
run
 ├─ 正确性、路由和日常测量
 ├─ failed / partial 时按条件自动补采公共诊断
 └─ 封存 task/result/summary/artifacts → report

封存的单个任务
 └─ diagnose
     ├─ 默认：离线校验和整理已有证据，不启动 Docker/NPU
     └─ --replay：原输入/原镜像上的参考检查 + 一次原路径 probe
        └─ 写入独立 diagnosis-* 目录，不覆盖原结论
```

`run --diagnostics failures` 默认开启。数值失败补查保存输入、模块状态、CPU 参考和误差分布；仅路由 `partial`
时至多补采一次 CPU profiler 调用链；初始化、OOM、执行异常只整理已有证据，避免在设备状态不确定时重复执行。
`--diagnostics off` 只关闭附加检查，不关闭原正确性、路由、错误和健康门禁。

公共 `diagnose` 默认服务所有 Case/dtype/路径，而非只服务 MM。检查完成表示工具流程完成，不表示原算子通过；
CPU 调用链只是路径线索，不能证明设备 kernel，也不能把 `partial` 提升为 `passed`。当前入口不执行数学模式干预、
专项根因归因、自动修复、重试或节点隔离。

### 2.2 报告、实时进度与终端查询

- 运行报告按“本次结论 → 判断依据 → 异常与下一步 → 性能怎么读 → 输入/计时配置 → 证据附录”组织。
  只有 probe/measurement 正确性、目标路由、无测量回退和中位耗时同时满足时，数据才进入通过性能表；其余耗时标为仅供诊断。
- `run`、Profiling、自动失败补采和 `diagnose` 向 stderr 输出当前组合、阶段、已完成数、任务耗时和累计耗时；
  心跳不进入算子测量窗口，`completed` 表示执行结束数，不是通过数。
- `list` 保留默认完整 JSON，并增加 `--names-only` 和 `--case CASE` 两种 shell 友好查询；查询只读取 catalog，
  不初始化厂商、Docker、NPU 或 legacy SSH。
- `report` 只从封存证据重建 Markdown，不补跑缺失检查，也不修改原 JSON、哈希或状态。

### 2.3 日常指标和 Ascend Profiling

默认 `--workload daily --profiling off`。日常测量新增轮间统计、中位数置信区间、逻辑有效带宽和独立分配器显存窗口；
主指标仍是同步主机批次耗时除以调用次数后取轮次中位数，不是纯设备 kernel 时间。

`--profiling timeline|full` 使用独立复放，不污染日常计时。timeline 通过 CANN HostToDevice 关系把 MSTX 目标窗口关联到
设备任务；full 再分组采集 ArithmeticUtilization、PipeUtilization、Memory、MemoryL0、MemoryUB 和资源冲突计数器。
累计时间、并集时间和首尾跨度分别保存，不能互相替代，也不能用“主机时间减 kernel 累计时间”直接推断提交开销。

性能对比只支持同一次 `--oplib both` 中、共享输入和身份且双方完整门禁通过的 nativetorch/FlagGems 配对。
不支持跨运行拼接、历史基线自动比较或仅凭路径名称判断底层 kernel 不同。

## 3. Inference V1–V1.3 新进展（2026-09-16 → 09-23）

本周把旧 inference 的“任务评分＋性能混合”入口重建为面向 FlagGems 的推理精度与性能工具。模型为 Qwen3-Embedding-0.6B，
执行链路 PyTorch/Transformers → torch_npu → 原生/选择性 FlagGems；vLLM 镜像只提供底座，不参与被测模型执行。
三个提交依次为 `7449aab9`（V1 单卡精度）、`b86ee2c3`（V1.2 单卡性能＋TP）、`d49d3910`（V1.3 TP 精度），均在
`FlagPerf_advance/dev-zkm`。实现细节见 [inference README](../../FlagPerf_advance/inference/README.md) 与
[实现文档](../../FlagPerf_advance/inference/implementation_docs/implementation_1.md)。V1.4 以来的续作（组件开关、
层级性能、A100 参考、推荐模型、监控旁路与成本优化）见下文 §4。

### 3.1 向量精度工具（V1）

- 旧“任务评分”流程与全部旧模型、厂商编译器、SSH 编排整体归档至 `inference/legacy`（184 个文件，SHA256 manifest
  校验一致）；配置集中到 `config/default.yaml`；输入改为参考 x-benchmark 设计的 12 条固定文本；权重需自行下载，
  程序离线加载，不自动下载模型或数据。
- `accuracy` 以同设备原生输出（off）为参照，比较按已验证策略选择性启用 FlagGems 的输出（on）；off/on 复用同一份
  tokenized inputs、独立进程执行。指标覆盖模型级（pooled vector＋归一化 embedding）与层级（全部 28 个 Transformer
  block 或指定模块）的 MSE、MAE、最大绝对/相对误差、余弦相似度，保留逐样本与各指标最差样本；不设精度阈值，
  off 不是数学真值或 NVIDIA Oracle。
- `preview` 自动探测 FlagGems 候选：先跑原生基线并把实际 ATen 调用映射到候选函数，再按稳定顺序逐个加入、以全量
  输入试验；只有完成、输出有限且实际命中的候选被接受。单卡实测 28 个候选接受 26、确认排除 2（`cat`：Triton 编译
  日志后进程 -11；`expand`：不支持 `implicit` keyword 的 TypeError）、未知 0；最终组合复验全部命中。
- 单卡实机（物理设备 0、BF16、batch 4、形状 `[4,20]/[4,16]/[4,256]`）：模型＋全部 28 层共 360 组样本/边界记录
  配对有效、无非有限值；embedding MSE 约 `3.45e-7`、最低逐样本余弦约 `0.9996`。仅模型、指定层分项与 FX/ONNX
  CPU 导出（round-trip 逐元素一致、checker 通过）另见实现文档；36 项 CPU 合同测试通过。

### 3.2 单卡总级性能与 TP 通信归因（V1.2）

新增与 `accuracy` 分离的 `performance` 子命令（total level）和单机张量并行（TP）支持：

- **单卡总级性能**：主时延覆盖设备驻留输入 → 前向/pooling/归一化 → 设备同步，默认 5 轮预热、30 轮测量、3 组重复，
  off/on 交替先后、每重复新进程隔离注册状态。显存按权重 storage（约 1136.35 MiB）、加载增量、预热后基线、测量峰值
  分列；输入/输出拷贝独立计时，不进入主时延；路由取证在独立进程完成，不污染正式计时。物理设备 1 实测
  off 平均 `32.242 ms/批`、on `4010.026 ms/批`，on 约为 off 的 **124.37 倍**（124.061 vs 0.9975 样本/s）；每侧
  270 批次、无执行失败，51/51 项独立复核一致。这是当前镜像、策略和输入下的观察，不是 FlagGems 普遍性能结论；
  同板另一 chip 当时有外部评测负载，本轮为功能与测量链路验收。
- **单机 TP**：`config/tp.yaml` ＋ torchrun，每 rank 独立进程、HCCL 承载模型通信、Gloo 仅用于测试协调；全局批次
  窗口取 `max(end)-min(start)`，样本/token 不按 rank 翻倍。派生镜像 `20260923-tp` 仅追加 `accelerate==1.13.0`
  （父镜像 ID 与 wheel SHA256 校验、逐包比对确认唯一差异）；解决普通用户容器 HCCL `ra init failed[19]`
  （任务专用 UID/GID passwd/group 只读映射）。物理设备 2、4 实测 off `48.257 ms`、on `2214.774 ms`，
  主窗口累计比 **45.895**；218/218 封存数据复核、57 项单元测试通过。
- **通信归因**：正式计时不带 profiler，另起相同配置进程组用 `torch_npu.profiler` 采样；通过
  correlation/connection 链把模块标记 → `Enqueue/Dequeue@HcclAllreduce` → CANN node → 设备事件逐段关联。
  12 份 rank 采样共 **2016/2016** 次 collective 事件全部归因到 Attention/MLP 输出聚合（各 84 次、各约 64 MiB 逻辑
  payload/rank/采样）；链路统计只取 CANN 矩阵 total 视图，避免 HCCS/SDMA 重叠字段重复计数。诊断观察：on 的计算
  任务区间并集（约 6246–6260 ms）远大于 off（约 49–52 ms）而通信区间未相应增加，指向**优先检查 FlagGems 计算路径**；
  未定位具体算子，非根因确认。
- **数值交叉**：同一 tokenized archive 下单卡原生、TP off、TP on 两两 embedding 余弦均 ≥ `0.9996`，描述性差异、
  无阈值判定。

### 3.3 TP 逐 rank 精度与策略身份复验（V1.3）

- `accuracy` 接入 TP：逐 rank 将 off/on 的同名样本与边界配对计算指标，不跨 rank 拼接分片输出，不平均跨 rank 误差；
  单卡策略不能复用于 TP，TP preview 必须覆盖全部 rank。配置迁移为 `config/default.yaml`＋`config/tp.yaml` 覆盖式。
- 实机（物理设备 4、6，普通用户容器 `privileged=false`）：模型＋全部 28 层每 rank 30 个边界、360 条逐样本记录，
  配对有效、异常 0；embedding MSE 约 `2.658e-7`、最大绝对差约 `0.00396`、最低样本余弦约 `0.9996`。另完成仅模型、
  指定 3 个 block＋`layers.0.self_attn.q_proj` 分片（每 rank `[4,20,1024]`，验证本地分片边界）、off-only/on-only
  分项，共 5 组运行。
- 执行源码变化使旧 TP 策略身份失效；本轮以“历史选择作为候选、重跑完整原生基线＋选择组合复验”形成新策略：
  26 接受、0 新确认排除、2 未知（`cat`/`expand` 未在新身份重试，不继承旧排除结论），覆盖状态 `partial`。
  身份差异与复验来源封存于 `reuse-decision.json`，正式入口保持严格身份校验，无自动策略迁移。
- 设备排查：原定设备 2、4 在 prepare 报不可用，syscall 跟踪定位 `/dev/davinci2` 打开返回 EBUSY（另一容器占用）；
  改映射 4、6 后完整运行，rank 0→物理 4、rank 1→物理 6，结论封存于 `device-diagnosis.json`。
- 独立校验脚本只读封存张量重算全部 FP64 误差：5 组运行 **12,200 项校验全部通过**；70 项离线合同测试通过
  （覆盖配置目录迁移、逐 rank 配对、padding 排除、失败证据和报告合同）。

## 4. Inference V1.4–V4 新进展（2026-09-24 → 10-09）

自 V1.3 之后两周，Inference 从“单模型、同设备差分”扩展为“多模型、三组件、A100 跨厂商参考、带监控与成本约束的
比对工具链”。九个功能提交依次为 `8e0091ad`（V1.4 组件开关）、`51de6408`（V1.5 preview 续探）、`0f2acf5b`
（V1.6 preview 分组优化）、`9cc0a57a`/`7c56a5da`（V2/V2.1 A100 精度与性能）、`cb12fb16`（V3 公共契约与
DeepFM）、`7386948f`（V3.1 DLRM 与 OneRec）、`376bf09c`（V3.2 A100 收口与 profiler 优化）、`76f1f733`
（V4 监控旁路），另有 10-07/10-08 的离线成本优化交付，均在 `FlagPerf_advance/dev-zkm`。V1.4–V1.6 已随
09-27/29 公开迁移进入独立 FlagPerf（`a5f26fe9`）；V2–V4 尚未同步。实现细节见
[inference README](../../FlagPerf_advance/inference/README.md) 与[指标来源矩阵](../../FlagPerf_advance/inference/implementation_docs/metrics-source-matrix.md)。

### 4.1 组件开关、层级性能与 preview 效率（V1.4–V1.6，09-26 → 09-29）

- **三组件开关（V1.4）**：FlagGems/FlagTree/FlagCX 显式 `off/on/both`（默认全 off、一次最多一个 both）。FlagTree
  通过新进程实际 triton 导入切换；FlagCX 切换模型通信组、Gloo 保留为测试协调组。联合 preview 各环境探测后取共同
  include 集合并逐侧、逐 rank 完整复验，避免不同算子覆盖混入比较。验收：83 项离线＋三组件单轴实机对照；使用体验
  增量（结果解释层、逐环境策略覆盖、形状/repeat 对照）另完成 99 项离线、9 项实机、2,044 项独立复算。
- **执行身份拆分与续探（V1.5）**：策略升为 schema 3，`identity_key`（执行身份）与 `analysis_key`（纯分析源码）
  分离，完整源码快照防止运行期间修改后发布新 verified 策略；`preview --resume-from --budget-seconds` 复用封存
  进度，恢复顺序含旧接受集合完整复验，未完成复验不发布策略。144 项离线回归、实机窗口 118.56 分钟。
- **层级性能**：`performance --level layer` 在完整模型内观察全部 28 个 block 或指定模块——层级 Event 计时、硬件
  profiler、分组显存分别执行，证据按相同输入、组件状态、repeat、batch 和 rank 关联。报告分摘要/逐层索引/完整明细
  三层，支持 `report --layers/--ranks/--shapes/--portable/--reanalyze`。179 项离线回归＋12,920 项视图核验；
  单卡 FlagGems 封存短测整体 on/off 比 86.89、TP 29.59 作为观察直接呈现并提供逐层追查入口。
- **preview 效率（V1.6）**：默认轻量证据（成功试验输出 tensor 约 4.09 GB → 0，全层有限值检查保留）、同 worker
  编译器身份复用、续探导入封存 Triton 缓存组（同源续探 876.90 s → 475.01 s，−45.8%）；新增分组搜索（最多 8 个
  候选原子接受、失败二分定位）与自适应预算。实机对照：首次完整 preview 墙时单卡 −53.8%/−54.5%、TP −55.4%；
  242 项离线回归＋4,962 项独立证据核验。这些是**搜索周转收益**，不是模型推理加速。

### 4.2 A100 参考与同规模离线比较（V2/V2.1，09-29 → 09-30）

- **精度参考（oracle）**：A100 原生 BF16 参考包保存固定 token/mask/批次、完整原 dtype 输出、模型摘要、环境与
  源码，逐文件 SHA256 校验后长期复用；`accuracy --oracle` 只读导入，新增 `oracle-comparison.json` 沿用 CPU
  FP64 逐样本/层指标；TP 仅接受完整 block/pooled/embedding，各 rank 分别比较、不拼接分片。
- **性能基线（performance-baseline）**：A100 单卡/TP=2 总级＋全部 28 层性能包，包含多次独立测量；`--baseline`
  同规模比较，倍率＝被测/A100；NCCL kernel 与 CANN collective elapsed 不混算时间倍率，不新增业务达标阈值。
- **入口收敛与结果精简（V2.1）**：A100 入口收敛为 `run_inference.py oracle` 与 `performance-baseline`；所有
  普通 accuracy/performance 默认在成功后清理本次原始输出（保留指标、日志、配置与来源），失败/partial/显式 keep
  保留完整现场，中断记为 `cleanup_partial`。
- 这是 **Oracle—Device 逐层差分链路的第一段落地**：A100 参考可复用、可被 910C 离线比较；但比较无阈值，误差
  传播与唯一根因归因仍未闭环。

### 4.3 推荐模型接入（V3/V3.1，10-01 → 10-05）

- **公共契约**：ModelAdapter 必需方法/能力检查、统一 LoadedModel/DeviceSession；参考包升为 schema 2，离线读取
  使用封存 workload 的边界与样本轴，不需要加载模型实现。
- **DeepFM（10-01 接入、10-04 TP＋hidden 差分）**：FP32、封存 421,452 参数、128 离散样本；默认差分对象改为
  MLP 最后隐藏块 `hidden[B,400]`（probability 显式开启）；TP=2 embedding 按特征维 AllGather＋4 个 Linear
  输入列 AllReduce；单卡→TP hidden max_abs 5.59e-8；384 项 CPU 回归＋17 项 Ascend 运行。
- **OneRec-8B（10-02）**：有状态生成契约——ExecutionSession/ExecutionStep、精度遵守 EOS 并截到首次 emitted
  token 分歧、性能分 prefill/decode/生成吞吐与独立 KV/allocator 测量。378 项 CPU 回归＋22 项实机主运行；
  12 条 token 轨迹与原生一致；每份通信 profile 完整归因 2,304 次 AllReduce＋32 次 AllGather。
- **DLRM（10-04）**：EmbeddingBag/dot/cat interaction、每字段变长 ID 袋；与 DeepFM 共享 RecommendationAdapter
  的封存加载/输出/参考/吞吐契约。398 项 CPU 回归；27 项主流程 26 completed、1 partial（TP FlagTree 在线
  profiler trace 截断，副本重导出后 12 份 profile 完整解析、每份 124/124 通信归因，原始 partial 保留）。
- 边界：合成资产验证的是测试工具，不评价真实 CTR/AUC、业务评分或稳定收益。

### 4.4 大 trace 分析、诊断语义与 A100 收口（10-05 → 10-06）

- **CUDA/Ascend 大 trace 分析与恢复**：按主机线程的区间树、作业内共享层归因与 NCCL 上下文，CUDA 分析放入默认
  900 秒独立子进程（`--analysis-timeout-seconds`）；超时/异常保留原始采集与既有计时、分析标记 partial。OneRec
  A100 single/TP2 消费队列按此恢复。
- **诊断语义更新**：零范数余弦、零参考 RMS 的归一化 RMSE 归入 `undefined_metrics`（6 条旧异常更正，绝对误差为
  零），真正 NaN/Inf、shape、空输出仍保留异常语义；性能报告新增独立重复 CV（阈值 0.10）与单次尖峰筛查（比值 5），
  需复查时保留完整现场、不改变执行状态、不剔除尖峰。
- **A100 验收收口**：已完成运行只读复核，原始证据与新视图分别归档。

### 4.5 统一监控旁路（V4，10-06 → 10-07）

- **机制**：`performance` 默认启动独立 collector，模型适配器不增加监控代码；厂商通过 `TelemetryProvider` 提供
  能力（首个实现 A3/910C DCMI＋CLI 降级）。默认每秒采集原生引擎/HBM 带宽活跃度、HBM 容量、共享功率域、频率/温度
  及 worker CPU/RSS；阶段事件关联加载/预热/测量/profiler，生成可离线筛选的 `monitor.html`。`--monitor off`
  关闭、`--monitor-interval-ms` 调间隔；监控状态独立于性能执行/分析状态。
- **实机验收**：Qwen3 TP=2（FlagTree/FlagCX on、FlagGems off/on 各一 repeat）执行/分析/监控均 completed；
  895 次采集、29,117 条记录；360 条逐 rank batch 时间戳与 latency 一致；profiler 冲突利用率保守屏蔽、短阶段
  样本不足保留 partial。
- **小规模开销对照（10-07）**：单卡、三组件全 off、默认 1000 ms 采集，6 组交替 on/off 共 1,080 个正式 batch；
  配对几何平均时延增幅 +0.0658%、单侧 95% t 上限 +0.6888%，通过本场景“平均增幅不超过 1%”判据。不外推 TP=2、
  其他负载或长期稳定性。

### 4.6 比对全流程成本优化与设备续验（10-07 → 10-09）

- **schema 4 存储与离线恢复**：比较核心包与摘要绑定的独立证据附件分离；OneRec single/TP2 比较核心从 18.78/
  72.80 GB 降至约 52.12/112.75 MB（完整附件 17.40/43.82 GB 保留诊断现场）。新入口：`--dry-run` 成本预检
  （不启动设备、输出 estimated 而非 verified）、`analyze` 离线恢复（检查点复用、失败不发布伪完成目录）、
  `repack-reference` 旧包迁移（schema 1–3 只读兼容）。475 项 CPU 回归、15 个旧 A100 包迁移等价检查、8 个正式
  消费任务离线复核通过。
- **10-08 耗时修正**：CUDA 在内存预算允许时整批解码＋直接索引（不足回退 SQLite）、worker 暂停循环 GC、确定性
  gzip level 1；OneRec 同一 2.62 GB trace 解析 165.70 s → 111.91 s（−32.5%），层归因与通信逐字段一致；
  482 项 CPU 回归。为热缓存共享主机单次观察，不外推稳定收益。
- **Ascend 续验（10-08）**：完成 21 个展开任务——DeepFM/DLRM/OneRec 原生单卡/TP2 精度＋层级性能 12 项、DeepFM
  FlagGems 单卡/TP2 新 preview 与 off/on 6 项、FlagTree 单卡 3 项；16,458 项交付检查＋695,910 项原始 trace
  核对通过。Qwen 原共享权重发现非有限值，健康副本按历史摘要恢复至独立输出目录，未改共享原文件。
- **10-09 续验**：Qwen 单卡/TP2 精度与性能 4 项复验通过（275/248/406/323 项交付检查＋3,259/13,133 项层级
  复核）；DeepFM FlagTree TP2 层级性能复核通过（2,125 项），联合 preview 存在候选组失败记录；进度快照状态为
  in_progress。
- **待资源**：DeepFM FlagTree/FlagCX TP2 剩余项与 A100 12 项设备短测。开跑前 A100 SSH 拒绝连接、本地 16 个
  Ascend 芯片被现有服务占用，按约定明确推迟、未停止他人服务；待补矩阵单独保存。

## 5. 验证结论与边界

### 5.1 52 Case 历史矩阵

- 52/52 个 Case 至少有一个输入类型和注册路径实测通过；这不表示完整矩阵全部通过。
- 最新历史汇总为 308 个适用组合：277 passed、24 blocked、5 failed、2 partial；另有 420 个类型组合不适用。
- 相比上一版的 277 passed、18 blocked、11 failed、2 partial，有 6 个 SPLIT_K/标量 mul 组合从 failed 改为 blocked。
  这是依赖错误分类更准确，不是底层执行能力提升或 kernel 修复。
- FP32 数值失败仍集中在 nativetorch 的 `addmm`、`bmm`、`linear`、`mm`、`mv`；当前证据不能确认唯一根因，
  也没有为得到通过结果而放宽原阈值。
- Torch-FL 注册缺口、FlagGems 注册/RNG/重载问题及 Triton `SPLIT_K` 问题继续保留原始错误；测试层不以 CPU 回退或自动切换路径伪装通过。

### 5.2 Operation V3 验证

- 性能与 Profiling 验收在标准 0.2.0 CANN 9 镜像、逻辑 Device14 上覆盖 abs 双路径 daily/full、
  mm/relu 双路径 FP16 timeline 和默认 daily/off/native；相关 99 项回归通过。该范围不是全部 52 Case 的新增硬件验收。
- 当前完整 Operation 文档记录的最终离线回归为 134 项通过，覆盖公共诊断、报告、进度、异常/中断和历史兼容；
  测试使用只读代码、无网络、未映射 NPU，因此只证明工具机制，不证明设备执行。
- 公共诊断曾对历史失败、blocked、partial、随机/离散输出及缺证据任务做离线检查；尚未新增公共 replay 的 NPU、
  多服务器或第二厂商验收。
- 最新报告样例中的合成成功不是硬件结果；历史失败副本和离线诊断只用于检查表达、链接、确定性和兼容性。

### 5.3 Inference V1–V1.3 验证

- 实机范围：单卡（设备 0 精度、设备 1 性能，镜像 `flagperf/inference-ascend:20260922`）与 TP=2（设备 2、4
  性能/通信；设备 4、6 精度，镜像 `20260923-tp`），均为 Qwen3-Embedding-0.6B、BF16、12 条固定输入、batch 4、
  三个形状。TP=4/8、其他模型/dtype/输入、NVIDIA 均无实机证据（NVIDIA 路径只有接口合同测试）。
- 性能数字是观察值不是基线：单卡 on/off 主窗口比 124.37、TP 比 45.895。on 变慢未归因到具体算子（需算子级
  profiling）；调用级路由取证（26 函数全部命中、单卡策略原生占比 35.437%）是 Python 调用证据，
  不是硬件 kernel fallback 率或耗时占比。
- 精度数字是描述性差异：单卡与 TP 的 embedding MSE 在 `e-7` 量级、最低样本余弦 ≥ `0.9996`、异常 0；
  off 是同设备原生参照，不是绝对正确性 Oracle，也不设达标阈值。
- 独立复核：单卡性能 51/51、TP 性能 218/218、TP 精度 12,200/12,200；离线合同测试随版本从 36 项演进到 70 项，
  只证明工具机制，不证明设备执行。
- 环境边界：单卡性能测量时同板另一 chip 有外部评测；TP 运行中被测设备无其他进程但不证明整机隔离。HCCL 通信
  归因适配的是锁定镜像、当前 trace 格式；通信 elapsed/wait/synchronization 字段不可相加、不可当通信占比除以主窗口。

### 5.4 Inference V1.4–V4 验证边界

- 实机范围仍限 Qwen3-Embedding-0.6B、DeepFM、DLRM、OneRec-8B 的固定/封存输入与单卡、TP=2；TP=4/8、其他
  dtype/输入、第三厂商无实机证据。A100 侧仅生成参考/基线包，不在 A100 上运行被测组件路径。
- A100 参考包锁定模型内容、输入与数值语义；oracle/baseline 比较无达标阈值，跨服务器独立重复不配对，NCCL kernel
  与 CANN collective elapsed 不混算倍率。
- preview 轻量证据/缓存续探/分组搜索是**搜索周转收益**，不是模型推理加速；组内成员共享一次组合证据，接受顺序与
  最终集合可能不同于逐个探测。
- 监控：秒级遥测用于趋势与阶段关联，短 batch 和算子耗时仍以原计时/profiler 为准；A3 功率含共享板卡域，不能按
  TP rank 相加；开销判据仅覆盖单卡默认间隔场景。
- 成本优化：核心包缩小与 trace 解析提速为特定任务观察，不宣称普遍降低解析时长或推理提速；`analyze` 恢复不重跑
  设备、不改写原执行状态；10-09 续验快照仍为 in_progress，待补设备矩阵未完成。
- 推荐模型为合成资产功能验收，不代表真实 CTR/AUC、业务评分或稳定收益；OneRec 生成指标不等于推荐质量。

### 5.5 尚不能宣称的能力

- Oracle—Device 差分已落地第一段（A100 精度参考与同规模性能基线的只读比较），但**误差传播、敏感算子归因与唯一
  根因闭环未实现**；Inference 的层级差分对象仍以同设备 off/on 为主，A100 参与的比较不设阈值。
- 已有 Operation 级设备时间线和计数器、Inference 的 HCCL 通信归因与 CUDA/Ascend 大 trace 分析（含独立时限与
  恢复），不等于模型图节点、Backend、通信与运行时状态的完整跨层 Profiling。
- 公共诊断、Inference 运行时与监控旁路均不包含任务自动重试、节点隔离、模型热加载、灰度切换或长期稳定性保障。
- 单机 Device14（Operation）与单机 TP=2/固定输入（Inference）的结果不能外推到更多设备、多机或其他国产设备；
  Inference 性能证据也不构成两段 Demo 的 vLLM 服务口径验收。

## 6. 环境和设备范围

| 类型 | 锁定或验证身份 | 用途与边界 |
| --- | --- | --- |
| Operation 研发/验证 runtime | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`，本机 ID `sha256:d9484109...d397` | CANN 9、Python 3.11、PyTorch 2.10、Torch-FL、Triton Ascend、FlagGems；用于当前 Operation 证据，不是两段 Demo 的替代验收镜像 |
| 生效训练腿 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.train` 为准；候选 1.0.0 血统尚未生效 |
| 生效推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.infer` 为准 |
| Inference 单卡 runtime | `flagperf/inference-ascend:20260922`，本机 ID `sha256:b4a5b2c8...f003` | 由生效推理腿基镜像派生：git archive 固定 FlagGems `f7ae8e6b` 源码＋ONNX 导出依赖；用于单卡精度/性能证据；被测引擎为 PyTorch/Transformers，vLLM 仅作底座，不等于锁定镜像本身 |
| Inference TP runtime | `flagperf/inference-ascend:20260923-tp`，本机 ID `sha256:16c26e9b...dc6d` | 在 20260922 基础上仅追加 `accelerate==1.13.0`（wheel SHA256＋逐包比对）；用于单机 TP=2 HCCL 性能/通信/精度证据 |
| Inference 三组件/TP runtime | `flagperf/inference-ascend:20260923-tp-flagtree-flagcx`，本机 ID `sha256:1147704d...7bf03` | 在 20260923-tp 基础上提取厂商编译器并集成 FlagTree/FlagCX 依赖；用于组件开关、层级性能、推荐模型、监控旁路与 10-08/10-09 续验证据 |
| A100 采集环境 | 既有 CUDA 容器内的独立虚拟环境（PyTorch/CUDA/NCCL） | 仅用于 `oracle` 精度参考与 `performance-baseline` 基线包生成；锁定身份与合同见实现文档六/七，不参与 910C 被测链路 |
| 宿主工具 | MindCluster ToolBox 26.1.0 | 通过配置的宿主路径只读挂载；版本门禁与容器镜像身份相互独立 |

Operation 历史实机验收限定为主机 `npu1-27`、逻辑 Device14、物理 NPU7 chip0；P2P 使用 NPU6/NPU7。
Inference 实机设备：单卡精度物理设备 0、单卡性能物理设备 1（同板设备 0 当时有外部负载）；TP 性能/通信物理
设备 2、4；TP 精度物理设备 4、6（设备 2 被其他容器占用，EBUSY）。V1.4 以后轮次：三组件与公开迁移验收单卡
设备 4、TP 设备 4/6（Operation 另用设备 8）；层级性能与 preview 优化对照单卡设备 4、TP 设备 4、8；监控验收
设备 2、4、开销对照设备 2；10-08 续验按任务分配，10-09 续验使用设备 4、5。
本机 image ID 不是 registry digest。结论性 Demo 验证必须遵循当前生效的 `dev/stack.lock.910c.v2.yaml`，
不能把 Operation 研发镜像或 Inference 派生镜像混入最终验收。

## 7. 使用入口

### 7.1 Operation V3

Operation V3 功能已于 2026-09-27 选择性公开迁移到独立 FlagPerf（自 `dev-zkm@1eba1052` 起）；以下命令是研发仓
`FlagPerf_advance` 的最新入口，公开仓入口见 FlagPerf 的 operation README：

```bash
# 终端友好地查看 Case 和支持类型；不访问 Docker/NPU
python3 operation/run.py list --names-only
python3 operation/run.py list --case mm

# 查看默认 52 Case 计划；dry-run 不证明镜像或设备可用
python3 operation/run.py run \
  --vendor ascend \
  --device-ids 14 \
  --dry-run

# 同次双路径日常测试，并单独采集设备 timeline
python3 operation/run.py run \
  --vendor ascend \
  --device-ids 14 \
  --case abs \
  --oplib both \
  --workload daily \
  --profiling timeline \
  --allow-privileged-root

# 默认离线检查一个封存的单任务目录
python3 operation/run.py diagnose \
  --source-task operation/result/<run-id>/<task-dir>

# 显式同输入复放；开始前重新确认原镜像、权限和空闲设备
python3 operation/run.py diagnose \
  --source-task operation/result/<run-id>/<task-dir> \
  --replay \
  --device-ids 14 \
  --allow-privileged-root

# 从现有封存证据重建报告，不重跑硬件
python3 operation/run.py report --run-dir operation/result/<run-or-diagnosis-id>
```

正式执行前必须核验镜像身份、获授权且空闲的物理设备、逻辑 Device 映射、设备租约和 privileged 容器影响。
`nativetorch` 在 Ascend 上对应 Torch-FL；`torch-fl` 不是当前 `--oplib` 的合法取值。
独立 FlagPerf 的公开入口与研发仓参数可能存在版本差（公开迁移基线为 `a5f26fe9`），跨仓使用前先核对各自 README。

### 7.2 Inference V1–V1.3

Inference V1–V1.6 已于 2026-09-27/29 公开迁移到独立 FlagPerf（`dev-zkm@a5f26fe9`）；V2–V4 仅在研发仓。以下命令从
`FlagPerf_advance/inference` 目录执行；宿主需 Python 环境并安装
`requirements.txt`（设备侧依赖在镜像内）：

```bash
# 先构建派生镜像（参数为本地 FlagGems 仓库路径；固定 commit 导出，不含未提交修改）
bash docker_images/ascend/build.sh /path/to/FlagGems

# 单卡：preview 生成策略 → 精度 → 性能（同一份策略）
python run.py list
python run.py preview --output /absolute/new-preview
python run.py accuracy \
  --policy /absolute/new-preview/preview/policy.yaml \
  --levels model layer --output /absolute/new-accuracy
python run.py performance --level total --flaggems both \
  --policy /absolute/new-preview/preview/policy.yaml \
  --output /absolute/new-performance

# 单机 TP（HCCL，设备数 2/4/8；须先构建 TP 镜像）
bash docker_images/ascend/build_tp.sh
python run.py preview --config config/tp.yaml --devices 4 6 \
  --output /absolute/new-tp-preview
python run.py accuracy --config config/tp.yaml --devices 4 6 \
  --flaggems both --levels model layer \
  --policy /absolute/new-tp-preview/preview/policy.yaml \
  --output /absolute/new-tp-accuracy
python run.py performance --config config/tp.yaml --devices 4 6 \
  --level total --flaggems both \
  --policy /absolute/new-tp-preview/preview/policy.yaml \
  --output /absolute/new-tp-performance
```

正式执行前必须核验镜像身份、获授权且空闲的物理设备、设备租约和容器权限。Inference 的策略身份包含模型、输入、
镜像、设备、依赖与执行源码；身份变化须重新完整 preview，`report` 展示层修改除外。`--device`（单卡）与
`--devices`（TP）不混用；TP 的 `export` 不支持。详细口径见
[inference README](../../FlagPerf_advance/inference/README.md)。

### 7.3 Inference V1.4–V4 新入口

以下命令同样从 `FlagPerf_advance/inference` 目录执行；参数口径与身份校验规则同 §7.2：

```bash
# 三组件开关：--flaggems/--flagtree/--flagcx 各自 off|on|both（默认全 off，一次最多一个 both）
python run.py preview --flaggems both --output /absolute/new-preview
python run.py performance --level layer --flagtree both \
  --policy /absolute/new-preview/preview/policy.yaml \
  --output /absolute/new-layer-run

# A100 参考（只读导入；倍率＝被测/A100，不设达标阈值）
python run.py accuracy --oracle /absolute/a100-oracle-package \
  --levels model layer --output /absolute/new-accuracy
python run.py performance --level total --baseline /absolute/a100-baseline-package \
  --output /absolute/new-performance

# preview 续探与分组搜索（周转优化，不是推理加速）
python run.py preview --resume-from /absolute/old-preview \
  --budget-seconds 600 --preview-search grouped --preview-budget adaptive \
  --output /absolute/new-preview

# 连续设备监控（默认 on；秒级趋势，短 batch/算子耗时仍读原计时与 profiler）
python run.py performance --level total --monitor on --monitor-interval-ms 1000 \
  --output /absolute/new-monitored-run

# 成本预检与离线恢复（不启动设备）
python run.py performance --config config/deepfm.yaml --level layer \
  --dry-run --output /absolute/new-run
python run.py analyze --source /absolute/saved-run --output /absolute/recovered-run
python run.py repack-reference --source /absolute/old-package \
  --output /absolute/migrated-reference
```

`--dry-run` 只输出 estimated/verified=false 的阶段与空间估计；`analyze` 仅恢复离线分析、比较与封包，不重跑设备，
失败保留 `.incomplete-*` 与 `recovery-failure.json`；`repack-reference` 输出 schema 4 的 core/evidence 结构。
推荐模型分别通过 `--config config/deepfm.yaml`、`config/dlrm.yaml`、`config/onerec.yaml` 进入，新增模型必须
重新 preview。详细机制见[实现文档十一至十六](../../FlagPerf_advance/inference/implementation_docs/implementation_11.md)。

## 8. 结果解释

| 状态 | 含义 |
| --- | --- |
| `passed` | 当前输入、运行身份和固定门禁下，测量及所需证据通过 |
| `partial` | 主测量可能有效，但监控、路由或其他所需证据不完整 |
| `blocked` | 已确认的底层注册、依赖或环境能力缺口阻止执行 |
| `failed` | 执行、正确性或明确门禁失败 |
| `not-applicable` | 该 Case 与输入类型组合不适用，不计为通过或失败 |

性能比较前必须对齐物理设备、workload、输入、rank、warmup、计时边界、软件栈和计算公式。
诊断的 `completed`、进程退出码 0 或一次 replay 通过都不能覆盖原任务的正确性/路由结论。
Inference 沿用同一语义：执行完成不等于精度达标（无阈值），也不等于 FlagGems 被实际调用（以独立路由取证为准）；
TP 样本数不乘卡数，策略 coverage 为 `partial` 时表示未重测候选，不表示已选组合复验失败。
V1.4 以后另区分：零范数余弦、零参考 RMS 等不可定义指标归入 `undefined_metrics`（不是异常）；性能报告的重复
CV/尖峰筛查标记“需复查”时不改变执行状态、不剔除尖峰；监控 `monitoring.status` 与性能执行/分析状态相互独立。

## 9. 下一步

1. 收拢待补设备矩阵：资源空闲后补 DeepFM FlagTree/FlagCX TP2 剩余项与 A100 12 项短测，收口 10-09 in_progress
   快照，并更新交付索引。
2. 将 Inference V2–V4（A100 参考、推荐模型、监控旁路、成本优化）审查后以最小增量选择性同步到 FlagPerf 个人
   分支并通过独立 PR 交付；Operation/Base 公开内容保持随 V1.4–V1.6 的迁移基线 `a5f26fe9` 演进。
3. Inference 优先补算子级 profiling 归因单卡 124×/TP 46× 变慢（A100 参考已可对照），并在无共享负载环境复测
   性能基线；新增证据继续与历史矩阵分开记录。
4. 按生效训练/推理镜像采集两段 Demo 的吞吐和时延：Inference 已有 PyTorch/Transformers 路径总级数据与 A100
   同规模比较，仍需 vLLM 服务口径与统一预热、并发、计时及状态口径。
5. 汇总 device-context、communication、memory、调度和监控证据，生成战略 §3 第 5 条要求的统一验收报告；基础
   验收稳定后，再逐步推进 Oracle—Device 误差传播与敏感算子归因闭环、完整跨层 Profiling、故障恢复和长稳/热加载
   能力。
