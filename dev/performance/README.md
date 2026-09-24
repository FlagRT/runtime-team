# performance — 性能评测与诊断

> 分支：`dev-zkm` ｜ PR 目标：`dev-1.0` ｜ 更新：2026-09-23
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
| Operation 公开基线 | 52 Case 已接入厂商中立 CLI | 最新历史矩阵：308 个适用组合中 277 passed、24 blocked、5 failed、2 partial；420 个不适用项单列 | 公开基线来自 `FlagPerf/dev-zkm@8ded0d74` |
| Operation V3 | 公共诊断、报告、进度、日常指标和可选 Profiling 已实现 | 代表算子硬件验证 + 锁定镜像离线回归；范围见下文 | 仅在 `FlagPerf_advance/dev-zkm@ad326754`，尚未同步到独立 FlagPerf |
| Inference 精度（单卡） | 向量精度工具已形成：off/on 差分、preview 策略、FX/ONNX 导出 | Qwen3-Embedding-0.6B、BF16、物理设备 0、12 条固定输入；28 候选 26 接受、2 排除；模型＋全部 28 层 360 组记录配对有效、异常 0 | 仅在 `FlagPerf_advance/dev-zkm@d49d3910`，未同步独立 FlagPerf |
| Inference 性能（单卡 total） | 时延、吞吐、显存、传输与独立路由取证链路已实测 | 物理设备 1、每侧 270 批次、无执行失败；off 32.242 ms vs on 4010.026 ms/批（124.37 倍，观察值）；51/51 独立复核 | 同上 |
| Inference TP（性能/通信/精度） | 单机 TP=2 HCCL：全局窗口计时、通信归因、逐 rank 精度已实测 | 性能与通信为物理设备 2、4（218/218 复核、2016/2016 事件归因）；精度为物理设备 4、6（12,200 项独立校验） | 同上 |
| 两段 Demo 统一验收报告 | 未完成 | Operation/Base/Inference 是支撑能力；Inference 证据为 PyTorch/Transformers 路径与专用固定输入，不等于 vLLM 服务口径验收；训练吞吐与跨方向证据尚未统一收拢 | 对应战略文档 §3 第 5 条，仍为下一阶段交付 |

当前最重要的边界是：**“研发目录已实现”“某组离线测试通过”“代表算子实机通过”“已发布到独立仓库”是四种不同状态。**
Operation V3 与 Inference V1–V1.3 均不能仅凭 `ad326754`/`d49d3910` 的存在被视为 FlagPerf 已发布能力，也不能替代两段 Demo 的最终验收。
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
[实现文档](../../FlagPerf_advance/inference/implementation_docs/implementation_1.md)。

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

## 4. 验证结论与边界

### 4.1 52 Case 历史矩阵

- 52/52 个 Case 至少有一个输入类型和注册路径实测通过；这不表示完整矩阵全部通过。
- 最新历史汇总为 308 个适用组合：277 passed、24 blocked、5 failed、2 partial；另有 420 个类型组合不适用。
- 相比上一版的 277 passed、18 blocked、11 failed、2 partial，有 6 个 SPLIT_K/标量 mul 组合从 failed 改为 blocked。
  这是依赖错误分类更准确，不是底层执行能力提升或 kernel 修复。
- FP32 数值失败仍集中在 nativetorch 的 `addmm`、`bmm`、`linear`、`mm`、`mv`；当前证据不能确认唯一根因，
  也没有为得到通过结果而放宽原阈值。
- Torch-FL 注册缺口、FlagGems 注册/RNG/重载问题及 Triton `SPLIT_K` 问题继续保留原始错误；测试层不以 CPU 回退或自动切换路径伪装通过。

### 4.2 Operation V3 验证

- 性能与 Profiling 验收在标准 0.2.0 CANN 9 镜像、逻辑 Device14 上覆盖 abs 双路径 daily/full、
  mm/relu 双路径 FP16 timeline 和默认 daily/off/native；相关 99 项回归通过。该范围不是全部 52 Case 的新增硬件验收。
- 当前完整 Operation 文档记录的最终离线回归为 134 项通过，覆盖公共诊断、报告、进度、异常/中断和历史兼容；
  测试使用只读代码、无网络、未映射 NPU，因此只证明工具机制，不证明设备执行。
- 公共诊断曾对历史失败、blocked、partial、随机/离散输出及缺证据任务做离线检查；尚未新增公共 replay 的 NPU、
  多服务器或第二厂商验收。
- 最新报告样例中的合成成功不是硬件结果；历史失败副本和离线诊断只用于检查表达、链接、确定性和兼容性。

### 4.3 Inference V1–V1.3 验证

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

### 4.4 尚不能宣称的能力

- 尚无 NVIDIA—国产设备的逐层中间结果差分、误差传播和唯一根因归因闭环；Inference 的层级差分是同设备
  FlagGems off/on，A100 参考导入未实现。
- 已有 Operation 级设备时间线和计数器、Inference 的 HCCL 通信归因，不等于模型图节点、Backend、通信与运行时
  状态的完整跨层 Profiling。
- 公共诊断与 Inference 运行时均不包含任务自动重试、节点隔离、模型热加载、灰度切换或长期稳定性保障。
- 单机 Device14（Operation）与单机 TP=2/固定输入（Inference）的结果不能外推到更多设备、多机或其他国产设备；
  Inference 性能证据也不构成两段 Demo 的 vLLM 服务口径验收。

## 5. 环境和设备范围

| 类型 | 锁定或验证身份 | 用途与边界 |
| --- | --- | --- |
| Operation 研发/验证 runtime | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`，本机 ID `sha256:d9484109...d397` | CANN 9、Python 3.11、PyTorch 2.10、Torch-FL、Triton Ascend、FlagGems；用于当前 Operation 证据，不是两段 Demo 的替代验收镜像 |
| 生效训练腿 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.train` 为准；候选 1.0.0 血统尚未生效 |
| 生效推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.infer` 为准 |
| Inference 单卡 runtime | `flagperf/inference-ascend:20260922`，本机 ID `sha256:b4a5b2c8...f003` | 由生效推理腿基镜像派生：git archive 固定 FlagGems `f7ae8e6b` 源码＋ONNX 导出依赖；用于单卡精度/性能证据；被测引擎为 PyTorch/Transformers，vLLM 仅作底座，不等于锁定镜像本身 |
| Inference TP runtime | `flagperf/inference-ascend:20260923-tp`，本机 ID `sha256:16c26e9b...dc6d` | 在 20260922 基础上仅追加 `accelerate==1.13.0`（wheel SHA256＋逐包比对）；用于单机 TP=2 HCCL 性能/通信/精度证据 |
| 宿主工具 | MindCluster ToolBox 26.1.0 | 通过配置的宿主路径只读挂载；版本门禁与容器镜像身份相互独立 |

Operation 历史实机验收限定为主机 `npu1-27`、逻辑 Device14、物理 NPU7 chip0；P2P 使用 NPU6/NPU7。
Inference 实机设备：单卡精度物理设备 0、单卡性能物理设备 1（同板设备 0 当时有外部负载）；TP 性能/通信物理
设备 2、4；TP 精度物理设备 4、6（设备 2 被其他容器占用，EBUSY）。
本机 image ID 不是 registry digest。结论性 Demo 验证必须遵循当前生效的 `dev/stack.lock.910c.v2.yaml`，
不能把 Operation 研发镜像或 Inference 派生镜像混入最终验收。

## 6. 使用入口

### 6.1 Operation V3

Operation V3 尚未同步到独立 FlagPerf，以下命令从 `FlagPerf_advance` 根目录执行：

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
独立 FlagPerf 当前仍使用其仓内公开基线 README；在 V3 同步完成前，不能直接复制上述新参数到旧入口。

### 6.2 Inference V1–V1.3

Inference 尚未同步到独立 FlagPerf，以下命令从 `FlagPerf_advance/inference` 目录执行；宿主需 Python 环境并安装
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

## 7. 结果解释

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

## 8. 下一步

1. 审查 `FlagPerf_advance@ad326754`（Operation V3）与 `@d49d3910`（Inference V1–V1.3）相对公开基线的最小增量，
   分别选择性同步到 FlagPerf 个人分支并通过独立 PR 交付。
2. Operation 在锁定身份和空闲设备上补代表性公共 replay、timeline/full Profiling 复验；Inference 优先补算子级
   profiling 归因单卡 124×/TP 46× 变慢，并在无共享负载环境复测性能基线；新增证据继续与历史矩阵分开记录。
3. 按生效训练/推理镜像采集两段 Demo 的吞吐和时延：Inference 已有 PyTorch/Transformers 路径总级数据，
   仍需 vLLM 服务口径与统一预热、并发、计时及状态口径。
4. 汇总 device-context、communication、memory、调度和监控证据，生成战略 §3 第 5 条要求的统一验收报告。
5. 基础验收稳定后，再逐步推进 Oracle—Device 逐层差分（含 A100 参考导入）、完整跨层 Profiling、故障恢复和
   长稳/热加载能力。
