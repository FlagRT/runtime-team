# performance — 性能评测与诊断

> 分支：`dev-zkm` ｜ PR 目标：`dev-1.0` ｜ 更新：2026-09-16
> 总组速览：[STATUS.md](STATUS.md) ｜ 本文档：详细进展、证据边界和使用入口
> 职责：统一国产设备的算子与基础规格评测入口，保留可归因的正确性、性能、路由、监控和运行环境证据。

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
| 两段 Demo 统一验收报告 | 未完成 | Operation/Base 是支撑能力，尚未完成训练吞吐、推理吞吐/时延与跨方向证据的统一收拢 | 对应战略文档 §3 第 5 条，仍为下一阶段交付 |

当前最重要的边界是：**“研发目录已实现”“某组离线测试通过”“代表算子实机通过”“已发布到独立仓库”是四种不同状态。**
Operation V3 不能仅凭 `ad326754` 的存在被视为 FlagPerf 已发布能力，也不能替代两段 Demo 的最终验收。

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

## 3. 验证结论与边界

### 3.1 52 Case 历史矩阵

- 52/52 个 Case 至少有一个输入类型和注册路径实测通过；这不表示完整矩阵全部通过。
- 最新历史汇总为 308 个适用组合：277 passed、24 blocked、5 failed、2 partial；另有 420 个类型组合不适用。
- 相比上一版的 277 passed、18 blocked、11 failed、2 partial，有 6 个 SPLIT_K/标量 mul 组合从 failed 改为 blocked。
  这是依赖错误分类更准确，不是底层执行能力提升或 kernel 修复。
- FP32 数值失败仍集中在 nativetorch 的 `addmm`、`bmm`、`linear`、`mm`、`mv`；当前证据不能确认唯一根因，
  也没有为得到通过结果而放宽原阈值。
- Torch-FL 注册缺口、FlagGems 注册/RNG/重载问题及 Triton `SPLIT_K` 问题继续保留原始错误；测试层不以 CPU 回退或自动切换路径伪装通过。

### 3.2 V3 验证

- 性能与 Profiling 验收在标准 0.2.0 CANN 9 镜像、逻辑 Device14 上覆盖 abs 双路径 daily/full、
  mm/relu 双路径 FP16 timeline 和默认 daily/off/native；相关 99 项回归通过。该范围不是全部 52 Case 的新增硬件验收。
- 当前完整 Operation 文档记录的最终离线回归为 134 项通过，覆盖公共诊断、报告、进度、异常/中断和历史兼容；
  测试使用只读代码、无网络、未映射 NPU，因此只证明工具机制，不证明设备执行。
- 公共诊断曾对历史失败、blocked、partial、随机/离散输出及缺证据任务做离线检查；尚未新增公共 replay 的 NPU、
  多服务器或第二厂商验收。
- 最新报告样例中的合成成功不是硬件结果；历史失败副本和离线诊断只用于检查表达、链接、确定性和兼容性。

### 3.3 尚不能宣称的能力

- 尚无 NVIDIA—国产设备的逐层中间结果差分、误差传播和唯一根因归因闭环。
- 已有 Operation 级设备时间线和计数器，不等于模型图节点、Backend、通信与运行时状态的完整跨层 Profiling。
- 公共诊断不包含任务自动重试、节点隔离、模型热加载、灰度切换或长期稳定性保障。
- 单机 Device14 的结果不能外推到 16 个逻辑 Device、多机或其他国产设备。

## 4. 环境和设备范围

| 类型 | 锁定或验证身份 | 用途与边界 |
| --- | --- | --- |
| Operation 研发/验证 runtime | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64`，本机 ID `sha256:d9484109...d397` | CANN 9、Python 3.11、PyTorch 2.10、Torch-FL、Triton Ascend、FlagGems；用于当前 Operation 证据，不是两段 Demo 的替代验收镜像 |
| 生效训练腿 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.train` 为准；候选 1.0.0 血统尚未生效 |
| 生效推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | 以 `dev/stack.lock.910c.v2.yaml` 的 `lock.infer` 为准 |
| 宿主工具 | MindCluster ToolBox 26.1.0 | 通过配置的宿主路径只读挂载；版本门禁与容器镜像身份相互独立 |

Operation 历史实机验收限定为主机 `npu1-27`、逻辑 Device14、物理 NPU7 chip0；P2P 使用 NPU6/NPU7。
本机 image ID 不是 registry digest。结论性 Demo 验证必须遵循当前生效的 `dev/stack.lock.910c.v2.yaml`，
不能把 Operation 研发镜像或候选血统混入最终验收。

## 5. 使用入口

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

## 6. 结果解释

| 状态 | 含义 |
| --- | --- |
| `passed` | 当前输入、运行身份和固定门禁下，测量及所需证据通过 |
| `partial` | 主测量可能有效，但监控、路由或其他所需证据不完整 |
| `blocked` | 已确认的底层注册、依赖或环境能力缺口阻止执行 |
| `failed` | 执行、正确性或明确门禁失败 |
| `not-applicable` | 该 Case 与输入类型组合不适用，不计为通过或失败 |

性能比较前必须对齐物理设备、workload、输入、rank、warmup、计时边界、软件栈和计算公式。
诊断的 `completed`、进程退出码 0 或一次 replay 通过都不能覆盖原任务的正确性/路由结论。

## 7. 下一步

1. 审查 `FlagPerf_advance@ad326754` 相对公开 Operation 基线的最小增量，将 V3 选择性同步到 FlagPerf 个人分支并通过独立 PR 交付。
2. 在锁定身份和空闲设备上补代表性公共 replay、timeline/full Profiling 复验；新增证据继续与历史矩阵分开记录。
3. 按生效训练/推理镜像采集两段 Demo 的吞吐和时延，统一预热、并发、计时及状态口径。
4. 汇总 device-context、communication、memory、调度和监控证据，生成战略 §3 第 5 条要求的统一验收报告。
5. 基础验收稳定后，再逐步推进 Oracle—Device 逐层差分、完整跨层 Profiling、故障恢复和长稳/热加载能力。
