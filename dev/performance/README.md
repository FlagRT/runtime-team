# performance — 性能评测与诊断

> 分支：`dev-zkm` ｜ PR 目标：`dev-1.0` ｜ 更新：2026-09-09
> 职责：统一国产设备的算子与基础规格评测入口，保留可归因的正确性、性能、路由、监控和运行环境证据。

本目录是 runtime-team 的性能方向协作入口。正式代码在独立的
[FlagRT/FlagPerf](https://github.com/FlagRT/FlagPerf) 仓库维护；本仓只同步阶段进展、环境约束、验证边界和后续任务，
不复制 FlagPerf 源码或实验产物。

## 1. 当前进展

| 能力 | 当前状态 | 已验证范围 | 交付状态 |
| --- | --- | --- | --- |
| Base Benchmark | 已完成 Ascend CANN 9/Torch-FL 适配 | 保留原 Case 配置、warmup、计时和结果语义；FP16 双 rank 在 NPU7/Device14,15 通过，两个 rank 各 `285.22 TFLOPS`，fallback=0 | 本地集成提交 `FlagPerf/dev-zkm@77b25848`，尚未发布到 FlagRT/FlagPerf 远程 |
| Base Toolkit | 已完成厂商工具执行与证据链 | MindCluster ToolBox 26.1.0、DMI、`npu-smi`、HCCL；测量、诊断和权限边界独立记录 | 同上 |
| 监控与报告 | 已完成基础协议 | 精确测量窗监控、pre/postflight、资源 lease、原始日志及确定性 Markdown/SVG 报告 | 同上 |
| FlagCX P2P | 声明范围内已晋级 | 单机双 rank、NPU6/NPU7、SIO/HCCS_SW、4/16/64/256 MiB；C0–C5 和 30-run compact formal 矩阵通过 | 晋级修改和证据仍位于本地 `FlagPerf_advance`，尚未发布 |
| Operation | 52 个 Case 已接入统一 CLI | 308 个适用组合中 277 passed、18 blocked、11 failed、2 partial；另有 420 个类型组合不适用 | 本地 `FlagPerf_advance/dev-zkm@a7530617` 后仍有未提交修改，尚未发布 |

这里的“已验证”表示有本地源码和实验记录支持，不等于已经进入团队远程仓。2026-09-09 核验时，
FlagRT/FlagPerf 远程只有 `main@66eb17e4`；后续必须先把 Base 和 Operation 变更整理为可审查的独立 PR。

## 2. 验证结论与边界

### Base 与 P2P

- 标准 runtime 只重跑了完整 FP16 Case，未覆盖全部 Base Case，因此整体验证仍为 `partial`。
- P2P 的 `c5-compact-v1` 证据包含 24 个规模曲线 run 和 6 个 monitor A/B run；30 个 run 的执行、测量与 postflight 均通过，最大 CV 为 2.063%，监控中位数差异为 0.0033%，fallback=0。
- P2P 晋级只适用于证据中声明的单机、双 rank、正向 SIO/HCCS_SW 范围；反向、多机和长稳未验证。
- P2P 报告沿用 FlagPerf 原有 `2 * bytes / elapsed_time` 公式。将其解释为原始单向有效载荷带宽时，数值应减半。

### Operation

- 52/52 个 Case 至少有一个输入类型和注册路径实测通过；这不表示完整矩阵全部通过。
- 默认 52 项单路径运行耗时 56.68 秒：45 passed、5 个 FP32 数值检查失败、2 个后端注册缺口；最长适用单组合约 116.14 秒。
- FP32 数值失败集中在 `addmm`、`bmm`、`linear`、`mm`、`mv`；当前证据不能证明严格 IEEE FP32 通过，也尚未形成唯一根因。
- `nativetorch` 的 `isnan`、`rsub` 存在 Torch-FL 注册缺口；部分 FlagGems 组合暴露注册、RNG、`SPLIT_K` 或重载兼容问题。测试层保留原算子调用，不以 CPU 回退或自动切换路径伪装通过。
- 当前计时是包含设备同步的主机批次时间；纯 kernel 时间和逐层 Oracle—Device 误差归因尚未实现。

## 3. 环境和设备范围

| 类型 | 锁定运行身份 | 当前边界 |
| --- | --- | --- |
| 标准 operator runtime | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` | CANN 9.0.0、Python 3.11.15、PyTorch 2.10、Torch-FL、Triton Ascend、FlagGems；本机 image ID `sha256:d9484109...d397` |
| FlagCX communication runtime | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | P2P 声明范围；本机 image ID `sha256:3b9e08f2...b6bc` |
| 宿主工具 | MindCluster ToolBox 26.1.0 | 通过宿主路径只读挂载，具体路径按执行机器配置 |

本机 image ID 不是 registry digest。镜像的团队 registry 发布状态和跨机可拉取性尚未验证。
CANN 8.5 不属于当前开发和证据范围。

Operation 实机验收限定为主机 `npu1-27`、逻辑 Device14、物理 NPU7 chip0；P2P 使用 NPU6/NPU7。
不能将这些单机结果外推为 16 个逻辑 Device、跨机或其他国产设备已经通过。

## 4. 使用入口

从独立 FlagPerf 工作树运行。Base 每次 `benchmark run` 只执行一个 Case；完整套件需要显式逐项调度。

```bash
# Base：先用 dry-run 检查静态计划
python3 base/run.py benchmark run \
  --config /tmp/flagperf-ascend910c.json \
  --case computation-FP16 \
  --npu-ids 7 \
  --monitor on \
  --dry-run

# Operation：查看默认 52 Case 计划
python3 operation/run.py run \
  --vendor ascend \
  --device-ids 14 \
  --dry-run

# 从已保存证据离线重建报告，不重跑硬件
python3 base/run.py report --run-id <benchmark-run-id>
python3 operation/run.py report --run-dir operation/result/<operation-run-id>
```

正式测量前必须核验镜像身份、获授权且空闲的物理设备、逻辑 Device 映射、ToolBox 路径和 privileged
容器影响。`--dry-run` 不执行 Docker/NPU preflight，也不证明设备可用。详细 CLI、Case 覆盖表和原始证据
随 FlagPerf 代码 PR 一并发布；在此之前以本地 `FlagPerf_advance/operation/README.md` 和
`FlagPerf_advance/operation/vendors/ascend/README.md` 为维护源。

## 5. 结果解释

| 状态 | 含义 |
| --- | --- |
| `passed` | 当前输入、运行身份和固定门禁下，测量及所需证据通过 |
| `partial` | 主测量可能有效，但监控、路由或其他所需证据不完整 |
| `blocked` | 已确认的底层注册或环境能力缺口阻止执行 |
| `failed` | 执行、正确性或明确门禁失败 |
| `not-applicable` | 该 Case 与输入类型组合不适用，不计为通过或失败 |

性能比较前必须对齐物理设备范围、workload、rank、warmup、计时边界、软件栈和计算公式。
Benchmark Case 与 Toolkit microbenchmark 未完成这些对齐时，只能分别报告事实，不能直接相除或据此声称唯一根因。

## 6. 下一步

1. 将 Base 集成提交整理并推送到 FlagRT/FlagPerf 个人分支，通过 PR 合入团队开发分支。
2. 完成 Operation 工作树审查和提交，以独立 PR 发布统一 CLI、Ascend 适配、覆盖表及可复核证据。
3. 修复或归因 FP32 数值偏差及后端注册问题，补充同条件重复测量，不放宽既有正确性门禁。
4. 发布两个 runtime 的可访问镜像和不可变 registry digest，并参数化宿主 ToolBox 路径。
5. 基础规格证据稳定后，逐步接入纯 kernel Profiling、Oracle—Device 逐层差分、故障恢复和长稳运行接口。
