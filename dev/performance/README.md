# performance — 性能评测与诊断

本目录是 runtime-team 中性能方向的协作入口，说明如何使用独立的 FlagPerf 仓库、采用哪套运行环境，以及怎样解释验证结果。它不复制 FlagPerf 源码，也不在 runtime-team 中维护另一套执行器或容器编排。

## 1. 仓库与职责边界

| 位置 | 职责 | 是否在本仓库维护 |
| --- | --- | --- |
| `dev/performance/README.md` | 团队入口、环境约束、执行流程、验证边界和后续任务 | 是 |
| `FlagPerf/base/` | Benchmark、Toolkit、监控、报告及 Ascend 适配的正式代码 | 否；在独立 FlagPerf 仓库通过其分支和 PR 维护 |
| `FlagPerf/base/result/` | 单次运行的原始日志、状态、配置快照和报告 | 否；运行产物，不提交到 runtime-team |

runtime-team 的 `dev-1.0` 只接收上述协作信息。FlagPerf 的代码变更、运行时锁文件和硬件证据应进入 FlagPerf 仓库，而不是把 `FlagPerf_advance` 或其他本地检出目录纳入本仓库。

当前入口覆盖 Ascend 910C/CANN 9 单宿主的基础规格评测：

- Benchmark：保留原 Base Case 的配置、warmup、计时和结果语义；
- Toolkit：执行 MindCluster ToolBox、DMI、`npu-smi` 或 HCCL 的厂商测量与诊断；
- report：从已有证据确定性重建 Markdown/SVG 报告，不重跑硬件。

精度差分、模型级 Profiling、故障恢复和热加载属于后续能力，不应由当前基础规格评测结果代替。

## 2. 执行架构

```text
runtime-team 宿主
  └─ FlagPerf/base/run.py
       ├─ BenchmarkExecutor -> 锁定容器 -> 原 Base Case
       ├─ ToolkitExecutor   -> 锁定容器 -> DMI/npu-smi/HCCL
       └─ report            -> 已保存证据 -> Markdown/SVG
```

Benchmark 与 Toolkit 只共享运行时锁、设备选择、preflight、资源 lease 和外层证据约定；二者的权限、计时公式、Case 执行和结果 schema 保持独立。

这里不新增 `docker-compose.yml`。FlagPerf 已由宿主入口按锁文件启动实际测量容器；再套一层开发容器会引入 Docker socket、宿主设备路径和结果目录的双重映射，反而模糊实际运行身份。

## 3. 环境基线和已验证边界

| 类型 | 运行时 | 当前边界 |
| --- | --- | --- |
| 标准 operator runtime | `flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64` | CANN 9.0.0、Python 3.11.15、PyTorch 2.10、Torch-FL、Triton Ascend、FlagGems；标准 Benchmark/Toolkit 基线 |
| FlagCX communication candidate | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | 仅用于明确允许的单机双 rank P2P Case；仍是 candidate，不具备生产资格 |
| 宿主工具 | MindCluster ToolBox 26.1.0 | 通过宿主路径只读挂载，路径必须按执行机器配置 |

以上镜像 tag 是 FlagPerf 仓库锁文件声明的运行身份，不等于镜像已发布到团队可访问的 registry。2026-09-02 检查时，当前 Docker daemon 中没有这些镜像，registry 发布状态和可拉取 digest 也尚未验证；执行者必须在正式运行前补做拉取/构建与镜像身份核验，不能把本地 image ID 当作 registry digest。

现有标准 runtime 和 communication candidate 的验证状态均为 `partial`。已有证据只覆盖声明设备上的静态检查及有限硬件路径；它不代表 16 个逻辑 Device、全部 Case、异常恢复、跨机通信、长稳或正式性能基线已验收。CANN 8.5 不属于本方向当前开发和证据范围。

权威细节以 FlagPerf 中以下文件为准：

- `base/vendors/ascend/torch_fl_2.10/stack.lock.yaml`
- `base/vendors/ascend/torch_fl_2.10/image-manifest.json`
- `base/vendors/ascend/torch_fl_2.10/validation-summary.json`
- `base/vendors/ascend/torch_fl_2.10_flagcx/validation-summary.json`

## 4. 宿主准备

正式运行前逐项确认：

1. Docker daemon 可用，目标镜像已按锁文件构建或拉取并核验身份；
2. `npu-smi info` 能看到本轮获授权的 NPU，并确认目标 Device 空闲；
3. 宿主驱动、固件、DMI、ToolBox 26.1.0 和设备节点存在；
4. 只选择本轮被授权的物理 NPU 或逻辑 Device；
5. 已审查 privileged 容器以及 Toolkit 命令对共享机器的影响。

不要为本机路径修改并提交 FlagPerf 的共享配置。可复制一份运行配置到仓库外：

```bash
cp FlagPerf/base/configs/ascend910_cann9_local.yaml /tmp/flagperf-ascend910c.json
# 按执行宿主修改 /tmp/flagperf-ascend910c.json 中的 toolbox_host_path、host_mounts 等字段。
```

该源文件虽然以 `.yaml` 结尾，当前内容和解析契约为 JSON；复制为 `.json` 是为了明确本地配置格式。`/tmp` 文件不进入任何仓库。

## 5. 推荐执行流程

所有命令从 FlagPerf 仓库根目录执行。以下 NPU 7 只是示例，必须替换成本轮获授权且空闲的设备。

先检查入口并生成静态计划：

```bash
cd FlagPerf
python3 base/run.py --help

python3 base/run.py benchmark run \
  --config /tmp/flagperf-ascend910c.json \
  --case computation-FP16 \
  --npu-ids 7 \
  --monitor on \
  --dry-run

python3 base/run.py toolkit run \
  --config /tmp/flagperf-ascend910c.json \
  --case computation-FP16 \
  --npu-ids 7 \
  --dry-run
```

`--dry-run` 只检查仓库配置并打印静态计划，不检查 Docker、NPU 占用或物理到逻辑 Device 映射。正式运行前仍必须完成 preflight。

确认静态计划、设备空闲和权限后，再运行实际测量：

```bash
python3 base/run.py benchmark run \
  --config /tmp/flagperf-ascend910c.json \
  --case computation-FP16 \
  --npu-ids 7 \
  --nproc-per-node 2 \
  --monitor on \
  --allow-privileged-root

python3 base/run.py toolkit run \
  --config /tmp/flagperf-ascend910c.json \
  --case computation-FP16 \
  --npu-ids 7 \
  --allow-privileged-root \
  --allow-disruptive-dmi
```

Toolkit 的 DMI/HCCL 命令具有独立授权边界。容量/OOM 等高风险 Case 还需显式使用 `--allow-high-risk-case`；communication candidate 还需 `--allow-candidate-runtime`。不能把这些开关写进默认配置以绕过逐次审查。

结果默认位于 `FlagPerf/base/result/<RUN_ID>/`。离线重建报告示例：

```bash
python3 base/run.py report --run-id benchmark-YYYYMMDDTHHMMSSZ
```

## 6. 结果解释与证据要求

| 状态 | 含义 |
| --- | --- |
| `execution_status` | 执行器和容器生命周期是否正常完成 |
| `measurement_status` | workload 或厂商工具是否产生有效测量 |
| `monitoring_status` | 请求的同窗监控证据是否完整；`--monitor off` 为 `not-run` |
| `postflight_status` | 运行后设备和宿主检查是否完成 |
| `report_status` | 报告是否从已保存证据成功生成 |

这些状态必须独立记录。监控不完整不能被写成测量失败，也不能把成功测量包装成完整验收。退出码 `2` 表示配置/授权错误，或主测量通过但所请求证据不完整而得到 `partial`。

性能比较前必须对齐物理设备范围、workload、并发 rank、warmup、计时边界、软件栈和计算公式。Toolkit microbenchmark 与 Benchmark Case 未完成上述对齐时，只能分别报告事实，不能直接相除或据此声称唯一根因。

一次可审计运行至少应保留：`summary.json`、`resolved-plan.json`、领域结果 manifest、原始 stdout/stderr、配置快照、pre/postflight、报告和 SHA-256 索引。

## 7. 后续协作任务

- 为两个 runtime 建立可访问的 registry 发布流程并记录不可变 digest；
- 将宿主 ToolBox 路径等机器差异参数化，在 FlagPerf 独立仓库提交最小 PR；
- 明确 FlagPerf 的共享开发分支和 PR 规则，避免以本地 `FlagPerf_advance` 目录代替正式变更；
- 在授权资源上补齐标准 runtime 全 Case、candidate 异常恢复、跨机、长稳和正式性能验收；
- 在基础规格证据稳定后，再逐步接入精度差分、模型级 Profiling、故障恢复与热加载接口。
