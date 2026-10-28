# SiLU 接入与安全回退：训练 v3 回归 case

## 用途与现状

10-23新增独立启动准入验证，见下方“启动准入三路”。关闭/允许原生各3步训练通过，严格要求目标为BLOCKED，未导入FlagGems；这不替代10-16目标编译失败结论。旧完整回归入口及历史日期说明保留。

本 case 消费总组训练镜像 `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`，不构建新镜像。
镜像 ID 锁定 `sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15`。
公共规范锚为 runtime-team `74197bf7d4b5dc8cdd3b2b89b15ac3561936dda6`：`dev/ENV-SPEC.md`、`dev/images/CASE-TEMPLATE.md`、训练 v3 demo。

**2026-10-16：11上已完成strict启动及最小NPU张量检查，回归结论见本次报告；不是正式验收。**
已由公共 `baseline.py export` 生成 [运行基线](demo/env.baseline.yaml)，实际记为 `dirty: true`；7项入口控制＋5项策略＋11项CPU桥接控制通过。11实机5类原生回退通过，目标SiLU首个FP32前向因缺shmem编译失败，三步训练未执行。CPU测试不代表真实FlagGems内核通过。

- 27 有 v3 镜像，但实际 ID 为 `7028028bb62c…`，与此 case 锁定 ID 不同，不能改文件让检查假通过；已有他人带卡容器。
- 11有锁定ID镜像；个人目录已准备Compose/PyYAML，通过sudo使用Docker，不改用户组或系统包。
- 本人容器完成测试后已停止，未进入他人容器，未更改公共镜像。完整记录见 [本次报告](../../docs/训练v3基线与回归准备-20261016.md)。
- 已知 shmem 编译路径问题仍可能影响 FlagGems；此 case **不挂载 stub、不改内核、不放宽数值容差、不在执行失败后重试原生**。
- 这是局部实验，不是完整三级回退、vLLM 或多卡训练验收。默认准入策略不变。

## 复现

### 启动准入三路

使用本方向0.3.0.dev0候选wheel、已核对device-context runtime快照、PyYAML/Compose和上述精确v3镜像，不需要模型权重。默认只映射一个已协调的设备，不挂shmem stub、不改变镜像包。

```bash
python3 dev/framework-adapter/cases/ascend-train-v3/assets/run_startup.py \
  --public-root /绝对路径/公共runtime-team \
  --source /绝对路径/本方向dev/framework-adapter \
  --runtime /绝对路径/device-context/prototype \
  --wheel /绝对路径/flagrt_framework_adapter-0.3.0.dev0-py3-none-any.whl \
  --device 0 --output /绝对路径/全新输出目录
```

入口生成image.baseline.yaml和demo，执行公共check/export/up.sh --strict，离线安装wheel，运行合成SiLU小模型并自动停止本人容器。原始结果、命令、输入hash、安装路径及退出状态保留在output。不修改公共启动脚本。

预期：off/degrade各3步原生训练，loss/参数逐值一致；strict在训练前BLOCKED，目标导入和调用均为0。外层`overall: PASS`只表示这三个控制行为符合预期，不表示目标内核成功。超时、原生环境损坏或其他异常为ERROR，不伪装成成功回退。最新结果见[1023报告](../../docs/环境准入与算子扩展验证-20261023.md)。

### 前置条件

| 项 | 要求与取得方式 |
|---|---|
| 主机与卡 | 1个空闲设备；10-16用户明确授权11/27空卡可用，本次只用11物理2，不进入他人容器。其他时段仍须核对分配规则和实时占用 |
| 镜像 | 从总组训练 v3 离线归档获取上述精确 image ID；不将27的同源镜像视作逐字节相同 |
| 宿主工具 | Python + PyYAML、Docker CLI + Compose v2；有 Docker 使用权限，不自动改用户组 |
| 公共代码 | 上述公共 commit 的 checkout/export，设为 `PUBLIC_ROOT`；本 case 的 `public-baseline` 由 prepare 生成软链，不复制公共镜像档案 |
| 本方向代码 | runtime-team 本人 checkout；当前代码未发布，基线中的8ecf3ae只表示HEAD，`dirty: true`不代表新文件在远端。对外验收前须授权发布、干净checkout并用公共工具 `export --require-clean` 重建基线 |
| 设备 runtime | 沿用10-14实测的 `0c59b9549d45d55b959294869f605a30dc3789c9`；本地Git必须已有该对象。入口自动导出该提交的runtime源文件到结果目录，SHA256锁定并记录；不会误用个人分支中的旧device-context子树 |
| 模型 | 原始 Qwen3-Embedding-0.6B，不是10步微调产物。27已核验副本 `/home/cgu135/framework-adapter-910c/acceptance-20260916/model-verified-20261014`；如在11运行，须先获准复制到本人目录。`model.safetensors` SHA256：`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| 数据 | 复用固定种子随机 token 的三步梯度/权重回归，不下载数据集，不声称正式业务训练 |
| 输出 | 宿主本人持久目录，挂载 `/case-output`；日志与完整JSON留此处，不进Git |

### 准备与启动（宿主、本人仓根）

以下为可替换路径的使用说明；本次11实际路径和执行结果见报告。

```bash
export PUBLIC_ROOT=/absolute/path/to/public-runtime-team
export WORKSPACE_ROOT="$PWD"
export FLAGRT_WORKSPACE_ROOT="$PWD"
export FLAGRT_MODEL_DIR=/absolute/path/to/verified-Qwen3-Embedding-0.6B
export FLAGRT_MODELS_DIR="$FLAGRT_MODEL_DIR"
export FLAGRT_CASE_OUTPUT=/absolute/path/to/own-results
export FLAGRT_DEVICE=2  # 宿主空闲物理编号；容器内单卡重新编号为0
mkdir -p "$FLAGRT_CASE_OUTPUT"
python3 dev/framework-adapter/cases/ascend-train-v3/assets/prepare.py --public-root "$PUBLIC_ROOT"
# 确认已协调带卡容器时段，再执行；up.sh 会真正启动容器。
bash "$PUBLIC_ROOT/dev/lib/up.sh" dev/framework-adapter/cases/ascend-train-v3/demo --strict
```

只核对时使用公共 `baseline.py check`，不要调用 up.sh；check会写 `.last-check.json`。
本demo消费公共镜像基线，但独立声明单卡配置，不继承公共demo的16卡映射。只读源码挂到`/case-workspace`，避开镜像默认`/workspace/FlagPerf`工作目录。`--device`必须与FLAGRT_DEVICE一致；入口核对物理挂载后在容器内使用逻辑0。

### 单入口验证（宿主、本人仓根）

```bash
python3 dev/framework-adapter/cases/ascend-train-v3/assets/run.py \
  --public-root "$PUBLIC_ROOT" --output "$FLAGRT_CASE_OUTPUT" \
  --model "$FLAGRT_MODEL_DIR" --device "$FLAGRT_DEVICE"
```

`--device`传宿主物理设备号。入口不启动/停止容器、不安装依赖；核对已有本人容器的精确image ID、FlagGems标签、挂载及包目录改动，随后执行公共baseline检查、模型哈希检查，记录代码哈希与runtime快照。
每个测试进程使用新Triton缓存；超时由容器内 `timeout` 终止，最多900秒＋10秒强制退出宽限。前一阶段失败不继续模型训练。
结束后仅在确认无其他任务使用时停止本人容器，不执行全机清理。

## 验证

| 顺序 | 验证内容 | 通过判据 |
|---|---|---|
| 1 | 5类执行前保留原生：开关关闭、环境不符、dtype、shape、layout | 每类0次Gems命中；前向/梯度与原生精确一致；注册恢复 |
| 2 | FP32/FP16/BF16单算子前反向，另含5类回退与一次单算子推理 | 真实Gems前反向命中、原阈值通过、退出后原生恢复；推理仅命中前向 |
| 3 | Qwen0.6B FP32原生/替换各3步 | loss、全部梯度、更新后参数按原 `atol=1e-5, rtol=1e-4` 对比；失败不继续更新 |

阶段2/3复用旧脚本会重复单算子门控，统计时不重复累加。
`overall: PASS`/0 = 本case全部检查完成；`FAIL`/1 = 验证脚本产出失败；`ERROR`/2 = 环境/启动/超时等无法完成判定。
完整结果写 `FLAGRT_CASE_OUTPUT/<UTC运行时间>/summary.json` 与阶段日志/JSON；失败栈须查看对应日志，不将所有失败归因于shmem。

本地测试命令：

```bash
python3 -B -m unittest discover -s dev/framework-adapter/cases/ascend-train-v3/assets -p test_runner.py -v
python3 -B -m unittest discover -s dev/framework-adapter/probes -p test_silu_policy.py -v
python3 -B -m unittest discover -s dev/framework-adapter/probes -p test_silu_training_adapter.py -v
```

公共脚本只在启动时核环境，不能证明运行中源码未变；本入口额外在运行前后核对探针源码哈希。正式发布还须冻结commit并重新生成clean基线。本次strict通过不等于全部回归通过，目标内核仍失败。
# 10-18准入更新（本地开发，未发布）

启动器现在传入经Docker检查的镜像ID；探针读取Triton及FlagTree/triton-ascend发行版本。
目标测试需要额外的`--admission-records`审核记录包（JSON位于仓内只读挂载，证据文件位于该JSON目录内）。
缺失记录或准入不通过时目标测试返回BLOCKED，模型阶段标记未运行，不算替换成功。5类原生回退测试仍独立运行。
10-16历史shmem失败结果保持原义；新控制逻辑不能证明shmem已修复。

记录格式由`probes/admission_records.py`校验：顶层schema=1、records数组；每条包含environment全字段、operator、dtype、shape、stages、evidence相对路径、evidence_sha256、status、revision。
通过记录必须同时覆盖forward/backward，匹配精确环境/dtype/shape及策略revision。failed/revoked优先否决。
这是经人工审核的本地配置，不是签名认证；文件哈希只证明证据没变，不能证明结论正确。不能用当前环境自动生成passed，更不能把CPU控制测试当作硬件资格记录。
目前没有新增获批硬件记录，旧硬件证据缺少新完整身份时不得自动迁移为passed。首次资格试验需要独立、明确的隔离验证流程，不能绕过生产准入静默放行；该流程及公共包仍在开发。
