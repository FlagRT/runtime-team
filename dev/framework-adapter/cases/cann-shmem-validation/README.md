# cann-shmem 训练 v3 修复验证

## 用途与现状

训练 v3 镜像 e13a15d0… 上，以本人隔离容器验证补装依赖。2026-11-06：cann-shmem 1.6.0 通过原始最小复现和 SiLU 前反向 6 项检查；默认 1.7.0 在下载校验阶段失败。完整结论见[报告](../../docs/cann-shmem修复验证-20261106.md)。

## 复现

前置条件：训练 v3 精确镜像、1 张已核空闲的 910C、Docker 权限、Python3/PyYAML/Compose、官方包索引网络；无需模型和数据集。

先使用[记录的 baseline/demo](demo/20261106/README.md)核对镜像。每轮使用本人新目录和新容器；不要在公共或他人的运行环境里安装。镜像 ID 与包叠加层是两份身份，必须一起保留。

在已核对的本人容器中，修复安装命令为：

```bash
python3 -m pip install cann-shmem==1.6.0 -i https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/
```

本轮实际采用更严格的阶段流程：`run.py --root <本轮目录> --stage start/before/plan/install/after/stop`，plan/install 传 `--version 1.6.0`。stage 每次只取一个值；plan 解析结果人工确认后再 install。root 内需有冻结公共 `public/dev/lib`、镜像目录及 `input/minirepro_shmem.py`、`input/silu_check.py`；按 HANDOFF 的冻结入口准备。stage 日志用排他创建，拒绝覆盖旧结果。start 拒绝已存在的同名容器，历史容器保留时请在新副本中使用新的本人容器名，不删除历史证据强行重跑。

最小验证单入口为 `python3 minirepro_shmem.py`；脚本原文来自公共提交 `0da9977086d07a867bf915b207f08e70f541683c` 的 `dev/issues/shmem-missing/assets/`。每次自动使用全新缓存。单算子追加验证为 `python3 silu_check.py`，输出目录固定 `/results`，需事先挂载本人可写结果目录。

## 验证

| 脚本 | 类别 | 判据 |
|---|---|---|
| 公共 minirepro_shmem.py | 复现/验证 | 0/PASS=编译与数值通过；1/FAIL=缺 shmem 复现；2/ERROR=其他问题 |
| [silu_check.py](assets/silu_check.py) | 验证 | 0=3 组 FP32 形状前反向通过；2=失败，详见 silu-result.json |
| [run.py](assets/run.py) | 阶段编排/收数 | 基线、安装步骤失败时中止；before/after 保留底层测试退出码，不把阶段编排结束等同测试通过 |
| [index_check.py](assets/index_check.py) | 收数 | 官方索引与最多 3 个 cp311/aarch64 wheel 的独立 SHA 对照；下载内容不安装、不执行 |
| [provenance.py](assets/provenance.py) | 收数/检查 | 核对前后包清单仅新增 cann-shmem 1.6.0；记录模块位置与版本 |

最小修复验证日志在 npu1-11 `/home/cgu135/framework-shmem-20261106/results/`，仓内只放[关键截选](assets/evidence/20261106.md)和[机器摘要](../../docs/evidence-cann-shmem-20261106/summary.json)。该轮不改准入；随后完成的补包身份与模型验证见下一节。

## 补包环境审核与模型回归

11-06后续本地dev4候选已完成补包身份测量、SiLU三路及Qwen0.6B双rank三路回归。原镜像缺包失败规则保留，精确核验的1.6.0叠加环境另做资格与审核。

- [结果报告](../../docs/训练v3补包审核与Qwen回归-20261106.md)、[机器摘要](../../docs/evidence-overlay-regression-20261106/summary.json)
- [baseline/demo及操作顺序](demo/20261106-overlay/README.md)
- [分阶段执行器](assets/regression.py)、[单算子资格与三路检查](assets/qualify_overlay.py)
- 模型资格探针和正常CLI检查器使用已有`probes/qualify_public_training.py`、`cases/ascend-official-unary/assets/check_training.py`的冻结副本，输入SHA与完整结果留存HANDOFF指定目录。
