# 框架接入与算子调用适配（framework-adapter）

负责人：顾宬 / cgu135。入口整理：2026-09-22。

复用框架入口和统一算子库，明确何时优先调用目标实现、何时保持原生，以及执行失败后的安全边界。本目录保存方向原型、部署配置与验证材料；正式算子内核和子库改动不搬入协调仓。

## 从哪里看

| 你想知道什么 | 只需先看 |
|---|---|
| 现在做到了哪、下一步做什么 | [STATUS.md](STATUS.md)（唯一当前进度入口） |
| 接着开发、恢复环境、运行测试 | [HANDOFF.md](HANDOFF.md) |
| 最新模型结果及RMSNorm问题细节 | [09-22实验报告](docs/910C模型接入与安全回退-20260922.md)，RMSNorm定位见§8 |
| 找对应日志、失败结果、版本和源码哈希 | [09-22证据目录说明](docs/evidence-model-20260922/README.md) |

日常不必逐个读日志。当前结论看STATUS，核对数字再进报告和对应证据。

## 代码地图与运行入口

以下路径相对本目录。统一命令从runtime-team根执行：
`bash dev/framework-adapter/probes/run_checks.sh <模式> [参数]`。不带参数只显示帮助，不会连接服务器或启动容器。

| 用途 | 文件（probes/下） | 模式 |
|---|---|---|
| 当前模型接入逻辑 | [qwen_scoped_adapter.py](probes/qwen_scoped_adapter.py) | 被测试脚本调用；默认不准入任何算子 |
| 原生模型与算子基线 | [qwen_embedding_baseline.py](probes/qwen_embedding_baseline.py) | `model-baseline` |
| 接入、回退与模型对照 | [qwen_gems_validation.py](probes/qwen_gems_validation.py) | `qwen-gems`，保守示例显式准入SiLU |
| RMSNorm同输入旁路诊断 | [qwen_rms_shadow.py](probes/qwen_rms_shadow.py) | `rms-shadow`，不替换模型输出 |
| RMSNorm按角色分组替换 | [qwen_rms_ablation.py](probes/qwen_rms_ablation.py) | `rms-ablation`，仅实验 |
| 控制测试 / 基线元数据测试 | [test_qwen_scoped_adapter.py](probes/test_qwen_scoped_adapter.py) / [test_qwen_probe_metadata.py](probes/test_qwen_probe_metadata.py) | `qwen-controls` / `metadata-tests` |
| 全目录语法检查 | [run_checks.sh](probes/run_checks.sh) | `local`，不用卡、不导入torch |

硬件模式必须在准备好的环境运行，显式提供模型、runtime路径和新的结果文件名；参数及依赖见HANDOFF/报告。语法通过不等于功能通过。诊断脚本的completed/退出0不等于所有数值对照通过。

## 历史材料（按需查，不代表当前状态）

| 日期 / 内容 | 报告 | 对应代码或证据 |
|---|---|---|
| 09-16 CPU模型基线与当时NPU阻塞 | [模型基线](docs/0.6B模型来源算子基线-20260916.md) | [证据](docs/evidence-model-20260916/)，旧[周报](docs/周报-20260916.md) |
| 09-09 多厂商探索 | [多芯片验证](docs/多芯片环境与算子验证-20260909.md) | [cross_vendor_smoke.py](probes/cross_vendor_smoke.py)，`cross-vendor` |
| 09-07 独立PyTorch原型 | [PyTorch接入](docs/PyTorch独立接入-20260907.md) | [pytorch_eager_adapter.py](probes/pytorch_eager_adapter.py)，`legacy-pytorch`；当前模型适配器仍复用其前置检查 |
| 09-03 / 09-07 vLLM探索 | [最小验证](docs/安全回退最小验证-20260903.md)、[单卡联调](docs/910C单卡联调-20260907.md) | `legacy-vllm`，[子库补丁](patches/vllm-plugin-FL-safe-fallback.patch) |
| 09-09 任务及接口草案 | [旧规划](docs/框架接入与安全回退-9月任务规划-20260909.md) | 月目标需结合09-17方案及09-21要求，不能仅据此排期 |
| 整理前的入口全文 | [历史入口快照](docs/历史入口快照-20260922.md) | 含当时README / STATUS / HANDOFF，旧结论不作为当前事实 |

`docker-compose*.yml`、`.env.example`为历史实验配置；[锁定推理容器脚本](probes/start_locked_infer_910c.sh)为已有部署复现入口，不要直接重建同名容器。所有资源与权限运行前重新检查。

## 维护规则

- README只管导航，STATUS只管当前成果/问题/下一步，HANDOFF只管环境和续接操作；不要在三处重复追加完整实验日志。
- 同一天同主题的追加实验写进已有日期报告；原始结果留在对应evidence目录，由该目录README说明最终版、失败版和替代关系。
- 当前代码保持稳定文件名；实验编号放结果文件名中。原始失败证据不覆盖，旧脚本/补丁不未经核对就合并删除。
- 共享协作遵循[仓库根README](../../README.md)，提交/合并/推送及OneDrive写入以当轮授权为准。本地整理不等于已上传。
