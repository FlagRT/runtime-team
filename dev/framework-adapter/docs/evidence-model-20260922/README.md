# 2026-09-22 真实模型接入证据

本目录保存原始结果。原生基线、控制测试、目标算子替换、模型输出对照分别计数。
失败结果保留；任何后续重跑使用新文件名，不能以放宽阈值覆盖原失败。

源环境：27，本人flagos-proto-infer-910c，v2锁定推理镜像加本人目录内的源码/依赖覆盖。
FlagGems commit f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4；Triton 3.5.0 + triton-ascend 3.2.1，未切换FlagTree构建物。

追加rms-shadow/rms-ablation为18:17前完成的同输入与分组诊断。JSON的completed表示执行完成，不表示所有数值对照通过；ablation隐藏层两项within_tolerance=false须保留。rms-final-state为本日后续停止记录；先前final-state为18:04历史快照。rms-control-regression重跑既有15项，不累加成绩。

## 优先查看（最终结论依据）

先读[实验报告](../910C模型接入与安全回退-20260922.md)，再按需打开下列材料。同名`.log`是运行日志，不是另一个独立测试。

| 想核对什么 | 文件 | 含义 |
|---|---|---|
| 原生模型基线 | [npu-float16-20260922.json](npu-float16-20260922.json) | 2组输入、8组模型来源算子用例 |
| 保守SiLU接入及回退 | [gems-model-float16-safe-v3-20260922.json](gems-model-float16-safe-v3-20260922.json) | 最终保守配置；8算子、4回退、4模型配置 |
| RMSNorm为何未准入 | [gems-model-float16-v2-20260922.json](gems-model-float16-v2-20260922.json) | 拆分SiLU/RMSNorm实验，包含失败，不能只挑通过项 |
| 相同输入的逐层差异 | [rms-shadow-v1-20260922.json](rms-shadow-v1-20260922.json) | 226次旁路观测，不替换模型输出 |
| Q/K与隐藏层的影响 | [rms-ablation-v1-20260922.json](rms-ablation-v1-20260922.json) | 4个分组模型实验，2个隐藏状态未通过 |
| 最终控制测试 | [qwen-control-v3-20260922.xml](qwen-control-v3-20260922.xml) | 15项；rms-control-regression日志是同一批回归，不累加 |
| 实验环境与版本 | [environment-20260922.txt](environment-20260922.txt)、[packages-20260922.txt](packages-20260922.txt) | 基座、源码和包版本 |
| 新诊断脚本哈希 | [rms-source-20260922.txt](rms-source-20260922.txt) | shadow和ablation，本地/远端已比对 |
| 最后收尾状态 | [rms-final-state-20260922.txt](rms-final-state-20260922.txt) | 18:17停止，不保证之后实时状态 |

## 保留追溯（通常不用读）

- `native-preflight-20260922.log`：最小NPU预检查，不替代模型结果。
- `gems-import-20260922.log`：首次缺依赖失败；`gems-import-deps-20260922.log`：补齐本人目录依赖后的导入。
- `gems-model-float16-20260922.json/.log`：首次模型替换失败；后续v2分开对照，safe-v3只验证保守准入，不能说v3“修好了RMSNorm”。
- `qwen-control-20260922.xml/.log`：早期14项；最终使用v3的15项，不相加。
- `final-state-20260922.txt`：18:04第一次停止；后续又启动诊断，最终看rms-final-state。
- [source-experiment-v2-20260922.tgz](source-experiment-v2-20260922.tgz)：失败实验当时的源码快照，不当作当前开发代码。

文件保持原名，便于对照报告及服务器结果。增加新结果时更新本表，不覆盖旧失败文件；不要为了“整洁”只保留通过结果。
