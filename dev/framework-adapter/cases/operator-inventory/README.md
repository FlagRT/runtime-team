# 三模型算子清单与ReLU扩展

目的：从实际训练调用选择下一批接入算子，而不是按模型名称猜测。实验日期2026-11-10，PPU v2，公共runtime 0.4.0。

## 文件与入口

- `probe.py` / `trace_ops.py`：各模型一次原生训练步骤，分前向、反向、优化器记录Python入口、ATen重载、形状/精度/stride、调用数与模块样本；不记录张量内容。
- `prepare.py`：从既有PPU模型/DeepFM冻结输入生成新输入，保留公共启动脚本、严格基线核对、三次空卡检查、单卡映射和退出清理。
- `compare.py`：与固定FlagGems wheel注册表和FlagOP源码注册入口对照；输出完整清单及机器JSON，不把源码存在当作已验证。
- `prepare_relu.py`：从PPU DeepFM输入生成ReLU单算子与模型实验；不修改公共模型源码、内核或模型容差。
- `review.py`：人工审阅后显式打包资格记录；测试runner不会调用它。模型候选仅用于资格采集，通过后才能签发限定模型审核。
- `test_trace.py` / `test_compare.py`：CPU控制；`demo/`保存实际启动基线，报告及结果入口见[验证报告](../../docs/三模型算子清单与ReLU接入验证-20261110.md)。

## 口径

DeepFM batch32/FP32，Qwen0.6B batch1 seq32/FP32，OneRec-8B batch1 seq16/BF16，后者开启非重入梯度检查点。三个均为完整模型、合成数据短训练。OneRec反向包含重算的前向调用。模块路径只留每种签名最多8个样本，算子计数和输入签名完整；超过20000签名直接报错。

ReLU新增接入`F.relu`/`torch.relu`/`Tensor.relu`非原位路径，反向为`aten.threshold_backward(..., 0)`。只审核本轮PPU FP32 32×400、有限输入、一阶梯度。模型候选只替换ReLU，sigmoid仍为原生，因此与之前sigmoid候选是不同实验。

## 复跑顺序

1. 检查本人权限、磁盘、镜像ID、实时卡况；用新输出目录冻结模型输入与wheel SHA，不改历史input。
2. `prepare.py --model-input <冻结Qwen输入> --deepfm-input <冻结DeepFM输入> --output <新输入>`。
3. 使用生成的`run.py --root <本人根> --run-id <唯一ID> --card <空卡> --stage inventory --kind deepfm|qwen|onerec --model <只读模型路径>`，逐个串行运行。
4. `compare.py --root <含deepfm.json/qwen.json/onerec.json的目录> --flagop <固定源码> --gems <固定wheel> --output <新目录>`。
5. `prepare_relu.py --source <冻结DeepFM输入> --wheel <候选适配包> --output <新输入>`；生成的runner先`--stage qualification`。审核三次独立前反向结果后，用`review.py --decision approve-operator`生成资格，再冻结新输入。
6. 新输入先`--stage routes`，再`--stage model --mode native|native_repeat|candidate`；后三者共享本轮原生参考，非native加`--reference <原生目录>`。候选通过后人工审核，生成包含模型审核的新输入，再运行off/on/unknown三路。失败停止并保留证据，不放宽容差、不自动回滚重试。

入口细节、不可覆盖的输入轮次、当前操作路径统一见[HANDOFF](../../HANDOFF.md#11-10-三模型清单与relu扩展)。所有测试只安装到本人容器结果目录的`packages/`，不改变公共镜像；自动生成新compose并执行公共`up.sh --strict`。
