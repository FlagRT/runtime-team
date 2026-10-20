# 训练入口配置开关

这三个配置由本方向 `tools/train_with_adapter.py` 读取，不是device-context已经发布的配置格式。包装器在作用域内执行未经修改的上游 `runtime/proto/proto_train_leg.py`，前向、反向和参数更新都保留在该作用域内。仅支持当前已核对哈希的昇腾双进程eager训练入口。

| 配置 | 行为 | 验证要求 |
|---|---|---|
| off.json | enabled=false；不导入FlagGems、不读取资格记录 | SiLU保持原生，原因experimental_disabled |
| on.json | 启用候选；读取/records/experimental-records.json及其哈希校验的证据 | 精确环境、形状、dtype及前反向资格匹配才进入FlagGems |
| unqualified.json | 启用候选但不给资格记录 | 执行前选原生，原因capability_unverified；不是先运行失败再重试 |

当前只声明SiLU FP32、形状(1,32,3072)，匹配0.6B的batch=1、seq=32验证。sigmoid公共机制已有独立测试，本次模型不宣称覆盖它。RMSNorm及其他算子保持原生。

`records`可为相对配置文件的路径或绝对路径；`null`表示无资格。记录缺失/损坏与合法的无资格配置不同，前者报错停止，后者走原生。配置未知字段、算子名、类型错误也停止。运行时异常不重试原生。

通过[官方环境case](../../cases/ascend-official-unary/README.md)执行，增加：

```text
--stage dc-train --device <空闲物理设备A> --second-device <空闲物理设备B>
--training-config off|on|unqualified --model <已核权重> --records <审核记录>
--output <全新结果目录>
```

每组使用独立容器、相同模型和固定随机token，确定性模式、3步、AdamW；不修改上游训练脚本。入口哈希变更会拒绝运行，需重新审阅后更新，不能跳过校验。

`adapter_rank*.json`只说明适配作用域执行情况；最终必须同时检查上游6项判据和真实GEMS内核日志，汇总以`result.json`及`summary.json`为准。不以loss下降代替算子数值一致性，也不以本次3步代替长训练验收。
