# 1014验证证据

完整环境、命令、判据与限制见[实验报告](../910C-v03推理与训练验证-20261014.md)。重复复跑不累计为新增用例；调用次数不是算子数量。

- 最新逐调用适配器：`scoped-silu-training-v4.json`为三dtype及模型三步对照通过；`scoped-silu-admission.json`为5类执行前原生路径与单算子推理通过。同名log为原始输出。
- `scoped-silu-training.json`、v2、v3分别记录Git身份核验、旧测试注册假设及反向工作线程检查问题，均失败，非通过证据。v3源码快照对应v4训练运行；before-engine-thread为修复前适配器。
- 旧整步注册实验：`silu-training.json`第4步梯度超差；`silu-native-control.json`原生控制10步通过；`silu-same-state.json`同权重10轮通过，不撤销独立轨迹失败。
- `embedding-native/gems.json`为真实文本两组各10步；reload为保存重载验证。检索指标两组均下降，不作业务模型效果验收。
- `native-inference.json`、`gems-inference.json`及`adapter-controls.xml`为旧推理入口在设备v0.3系列的回归。
- `native-training-v2.json`为原生训练成功；无v2的native-training为第一次序列化失败保留结果。
- `silu-adapter-cpu-controls.log`记录早期5＋10项CPU测试；修复引擎线程后已增至5＋11项，见本次发布时复跑日志`publish-controls.log`。CPU模拟注册不是NPU结果。

脚本快照用于追溯，不是新的开发入口；当前代码在../../probes。数据集和模型权重不入仓。
