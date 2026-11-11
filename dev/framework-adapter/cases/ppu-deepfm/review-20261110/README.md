# PPU DeepFM 限定审核

operator-records.json仅批准11-10独立资格通过的sigmoid FP32 shape32前反向，3次重复证据及SHA绑定在同目录。不能据此启用整个模型。

DeepFM候选在第1步参数更新后超差，因此本目录没有passed模型审核，也没有可启用的on配置。off关闭适配；unknown保留算子审核但缺模型审核，必须选择原生并报告workload_unverified。模型独立数值资格失败后不自动回滚或重试，该进程停止，其他测试另起容器并重新初始化。

model-reviews.failed.json记录上述精确环境及模型配置的失败资格，绑定model-failed-evidence.json。strict-denied.json要求目标，必须由启动准入拒绝并给出workload_denied；该失败记录的严格拒绝已做CPU控制，不能算PPU额外训练。模型源码、种子、优化器、步数和执行代码指纹变化后不能直接继承本记录。
