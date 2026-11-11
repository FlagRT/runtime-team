# 实际启动基线

本目录保存本轮成功运行生成的compose、env.baseline.yaml及严格启动回执：20261110-relu-on为模型开启，20261110-onerec-inventory为完整OneRec算子采集。镜像声明在上一级image.baseline.yaml。

这些文件中的绝对路径与容器名是历史审计信息。复跑用prepare/runner生成新输入、新输出和新容器，不直接启动旧compose；先重核空卡、权限、磁盘、模型和wheel SHA。每轮三次空闲采样、单卡映射、网络关闭，使用公共`up.sh --strict`，结束停止本人容器。

清单采集给Qwen/OneRec容器64GiB内存、1800秒超时；ReLU/DeepFM容器8GiB、480秒超时。模板不包含模型权重、wheel或凭证，其来源与冻结路径见HANDOFF和日期报告。
