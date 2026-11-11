# 2026-11-10 实际启动配置

本目录归档两模型“开启但缺模型审核”测试的实际配置：`qwen-unknown-01/`、`onerec-unknown-01/`。每个子目录包含当次生成的docker-compose.yml、env.baseline.yaml及基线检查记录；镜像定义见上级[image.baseline.yaml](../image.baseline.yaml)。完整运行命令和结果见[模型报告](../../../docs/PPU两模型接入与安全准入验证-20261110.md)。

这些配置保留原始绝对路径，仅用于审计当次运行，不能直接docker start重用历史输出。复跑按[case说明](../README.md)生成新的冻结输入、运行ID和demo；重新核卡、权重、容量和镜像身份，执行baseline export及up.sh --strict。

缺模型审核时保留原生是预期路径；不把该demo当作目标算子获得模型数值批准的证据。
