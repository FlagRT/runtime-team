# 2026-11-04 DeepFM开启路径的启动记录

本目录的docker-compose.yml与env.baseline.yaml是152新机实测生成物，原样保留作为证据；含当次宿主绝对路径，不是可直接照抄到任意主机的推荐配置。

源码候选6ee9129，公共快照a7e90e7，v2镜像cac5ddae5f62，GEMM兼容补丁关闭。单卡SMI4→PCI0000:84:00.0→/dev/xpu4→逻辑0。容器已停止；不要直接重启本记录。

复跑先按[case说明](../../README.md)重新核空卡、镜像、输入与资格，使用assets/run.py生成新的输出目录及baseline，再由公共up.sh --strict启动。完整结果见[本轮报告](../../../../docs/P800新机三路回归-20261104.md)。
