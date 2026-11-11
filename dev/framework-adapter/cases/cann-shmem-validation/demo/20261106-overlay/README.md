# 11-06补包审核与Qwen三路：实际启动记录

本目录保留训练v3双rank回归的实际`image.baseline.yaml`、`demo/docker-compose.yml`、`demo/env.baseline.yaml`与strict检查回执。`runtime-overlay.json`单独记录容器内新增包：公共镜像baseline不因pip安装而改写。

记录主机npu1-11，物理14/15→逻辑0/1；容器`cgu135-overlay-framework-overlay-20261106`已停止。绝对挂载路径是当次现场记录，不能直接当作另一台机器可用配置或空卡预约。

## 复现顺序

1. 重核本人权限、镜像ID、空卡/映射、模型与存储；新建本人运行目录，不能覆盖本轮。
2. 按HANDOFF冻结公共工具、公共runtime、实测wheel、1.6.0补包和case脚本，核输入manifest。模型只读挂载。
3. 用`assets/regression.py --root <新运行目录> --stage init --devices <两物理ID> --model <权重目录>`生成该轮baseline/demo、strict启动并离线安装两个wheel。原目录名称会派生容器名，必须唯一。
4. `--stage operator`取得单算子资格；人工核对数值、命中和身份后建立算子审核，再运行`controls`。
5. 依次运行`native`、`repeat`、`candidate`，检查两rank逐张量结果；只有资格通过并审查证据后才建立模型审核。资格脚本本身不产生通过授权。
6. 正常CLI依次运行`off`、`on`、`unqualified`；检查6/6、3步、前反向命中和退出恢复。
7. `--stage stop`停止本人容器，取证并核SHA。原生参考张量保留远端；轻量备份带张量哈希清单。

阶段命令记录在完整证据`results/*.command.json`。环境变量`FLAGRT_CANN_SHMEM_WHEEL`指向只读wheel；正式CLI测量实际安装内容后写入`compiler_build`，算子和模型审核都使用这个新身份。

公共工具版本`0da9977086d07a867bf915b207f08e70f541683c`；runtime版本`719de8cbaea75691b38ad87d6706a949e045b892`。精确实测wheel/SHA及结果见[报告](../../../../docs/训练v3补包审核与Qwen回归-20261106.md)；本地冻结入口见[HANDOFF](../../../../HANDOFF.md#11-06补包环境审核与模型三路回归)。
