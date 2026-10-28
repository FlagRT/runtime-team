# 910C长序列重复验证环境

_按现行FlagCX挂载覆盖口径，复用本方向已验证的窄设备装配。_

## 🔧 条件与入口

使用[910c-bucketed](../910c-bucketed/README.md)的相同镜像、实体模型目录和干净FlagCX `62fe615`源码，选择经检查且可真实初始化的两个die。本目录经Compose `extends`复用该配置，更换项目及容器名；私有IPC、1 GiB SHM、4核限额及只读驱动/权重挂载保留。基线由公共工具自动生成。

```bash
export COMM_DIE_A=10 COMM_DIE_B=11
export COMM_MODEL_ROOT=/absolute/path/OneRec-8B
mkdir -p evidence
python3 dev/lib/baseline.py export dev/communication/env/910c-repeat --require-clean
bash dev/lib/up.sh dev/communication/env/910c-repeat --strict
```

die编号每次重新检查。覆盖安装完成后运行`dev/communication/training/probe_training_env.py`确认加载路径及两个真实设备。模型验收见[本周报告](../../docs/TRAINING_REGRESSION_20261028.md)。日志保留共享存储，结束后仅停止自己的`flagrt-communication-1028-910c`容器。
