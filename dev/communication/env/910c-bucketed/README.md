# 910C 有界分桶训练环境

_通信消费侧验证；不发布新镜像，不修改设备原型。_

---

## 📋 前置条件

| 项目 | 要求 |
| --- | --- |
| 镜像 | 公共 ascend-train-comm v3；由同目录基线核对 image id |
| FlagCX | 仓根 `FlagCX/`：组织 `dev-1.0` 的 `62fe6153852658f17e729a06be4da1c766201ef7`，含设备方向修复 |
| 源码 | Mac 下载并同步，无需服务器 GitHub 凭据；同时同步固定的 json/googletest 子模块 |
| 设备 | 两个无计算占用、无其他容器映射且可实际初始化的 die；不停止他人容器 |
| 模型 | OneRec-8B 完整三分片检查点；revision `29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff` |
| 工具 | Python 3 + PyYAML、Docker Compose；允许使用用户测试目录中的工具 |

环境遵循 [ENV-SPEC](../../../ENV-SPEC.md) 与设备方向的
[挂载覆盖机制](../../../device-context/docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md)。
不复用 demo 的全部设备映射：Compose 的 `extends` 不能将设备列表收窄。
本环境使用私有 IPC、1 GiB SHM、4 核 CPU 上限及只读驱动、只读模型挂载。
`MAX_JOBS` 只约束扩展构建；上游 core make 会探测宿主核数，CPU 配额负责限制实际占用。

## 🔧 启动与版本核对

在已同步的仓根执行，环境变量必须指向真实空闲设备和实体目录：

```bash
export COMM_DIE_A=6 COMM_DIE_B=7
export COMM_MODEL_ROOT=/absolute/path/OneRec-8B
mkdir -p evidence
python3 dev/lib/baseline.py export dev/communication/env/910c-bucketed --require-clean
bash dev/lib/up.sh dev/communication/env/910c-bucketed --strict
```

设备编号只是示例，每次重新检查。编译日志在仓根 `evidence/flagcx-build.log`；
容器进入 `sleep infinity` 才表示覆盖安装完成。安装成功后须确认实际加载的是
`/workspace/FlagCX/plugin/torch/flagcx`，而非镜像内旧副本。
生成的 `env.baseline.yaml` 不手工改写；未通过 `--strict` 不启动验收。

## ✅ 单例验证

容器内、CANN 环境已加载后：

```bash
python3 dev/communication/training/run_onerec_case.py \
  --runtime-root /workspace/dev/device-context/prototype \
  --model-root /workspace/models/OneRec-8B \
  --collective flagcx --bucket-mib 64 --seq-len 16 --steps 50 \
  --output /workspace/evidence/new-onerec-case
```

输出目录必须不存在。退出码：0/PASS、1/FAIL、2/ERROR；启动器非零、超时、
未正常销毁通信组、两端哈希不一致、步数或载荷计数不符，均不通过。
`--collective hccl` 仅是诊断对照，不能替代 FlagCX 验收。
完整日志留执行机器，公共报告仅收录摘要、截选及来源路径。

缓冲区上限只指新增梯度通信 scratch，不是模型总显存上限。
测试采用合成 token、全参数 BF16/SGD；不代表推荐质量或业务收敛。

## 📍 收尾

确认自己的测试进程已结束后停止本环境，不操作其他容器；共享存储中的模型和证据保留。

```bash
docker stop flagrt-communication-1021-910c
```
