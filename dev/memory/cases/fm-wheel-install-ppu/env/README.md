# env 引用方式与加装说明

本样例不自建运行环境目录，直接引用 FM 在 PPU 的现行定档环境
[`dev/memory/cases/tiering-async-ppu/env/`](../../tiering-async-ppu/env/)
（docker-compose.yml + env.baseline.yaml），并在其上运行时加装 FlagMemory v0.3.0 wheel。

## 环境选型依据

- 该环境挂载 FlagMemory 源树（pip-editable，commit 守门 58e60a9）——本样例在其上装
  wheel 形成双形态对照；版本事实唯一来源 = 所引目录 `env.baseline.yaml`，
  `up.sh --strict` 机器核对；
- 镜像 = flagrelease vllm0.24 定档（FM connector 版本依赖 ≥0.24；thead-ppu env/v1 的
  vllm 0.19 不满足，见 FM RUNBOOK 边界登记）；
- PPU 无带卡容器独占，选卡 = ppu-smi 挑空闲卡 + CUDA_VISIBLE_DEVICES 授予。

## 本样例的加装方式（run.sh 内执行）

docker cp wheel 进容器（**保留原文件名**——文件名是 pip 元数据，改名即拒装）→
`pip install`。版本责任在本样例：run.sh 头部登记 wheel 下载 URL + sha256 锚，执行前自证。
若加装需升级为环境级承诺，应改走 compose 的 `x-flagrt.env_packages`（层包登记）。

## 换环境 / 升级组件

- 换环境：run.sh 的 `ENV_DIR` 一处；
- 升级 FM 版本：换 WHEEL 路径 + run.sh 头部 sha256 锚 + C2b 期望版本号，共三处。
