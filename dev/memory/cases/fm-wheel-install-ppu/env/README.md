# env 引用方式与加装教学注解

本样例**不自建运行环境目录，直接引用现行环境目录**
[`dev/images/thead-ppu-runtime/v2/env/v1`](../../../../images/thead-ppu-runtime/v2/env/v1/)
（docker-compose.yml + env.baseline.yaml），并在其上**运行时加装** FlagMemory v0.3.0 wheel。

## 为什么选这个环境做「加装」演示

- 该环境的 `mounted_repos: {}` **不挂载任何代码仓**（镜像全部资产固化）——干净基底，
  装进什么就是什么，没有源码挂载会遮蔽 pip 安装结果；
- 版本事实唯一来源 = 所引目录 `env.baseline.yaml`，`up.sh --strict` 机器核对；
- PPU 无带卡容器独占，多容器可并存，选卡纪律 = ppu-smi 挑空闲卡 + CUDA_VISIBLE_DEVICES 授予。

## 「继承环境之后自己加装」的三种方式（本样例演示第 ③ 种）

| 方式 | 登记处 | 适合谁 | 本仓实例 |
|---|---|---|---|
| ① 环境层包（起容器后幂等 pip 装发行包） | compose 的 `x-flagrt.env_packages`（version+sha256 双锚） | 环境维护者给**所有人**统一加装 | cann-shmem 1.6.0（910C 训练腿 env/v2） |
| ② 挂载源码（pip-editable / pythonpath-source） | compose 的 `x-flagrt.mounted_repos`（commit 守门） | **开发者**改代码即生效 | FlagCX/dc/FM 三挂载（910C 训练腿 env/v2） |
| ③ 运行时手动加装（本样例：docker cp + pip install） | 样例 run.sh 自己（wheel sha256 自证） | **消费方/验收方**临时验证一个交付组件 | 本样例（FM wheel 消费形态） |

教学要点：方式 ③ 的版本责任在样例自己（所以 run.sh 头部声明了 wheel 下载地址 + sha256 锚
并在执行前自证）；要变成「环境承诺」就该升为方式 ① 登记进 compose——两种形态的差别是
**承诺范围**，与昆仑芯 env/v1 的 dc「登记 vs 不登记」同构。

## 安装面怎么自证（C2 判据的用心）

只验 `version == 0.3.0` 不够——源码挂载也能给出版本号。本样例加验
`flagmemory.__file__` 落在 **site-packages**：证明生效的确实是 wheel 装的那份，
没有被挂载源树遮蔽（PYTHONPATH 优先级高于 site-packages，这是常见坑）。

## 换环境/换组件时改哪里

- 换环境：`run.sh` 顶部 `ENV_DIR` 一处；
- 升级 FM 版本：换 `WHEEL` 路径 + `run.sh` 头部的 sha256 锚与 C2 的期望版本号——
  这就是「安装新版本」的全部动作（分层管理器后续版本同法）。
