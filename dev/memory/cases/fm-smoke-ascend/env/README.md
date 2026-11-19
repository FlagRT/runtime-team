# env 引用方式

本样例不自建运行环境目录，直接引用现行环境目录
[`dev/images/ascend-train-comm/v3/env/v2`](../../../../images/ascend-train-comm/v3/env/v2/)
（docker-compose.yml + env.baseline.yaml）。

- 该目录即本样例的验收对象环境（镜像 + FlagCX/flagos-device-context/FlagMemory 三挂载 +
  cann-shmem 层包），无需裁剪或叠加；
- 不复制装配文件（不产生第二份会漂移的真相）；版本事实唯一来源 = 该目录
  `env.baseline.yaml`，由 `dev/lib/up.sh --strict` 机器核对。

## 环境事实依赖

| 依赖 | 锚点 |
|---|---|
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-…-arm64`，id 见 env.baseline.yaml `image.id` |
| FlagCX | `mounted_repos.FlagCX.commit`（覆盖镜像内固化副本） |
| flagos-device-context | `mounted_repos` 同上 |
| FlagMemory | `mounted_repos` 同上（v0.3.0，58e60a9） |
| cann-shmem | `x-flagrt.env_packages`（1.6.0，起容器后幂等安装） |
| 选卡 | run.sh 自理（TRAIN_DIE_A/B；compose 只映射这两张 die） |

换环境：改 run.sh 的 `ENV_DIR` 一处并重跑（镜像自证与 `--strict` 会拦截不一致）。
