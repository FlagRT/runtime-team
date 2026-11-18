# env 引用方式

本样例**不自建运行环境目录，直接引用现行环境目录**
[`dev/images/ascend-train-comm/v3/env/v2`](../../../../images/ascend-train-comm/v3/env/v2/)
（docker-compose.yml + env.baseline.yaml）。

选择理由（ENV-SPEC §3.2 允许使用方自建或直接引用，本样例取最简且合规的直接引用）：

- 该目录即本样例的验收对象环境（镜像 + FlagCX/flagos-device-context/FlagMemory 三挂载 +
  cann-shmem 层包），无需任何裁剪或叠加；
- 不复制装配文件＝不产生第二份会漂移的真相；版本事实唯一来源 = 该目录
  `env.baseline.yaml`，由 `dev/lib/up.sh --strict` 机器核对（镜像 id、三挂载 commit、
  cann-shmem 容器内实装版本）。

样例对环境的事实依赖（如需换环境，改 run.sh 的 `ENV_DIR` 并重跑）：

| 依赖 | 锚点 |
|---|---|
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-…-arm64`，id 见 env.baseline.yaml `image.id` |
| FlagCX | `mounted_repos.FlagCX.commit`（覆盖镜像内固化副本） |
| flagos-device-context | `mounted_repos.flagos-device-context.commit`（v0.4.0 发版提交） |
| FlagMemory | `mounted_repos.FlagMemory.commit`（v0.3.0 发版提交；被测组件本体） |
| cann-shmem | `env_packages[0]`（1.6.0，version+sha256 双锚，up.sh 起容器后安装） |
