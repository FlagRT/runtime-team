# env 引用方式（教学注解版）

本样例**不自建运行环境目录，直接引用现行环境目录**
[`dev/images/ascend-train-comm/v3/env/v2`](../../../../images/ascend-train-comm/v3/env/v2/)
（docker-compose.yml + env.baseline.yaml）。

## 为什么引用而不是复制（做首件样例时最常问的一条）

ENV-SPEC §3.2 允许「直接引用」或「自建」两种形态，本样例取最简且合规的**直接引用**：

- 该目录就是验收对象环境（镜像 + FlagCX / flagos-device-context / FlagMemory 三挂载 +
  cann-shmem 层包），无需任何裁剪或叠加；
- 复制一份 compose ＝ 制造第二份会漂移的真相——上游改了你这份不会跟着改；
- **版本事实唯一来源 = 所引目录的 `env.baseline.yaml`**，由 `dev/lib/up.sh --strict`
  机器核对（镜像 id、三挂载 commit、cann-shmem 容器内实装版本），本样例不自记版本号。

## 引用要在 run.sh 里做什么（三行的事）

```bash
ENV_DIR="$ROOT/dev/images/ascend-train-comm/v3/env/v2"   # ① 指向被引目录
bash dev/lib/up.sh "$ENV_DIR" --strict                   # ② 用它的 up.sh 起容器（含核对）
docker compose -f "$ENV_DIR/docker-compose.yml" down     # ③ 收尾用它的 compose down
```

## 读懂数据流（对照本样例你会看到的东西）

| 环境里的机制 | 在哪登记 | 一句话 |
|---|---|---|
| 镜像是哪个 | `env.baseline.yaml` 的 `image.id` | run.sh 第 1 步逐字自证的就是它 |
| 三个仓怎么进容器 | compose 的 `x-flagrt.mounted_repos`（`activation: pip-editable`） | 起容器时以可编辑方式装入；宿主检出必须停在基线登记的 commit，`--strict` 会拒绝不一致启动 |
| 层包（cann-shmem） | compose 的 `x-flagrt.env_packages` | 起容器后幂等安装、`--strict` 装完复核容器内版本 |
| 选卡 | run.sh 自己的事（`TRAIN_DIE_A/B`） | compose 只映射这两张 die；容器内 0 号设备 = TRAIN_DIE_A |

## 换环境时改哪里

样例对环境的事实依赖只有 `run.sh` 顶部的 `ENV_DIR` 一处；换环境 = 改它并重跑
（镜像自证与 `--strict` 会立即拦截不一致）。
