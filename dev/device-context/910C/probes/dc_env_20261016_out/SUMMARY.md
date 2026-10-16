# 证据索引 · 910C 训练腿运行环境的实质验证（2026-10-16）

**这次验的是什么**：`dev/device-context/910C/env/train/docker-compose.yml` 声明的**挂载覆盖**
路径是否在真机上成立 —— 即「镜像内固化的 FlagCX 换成宿主 checkout、容器启动时 pip editable 安装、
判据随之由 FAIL 变 PASS」。

**这次没验什么**（如实登记）：**没有**跑 `dev/lib/up.sh <env-dir> --strict`（当时该机缺前置；**当日已补齐并在 `up.sh` 上跑通**，另见
[`../baseline_mechanism_20261016_out/SUMMARY.md`](../baseline_mechanism_20261016_out/SUMMARY.md)，其中含 `--strict` 的实测结论）。
npu1-27 缺 `docker compose` v2 插件与 `pyyaml`（实测见
[`../../env/README.md`](../../env/README.md) §2），`baseline.py` 起不来，因此也没有产出 `env.baseline.yaml`。
本次用与 compose **逐项等价**的 `docker run` 复刻其参数（唯一额外挂载是 A 臂的影子目录，
该偏差写在驱动脚本头部）。⇒ 结论只覆盖「设备映射 + 挂载生效方式 + 判据」，**不覆盖**基线核对。

## 装置

| 项 | 值 |
|---|---|
| 机器 / 卡 | npu1-27 · die 5 与 die 0（`probe_all_dies_910c_20261016.sh` 逐 die 试 INIT 实测唯一的 FREE 对） |
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` |
| 容器 | `flagos-dc-train-910c-dev`（跑完已 `docker stop`，`Exited(137)`，无残留进程，我方运行容器 0） |
| 被测源码（挂载覆盖） | FlagCX `kistich/ascend-flagcx-adapt` @ `b576fa6a`（宿主 checkout，工作区 0 项未提交） |
| 对照（镜像固化版） | FlagCX `4e0e0cbc`（从镜像内解出，影子到 `/opt/baked-flagcx`） |
| 驱动脚本 | [`run_dc_env_910c_20261016.sh`](../run_dc_env_910c_20261016.sh) · 单入口验收 [`run_910c_p9_acceptance.sh`](../run_910c_p9_acceptance.sh) |
| 完整日志 | `npu1-27:/mnt/raid/hliu553/dc_env_20261016/log/`（含 `dc_env_run.log`、`arm_{A,B}_final.log`）· 宿主工作区快照 `/mnt/raid/hliu553/dc-ws/` |

## 判据（A/B 对照，同一容器、同一条命令）

| 臂 | 容器内加载的 FlagCX | `flagcx_sync_test.py --iters 20` | `ddp_native_test.py --steps 10` | 合并 |
|---|---|---|---|---|
| **B = 本环境**（宿主挂载覆盖，editable） | `/workspace/FlagCX/plugin/torch/flagcx` | **`overall: PASS`**（八组全 PASS） | **`overall: PASS`**（四项全 PASS） | **PASS / rc=0** |
| **A = 镜像固化版**（`PYTHONPATH` 影子） | `/opt/baked-flagcx/flagcx` | `overall: FAIL`（八行图案与 issue 一致） | `overall: FAIL` | FAIL / rc=1 |

## 截选文件

| 文件 | 内容 |
|---|---|
| [`B_sync_verdict.txt`](B_sync_verdict.txt) | B 臂 `flagcx_sync_test.py` 判定段原文（八组全 PASS） |
| [`B_ddp_verdict.txt`](B_ddp_verdict.txt) | B 臂原生 DDP 双卡判定段原文（四项全 PASS） |
| [`A_sync_verdict.txt`](A_sync_verdict.txt) | A 臂同一命令判定段原文（缺陷复现，图案同 issue） |
| [`A_ddp_verdict.txt`](A_ddp_verdict.txt) | A 臂原生 DDP 判定段原文（复现 issue 类别①） |
| [`device_selfproof.txt`](device_selfproof.txt) | 两臂各自加载的 FlagCX 路径与 `.so` sha256 前 16 位（装置自证） |
| [`install_proof.txt`](install_proof.txt) | 挂载覆盖的安装自证（`Successfully installed flagcx-0.13.0`、`flagcx -> /workspace/FlagCX/...`）+ 被测源码自证（HEAD、工作区、两处构建开关默认值） |

## 后续补充（同日 12:0x，完整证据见 [`../baseline_mechanism_20261016_out/SUMMARY.md`](../baseline_mechanism_20261016_out/SUMMARY.md)）

- **机制侧已补齐并在 `up.sh` 上跑通**：装上用户级的 compose v2.29.7 + pyyaml 6.0.2 后，
  `baseline.py export --require-clean` 双腿 rc=0；`up.sh <env>` rc=0 起容器（compose 的 `Cmd`
  逐字等于声明的 pip-editable 命令）；**同一单入口验收在本机制起的容器里仍 `overall: PASS`**。
- ⚠️ **`_C…so` 的 sha256 不能当"某次构建"的稳定指纹**：同一 commit（`b576fa6a`）两次 editable 构建，
  `_C…so` 分别是 `098f3227…`（本轮手工 `docker run`）与 `ac507f07…`（`up.sh` 那轮），
  而 `libflagcx.so` 两次都是 `5af25cc9…` ⇒ 该指纹只用于**区分"镜像内固化版 vs 宿主挂载版"**，
  装置自证必须**同时**打印加载路径。
- ⚠️ **从手工 `docker run` 切到 `up.sh` 前要 `docker rm -f` 同名容器**，否则 compose 报
  `Conflict. The container name … is already in use`（本轮实测）。

## 过程中修掉的两处**脚本自身**缺陷（都是本轮新引入，不是被测对象的问题）

1. **合并判定取值错**：`v1=$(grep -oE 'overall: (PASS|FAIL)' …)` 取出的是 `overall: FAIL` 整串，
   与 `FAIL` 比较不相等 ⇒ A 臂两项都 FAIL 时反而报 `ERROR`。改为取词（`awk '{print $2}'`）后
   A 臂正确报 `FAIL / rc=1`。
2. **证据目录落在宿主路径上**：`OUT` 传了宿主路径 ⇒ 容器内创建了**同名但属另一文件系统**的目录，
   宿主侧 `out/` 是空的。属宿主/容器路径混用（本项目已知坑），改为落在挂载内再拷出。
