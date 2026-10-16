# 证据索引 · 910C 上把总组「环境基线机制」真正跑一遍（2026-10-16）

**这次验的是什么**：不是"我方的判据能不能过"，而是**总组的环境基线机制（`dev/lib/baseline.py` +
`dev/lib/up.sh`）在 npu1-27 上能否真正跑通** —— `export` 生成运行环境基线、`check` 核对、
`up.sh` 起容器，以及容器起来后挂载覆盖是否生效。上一轮（[`../dc_env_20261016_out/SUMMARY.md`](../dc_env_20261016_out/SUMMARY.md)）
因该机缺 `docker compose` v2 与 `pyyaml` 只能退化为等价 `docker run`，本次把前置补齐后走完整路径。

## 前置怎么补的（**用户级，不需要 sudo、不改宿主包**）

用户提示"27 账户有 sudo"，但实测 `sudo -n true` 报 `sudo: a password is required`
（非交互不可用），故未走 sudo，改用**用户级安装**——对共享机更轻，且**版本与总组验证过的环境一致**：

| 组件 | 来源 | 版本 | 落点 |
|---|---|---|---|
| `docker compose` v2 插件 | 官方 GitHub Release `v2.29.7` 的 `docker-compose-linux-aarch64`（静态二进制 62095190 字节） | **v2.29.7**（与总组在 npu1-11 上验证的 Compose v2.29.7 同版本） | `~/.docker/cli-plugins/docker-compose`（`install -m 0755`） |
| `pyyaml` | PyPI `PyYAML-6.0.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 6.0.2 | `~/.local/lib/python3.9/site-packages/yaml`（`pip install --user`） |

**下载件校验**：`sha256sum docker-compose-linux-aarch64` = `6e9fbd5daa20dca5d7d89145081ae8155d68ef2928b497d9f85b54fe0f9dbb2c`
= 官方 `docker-compose-linux-aarch64.sha256` 文件的值，**逐字一致**。

> ⚠️ **npu1-11 的网络身份不可达**：该机是 `10.120.72.11`（在 27 的 `/etc/hosts` 里，ping 通、22 开），
> 但 `hliu553@10.120.72.11` 与本地三把私钥（`id_kistich` / `id_rsa` / `id_ed25519`）一律
> `Permission denied (publickey,password)`；从 27 上直连同样被拒 ⇒ **无法从 11 直接拷环境**，
> 故按"**同版本官方件**"补齐（这正是"迁过来"的等价做法，且版本号与 11 上验证过的一致）。

## 装置

| 项 | 值 |
|---|---|
| 机器 | npu1-27（`910C` 别名）· die 5 + die 0（`probe_all_dies_910c_20261016.sh` 实测唯一的 FREE 对） |
| 工作区 | `WORKSPACE_ROOT=/mnt/raid/hliu553/dc-ws`（dev-1.0 的 `dev/lib` + `dev/images` + 本方向的 `dev/device-context` + `FlagCX` checkout @ `b576fa6a`） |
| 环境目录 | `dev/device-context/910C/env/train`（训练腿）· `dev/device-context/910C/env/infer`（推理腿） |
| 驱动脚本 | [`run_mechanism_910c_20261016.sh`](../run_mechanism_910c_20261016.sh)（[0]–[10] 全程）· [`resume_mechanism_910c_20261016.sh`](../resume_mechanism_910c_20261016.sh)（[5b]–[10] 重跑并落盘） |
| 完整日志 | `npu1-27:/mnt/raid/hliu553/dc_env_20261016/mechanism/{nohup.out,resume.log}` |

## 结果

| # | 步骤 | 结果 |
|---|---|---|
| [0] | 工具自证 | compose **v2.29.7** · pyyaml **6.0.2** · 用户级落点 · `sudo` 未使用（证据 `01`） |
| [1] | `baseline.py export <env> --require-clean`（训练腿） | **rc=0**，生成 `env.baseline.yaml`（证据 `02`、`08`） |
| [1b] | 同上（推理腿） | **rc=0**（证据 `02`、`09`） |
| [2] | `baseline.py show` | 挂载覆盖关系清楚：`FlagCX b576fa6ad977 pip-editable 否 是 <- 4e0e0cbcbf72 /workspace/FlagCX` |
| [3] | `baseline.py check` | **rc=1**，唯一 `FAIL` = **镜像 id 不一致**（跨机事实，见下）；`[PASS] FlagCX: commit=b576fa6ad977` + `[INFO] 覆盖镜像内固化版本…pip-editable`（证据 `03`、`07`） |
| [4] | `dev/lib/up.sh <env> --strict` | **rc=1，拒绝启动** —— 原因就是 [3] 的那条 `FAIL`（证据 `03`） |
| [6] | `up.sh <env>`（不加 `--strict`，spec 允许） | **rc=0**，容器 `flagos-dc-train-910c-dev` Created → Started（证据 `04`） |
| [7] | 挂载生效 | `docker inspect` 的 `Cmd` **逐字等于** compose 声明的 pip-editable 命令；日志 `Successfully installed flagcx-0.13.0`；`import flagcx` → **`/workspace/FlagCX/plugin/torch/flagcx`**（宿主 checkout，非镜像内）（证据 `04`） |
| [8] | 两项验收（单入口，跑在**本机制起的**容器里） | **`overall: PASS` / rc=0** —— `flagcx_sync_test` 八组全 PASS + 原生 DDP 四项全 PASS（证据 `05`） |
| [10] | `docker compose down` | **rc=0**，容器 Removed，我方运行中容器 0、残留进程 0 |

## ⚠️ 唯一未通过的一项：`--strict`，原因是**跨机镜像 id 不同**

```
[FAIL] image: id 不一致: 基线 sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15
                         本机 sha256:7028028bb62ce06322f2b53379891c062caf8ac6b3f2b1a469be6dafe9af7913
        (ref=flagrt/ascend-operator-runtime-comm:2.0.0-…g4e0e0cb-arm64)
```

- 这不是本方向的配置问题：`image_list.md` 早已登记"跨机 image id 不同属预期（内容同源）：npu1-27 侧为
  `be30a952c2eb` / `7028028bb62c`"。**本次把它在新机制下的后果钉死了**：env 基线的 `image.id` 抄录自镜像
  基线（= 制造机 npu1-11 的 id），所以**除制造机以外的机器上 `check` 必然 FAIL、`up.sh --strict` 必然拒绝启动**。
- 本轮处置（如实登记，不绕）：出结论的这一次**没有**满足 `--strict`；用 `up.sh`（spec 明文"默认只报告、
  不阻止启动"）完成容器路径与判据验证，并在本文件与看板里写明这一点。修复属镜像/机制侧（见待办）。

## 截选文件

| 文件 | 内容 |
|---|---|
| [`01_tool_selfproof.txt`](01_tool_selfproof.txt) | compose v2.29.7 / pyyaml 6.0.2 落点与 `sudo` 未使用 |
| [`02_export_show.txt`](02_export_show.txt) | `export --require-clean` 双腿 rc=0 + `show` 的挂载覆盖表 |
| [`03_check_and_strict.txt`](03_check_and_strict.txt) | `check` rc=1 的唯一 FAIL（跨机 id）+ `up.sh --strict` 拒绝启动 |
| [`04_up_plain_and_install.txt`](04_up_plain_and_install.txt) | `up.sh` 起容器 rc=0 + compose `Cmd` 自证 + 安装/加载自证 |
| [`05_acceptance_in_env.txt`](05_acceptance_in_env.txt) | 机制起的容器内两项验收（八组 + 四项全 PASS） |
| [`06_down.txt`](06_down.txt) | 收尾 `down` rc=0（含"down 也需 `WORKSPACE_ROOT`"的实测） |
| [`07_last_check.json`](07_last_check.json) | `check` 留痕（`FAIL: image` / `PASS: FlagCX` / 两条 INFO） |
| `08_env_baseline_train.yaml` · `09_env_baseline_infer.yaml` | 生成物副本；入库位置见 [`../../env/train/env.baseline.yaml`](../../env/train/env.baseline.yaml) 与 [`../../env/infer/env.baseline.yaml`](../../env/infer/env.baseline.yaml) |

## 本轮踩到的 3 个坑（都是"自己引入 + 按规矩修"）

1. **`docker compose down` 也要 `WORKSPACE_ROOT`**：`ENV-SPEC §4.3` 只写了绕开 `up.sh` 起容器时要 export，
   停容器同样需要。第一次 `down` 只 export 了 `FLAGRT_WORKSPACE_ROOT` ⇒ `${WORKSPACE_ROOT}` 插值成空 ⇒
   `invalid spec: :/workspace: empty section between colons`、`down rc=15`，容器**留了下来**（die 未释放）。
   补上后 rc=0。
2. **同名残留容器会挡住新机制起容器**：上一轮实质验证用了手工 `docker run`（容器名与 compose 相同）且只
   `docker stop` 未 `docker rm` ⇒ 这次 `up.sh` 报 `Conflict. The container name … is already in use`。
   ⇒ **凡用 compose/`up.sh` 起名与手工 `docker run` 相同的容器，切换路径前必须先 `docker rm -f`。**
3. **editable 构建产物不可字节复现**：同一 commit（`b576fa6a`）两次 editable 构建，`_C…so` 的 sha256
   前 16 位分别是 `098f3227…`（手工 `docker run` 那轮）与 `ac507f07…`（本轮 `up.sh` 那轮），而
   `libflagcx.so` 两次都是 `5af25cc9…`。⇒ **`.so` sha256 可用于区分"镜像内固化版 vs 宿主挂载版"，
   但不能当作"这份二进制就是某次构建"的稳定指纹**；装置自证应同时打印**加载路径**。

## 遗留（与本文相关的待办）

| # | 项 | 归属 |
|---|---|---|
| 1 | **`image.id` 的跨机口径**：`ENV-SPEC` 现只记一个 id ⇒ 非制造机上 `--strict` 恒不通过。可选处置：① 基线允许记多个等价 id；② 分发统一用 `docker load` 离线包（content-derived id，跨机一致）；③ 文档明确"`--strict` 只在制造机可满足"，并在别机改用其它可满足的判据 | **总组 / 镜像侧**（本方不擅改） |
| 2 | npu1-27 的用户级前置（compose 插件 + pyyaml）是**本机状态、不入仓**：换机器/换账号要重做一遍（步骤已写入 [`../../env/README.md`](../../env/README.md) §2） | 本方向（已在 env/README 记步骤） |
