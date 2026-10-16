# device-context 方向 · 910C 运行环境目录

> 规范：[`dev/ENV-SPEC.md`](../../../ENV-SPEC.md) §3.2（运行环境目录）与 §5（挂载资产生效方式）。
> 本目录替换的是本方向此前的旧写法（`docker compose -f ../compose.base.yml -f docker-compose.yml`，
> 该写法与 `compose.base.yml` 已于 2026-09-23 退役，见 [`使用与维护指南`](../../../../docs/运行时基座-使用与维护指南.md) §6）。

## 1. 本目录是什么

一个目录 = 一套完备运行环境 = `docker-compose.yml`（启动与装配）+ `env.baseline.yaml`（版本事实）。
本方向的 910C 是**两腿分栈**（训练腿与推理腿各一个镜像），故两腿各一套：

| 目录 | 腿 | 镜像（现行发布物） | 挂载资产 | 容器名 |
|---|---|---|---|---|
| [`train/`](train/docker-compose.yml) | 训练腿 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | **FlagCX**（pip-editable 覆盖固化版本） | `flagos-dc-train-910c-dev` |
| [`infer/`](infer/docker-compose.yml) | 推理腿 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（华为昇腾官方，digest 锁定） | 无（`{}`） | `flagos-dc-infer-910c-dev` |

两腿镜像的事实依据：镜像基线 [`ascend-train-comm/v3`](../../../images/ascend-train-comm/v3/image.baseline.yaml) ·
[`ascend-infer-vllm/v1`](../../../images/ascend-infer-vllm/v1/image.baseline.yaml)；
发版口径与已知限制见 [`基座发版说明-20261014`](../../../../docs/基座发版说明-20261014.md)。

**分工**：本目录只管容器生命周期；**推理服务一律走组内唯一入口**
[`prototype/scripts/serve_standard.sh`](../../prototype/scripts/serve_standard.sh)
（demo/环境起「毛坯房」，serve 在里面「点火」）。训练腿的判据入口见
[`../README.md`](../README.md) 与 [`prototype/scripts/`](../../prototype/scripts/)。

## 2. 机制前置：npu1-27 **已补齐**（用户级，无 sudo、不改宿主包）

新机制的工具链依赖下列宿主能力。2026-10-16 实测该机**原本都不满足**，当日已按下面的方式补齐，
**两条腿的 `export` 已 rc=0**（证据见 [`../probes/baseline_mechanism_20261016_out/SUMMARY.md`](../probes/baseline_mechanism_20261016_out/SUMMARY.md)）。

| 依赖 | 用途 | 补齐方式与落点 | 版本 |
|---|---|---|---|
| `docker compose` **v2 插件** | `baseline.py` 靠 `docker compose config --format json` 解析 compose 的真实 bind 挂载；`up.sh` 靠它起/停容器 | 官方 Release 静态二进制（aarch64）→ `~/.docker/cli-plugins/docker-compose`（`install -m 0755`，**用户级，不需 root**） | **v2.29.7**（与总组在 npu1-11 上验证过的 Compose 同版本） |
| `pyyaml` | `baseline.py` 顶部 `import yaml` | `pip install --user` 一份 aarch64 manylinux wheel（cp39 对应本机 `/usr/bin/python3` 3.9.9） | 6.0.2 |

```bash
# 复现该前置（新机器/新账号要重做一遍；本机状态不入仓）
curl -sSL -C - -o /tmp/docker-compose-linux-aarch64 \
  https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-aarch64
curl -sSL -o /tmp/docker-compose.sha256 \
  https://github.com/docker/compose/releases/download/v2.29.7/docker-compose-linux-aarch64.sha256
sha256sum -c /tmp/docker-compose.sha256          # 期望 6e9fbd5daa20dca5d7d89145081ae8155d68ef2928b497d9f85b54fe0f9dbb2c
mkdir -p ~/.docker/cli-plugins
install -m 0755 /tmp/docker-compose-linux-aarch64 ~/.docker/cli-plugins/docker-compose
docker compose version                            # 期望 v2.29.7
python3 -m pip install --user --no-index --no-deps \
  PyYAML-6.0.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
python3 -c "import yaml; print(yaml.__version__)"  # 期望 6.0.2
```

> ⚠️ **别指望从 npu1-11 拷环境**：该机是 `10.120.72.11`（在 27 的 `/etc/hosts` 里、ping 通、22 开），
> 但 `hliu553@10.120.72.11` 与本地三把私钥一律 `Permission denied (publickey,password)` ⇒ 拿不到。
> 上面走的是"**同版本官方件**"，版本号与 11 上验证过的相同。
> 另：本机 `sudo -n true` 报 `sudo: a password is required`（非交互不可用）⇒ 未走 sudo。

### 2.1 ⚠️ `up.sh --strict` 在本机**必然拒绝启动**（跨机镜像 id 不同）

`env.baseline.yaml` 的 `image.id` 是**抄录自镜像基线**的，而镜像基线记的是**制造机（npu1-11）**的 id：

```
[FAIL] image: id 不一致: 基线 sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15
                         本机 sha256:7028028bb62ce06322f2b53379891c062caf8ac6b3f2b1a469be6dafe9af7913
```

- 这不是本方向的配置问题——[`image_list.md`](../../../images/image_list.md) 早已登记"跨机 image id 不同属预期
  （内容同源）：npu1-27 侧为 `be30a952c2eb` / `7028028bb62c`"。**但它的后果要写清**：`check` 在非制造机上
  恒 `FAIL`，`--strict` 因此恒拒绝启动。
- 本方向处置（如实登记，不绕）：本机出结论的运行**用 `up.sh`（不加 `--strict`）**，并在证据里注明
  "本次未满足 `--strict`"；其余各项仍照核对（`[PASS] FlagCX: commit=…` 是逐字通过的）。
- 正解在镜像/机制侧（允许记多个等价 id，或统一用 `docker load` 离线包分发使 id 跨机一致），
  已记入待办；**本方不擅改基线或 `baseline.py`**。

## 3. 选卡：唯一可靠的判据是「逐 die 建容器试 INIT」

910C 的带卡容器**独占**，且**「空闲卡」的依据是容器映射表而不是 `npu-smi` 的进程表**
——已被 init 但当前空闲的容器在进程表里是**隐形**的，却足以让你 `aclInit` 失败。
故 16 张 die 都要挂会与他人活跃容器相交，容器内直接
`dcmi model initialized failed, because the device is used`、`device_count=0`。

```bash
# 逐 die 单独建容器试 ACL INIT，不早退，输出 FREE 列表
bash dev/device-context/910C/probes/probe_all_dies_910c_20261016.sh
# 把 FREE 里的两个填进 dev/device-context/910C/env/{train,infer}/.env（或直接 export）
```

口径全文：[`../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`](../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md) §6
（含「同一时刻只留 1 个」「与挑哪张卡无关」两条旧口径被推翻的实测）。

`DC_DIE_A` / `DC_DIE_B` **不给默认值**：留空即让 compose 报错。给默认值等于让默认值替你选卡，
而选卡必须每次按实测结果来。也可以直接用宿主 shell 环境传（优先级高于 `.env`，且不受 compose
版本对「`.env` 取自项目目录还是当前目录」的行为差异影响）：

```bash
export DC_DIE_A=5 DC_DIE_B=0
```

## 4. 为什么不用 `extends` 复用镜像 demo

两腿的镜像 demo（[`ascend-train-comm/v3/demo/`](../../../images/ascend-train-comm/v3/demo/docker-compose.yml)、
[`ascend-infer-vllm/v1/demo/`](../../../images/ascend-infer-vllm/v1/demo/docker-compose.yml)）都把
**全部 16 张 davinci** 映射进容器。compose 的 `extends` 对 `devices` 这类序列**只能并集、不能收缩**，
而本方向在共享机上必须只挂自己要用的 die ⇒ 无法用 `extends` 表达。

因此本目录的两个 compose 是**自包含**的（属性逐项取自 demo 的实测配置，差异只有 devices 收窄与容器名），
并在文件头写明这一处偏差的理由。镜像 demo 是「单租户整机」口径的推荐用法；本目录是「共享机按 die 分区」口径。

## 5. 挂载资产：训练腿的 FlagCX

训练腿镜像里固化的 FlagCX 是公开主干 `4e0e0cbcbf721169ca82348080f8353aebfe2c31`
（= 本组分支的分叉点），**不含** c10d 完成语义与异构修复（缺陷 case：
[`dev/issues/flagcx-c10d-collectives/`](../../../issues/flagcx-c10d-collectives/README.md)）。
本方向需要那份修复，故按 ENV-SPEC §2.2「同一 repo 既固化又挂载」以 pip-editable 覆盖：

- 宿主 checkout：`<WORKSPACE_ROOT>/FlagCX`（ENV-SPEC §3.2 的默认表；远端 `git@github.com:FlagRT/FlagCX.git`）
- 分支/提交：`kistich/ascend-flagcx-adapt`，见 [`train/env.baseline.yaml`](train/env.baseline.yaml) 里的
  `overrides_baked` / `baked_commit` / `commit` / `pushed`
- 生效方式：容器启动时 `USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation -e /workspace/FlagCX`
  （FlagCX 含 C++ ⇒ 改代码后须重启容器触发重装）
- 构建产物（`build/`、`*.egg-info`）与 `__pycache__` 不污染 checkout：前者已在 FlagCX 自己的
  `.gitignore` 里，后者由 `PYTHONDONTWRITEBYTECODE=1` 挡住

改动稳定后按 ENV-SPEC §2.3「晋升」固化进下一个镜像版本，本环境再删掉该挂载声明。

## 6. 常用命令（仓根执行）

```bash
# 首次或 x-flagrt / 挂载 repo 变更后：导出运行环境基线
python3 dev/lib/baseline.py export dev/device-context/910C/env/train
python3 dev/lib/baseline.py export dev/device-context/910C/env/infer

# 只读查看 / 核对（check 结果写本目录 .last-check.json，已被根 .gitignore 挡掉）
python3 dev/lib/baseline.py show  dev/device-context/910C/env/train
python3 dev/lib/baseline.py check dev/device-context/910C/env/train

# 启动：验收与出结论的运行应当加 --strict
#   ⚠️ 本机（npu1-27）例外：镜像 id 与制造机不同 ⇒ check 恒 FAIL、--strict 恒拒绝启动（见 §2.1）。
#      本机出结论的运行用下面第 2 条，并在证据里注明「本次未满足 --strict」。
docker ps        # 昇腾驱动对带卡容器独占：确认本机无其他带卡容器
export DC_DIE_A=<die> DC_DIE_B=<die>          # 见 §3；也可写进本目录 .env
FLAGRT_MODELS_DIR=<实体模型目录> dev/lib/up.sh dev/device-context/910C/env/train --strict  # ① 制造机上
FLAGRT_MODELS_DIR=<实体模型目录> dev/lib/up.sh dev/device-context/910C/env/train           # ② 本机

# 复现他人的环境：先按基线切换挂载 repo，再启动
python3 dev/lib/baseline.py load dev/device-context/910C/env/train
dev/lib/up.sh dev/device-context/910C/env/train

# 收尾（⚠️ 绕开 up.sh 时必须**自己** export WORKSPACE_ROOT 与 die 号 —— 起容器与停容器都要；
#        只设 FLAGRT_WORKSPACE_ROOT 不够：compose 插值取的是 WORKSPACE_ROOT，
#        缺它会报 invalid spec: :/workspace: empty section between colons 且容器留不下来被释放）
export WORKSPACE_ROOT=$PWD DC_DIE_A=<die> DC_DIE_B=<die>
docker compose -f dev/device-context/910C/env/train/docker-compose.yml down
```

- `check` 的留痕写 `<env-dir>/.last-check.json`（已被根 `.gitignore` 挡掉，不入仓；本轮实例见
  [`../probes/baseline_mechanism_20261016_out/07_last_check.json`](../probes/baseline_mechanism_20261016_out/07_last_check.json)）。
- 出结论的运行另须 `export --require-clean`（工作区不干净或 commit 未推送即拒绝导出）。
- ⚠️ 从手工 `docker run` 切到 `up.sh` 前，**先 `docker rm -f` 同名容器**：compose 遇到同名残留会
  `Conflict. The container name … is already in use` 而起不来（本轮实测踩到）。

## 7. 与方向内其他文档的关系

- 方向看板与四实例状态：[`../../README.md`](../../README.md) · [`../../STATUS.md`](../../STATUS.md)
- 910C 实例结论与证据目录：[`../README.md`](../README.md)
- 选卡与名额口径：[`../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`](../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md)
- 基座分层的职责边界与「叠上来」的标准动作：本方向汇总文档
  [`../../docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md`](../../docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md)
