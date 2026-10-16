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

## 2. ⚠️ 机制前置：npu1-27 当前**不具备**

新机制的两个工具依赖下列宿主能力，**npu1-27 实测都不满足**（2026-10-16）：

| 依赖 | 用途 | npu1-27 实测 | 有该能力的机器 |
|---|---|---|---|
| `docker compose` **v2 插件** | `baseline.py` 靠 `docker compose config --format json` 解析 compose 的真实 bind 挂载；`up.sh` 靠它起容器 | ❌ `docker 20.10.8`，`docker compose` 不是 docker 子命令；`/usr/libexec/docker/cli-plugins/` 下只有 `docker-app`/`docker-buildx`；`~/.docker/cli-plugins/` 存在但为空 | npu1-11（Compose v2.29.7，总组验证环境）、P800（Compose v2.40.3） |
| `pyyaml` | `baseline.py` 顶部 `import yaml` | ❌ `/usr/bin/python3`(3.9.9) 无 `yaml`；无其他可用解释器 | P800（pyyaml 6.0.1） |

⇒ **在 npu1-27 上，`baseline.py export/check` 与 `up.sh --strict` 目前都跑不了**（会在
`import yaml` 或 `docker compose` 处直接失败）。两条可选路径：

1. **补齐 npu1-27 的用户级前置**（不需要 root、不改驱动）：
   `~/.docker/cli-plugins/docker-compose`（aarch64 静态二进制）+ 一份私有目录的 pyyaml
   （`pip install --target` 后用 `PYTHONPATH` 指过去）。装法待用户确认后写入本节。
2. **在具备前置的机器上做机制侧验证**（如 P800 可跑 `docker compose config` 与 `baseline.py`），
   910C 侧只做实质验证（等价 `docker run` + 判据）。

本方向当前状态：**镜像侧档案与运行环境目录已就位，机制侧的 export/check/up 端到端待前置补齐后执行**；
在此之前不得声称「已在 910C 上按 `--strict` 起过该环境」。

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
- 分支/提交：`kistich/ascend-flagcx-adapt`，见 `train/env.baseline.yaml`（**待生成**，见 §2）里的
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

# 启动：验收与出结论的运行必须加 --strict（不一致即拒绝启动）
docker ps        # 昇腾驱动对带卡容器独占：确认本机无其他带卡容器
FLAGRT_MODELS_DIR=<实体模型目录> dev/lib/up.sh dev/device-context/910C/env/train --strict

# 复现他人的环境：先按基线切换挂载 repo，再启动
python3 dev/lib/baseline.py load dev/device-context/910C/env/train
dev/lib/up.sh dev/device-context/910C/env/train --strict

# 收尾
docker compose -f dev/device-context/910C/env/train/docker-compose.yml down
```

出结论的运行另须 `export --require-clean`（工作区不干净或 commit 未推送即拒绝导出）。

## 7. 与方向内其他文档的关系

- 方向看板与四实例状态：[`../../README.md`](../../README.md) · [`../../STATUS.md`](../../STATUS.md)
- 910C 实例结论与证据目录：[`../README.md`](../README.md)
- 选卡与名额口径：[`../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`](../docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md)
- 基座分层的职责边界与「叠上来」的标准动作：本方向汇总文档
  [`../../docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md`](../../docs/BASELINE_LAYERS_AND_ADOPTION_20261016.md)
