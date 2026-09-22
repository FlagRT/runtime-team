# 910C 的 FlagOS 官方对应镜像（`flagos-runtime-ascend-cann9.0.0-910c`）

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文
> 性质：**910C 实例专属记录（不迁移）** ｜ 状态：**仅登记与对照，未切换**
> 触发：镜像血统对齐核查时发现 FlagOS 官方已有 910C 专用运行时镜像
> （背景与三家对照见 `../../prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`）

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| FlagOS 官方有 910C 专用运行时镜像吗？ | ✅ **有**：`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0` |
| 与我们锁定栈一致吗？ | ✅ **软件栈逐项一致**（CANN 9.0.0 / Python 3.11 / torch 2.10.0 / triton 3.5.0 / flagtree **0.7.0rc2+ascend3.5**） |
| 现在要切换吗？ | ❌ **不切换**。本方向结论已在锁定镜像上取得，切换需总组裁定；本文件**只做登记** |
| 能直接替代训练腿吗？ | ⚠️ **不能** —— 官方 runtime 镜像**不含 FlagCX**，而训练腿必需（与「官方推荐镜像不含 FlagCX」是同一个已登记诉求） |
| 它的价值 | ① 为 `dev/stack.lock.910c.yaml` 的 `candidates:` 提供**官方对应物**（此前候选是组内自建）；② 其 `base/runtime/<backend>.md` 明确了**宿主驱动前置**，是基座登记的规范做法；③ 附 **`compiler` 命令**可切 FlagTree/Triton |

---

## 2. 逐项对照

| 项 | 我们锁定的训练腿 | 我们锁定的推理腿 | **官方 runtime 镜像** |
|---|---|---|---|
| tag | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0` |
| digest | **无 registry**（靠 `docker save` + 重建配方） | `sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7` | `sha256:1048d622c928e86dd004ddb58b8b88602d91dc3c15458fda0262ca091e3ffb35` |
| 大小 | — | — | **5.4 GiB**（镜像层） |
| 架构 | aarch64 | aarch64 | **aarch64** |
| Python | 3.11 | — | **3.11** |
| CANN | 9.0.0 | 9.0（`vllm-ascend` 自带） | **CANN Toolkit 9.0.0 + CANN 910C Ops 9.0.0（`Ascend-cann-A3-ops`）+ CANN NNAL 9.0.0** |
| torch | 2.10.0 | — | **torch 2.10.0+cpu / torch-npu 2.10.0** |
| triton | 3.5.0 | — | **triton 3.5.0**（备 `triton_ascend==3.2.1`） |
| FlagTree | v1 未含（用 pip triton wheel）；`candidates.train` v2 为 flagtree 3.5 线 | — | **flagtree 0.7.0rc2+ascend3.5** |
| FlagGems | 主干 commit | — | **flag_gems 5.4.0-rc2.post3** |
| numpy / torchvision / torchaudio | — | — | 1.26.4 / 0.25.0+cpu / 2.10.0+cpu |
| FlagCX | ✅ 镜像内（训练腿必需） | — | ❌ **不含**（见 §4） |
| vLLM | — | 0.20.2 | runtime 层不含；**应用镜像另有**（见 §3.3） |
| 设备后端 | `flagos`（torch_fl，权宜例外） | `npu`（torch_npu） | **`npu`（torch_npu 2.10.0，即 Route A）** |
| **宿主驱动前置** | —（现网在用） | — | **26.0.rc1**（官方 `runtime/ascend-cann9.0.0-910c.md` 明示） |
| 容器工具链前置 | — | — | `Ascend-docker-runtime >= 6.0.RC3`（可选） |
| 起容器（官方给法） | — | — | `docker run --rm -it -e ASCEND_VISIBLE_DEVICES=0,1 <image> bash`（无 toolkit 时给全 `--device /dev/davinci*` 等） |

> **⇒ 关键观察**：官方 runtime 镜像的设备后端是 **`npu`（torch_npu）**，
> 即与我们**推理腿**同路线（Route A）；而我们**训练腿**因锁定镜像约束走 `flagos`（torch_fl）——
> 后者在 `stack.lock.910c.v2.yaml` 中登记为 **「权宜例外」**，并有「10 月起评估切回 Route A」的 TODO。
> **官方镜像天然是 Route A** ⇒ 若切到它，**训练腿的 flagos 例外可以取消**，TODO 随之关闭。
> 但前提是解决 §4 的 FlagCX 缺口。

---

## 3. 官方镜像详情与可复现入口

### 3.1 来源与出处

- 构建仓：`flagos-ai/build-infra`（公开）；**`configs.yaml` 是其镜像体系的唯一 source of truth**，
  文档站 `https://flagos-ai.github.io/release-info/` 由它自动生成（**不会漂移**）
- registry 前缀定义：`harbor.baai.ac.cn/` 下 `flagos-base`（厂商 SDK + OS 底座）→
  `flagos-runtime`（+ torch / 厂商 torch 插件 / triton / flag_gems）→ `flagos-app`（vLLM / SGLang / Megatron）
- 本镜像 `base` 层：`harbor.baai.ac.cn/flagos-base/flagos-base-ascend-cann9.0.0-910c:2.2.0`
- 当前体系版本 **2.2.0**；本镜像 push 时间 **2026-09-20**

### 3.2 `configs.yaml` 中的 910C 条目（原文摘录）

```yaml
    # 910C twin of cann9.0.0. The ops package is the only chip-split piece
    # (Ascend-cann-A3-ops_<ver>_linux-aarch64.run)
    cann9.0.0-910c:
      extras: ascend-cann900-910c
      hardware: ["Ascend 910C"]
      driver: "26.0.rc1"
      python: "3.11"
      cmake_backend: NPU
      triton: triton==3.5.0
      flagtree: flagtree==0.7.0rc2+ascend3.5
      triton_post_install:
        - triton_ascend==3.2.1
      deps:
        - torch==2.10.0+cpu
        - torch-npu==2.10.0
        ...
      deps_app:
        vllm0.20.2: []
        vllm0.24.0: []
      sdk:
        - CANN Toolkit 9.0.0 (aarch64)
        - CANN 910C Ops 9.0.0 (aarch64)   # Ascend-cann-A3-ops_9.0.0_linux-aarch64.run
        - CANN NNAL 9.0.0 (aarch64)
```

> 注意 `# 910C twin of cann9.0.0` 的注释：**910C 与 910B 共用 toolkit/NNAL/依赖，
> 唯一芯片切分件是 ops 包**（`Ascend-cann-A3-ops`）。这与我们既有认知一致，可作为对外口径依据。

### 3.3 同线的官方应用镜像（供推理腿/训练腿评估）

| 镜像 | tag | push | digest | 大小 |
|---|---|---|---|---|
| `flagos-app/vllm0.20.2-ascend-cann9.0.0-910c` | `2.2.0-0.2.2rc2.post2` | 2026-09-20 | `sha256:08fc5d93314568ca677…` | 5.8 GiB |
| `flagos-app/vllm0.24.0-ascend-cann9.0.0-910c` | `2.2.0-0.3.0rc2.post2` | 2026-09-20 | `sha256:70272c56ea65dd0f243…` | 5.8 GiB |
| `flagos-app/megatron_training0.17.1-ascend-cann9.0.0` | `2.1.2-0.2.1_9.g48b97a13f` | 2026-08-20 | `sha256:84cfc3a0974ee5afdd6…` | 5.5 GiB |

> 我们推理腿现用 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（华为官方，vLLM **0.20.2**）——
> 官方 `flagos-app/vllm0.20.2-ascend-cann9.0.0-910c` 与之**同 vLLM 小版本**，可作对照候选。

### 3.4 一键复核（任何人可复跑，**无需凭据**）

```bash
R=flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c; T=2.2.0
TOK=$(curl -s "https://harbor.baai.ac.cn/service/token?service=harbor-registry&scope=repository:$R:pull" \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -sD- -o/dev/null -H "Authorization: Bearer $TOK" \
     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
     "https://harbor.baai.ac.cn/v2/$R/manifests/$T" | grep -i docker-content-digest
```

---

## 4. ⚠️ 训练腿缺口：官方 runtime 镜像不含 FlagCX

- 官方 `runtime/ascend-cann9.0.0-910c.md` 列出的 major Python packages 为
  `flag_gems` / `flagtree` / `numpy` / `torch-npu` / `torch` / `torchaudio` / `torchvision` / `triton`
  —— **无 flagcx**；
- Harbor 上确有 FlagCX 镜像线（`flagcx/flagcx-{ascend,nvidia,metax,tsingmicro}`），
  但 **`flagcx-ascend` 最后 push 为 2026-02-02（tag `latest`）**，明显早于我们使用的 FlagCX 0.13.0 线
  ⇒ **不构成训练腿的可用替代**；
- 因此：**「官方推荐镜像不含 FlagCX，而训练腿必需」这一诉求不变**
  （已登记于 `../../prototype/docs/IMAGE_REQUIREMENT_SPEC_20260920.md`「请总组裁定的三件事」第 3 项）。
  若未来评估切到官方 runtime 镜像，仍需：① 官方出带 FlagCX 的 910C 变体，或
  ② 由我们提供**可复现的叠加配方**（官方 base/runtime + FlagCX），二者都需总组裁定。

---

## 5. 处置建议（本方向只提诉求，不自行切换）

| # | 建议 | 说明 |
|---|---|---|
| 1 | **登记为 `candidates.train` 的「官方对应物」**（不生效） | 已在 `dev/stack.lock.910c.yaml` 工作草稿中登记；与既有组内自建候选**并列**，由总组裁定 |
| 2 | **是否切换由总组裁定** | 本方向**不自行切换**（结论已在锁定镜像上取得；切换会引入重跑成本与新的不确定性） |
| 3 | **若总组倾向切换，本方向建议的前提条件** | ① 先补齐 FlagCX 缺口（见 §4）；② 确认宿主驱动 26.0.rc1 前提（我们现网驱动版本**未在本次核查中采集**，须实测）；③ 切换后重跑 `VERIFICATION_MANIFEST` 的 9 条复核命令 |
| 4 | **采用其「明示宿主驱动前置」的登记做法** | 官方 `base|runtime/<backend>.md` 每份都写 **Host driver**，正是我们 `dev/images/<name>/v<N>/lock.yaml` 可以吸收的字段（建议后续补入） |
| 5 | **顺带可用于训练腿 Route A 回归评估** | 官方镜像设备后端为 `npu`（torch_npu）⇒ 若切它，`stack.lock.910c.v2.yaml` 里「训练腿走 flagos 的权宜例外」可一并取消，关闭 10 月 TODO（仍需 FlagCX 到位） |

---

## 6. 未验证 / 未取得（如实标注，不补零）

- ⚠️ **未实际 `docker pull`**，也未起容器。本次仅做 **Registry v2 接口级验证**（匿名 token 取 manifest 成功）；
  镜像内**实际包版本未在本机复核**（§2 的版本列取自官方 `runtime/ascend-cann9.0.0-910c.md`）。
- ⚠️ **我们现网 910C 宿主驱动版本未采集** ⇒ 无法判断是否满足官方标称的 `26.0.rc1`，
  切换前须实测（`npu-smi info` 首行 / `/usr/local/Ascend/driver/version.info`）。
- ⚠️ 官方 runtime 镜像**是否已含我们原型所需的全部前提**（如 `transformers`、`vllm` 的安装位置，
  参考 P800 曾踩到的「`vllm` 不在默认 `PATH`」类问题）**未验证**。
- ⚠️ `flagos-app/megatron_training0.17.1-ascend-cann9.0.0` 是否为 **910B/910C 通用**（目录名无 `-910c` 后缀）
  **未确认**，其与 910C 的配套关系需向官方确认后再用。
