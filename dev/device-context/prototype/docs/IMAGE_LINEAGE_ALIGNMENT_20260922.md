# 镜像血统对齐核查（昆仑芯 / 寒武纪 / 昇腾）—— 对照类脑基线与 FlagOS 官方镜像体系

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文
> 触发：评估「测试新芯片可对齐类脑（x-benchmark）的镜像版本」这一说法，核对本方向用过的
> **昆仑芯 P800** 与待接入的 **寒武纪 MLU590** 镜像选型**是否需要重新调整**
> 依据：类脑仓库文档 + FlagTree/FlagGems 上游 + FlagOS 官方镜像构建仓（`flagos-ai/build-infra`）
> **+ BAAI Harbor 实机探测（本项目首次）**。写法：依据自包含，不写成「见某文件某节」。

---

## 0. 结论速览

| 芯片实例 | 我们现用镜像 | 是否需要调整 | 一句话理由 |
|---|---|---|---|
| **第 1 家 昇腾 910C** | 组内 `flagrt/ascend-operator-runtime-comm:0.1.3-…`（训练腿）+ 华为 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（推理腿） | **不需要**（仅新增一条「官方对应物」备选） | 栈逐项落在 **ascend3.5 / CANN 9.0** 线，与类脑文档口径一致；FlagOS 官方另有同线镜像可作候选 |
| **第 2 家 昆仑芯 P800** | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base` 血统 | **镜像本身不需要**；**入锁候选口径需要补充** | 该镜像**就是** FlagTree 手册给 P800 的那一份，与类脑指向同一条 **xpu3.6** 线；但 FlagOS 官方已有更新血统（**XRE 5.37.1**），前置是宿主要 5.37.1，我们实测是 **5.0.21.47** |
| **第 3 家 寒武纪 MLU590** | 尚未选定（此前结论：须走寒武纪官方渠道申请） | **需要，且结论要更正** | ① **FlagOS 官方在 BAAI Harbor 上已有寒武纪三代镜像**，且**实测可匿名拉取** ⇒「只能走寒武纪私仓」不成立；② 但**档位由宿主驱动决定**：我们驱动 **v6.2.29** 只能走 **neuware4.4.3**（要求 6.2.15），**neuware4.7.2 要求 6.5.48** |

**最关键的一条**：**寒武纪的镜像档位不能只看 `torch-mlu` 版本，必须先看宿主驱动版本**。
我们原选型文档锚定的 `torch 2.11.0 / torch-mlu 1.33.1 / py3.12` 正是 **neuware4.7.2** 档，
而该档官方标注要求**宿主驱动 6.5.48**——**我们两台测试机是 6.2.29，不满足**。

---

## 1. 被参考的那份文档到底写了什么

`https://github.com/flag-custom-stack/x-benchmark/blob/master/docs/non-ai-docs/基础测试环境配置之算子.md`
（仓库 `flag-custom-stack/x-benchmark`，私有，全仓 123 个文件；该文 1473 B）正文只有 4 节：

| 节 | 平台 | 指向 | 版本线 | FlagGems 安装 extra |
|---|---|---|---|---|
| 「Ascend 910 (CANN 9.0)」 | 昇腾 | FlagTree wiki `User-manual-for-ascend` | **ascend3.5** | `.[ascend-cann900]` |
| 「Ascend 910 (CANN 8.5)」 | 昇腾 | 同上 | **ascend3.2** | `.[ascend-cann850]` |
| 「昆仑芯」 | 昆仑芯 | FlagTree wiki `User-manual-for-xpu` | **xpu3.6** | `.[kunlunxin]` |
| 「平头哥」 | 平头哥 | FlagTree wiki `User-manual-for-ppu` | **ppu3.6** | `.[thead]` |

**⚠️ 关键事实：这份「类脑基础测试环境配置之算子」里没有寒武纪这一节。**
交叉复核（两法一致，非单点判断）：
- 该仓库 `docs/non-ai-docs/` 仅 5 个文件（mul 算子问题描述 / 演示流程说明 / 基础环境配置之模型 /
  基础环境配置之算子 / 特殊模型记录），**逐份看过，无 cambricon / MLU 字样**；
- 该仓库全 123 个文件清单里亦无寒武纪相关文件。

⇒ 结论：**类脑走的是「FlagTree 算子线」的镜像口径**，而 **FlagTree 本身没有寒武纪后端手册**
（wiki 26 页中 `User-manual-for-*` 共 17 个后端：aipu/amd/ascend/cpu/enflame/hcu/iluvatar/metax/
mthreads/nvidia/ppu/rpu/spacemit/sunrise/tileir/tsingmicro/xpu，**无 cambricon**；
`User Manual` 索引页的安装表亦无 cambricon 行）。
⇒ **「对齐类脑的镜像版本」这条路，在寒武纪这一支上没有可对齐的目标**。

---

## 2. 昆仑芯 P800：镜像本身无需调整，但入锁口径要补一条

### 2.1 我们用的就是手册给的那一份

FlagTree `User-manual-for-xpu`（2026-09-18 编辑，12 次修订）标题为 **「💫 KLX xpu 3.6 —— Based on Triton 3.6, x64，Available for P800」**，
其 §1.1「Use the image (P800)」给的**唯一**镜像就是：

```text
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base
# Plan A: docker pull (59.9GB)   Plan B: docker load (32GB)
```

与我们现用镜像（`flaggems-main-dev:202608` 变体，等价性已验证）**同一血统**。
类脑 `docs/p800_flag_gems_copy_unavailable_reproduction.md` 记录的 P800 实测栈为
`torch 2.9.0+cu129 / flagtree 0.6.1+xpu3.6`，与 FlagGems 的 `kunlunxin` extra
（`flagtree==0.6.1+xpu3.6`）一致。**⇒ 三者同线，镜像选型不需要调整。**

> **⚠️ 一处待核对的差异（不臆断）**：类脑该文档记 `Triton 3.5.0`，而我们 P800 基线实测记
> `triton 3.6.0`（两者 `flagtree` 均为 `0.6.1+xpu3.6`）。**未查明原因**，标注为待核对项；
> 它只影响算子编译路径，**不影响本方向（设备上下文）结论**。

### 2.2 但 FlagOS 官方已有更新血统：XRE 5.37.1

`harbor.baai.ac.cn/flagos-base/`…`flagos-runtime/`…`flagos-app/` 三代体系（下详 §4）里，
昆仑芯当前档是 **`xre5.37.1`**：

| 项 | 我们现用（FlagTree xpu3.6 线） | FlagOS 官方 flagos-runtime 线 |
|---|---|---|
| 镜像 | `…/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base` | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.2.0` |
| Python | 3.10 | 3.10 |
| torch | 2.9.0+cu129 | 2.9.0+cu129（**一致**） |
| flagtree | 0.6.1 → 手册现行建议 0.7.0rc3 | **0.7.0rc2+xpu3.6** |
| triton | 3.6.0 | 3.6.0+gitcd2d6c1b |
| 底层 SDK | FlagTree xpu 线自带 | **CUDA 12.9.0_575.51.03 / XRE-CUDA12 5.37.1.0 / XCUDART 5.13.0** |
| **宿主驱动前置** | —（已在用） | **5.37.1** |
| 我们实机 | — | 宿主 `xpu-smi` = **5.0.21.47**（XPU-RT 5.0.21，SDK `/usr/local/xpu-5.0.21.47`） |
| digest | `sha256:ea6d797a…`（`-base`） | `sha256:0f488c7bca77bb613547bee6b61e87bf91c85a19ea0cab4ffb3036b3da32f938` |

**⇒ 处置建议**：**不切换**（现有结论是在已等价验证过的镜像上取得的，切换要动驱动、收益不明确），
但**入锁材料里要把「两条血统」都写清**，并注明 FlagOS 官方线的前置是宿主驱动 5.37.1
（我们**未实测** 5.0.21.47 能否跑 5.37.1 的用户态栈，**不臆断兼容性**）。

### 2.3 ⭐ 顺带得到一条对本方向缺陷的官方印证

FlagOS 官方构建仓 `flagos-ai/build-infra` 的 `configs.yaml` 里，昆仑芯 vLLM 应用层环境变量原文：

```yaml
          vllm:
            # XPU_EVENT_KL3_ENABLE deliberately NOT set: it is the P1 fake-hang trigger
            # (device timeout) on this XRE stack — default env is clean, keep it so.
            VLLM_FL_PLATFORM: kunlunxin
            VLLM_FL_PREFER: flagos
            USE_FLAGGEMS: "1"
            VLLM_FL_FLAGOS_WHITELIST: silu_and_mul,rms_norm,rotary_embedding
```

—— 官方**明确不设 `XPU_EVENT_KL3_ENABLE`**，并称之为 **「P1 fake-hang trigger（device timeout）」**。
这与本方向独立定位的结论一致（该变量 + 设备侧集合通信 ⇒ ≈89% 概率性永久自旋，自旋点在厂商
`libxpucuda.so` 内，偏移 `+0x94080`）。⇒ **我们上报的缺陷方向正确，且上游已把它当作已知问题对待**；
对外提交时应引用该注释作为「上游已承认该触发器」的旁证。

---

## 3. 寒武纪 MLU590：结论需要更正，且档位受宿主驱动硬约束

### 3.1 更正：「必须走寒武纪官方渠道、无公开上游」不成立

`flagos-ai/build-infra` 的 `configs.yaml` 与 BAAI Harbor 实探结果（**本项目首次对 Harbor 做接口级探测**）
显示，**寒武纪在 FlagOS 官方镜像体系里是完整的一等公民**：

| 层 | 镜像仓（BAAI Harbor） | 说明 |
|---|---|---|
| 基座 | `flagos-base/flagos-base-cambricon-neuware4.4.3` 与 `…-neuware4.7.2` | 厂商 SDK + OS 底座 |
| 运行时 | `flagos-runtime/flagos-runtime-cambricon-neuware4.4.3` 与 `…-neuware4.7.2` | 加 torch/torch-mlu/triton/flag_gems |
| 应用 | `flagos-app/{vllm0.20.2,vllm0.24.0,sglang0.5.18,megatron_training0.17.1}-cambricon-{neuware4.4.3,neuware4.7.2}` | **8 个仓**，含推理与训练 |
| 周测 | `flaggems/cambricon-flaggems-test-mlu590-m9de-triton3-2-1`、`…-mlu590-m9de` | FlagGems 周测专用，`maintainer=cambricon` |

**匿名可拉实测**（Docker Registry v2 标准鉴权路径，**未使用任何凭据**；`/v2/` 直接访问返回 401 是
Registry 的正常 challenge，不代表私有）：

| 镜像 | tag | digest（实取 manifest 的 `Docker-Content-Digest`） |
|---|---|---|
| `flagos-runtime/flagos-runtime-cambricon-neuware4.7.2` | 2.2.0 | `sha256:a37f46e331d638f5901c1ae30ac10b79fb41d76451822eee40c470be712a5e20` |
| `flagos-runtime/flagos-runtime-cambricon-neuware4.4.3` | 2.2.0 | `sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2` |
| `flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1` | 2.2.0 | `sha256:0f488c7bca77bb613547bee6b61e87bf91c85a19ea0cab4ffb3036b3da32f938` |
| `flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c` | 2.2.0 | `sha256:1048d622c928e86dd004ddb58b8b88602d91dc3c15458fda0262ca091e3ffb35` |
| `flaggems/cambricon-flaggems-test-mlu590-m9de-triton3-2-1` | 20260618 | `sha256:74fef7dfad1000e74fa3cdd5b5c0f22d2a97f84cd947b6ddded84ef48a84fbb5` |

同法探测 `harbor.baai.ac.cn/api/v2.0/projects`（匿名可读）确认 `flaggems` / `flagtree` / `flagos-app`
等项目 `public = true`。
⇒ **寒武纪私仓凭据（`docker.cambricon.com` 等）不再是接入的硬前置**；可以直接从 BAAI Harbor 拉。
（**未做**：在带卡机器上真正 `docker pull` —— 那需要 docker 权限，见 §5。）

### 3.2 ⭐ 两档版本组合（官方 `configs.yaml` 原文口径）

| | `cambricon-neuware4.4.3` | `cambricon-neuware4.7.2` |
|---|---|---|
| Python | 3.10 | 3.12 |
| torch | 2.7.1+cpu | 2.11.0+cpu |
| torch-mlu | 1.29.2+torch2.7.1 | **1.33.1+torch2.11.0** |
| torch-mlu-ops | 1.8.0+torch2.7.1 | 1.12.1+torch2.11.0 |
| triton | 3.2.0+mlu1.7.2 | 3.4.0+mlu2.1.1 |
| 其他 | cambricon_dali 0.13.0 | pandas 3.0.5 / numpy 2.2.6 |
| **宿主驱动前置** | **6.2.15** | **6.5.48** |
| 基座 OS | ubuntu 22.04 | ubuntu 24.04 |
| SDK 组件 | cntoolkit 4.4.3 / cncl 1.29.4 / cnnl 2.1.829 / mluops 1.8.1 / cnmon 6.2.15 | cntoolkit 4.7.2 / cncl 1.30.8 / cnnl 2.2.14 / mluops 1.12.0 / cnmon 6.5.48 |
| FlagGems extra 名 | `.[cambricon-neuware443]` | `.[cambricon-neuware472]` |

### 3.3 ⚠️ 硬约束：我们两台测试机是驱动 **v6.2.29**

实测（只读）：`8 × MLU590-M9`、**驱动 `v6.2.29` / 固件 `v1.5.0`**、`cnmon` = CNMON v6.2.29、
设备节点 `/dev/cambricon_dev0..7`。

- v6.2.29 落在 **6.2.x 线** ⇒ 与 `neuware4.4.3`（要求 6.2.15）**同线**；
- **`neuware4.7.2` 要求 6.5.48，我们**不满足**。

**⇒ 处置（两条路，二选一，写进申请清单）**：

| 方案 | 动作 | 代价 / 风险 |
|---|---|---|
| **A（推荐先走）** | 用 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（py3.10 / torch 2.7.1 / torch-mlu 1.29.2 / triton 3.2.0+mlu1.7.2） | 版本较旧；`torch-mlu` 对 2.7.1 的配套此前我方未验证过；**但驱动同线、起容器风险最低** |
| **B** | 请管理员把两台机器驱动升到 **6.5.48**，再用 `…-neuware4.7.2:2.2.0`（py3.12 / torch 2.11.0 / torch-mlu 1.33.1） | 与测试机上他人资产（`/srv/data/build_base_venv.sh` 里就是 4.7.2 档的组合）一致；**但要动宿主驱动 ⇒ 会影响他人，需协调** |

> **诚实标注**：`6.2.29` 能否实际跑 `neuware4.4.3`（官方标 6.2.15）**我们未实测**——
> 同 minor 线属**推断**，须起容器后以 `cnmon` / `torch.mlu.device_count()` 实测定论。
> 同理 `5.0.21.47`（P800）与 XRE 5.37.1 的兼容性亦**未实测**。

### 3.4 对本方向选型文档的具体影响

原 `MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` 的 §3.4 把目标档锚在
「`torch2.11.0` + `torchmlu1.33.1` + `ubuntu22.04` + `py312`」——
**方向正确**（就是 `neuware4.7.2` 档），但：
① 缺了**官方档位名 `neuware4.7.2`**；
② 缺了**宿主驱动 6.5.48 这一硬前置**；
③ 未登记 `neuware4.4.3` 这条**驱动同线的可用备选**；
④ 把「镜像来源」写成「只能走寒武纪官方渠道」，**与 §3.1 实测不符**。
⇒ 已在该文档顶部追加更正段（见 §6 文档清单）。

---

## 4. FlagOS 官方镜像体系（本次新发现，建议作为基座对齐的**权威口径**）

来源仓：`flagos-ai/build-infra`（公开；`configs.yaml` 是唯一 source of truth，
文档站 `https://flagos-ai.github.io/release-info/` 由其自动生成，避免漂移）。
registry 前缀定义（`.github/build-config.yml`）：

```yaml
registry:
  host: harbor.baai.ac.cn
  prefixes:
    base:    flagos-base      # 厂商 SDK + OS 底座
    runtime: flagos-runtime   # + torch / 厂商 torch 插件 / triton / flag_gems
    builder: flagos-dev       # 构建工具链镜像
    app:     flagos-app       # vLLM / SGLang / Megatron 应用镜像
```

当前发布版本 **2.2.0**（各镜像 push 时间 2026-09-17 ~ 2026-09-20）。三家对照（均取自 `configs.yaml` / `base|runtime/*.md`）：

| | 第 1 家 昇腾 910C | 第 2 家 昆仑芯 P800 | 第 3 家 寒武纪 MLU590 |
|---|---|---|---|
| backend key | `ascend-cann9.0.0-910c` | `kunlunxin-xre5.37.1` | `cambricon-neuware4.7.2` / `…4.4.3` |
| runtime 镜像 | `flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0` | `…/flagos-runtime-kunlunxin-xre5.37.1:2.2.0` | `…/flagos-runtime-cambricon-neuware4.7.2:2.2.0` |
| 宿主驱动前置 | `26.0.rc1` | `5.37.1` | `6.5.48`（4.4.3 为 6.2.15） |
| Python | 3.11 | 3.10 | 3.12（4.4.3 为 3.10） |
| torch / 插件 | 2.10.0+cpu / torch-npu 2.10.0 | 2.9.0+cu129 / `torch_plugin`+`torch_xray` | 2.11.0+cpu / torch-mlu 1.33.1 |
| flagtree | **0.7.0rc2+ascend3.5** | **0.7.0rc2+xpu3.6** | —（寒武纪无 flagtree 线） |
| triton | 3.5.0（备 triton_ascend 3.2.1） | 3.6.0+gitcd2d6c1b | 3.4.0+mlu2.1.1 |
| flag_gems | 5.4.0-rc2.post3 | 5.4.0-rc2.post3 | 5.4.0-rc2.post3 |
| 特有便利 | — | 镜像内含 `compiler` 命令可切 FlagTree/Triton | 无容器 toolkit，用 `--device /dev/cambricon_dev0 --device /dev/cambricon_ctl` |

**对我们三家的意义**：
- **910C**：官方线（CANN 9.0 / torch 2.10 / triton 3.5 / **ascend3.5**）与我们锁定栈**逐项一致**，
  只是来源不同（我们 = 组内 `flagrt` 镜像 + 华为 `vllm-ascend`）。⇒ 可作为训练腿「候选新血统」的
  **官方对应物**，一并报总组（我们已在 `stack.lock.910c.v2.yaml` 的 `candidates:` 段登记过 v2 候选）。
- **P800**：官方线在小版本上更靠后（`flagtree 0.7.0rc2` vs 我们的 `0.6.1`），但**底层 SDK 换代**
  （XRE 5.37.1）⇒ 见 §2.2。
- **寒武纪**：**只有官方线可用**（无 FlagTree 线）⇒ 见 §3。

---

## 5. 未验证 / 未取得（如实标注，不补零）

- ⚠️ **未在带卡机上真正 `docker pull`** 上述任何镜像。本次只做了 Registry v2 接口级验证
  （匿名 token + manifest 实取）——**这证明了鉴权路径可通过**，但**不等于**已确认带卡机可达
  `harbor.baai.ac.cn`（寒武纪两台机器的出网策略未测）。
- ⚠️ **驱动兼容性未实测**：① 6.2.29 能否跑 neuware4.4.3（官方标 6.2.15）；② P800 的 5.0.21.47
  能否跑 XRE 5.37.1。**均为推断，须实测定论**。
- ⚠️ 未查 `flagos-app` 里寒武纪应用镜像的内置 vLLM 版本与其厂商移植关系
  （`harbor.baai.ac.cn/flagos-app/vllm0.24.0-cambricon-neuware4.7.2` 现有 tag
  `2.2.0-0.3.0rc2.post2`，push 时间 2026-09-20）——即「推理腿是厂商移植版还是社区版 + 插件」这一
  分叉点**仍未落实**，直接影响接入人天。
- ⚠️ 类脑 P800 记 `Triton 3.5.0` 与我方记 `triton 3.6.0` 的差异**未查明原因**（§2.1）。
- ⚠️ 寒武纪**三个私仓**（`docker.cambricon.com` 等）与 BAAI Harbor 的关系未确认；
  本轮结论是「**不需要**它们」，而非「它们不可用」。

---

## 6. 本次已同步修改的文档

| 文档 | 改动 |
|---|---|
| `MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` | 顶部追加 **2026-09-22 更正段**：镜像来源由「寒武纪私仓/官方渠道」更正为「FlagOS 官方 BAAI Harbor 已有且可匿名拉取」；§1 结论表两行更正 |
| `prototype/docs/IMAGE_SELECTION_GUIDE_20260920.md` | §2 来源优先级**新增 `flagos-base / flagos-runtime / flagos-app` 三代体系**；§6 寒武纪镜像建议改为**按宿主驱动选档** |
| `../STATUS.md` | 「基座与约束」新增 2 条（寒武纪两档驱动门槛 / FlagOS 官方镜像体系）；「阻塞与协调」更正寒武纪镜像获取一项 |
| 本文件 | 新建 |

## 7. 附：一键复核命令

```bash
# ① FlagOS 官方镜像体系（唯一 source of truth）
curl -s https://raw.githubusercontent.com/flagos-ai/build-infra/main/configs.yaml | \
  grep -nA30 '^  cambricon:'

# ② Harbor 匿名可拉验证（无需凭据；把 <repo>:<tag> 换成上表任一）
R=flagos-runtime/flagos-runtime-cambricon-neuware4.7.2; T=2.2.0
TOK=$(curl -s "https://harbor.baai.ac.cn/service/token?service=harbor-registry&scope=repository:$R:pull" \
      | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
curl -sD- -o/dev/null -H "Authorization: Bearer $TOK" \
     -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
     "https://harbor.baai.ac.cn/v2/$R/manifests/$T" | grep -i docker-content-digest

# ③ 类脑那 4 条线（FlagTree 手册，确认无寒武纪）
curl -sL https://github.com/flagos-ai/FlagTree/wiki/_pages | grep -oiE 'User-manual-for-[a-z]+' | sort -u
```
