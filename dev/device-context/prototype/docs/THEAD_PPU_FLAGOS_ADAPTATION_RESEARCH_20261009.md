# FlagOS 对平头哥（T-Head PPU）的适配现状 · 调研报告

**日期**：2026-10-09 · **性质**：调研（**未上机、未接入**）· **面向**：第 4 家实例接入的可行性评估

> **证据口径**：本报告分两类来源，逐条标注 ——
> **【实测】** = 本方向于 2026-10-09 亲自拉取/查询所得（Harbor API 全量、官方仓库原始文件、pypi 源）；
> **【公开】** = 厂商/社区公开文档与报道（**未由本方向独立验证**）。
> 凡本方向**没有取到**的，一律写「未记录 / 未验证」，**不补零、不推测**。

---

## 0. 结论先行

| 问题 | 结论 |
|---|---|
| FlagOS 对平头哥适配到哪一步了？ | **组件层已全面覆盖，运行时层完全空白**。编译器 / 算子库 / 推理插件都有平头哥后端；**`flagos-runtime` 与 `flagos-base` 下没有任何平头哥镜像**。 |
| 我们的接入路径能直接复用吗？ | ❌ **不能直接复用**。三家的「`flagos-runtime-*` + `flagos-app/*`」双镜像路径在平头哥上**断了**（详见 §3）。 |
| 接入形态是哪一路？ | **路径 B（复用 `torch.cuda` 命名空间）**——与已接入的 **P800（昆仑芯）同构**，证据链完整（§4）。 |
| 属不属于本层职责？ | ✅ **正是本层的位置**：组件层已被 FlagOS 覆盖，而**设备上下文 + 多流 stream 这一层在平头哥上没有任何现成实现**可用。 |
| 最大风险是什么？ | **机器/镜像从哪来**（§5 第 1–3 条）。按既有经验，**接入动作只花一天，环境打通才是工期风险**。 |

---

## 1. FlagOS 各组件对平头哥的支持面

| 层 | 组件 | 支持情况 | 证据 |
|---|---|---|---|
| 统一编译器 | **FlagTree** | ✅ **完整** | 【实测】wiki `User-manual-for-ppu` 已 14 次修订，标题即「T-Head（平头哥）ppu **3.6**」；文中写明 *"Based on Triton 3.6, x64"*、*"Available for **810E, 890P**"*，并给出 `FLAGTREE_BACKEND=ppu` 的源码构建路径 |
| 算子库 | **FlagGems** | ✅ **有专门厂商后端目录** | 【实测】`src/flag_gems/runtime/backend/**_thead**/` 存在，内含 `fused/`、`ops/`、`tune_configs.yaml`、`mm_ppu_expand.yaml`、`general_ops_thead_expand.yaml`。⭐ 设备识别方式：`device_finder.py` 判 **环境变量 `PPU_SDK` 是否存在**（不是判设备名） |
| 集合通信库 | **FlagCX** | ⚠️ **未覆盖** | 【实测】README「Backend Support」表的 12 个后端里**没有平头哥**。⚠️ 见 §5 第 5 条：表里的 **PCCL 指向 Sunrise**（`sunrise-ai.com`），而平头哥自家的集合通信库**也叫 PCCL** —— **同名不同家** |
| 推理插件 | **vllm-plugin-FL** | ✅ **Supported** | 【实测】README「Supported Chips」表 T-Head = **Supported**；仓库内有 `docker/**thead**/Dockerfile`、`.github/configs/**thead**.yml`、`.github/scripts/thead/{setup.sh,check.sh}` |
| 推理插件 | **sglang-plugin-FL** | ⚠️ 有镜像、细节未查 | 【实测】Harbor `flagos-app` 下有 `sglang0.5.12-thead-ppu2.1.0` 仓（见 §3） |
| 训练框架 | Megatron-LM-FL | **未记录** | 本方向**未查**该仓库的厂商矩阵，不作结论 |
| 已废弃路线 | Torch-FL | ⚠️ 有 `ppu` 但状态 Experimental | 【公开】本项目 2026-09-22 留档的引用：torch_fl 平台矩阵实有 9 家，其中 **`ppu`（平头哥）状态为 Experimental**；且该路线**本方向已整体退出**，不作为可选项 |
| 模型发布 | **FlagRelease** | ✅ 多次 Day0 覆盖平头哥 | 【公开】2026-08-13 报道：Qwen3.8-2.4T-A95B 完成 9 家芯片适配，**列表首位即平头哥**；2026-04-25 报道：DeepSeek-V4-Flash 完成 8 款以上芯片适配，含「**平头哥真武**」 |

⭐ **一句话**：**平头哥在 FlagOS 里是"组件齐全、运行时缺位"** —— 这恰好是本方向（设备上下文 + 多流 stream）的空白位。

---

## 2. 官方构建配置（唯一 source of truth）里的平头哥段

【实测】拉取 `raw.githubusercontent.com/flagos-ai/build-infra/main/configs.yaml`（**version `2.3.0`**，50213 字节，HTTP 200），`vendors:` 下 `thead` 段**全文如下**：

```yaml
  # No base image yet. Has to run on PAI using Aliyun images
  thead:
    ppu2.0.0:
      extras: thead
      hardware: ["T-Head ZW810E"]
      python: "3.12"
      triton: triton==3.5.0+ppu2.0.0.oe
      deps:
        - torch==2.9.0+ppu2.0.0.oe
        - numpy==1.26.4
```

**逐字读出的三件事**：

1. ⛔ 注释原文 **`# No base image yet. Has to run on PAI using Aliyun images`** —— 官方**自己声明没有基座镜像**，要跑在阿里云 PAI 上。
2. 该段**只有 8 行**，且**缺 `driver` / `flagtree` / `sdk` / `deps_app` / `env` 五个字段**。
3. 硬件标识 = **`T-Head ZW810E`**（真武 810E）。

**成熟度横向对照**（【实测】同一文件逐厂商段统计）：

| 厂商段 | 段行数 | `driver` | `flagtree` | `sdk` | `deps_app` | `env` |
|---|---:|---|---|---|---|---|
| nvidia | 107 | ✓ | ✓ | ✓ | ✓ | ✓ |
| ascend | 153 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **kunlunxin（我们已接入）** | 55 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **cambricon（我们已接入）** | 89 | ✓ | ✗ | ✓ | ✓ | ✓ |
| **thead（本次调研）** | **8** | **✗** | **✗** | **✗** | **✗** | **✗** |
| spacemit | 7 | ✗ | ✗ | ✗ | ✗ | ✗ |

⇒ **13 家里只有 `thead` 与 `spacemit` 是"骨架级"**，其余 11 家字段齐全。**平头哥在官方构建配置里排在成熟度最低的一档。**

---

## 3. ⛔ 最关键发现：平头哥没有运行时层镜像

【实测】Harbor `harbor.baai.ac.cn` API **全量**查询（`/api/v2.0/projects/<proj>/repositories?page_size=100`，已翻页确认未截断）：

| Harbor 项目 | 仓总数 | 平头哥相关 | 说明 |
|---|---:|---|---|
| **`flagos-runtime`** | 20 | **无（0 个）** | ⛔ 该项目下三个已接入实例的镜像**都在**：`flagos-runtime-ascend-cann9.0.0`、`flagos-runtime-kunlunxin-xre5.37.1`、`flagos-runtime-cambricon-neuware4.4.3` —— **唯独没有 thead** |
| **`flagos-base`** | 22 | **无** | ⛔ 基座层也没有 |
| `flagos-dev` | 32 | **无** | 只查到 CI 用的 `flagos-dev/vllm-plugin-fl:v0.24.0-thead-ci` 这一**个引用**（在 `configs/thead.yml` 里），仓列表中未列出 |
| `flagos-app` | **127**（100+27） | ⚠️ **仅 1 个** | `sglang0.5.12-thead-ppu2.1.0`，tag `2.2.0-0.2.0`，**22.72 GiB**，push 时间 **2026-09-24** |
| `flagtree` | 40 | ✅ 有完整镜像 | 见下 |

**`flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04` 的可用 tag**（【实测】artifacts API）：

| tag | 大小 | push 时间 |
|---|---:|---|
| `202607-3.6-vllm0.24.0` | 17.52 GiB | 2026-08-24 |
| `202608-3.6-base` | 17.50 GiB | 2026-08-24 |
| `202608-clean` | 17.14 GiB | 2026-08-24 |
| `202607-3.6-base` | 16.58 GiB | 2026-07-29 |
| `202607-clean` | 16.21 GiB | 2026-07-29 |

### 3.1 三条可用镜像线（以及它们的代价）

| 线 | 镜像 | 来源 | 代价 |
|---|---|---|---|
| **A · FlagTree 线**（**唯一在 Harbor 上的**） | `harbor.baai.ac.cn/flagtree/flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-vllm0.24.0` | 【实测】Harbor | 17.52 GiB；档位 **ppu3.6 / torch 2.10 / SDK 2.1.0**，**与 configs.yaml 的 ppu2.0.0 档不一致** |
| **B · 官方 CI 实际用的基座** | `egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/inference-xpu-pytorch:26.04-v2.1.0-vllm0.23.0-torch2.10-cu130-20260710` | 【实测】`vllm-plugin-FL/docker/thead/Dockerfile` 的 `ARG THEAD_BASE_IMAGE` 默认值 | ⚠️ **在阿里云 ACR，不是 Harbor** ⇒ 拉取凭据/网络**未验证** |
| **C · 阿里云 PAI 官方镜像** | `pai-pg1-training-1.5.2-ubuntu` 等 | 【公开】阿里云 PAI 镜像 Release Note | ⛔ 官方原文：*"PAI-PPU 训练镜像**仅支持在 PAI 平台内**使用；推理镜像仅支持在 PAI-EAS 中使用，**不支持其他环境**"* ⇒ **拿不出 PAI 就用不了** |

### 3.2 pypi 侧：平头哥有专用源，且确实分发定制 torch

【实测】`https://resource.flagos.net/repository/flagos-pypi-thead/simple/`（HTTP 200，753 字节）**顶层全文**：

```
filelock  flag-gems  flagtree  fsspec  jinja2  markupsafe  mpmath
networkx  numpy  packaging  sympy  torch  torch-fl  triton  typing-extensions
```

`.../simple/torch/` 页**全文列出的两个版本**：

| 包 | SHA256 |
|---|---|
| `torch-2.9.0+ppu2.0.0.oe.tar.gz` | `3b179378cdbdb6b04e5877df4942e0790d1ffb1a3e26b7adfa7942e61d3d014f` |
| `torch-2.10.0+v0.1.0.ppu2.1.1.tar.gz` | `e62e0e0ee6fc87bf2de07f693571a1f061e2d269c305499deb6d7dd8eb1fe815` |

⇒ **两件事**：① 平头哥的 torch 是**厂商定制构建**（本地版本号 `+ppu2.0.0.oe` / `+v0.1.0.ppu2.1.1`），**不是**独立命名的插件包（不像 `torch_npu` / `torch_mlu`）；② 该源**同时提供 `torch-fl`**，说明平头哥也在 torch-fl 路线上有投入。

### 3.3 ⚠️ 档位分叉（三套栈，版本不一致）

| 来源 | SDK / 档位 | triton | torch | 备注 |
|---|---|---|---|---|
| `build-infra/configs.yaml`【实测】 | ppu **2.0.0** | `3.5.0+ppu2.0.0.oe` | `2.9.0+ppu2.0.0.oe` | 无基座镜像 |
| FlagTree wiki 手册【实测】 | ppu **3.6**（镜像名含 sdk **2.1.0**） | 3.6 | 2.10.0 | 有镜像 |
| `flagos-pypi-thead` 源【实测】 | **2.0.0 与 2.1.1 并存** | — | 2.9.0 / 2.10.0 | — |
| 阿里云 PAI 官方【公开】 | **1.5.2** | — | 2.4.0 / 2.6.0 | 仅 PAI 内可用 |

⇒ **这与 P800 当年的形态同构**（FlagTree 线 vs FlagOS 官方线不一致，见本项目 `IMAGE_LINEAGE_ALIGNMENT_20260922.md`）。
⛔ **纪律不变：选档第一判据是宿主驱动版本，不是 Python 包版本** —— 而平头哥**恰恰是唯一连 `driver` 字段都没有的段**（§2）。
⇒ **上机第一件事必须是读宿主驱动版本**，在此之前**任何档位结论都不成立**。

---

## 4. 接入形态判定：路径 B（复用 `torch.cuda` 命名空间）

这一步决定我们后端 `device_type` 怎么填，是本次调研**证据最硬**的部分。五条独立证据：

| # | 证据 | 原文 / 读数 |
|---|---|---|
| 1 | 厂商 UMD/运行时**兼容 CUDA API** | 【公开】阿里云 T-Head SAIL SDK v2.1 文档：*"兼容绝大多数 cuda runtime api (**cudaXXX**) 和 cuda driver api (**cuXXX**)"* |
| 2 | 官方 CI **用 `torch.cuda` 判定设备** | 【实测】`vllm-plugin-FL/.github/scripts/thead/setup.sh` 末尾：`print(f"Accelerator available: {torch.cuda.is_available()}")`、`print(f"Accelerator count: {torch.cuda.device_count()}")` |
| 3 | vllm-plugin-FL 把 thead 归入 **cuda 家族** | 【实测】`vllm_fl/platform.py`：`use_custom_op_collectives()` 返回 `cls.vendor_name in ("nvidia", "thead", "iluvatar")`；`get_device_uuid()` 的 `cls.device_type == "cuda"` 分支对平头哥**走 `pynvml`（NVML）** |
| 4 | `nvidia-smi` 被**替换成 `ppu-smi`** | 【实测】`.../thead/check.sh` 注释原文：*"**nvidia-smi is a symlink to ppu-smi** on T-Head PPU SDK"*，脚本直接调 `nvidia-smi` 即完成 PPU 可用性检查 |
| 5 | 镜像名自带 **`cu130`**、官方用 `CUDA_VISIBLE_DEVICES` | 【实测】FlagTree wiki 手册的 vLLM benchmark 段：`export CUDA_VISIBLE_DEVICES=0,1  # 810E * 2`；三个镜像名均含 `cu130` |

### 4.1 与已接入三家的对照

| 实例 | `device_type` | torch 包 | 集合通信（厂商侧） | 选卡变量 | 设备工具 |
|---|---|---|---|---|---|
| 910C（第 1 家） | `npu` | `torch_npu` | HCCL | `ASCEND_RT_VISIBLE_DEVICES` | `npu-smi` |
| **P800（第 2 家）** | **`cuda`** | **XPytorch** | XCCL（走 FlagCX） | `CUDA_VISIBLE_DEVICES` | `xpu-smi`【本项目已有实测】 |
| MLU590（第 3 家） | `mlu` | `torch_mlu` | CNCL | `MLU_VISIBLE_DEVICES` | `cnmon` |
| **平头哥 PPU（候选第 4 家）** | **`cuda`**（本报告判定） | torch 定制构建（`+ppu2.0.0.oe`） | **PCCL**【公开】 | **`CUDA_VISIBLE_DEVICES`**（官方 CI 用法） | **`ppu-smi`**（=`nvidia-smi` 符号链接） |

⚠️ **两条必须记住的推论**：
1. 平头哥与 P800 **同属 `cuda` 命名空间** ⇒ **两者不能在同一进程内混用**（与我们后端目录里对 `cuda` 家族的既有约束一致）。
2. 厂商标识**不能靠命名空间**（都叫 `cuda`），**只能靠设备名 / `name` 字段** —— 这正是 skill 里"路径 B：厂商标识靠 `name`"那条的适用场景。

### 4.2 设备与容器形态（【实测】官方 CI 配置）

```
platform: thead
ci_image: harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.24.0-thead-ci
runner_labels:
  -  flagcicd-810e                      # ← 真武 810E
container_options: >-
  --device /dev/alixpu
  --device /dev/alixpu_ctl
  --device /dev/alixpu_ppu0 … /dev/alixpu_ppu15      # ← 0..15，共 16 张
```

- 设备节点前缀 = **`alixpu`**；单机 **16 卡**（与 910C 同量级，多于 P800 / MLU590 的 8 卡）。
- `container_volumes: /mnt/airs-business/cicd/models:/data/models` ⇒ 该 CI 机的模型挂载点。
- FlagGems 侧厂商标识 = 环境变量 **`GEMS_VENDOR=thead`**（【实测】`docker/thead/Dockerfile`）。

---

## 5. 风险与前置条件（按优先级）

| # | 事项 | 现状 | 影响 |
|---|---|---|---|
| **1** | ⛔ **机器从哪来**（**最大的未知**） | **未记录** —— 本方向**没有**平头哥机器的任何信息（IP / 权限 / 是否共享） | 决定一切。按既有经验：**接入动作一天，环境打通才是工期风险** |
| **2** | ⛔ **运行时层镜像不存在** | 【实测】`flagos-runtime` / `flagos-base` 均无 thead | 三家的"双镜像路径"**断了**：要么走 FlagTree 线镜像（档位不同），要么自建 |
| **3** | ⚠️ **无 vLLM 应用镜像** | 【实测】`flagos-app` 只有 **sglang** 的（`sglang0.5.12-thead-ppu2.1.0`），**无 `vllm*-thead-*`** | 推理腿服务化两条路：换 sglang，或自建 vLLM 应用镜像。⚠️ 注意本项目**服务化统一入口**是 `serve_standard.sh`（vLLM 路径） |
| **4** | ⚠️ **镜像在阿里云 ACR** | 【实测】`egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/...` | 拉取凭据 / 网络**未验证**；且 **Harbor 匿名可拉**这条既有便利**不适用** |
| **5** | ⚠️ **集合通信库缩写撞名** | 【实测】FlagCX 表里的 **PCCL 指向 Sunrise**（`sunrise-ai.com`）；而平头哥自家库**也叫 PCCL**（PPU Collective Communications Library）【公开】 | 引用时必须写全称。⚠️ 同族教训：**不要按名字认库**（本项目已有"不要按名字认容器"的同型坑） |
| **6** | ⚠️ **FlagCX 未覆盖平头哥** | 【实测】README 表格 12 个后端无平头哥 | **跨厂商通信**这条线在平头哥上是否可用 ⇒ 需实测，**不得外推** |
| **7** | ⚠️ **三套档位不一致**（§3.3） | 2.0.0 / 2.1.0 / 1.5.2 并存 | 选档必须**先读宿主驱动**；平头哥是唯一连 `driver` 字段都没有的段 |
| **8** | ⚠️ **集合通信后端名未知** | 【公开】平头哥 PCCL *"兼容支持绝大多数 nccl api 和环境变量"* | 训练腿的 `DC_DIST_BT` **必须实测探测**；⛔ 拿不到就**报错退出**，**绝不兜底 `gloo`**（会静默退化为纯 CPU，训练照样跑完 ⇒ 比失败更糟） |
| **9** | ⚠️ **选卡变量未实测** | 官方 CI 用 `CUDA_VISIBLE_DEVICES`【实测】 | 与 P800 相同写法，但仍须**实测确认**（`device_count()` 是否随之变化），**不照抄** |

---

## 6. 对第 4 家接入的初步判断

**好消息（组件层成熟）**：
- FlagTree / FlagGems / vllm-plugin-FL 三件都有平头哥后端，**算子与编译这条线不必我们从零做**；
- torch 有官方专用 pypi 源，**依赖获取路径清晰**（`flagos-pypi-thead`）；
- 官方 CI 已跑通（`flagcicd-810e` + 16 卡容器），**容器形态有现成参照**；
- 形态判定为**路径 B**，而路径 B **本方向已有一家完整实现（P800）** ⇒ `kunlun` 后端是**最贴近的写法参照**（⚠️ **只参照写法，不照抄实现与结论** —— 铁律）。

**坏消息（运行时层空白 + 环境未定）**：
- **运行时层镜像不存在** ⇒ 我们三家用的"标准双镜像路径"在平头哥上**不成立**，需要先解决镜像来源；
- 服务化（vLLM 应用镜像）同样缺失；
- 机器从哪来仍是空白。

**建议的推进顺序**（**待确认后再动手，本轮不做实现**）：
1. **先确认机器**（IP / 权限 / 是否共享 / 卡数）—— 这是唯一的前置阻塞项；
2. 上机**第一件事读宿主驱动版本**（决定档位，§3.3），第二件读 `ppu-smi` 输出与 `torch.cuda.device_count()`；
3. 用 `prototype/scripts/preflight_env.sh` 快照环境事实（**7 项**，见接入手册）；
4. 决定镜像线（FlagTree 线 / 自建 / 阿里云 ACR），并把结论写进选档文档；
5. 之后才进入常规接入流程（新建 backend → 离线自检 → conformance 13+6 → 多流 16 → 职责审计 → 两条腿 → 服务化 → 错误闭环）。

---

## 7. 待确认项（需要你来定 / 提供）

| # | 待确认 | 为什么必须先定 |
|---|---|---|
| 1 | **有没有平头哥的机器？**（IP、凭据、是否共享机、几张卡） | 唯一的前置阻塞项；无机器则本报告止于调研 |
| 2 | 机器是**阿里云 PAI 环境**还是**自有/裸机**？ | 若是 PAI，官方镜像仅限 PAI 内用（§3.1 线 C）；自建则需解决 SDK 与驱动 |
| 3 | 走**哪条镜像线**（FlagTree 线 / 阿里云 ACR / 自建） | 决定档位与后续所有结论的可比性 |
| 4 | 芯片型号是 **810E** 还是 **890P / M890**？ | configs.yaml 只登记了 `ZW810E`；FlagTree 手册写 *"Available for 810E, 890P"* |
| 5 | 服务化这条腿**用 vLLM 还是 sglang**？ | 平头哥只有 sglang 应用镜像；与本项目统一入口不一致时需要定口径 |

---

## 附：本次调研的取证命令（可复核）

```bash
# ① 官方构建配置（唯一 source of truth）
curl -sL https://raw.githubusercontent.com/flagos-ai/build-infra/main/configs.yaml -o /tmp/bi.yaml
grep -n -A8 "  thead:" /tmp/bi.yaml          # → 8 行骨架 + "No base image yet"

# ② Harbor 镜像全量（逐项目，注意翻页）
for p in flagos-runtime flagos-base flagos-app flagos-dev flagtree; do
  curl -s "https://harbor.baai.ac.cn/api/v2.0/projects/$p/repositories?page_size=100&page=1"
done

# ③ 平头哥专用 pypi 源
curl -s https://resource.flagos.net/repository/flagos-pypi-thead/simple/
curl -s https://resource.flagos.net/repository/flagos-pypi-thead/simple/torch/

# ④ 形态判定（路径 B）的三处原始文件
curl -sL https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/.github/scripts/thead/setup.sh
curl -sL https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/.github/scripts/thead/check.sh
curl -sL https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/vllm_fl/platform.py

# ⑤ 设备与容器形态
curl -sL https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/.github/configs/thead.yml
curl -sL https://raw.githubusercontent.com/flagos-ai/vllm-plugin-FL/main/docker/thead/Dockerfile

# ⑥ FlagGems 的厂商标识方式
curl -sL https://raw.githubusercontent.com/flagos-ai/FlagGems/master/src/flag_gems/runtime/backend/device_finder.py

# ⑦ FlagTree 手册（网页）
# https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-ppu
```
