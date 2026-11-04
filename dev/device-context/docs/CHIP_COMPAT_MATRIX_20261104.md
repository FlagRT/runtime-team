# 国产芯片兼容性对照表（设备执行上下文层 · v0.1 初表）

> 起草：2026-11-04 ｜ 责任人：Kistich（hliu553）｜ 状态：**初表（未上机复核）**
> 范围：**设备执行上下文层（device-context / 多流 Stream）** 在四款国产芯片上的软件栈兼容性。
> 口径：**只登记本方向实际跑过的组合**；没跑过的一律标 🟡/⚪，**不按厂商宣传或镜像名推断**。

## 0 怎么读这张表

四态判据见 §1。先看这三句：

| 你要问的 | 答案 |
|---|---|
| **哪些是确定的（可直接对外声明）** | **8 格 ✅**：910C 开发包；P800 开发包/框架；MLU590 驱动/开发包/框架；PPU 驱动/开发包。每一格都有仓内可复现的验收证据 |
| **哪些要上机核验** | **10 格 🟡 + 1 格 ⚪**。最集中的缺口是**三个「宿主 OS / 驱动未采集」**（910C 的 OS 与驱动、PPU 的 OS、P800 新机的 OS 与驱动），以及 **MLU590 的算子库从未验过** |
| **哪些是不支持的** | **§5 共 9 条**。其中真正「芯片级不支持」的只有 2 条：MLU590 的 **FP8/MX 量化**（设备能力 `(5,0)` 不满足）和 **FlagTree 构建物无寒武纪线** |

⚠️ **本表不声明的事**：算子库的**逐算子数值正确性**、性能、以及「另三家做不到」这类横向推断。这些属算子/数值域，超出本方向职责（见 §7）。

## 1 状态定义（四态）

| 标记 | 含义 | 判据（三者同时满足才算 ✅） |
|---|---|---|
| ✅ **已验证** | 该版本组合下，本方向跑过**完整验收**且 PASS | ① 有单入口全量或多流/两腿判据 PASS；② 证据文件在仓内、可复现；③ 版本号来自**实测读取**（非镜像名推断） |
| 🟡 **部分验证** | 能用，但**证据不全**或**带已知限制** | 以下任一：只跑了部分判据 / 版本号来自**历史单次采集**且未复核 / 存在已登记缺陷限定使用面 |
| ⚪ **待核验** | **从未验证**，且当前无任何实测依据 | 只有镜像声明或上游文档，本方向零实测 |
| ⛔ **不支持** | 实测不可用 / 厂商官方明确无此线 / 本层明确不承诺 | 有明确失败记录或厂商矩阵排除依据 |

> **⚠️ 未采集 ≠ 不支持。** 本表用 ⚪ 区分「没测过」与「测过不行」，因为把前者写成后者会让别人白白返工。
> 初表里 🟡 与 ⚪ 都属「**上机要核验的**」，区别是 🟡 已有部分证据、⚪ 一片空白。

## 2 主表：四芯片 × 五层

> 每格 = **版本 ｜ 状态**。状态是对「该版本在本层可用」的判定，不是对芯片厂商的判定。
> 证据指针在 §3。**路径基准**：`910C/` `P800/` `MLU590/` `PPU/` 开头的相对 `dev/device-context/`；`dev/images/…` 开头的相对**仓库根**。

| 层 | **昇腾 910C** | **昆仑芯 P800** | **寒武纪 MLU590** | **平头哥 PPU（真武 ZW810E）** |
|---|---|---|---|---|
| **操作系统** | 容器基座 **Ubuntu 22.04**（aarch64）<br>宿主 OS/内核 **未采集**<br>🟡 部分验证 | 宿主 **Ubuntu 24.04.2 LTS** / kernel `6.8.0-87`<br>容器基座 **Ubuntu 22.04**<br>🟡 部分验证（**新机未采集**） | 宿主 **Ubuntu 20.04.6 LTS** / kernel `5.15.0-139`<br>容器 **Ubuntu 22.04**（由包名推断）<br>🟡 部分验证 | 容器基座 **Ubuntu 24.04**（x86_64）<br>宿主 OS/内核 **未采集**<br>🟡 部分验证 |
| **驱动** | `25.5.0`（ascendhal `7.35.23`）<br>⚠️ **2026-09-08 单次采集，未复核**<br>🟡 部分验证 | 宿主 `5.0.21.47`（XPU-RT `5.0.21`）<br>⚠️ **老机实测；新机未采集**<br>🟡 部分验证 | **`v6.2.29`** / 固件 `v1.5.0` / CNMON `v6.2.29`<br>✅ 已验证 | KMD **`1.5.1-1d747a`** / HGGC `13.0`<br>✅ 已验证 |
| **开发包（SDK）** | **CANN `9.0.0`**<br>✅ 已验证 | **XTDK `3.6.0.1`** · XRE `5.13.0.0` · XCCL `20260209_9` · xmlir `1.0.0.1`<br>✅ 已验证 | **NeuWare `4.4.3`**<br>cntoolkit `4.4.3-1` · cnnl `2.1.829` · cnnlextra `2.2.7` · mluops `1.8.1` · cncl `1.29.4` · cnclep `1.1.1`<br>✅ 已验证 | **PPU SDK `2.1.0-a5f865`**（CUDA wrapper `cu130`）· PCCL `2.1.0`<br>✅ 已验证 |
| **框架（PyTorch）** | torch **`2.10.0+cpu`** · torch_npu **`2.10.0`** · py `3.11.15`<br>⚠️ `2.10.0` 命中退出期 weakref UAF（`post7` 已修，镜像未升）<br>🟡 部分验证 | torch **`2.9.0+cu129`** · py `3.10.18`<br>设备 API 走 **`torch.cuda`**（`torch.xpu` 不可用，`USE_XPU=OFF`）<br>✅ 已验证 | torch **`2.7.1+cpu`** · torch_mlu **`1.29.2+torch2.7.1`** · py `3.10`<br>退出期 weakref **未中招**<br>✅ 已验证 | torch **`2.10.0`** · py `3.12.3` · transformers `4.57.0`<br>⚠️ 退出期 weakref UAF 为**平台缺陷**，我方运行时已自保<br>🟡 部分验证 |
| **算子库** | FlagGems `f7ae8e6b`（5.4.0-rc0 线）<br>triton **`3.5.1`**（FlagTree `triton_v3.5.x` @`15ec1a6c`）<br>🟡 部分验证 | FlagGems `73c5aff1`（挂载覆盖，含 GEMM compat）<br>flagtree **`0.6.1+xpu3.6`**<br>⚠️ GEMM **数值缺口未解**（`K-5`，禁用于关键路径）<br>🟡 部分验证 | flag_gems **`5.4.0rc2.post3`**<br>triton **`3.2.0+mlu1.7.2`**（厂商 triton-mlu 变体）<br>torch_mlu_ops `1.8.0`<br>⚪ **本组未做结论性验证** | FlagGems **`5.4.0rc2`**<br>triton **`3.6.0`**（v2 起由 flagtree `0.7.0+ppu` whl 固化）<br>⚠️ v1 无 triton ⇒ 服务化必挂，v2 已修<br>🟡 部分验证 |

### 2.1 计数

| 状态 | 格数 | 分布 |
|---|---|---|
| ✅ 已验证 | **8** | 910C 开发包｜P800 开发包·框架｜MLU590 驱动·开发包·框架｜PPU 驱动·开发包 |
| 🟡 部分验证 | **11** | 三家 OS 全 🟡；910C 驱动·框架·算子库；P800 驱动·算子库；MLU590 OS；PPU 框架·算子库 |
| ⚪ 待核验 | **1** | MLU590 算子库 |
| ⛔ 不支持 | **0**（格级） | 能力级不支持见 §5 |

## 3 逐格证据指针

> 原则：**每格都要能落到一个文件**。落不到的，就是 🟡/⚪ 而不是 ✅。

### 3.1 昇腾 910C

| 层 | 版本事实来源（仓内路径） | 验收证据 |
|---|---|---|
| OS / 开发包 | `dev/images/ascend-operator-runtime/v3/image.baseline.yaml` **L13–L15**（基座 ref 含 `cann9.0.0-ubuntu22.04-aarch64`；`provides: {cann: 9.0.0, python: 3.11.15, torch: 2.10.0+cpu, torch_npu: 2.10.0}`） | `910C/docs/ASCEND_910C_RERUN_20261027.md`（离线 92/0/2 · 审计 73/0/5 · 推理 14/14 · 训练 50 步 6/6） |
| 驱动 | `910C/docs/DIAG_TRAIN_IMAGE_NPU_20260908.md` **L64**（`25.5.0（ascendhal 7.35.23）`） | ⚠️ **仅此一处**，且 `910C/docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md` **L149** 明确写「现网驱动版本未采集」⇒ 两者矛盾未解，必须复采 |
| 框架 | 同 image.baseline **L15**；weakref 结论见 `910C/docs/ASCEND_WEAKREF_EXIT_UAF_20261103.md` | 同上；weakref 判据见 `910C/probes/weakref_check_20261103_out/MANIFEST.md` |
| 算子库 | 同 image.baseline **L30–L41**（FlagTree `15ec1a6c` / FlagGems `f7ae8e6b`） | 随全量验收间接跑通（无逐算子对照）；构建期需 `FLAGGEMS_KERNEL=OFF` 补丁，见 `910C/docs/PROGRESS_20260822.md` **L31** |

### 3.2 昆仑芯 P800

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS / 驱动 | `P800/docs/KUNLUN_P800_ENV_REPORT_20260914.md` **L35**（`Ubuntu 24.04.2 LTS`）、**L68**（驱动 `5.0.21.47`）；`P800/docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md` **L79**（kernel `6.8.0-87-generic`） | ⚠️ 均为**老机**（`43.180.254.67`，**11-07 到期**）。新机 `81.70.153.29` 未采集，见 `P800/docs/KUNLUN_P800_MIGRATION_20261103.md` |
| 开发包 / 框架 | `dev/images/kunlun-operator-runtime/v2/image.baseline.yaml` **L14–L16**（基座 `ubuntu22.04`；`provides: {xtdk: 3.6.0.1, xre: 5.13.0.0, xccl: 20260209_9, python: 3.10.18, torch: 2.9.0+cu129, xmlir, xtorch_ops}`） | `P800/docs/KUNLUN_P800_MIGRATION_20261103.md` **§5.1b**（新机归位后：离线 43/0/1 · 推理 13/13 · 区分度 0.6392） |
| 框架（命名空间） | `P800/README.md` **L74**（`torch.xpu.is_available()=False`、`torch.cuda.device_count()=8`、`USE_XPU=OFF`） | 同上 |
| 算子库 | `dev/stack.lock.kunlunxin.v1.yaml` **known_issues K-1/K-5**；`dev/images/kunlun-operator-runtime/v2/image.baseline.yaml` FlagGems `73c5aff1` | GEMM 编译维度已在 `env/v1` 修复（minirepro 12/12、pytest mm 18/18、扩展 99/99，2026-10-28）；**数值缺口 K-5 未解** |

### 3.3 寒武纪 MLU590

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS / 驱动 | `MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md` **L33**（`Ubuntu 20.04.6 LTS / 5.15.0-139-generic`）、**L38**（驱动 `v6.2.29` / 固件 `v1.5.0`）、**L41**（CNMON `v6.2.29`） | `MLU590/docs/CAMBRICON_MLU_RERUN_20261103.md`（单入口 9 段 `overall: PASS`：离线 103/0/0 · smoke 46/0 · 审计 63/0/15） |
| 开发包 | `dev/images/cambricon-mlu-runtime/v1/image.baseline.yaml` **L72–L82**（neuware `4.4.3` + 六个包级版本，均为 `.ubuntu22.04`） | 同上 |
| 框架 | 同上 **L25–L43**（torch `2.7.1+cpu` / torch_mlu `1.29.2+torch2.7.1` / torch_mlu_ops `1.8.0`） | 同上；weakref 未中招见 `MLU590/probes/weakref_check_20261103_out/MANIFEST.md` |
| 算子库 | 同上 **L45–L58**（triton 在 `/opt/triton`、flag_gems `5.4.0rc2.post3`，文件原话：**「本组尚未在本芯片上做结论性验证」**） | **无** ⇒ ⚪ |

### 3.4 平头哥 PPU

| 层 | 版本事实来源 | 验收证据 |
|---|---|---|
| OS / 开发包 | `dev/images/thead-ppu-runtime/v2/image.baseline.yaml` **L18–L21**（基座 `…-ubuntu24.04:202607-3.6-base`；`provides: {python: 3.12.3, torch: 2.10.0, vllm: 0.19.0+cu130, transformers: 4.57.0, ppu_sdk: v2.1.0}`） | `PPU/docs/PPU_V2_ACCEPTANCE_20261103.md`（13 段 `overall: PASS`：训练 loss 11.1530 · 推理 13/13 · 区分度 0.6391 · 审计 73/0/5） |
| 驱动 | `PPU/docs/PPU_BACKEND_ONBOARDING_20261010.md` **L33**（`KMD 1.5.1-1d747a` · SDK `2.1.0-a5f865` · HGGC `13.0` · PCCL `2.1.0`） | 同上 |
| 框架 | image.baseline **L40–L55** | 同上；weakref 平台缺陷见 `PPU/docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md` |
| 算子库 | image.baseline **L25–L37**（flagtree `0.7.0+ppu.gite223f42c` whl 提供 triton 3.6.0） | ⚠️ **v1 曾无 triton**（`import triton` 报 `ModuleNotFoundError`，见 `PPU/README.md` **L105**）⇒ v2 加派生层修复，服务化才通 |

## 4 补充：推理服务栈（vLLM 线）单列

> 单独列出的理由：四家差异最大、踩坑最多，且**它与「框架」不是同一件事**（同一 torch 版本可配不同 vLLM 插件策略）。

| 项 | 910C | P800 | MLU590 | PPU |
|---|---|---|---|---|
| vLLM | **0.20.2** | `0.13.0` | **0.20.2** | `0.19.0+cu130` |
| 插件 | `vllm-plugin-FL` @`8b05912`（+1 补丁） | `vllm-plugin-FL`（镜像内置） | `vllm-plugin-FL **0.2.2**` | 无（镜像自带） |
| **必设环境变量** | ⚠️ 必须 **`unset VLLM_PLUGINS`** | ⚠️ 必须 **`VLLM_FL_PREFER=vendor`** + `PYTHONPATH=/env/FlagGems/src` | ⚠️ 必须 **启用**插件（勿套 910C 的 unset） | 两者都**不需要** |
| 服务化状态 | ✅ `SERVE_STANDARD_PASS` | ✅ `SERVE_LEG_PASS 10/10` | ✅ `SERVE_STANDARD_PASS`（ready=1 smoke=1） | ✅ `SERVE_LEG_PASS 10/10` |

⭐ **这四行是反的**：910C 要求 **关闭**插件、MLU590 要求 **开启**插件。**「按芯片相反」这条已写成实测结论，别照抄隔壁家。**

## 5 不支持 / 已知限制清单（能力级）

> 这张表才是真正的「不支持」。**分三类**，别混写：① 芯片级不支持（厂商能力边界）② 我方不承诺（职责边界）③ 已知缺陷但可规避。

### 5.1 芯片级不支持（厂商能力边界，不可通过我方规避）

| # | 芯片 | 项 | 依据 | 影响 |
|---|---|---|---|---|
| N-1 | MLU590 | **FP8 / MX 量化不可用** | `torch.mlu.get_device_capability()` 返回 **`(5, 0)`**；厂商 vLLM 手册标 FP8 需 ≥`(6,0)`、MX 需 ≥`(6,1)` | 量化路径只能用 SmoothQuant 一类；**别按 910C/PPU 的量化方案平移** |
| N-2 | MLU590 | **无 FlagTree 构建物线** | BAAI·FlagTree 官方手册**无寒武纪条目**；镜像内 triton 是厂商 `triton-mlu` 变体 | 不能用 FlagTree 统一编译层；版本锚只能靠 Harbor digest |

### 5.2 我方不承诺（职责边界，不是芯片缺陷）

| # | 项 | 说明 |
|---|---|---|
| N-3 | **算子逐算子数值正确性** | 四款芯片均**未做**逐算子对照。本方向只保证「上下文层/多流」的行为，算子正确性归算子/数值域 |
| N-4 | **性能 / 吞吐横向可比** | 四家基准模型相同（`Qwen3-Embedding-0.6B`），但**卡型、卡数、单请求延迟量级差异极大**（MLU590 单请求 p50 约 645 ms、P800 约 57 ms）⇒ **吞吐数字不可横比** |
| N-5 | **「另三家做不到」类推断** | 禁止从两家结论外推第三家。每家结论只对本机、本镜像、本卡成立 |

### 5.3 已知缺陷（可规避，已登记）

| # | 芯片 | 缺陷 | 规避 | 状态 |
|---|---|---|---|---|
| N-6 | 910C | `torch_npu 2.10.0` 退出期 weakref UAF（`rc=139`） | 升 `2.10.0.post7`（Ascend PR #41596）；或我方运行时侧已不依赖弱引用语义 | **锁定镜像未升版** ⇒ 复现代码仍会命中 |
| N-7 | PPU | `torch 2.10.0` 同类 weakref UAF（平台侧） | 运行时等价补丁（`PPU/probes/weakref_check_20261103_out/wrfix.c`）—— **止血，不进镜像/不进交付包** | 已对外提交厂商，**未修** |
| N-8 | P800 | FlagGems GEMM **数值缺口**：64³ fp32 非确定性乱值（同输入 rel 0.29~1.02） | 禁用于关键路径；可用面限 pytest 18/99 矩阵 + minirepro 12 组 | **未解**，`K-5` |
| N-9 | P800 | 流优先级**退化单点**：`range=(0,0)`，「能设置 ≠ 有效果」 | 不做基于优先级的调度决策 | 已登记 `KUNLUN-STREAM-PRIORITY-SINGLE-LEVEL` |

## 6 上机核验清单（下一步逐芯片执行）

> 目标：把 **11 格 🟡 + 1 格 ⚪** 收敛成 ✅ 或确认的限制。
> ⚠️ 纪律：**先采事实再下判定**；采集失败写「未采集」，不写「不支持」。

### 6.1 三级采集（每家都跑同一套）

| 级别 | 采集内容 | 命令（示例，按芯片替换） | 期望/判据 |
|---|---|---|---|
| **L0 宿主** | OS / 内核 / 驱动 / 固件 | `cat /etc/os-release \| head -3`<br>`uname -r`<br>910C: `cat /usr/local/Ascend/driver/version.info`<br>P800: `xpu-smi \| head -20`<br>MLU590: `cnmon \| head -20`<br>PPU: `ppu-smi \| head -20` | 与表中版本逐字对齐；**不一致即标「版本已漂移」并更新表** |
| **L1 容器** | 基础镜像 OS / py / torch / SDK | `grep PRETTY_NAME /etc/os-release`<br>`python3 -c "import torch;print(torch.__version__)"`<br>`pip list \| grep -Ei "torch\|triton\|flag"` | 镜像声明的版本必须与容器内 `pip`/`import` 实测一致 |
| **L2 行为** | 设备可见性 + 最小算子 + 本层判据 | `torch.cuda.device_count()`（P800/PPU）/ `torch.npu.device_count()`（910C）/ `torch.mlu.device_count()`（MLU）<br>最小 `zeros + matmul + sync`<br>`runtime/smoke_runtime.py --backend <x>` | 见 §6.2 |

### 6.2 分芯片的核验重点（按现有缺口排序）

| 优先 | 芯片 | 要核的具体项 | 为什么它能改变结论 |
|---|---|---|---|
| **P0** | 910C | ① 宿主 OS + 内核 ② **驱动真实版本**（`25.5.0` 与 `OFFICIAL_RUNTIME_COUNTERPART` 的「未采集」矛盾） | 驱动是芯片选型第一判据；现在这格是**两个互相矛盾的记录**，必须定一个 |
| **P0** | P800 | 新机 `81.70.153.29` 的 **宿主 OS / 内核 / 驱动**（现有值全来自 11-07 到期的老机） | 老机一关，表里这格就变成无依据 |
| **P1** | PPU | 宿主 OS + 内核（现在只有容器基座 OS） | 四家里唯一**连宿主 OS 都没有**的 |
| **P1** | MLU590 | ① 容器 `/etc/os-release`（现在是**由包名 `.ubuntu22.04` 推断**）② **FlagGems 算子对照**（唯一一格 ⚪） | 前者把「推断」变「实测」；后者从零建立证据 |
| **P2** | 全部 | 逐格把「随验收间接可用」升级为「逐算子对照」 | 属算子域，**建议与算子方向联合，本方向不单独下结论** |

### 6.3 核验记录写法（避免这轮的老毛病）

- 每条读数必须带 **采集时间 + 机器名 + 命令原文**（例：`npu1-27 · 2026-11-04 10:12 · cat /usr/local/Ascend/driver/version.info`）
- **同一事实出现两处冲突时，两处都保留并标「冲突待定」**，别悄悄改成新值（本轮 910C 驱动就是这种状态）
- 采到新值后：改本表 + 在 §8 记修订；**同时全仓扫旧值**（避免某份文档还留着老版本号）

## 7 边界声明

1. **本表只覆盖设备执行上下文层**（init/context/queue、Stream/Event、D2D/H2D、错误恢复、多流）。算子库、集合通信、显存管理、调度、性能均**只登记不结论**。
2. **✅ 的含义是「本方向在该组合上跑通了完整验收」**，不是「该组合在所有场景都正确」。
3. **未采集不写不支持**；**不支持必须带失败记录或厂商矩阵依据**。
4. 四家判定**互不外推**：某芯片「能」不证明另一家「能」，某芯片「不能」也不证明另一家「不能」。

## 8 修订记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-11-04 | v0.1 | **初表**：四芯片 × 五层共 20 格，✅ 8 / 🟡 11 / ⚪ 1 / ⛔ 0；梳理出 9 条不支持与限制；给出 §6 上机核验清单（P0 两项：910C 驱动矛盾、P800 新机未采集） |
