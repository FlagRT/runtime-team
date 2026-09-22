# 设备方向的镜像选择与确定指南

> 版本：v1.0（2026-09-20）｜ 负责人：Kistich（hliu553）
> **用途**：回答"设备方向需要一个镜像时，怎么选、怎么验、谁定"。适用于第三家芯片（寒武纪等）接入，
> 也适用于现有两实例镜像的补齐。
> **写法约定**：依据写成可独立阅读的实测事实，不写成"见某文件某节"。

---

## 1. 先明确：我们要的镜像到底要满足什么

设备方向（设备抽象与执行上下文 + 多流 Stream + 错误翻译 + 状态恢复）在镜像里只做三件事：
**跑到设备上、跑通 conformance、跑通训推两条腿**。所以对镜像的要求是"**能被我们的原型接管**"，而不是"算子多、模型多"。

| 需求 | 具体内容 | 为什么 |
|---|---|---|
| ① 设备可见 | 镜像内能枚举到卡 | 这是所有后续工作的前提；不可见时连 smoke 都跑不了 |
| ② 厂商 torch 栈可导入 | `torch_npu`（昇腾）/ `torch.cuda` 兼容层（昆仑芯）/ `torch_mlu`（寒武纪） | 我们的 backend 是包在厂商 torch 命名空间之上的，没有它就没有"设备抽象"可包 |
| ③ 可承载我们原型 | 纯 Python 的 `runtime/` 能被 import（无额外依赖） | 原型零第三方依赖，只要 python 能跑 |
| ④ 推理腿有可用 vLLM 路径 | vLLM 能认出设备（厂商移植版 或 有平台插件） | 服务化形态要用 OpenAI 兼容接口 |
| ⑤ 训练腿有可用集合通信 | FlagCX 或厂商 CCL 能建进程组 | 2 卡 DDP 需要 |
| 不需要 | 算子库、模型转换器、调度器 | 分别属算子 / 编译 / 调度方向 |

**一句话判据**：镜像装好、容器起好之后，两条命令不发散即可认为"方向候选可用"：

```bash
python -c "import torch; print('devs', torch.cuda.device_count() if hasattr(torch,'cuda') else torch.npu.device_count())"
python -c "import vllm; from vllm.platforms import current_platform; print('platform', current_platform)"
```

第二条若打印 `UnspecifiedPlatform`，就**不能**直接 `vllm serve`——需要额外平台插件（P800 就是这种情况，
它靠 vllm-plugin-FL 提供平台；昇腾不需要，因为 `vllm-ascend` 是厂商官方移植版）。

---

## 2. 来源优先级（按可复现性从高到低）

> **2026-09-22 补充**：FlagOS 官方镜像体系（`flagos-ai/build-infra` 构建，
> registry 前缀 `flagos-base` / `flagos-runtime` / `flagos-dev` / `flagos-app`，当前版本 **2.2.0**）
> 覆盖 **14 个后端**（含**寒武纪**，这是 FlagTree 手册覆盖不到的一家），**是首选的官方对齐口径**。
> **判据：同一芯片既有 FlagTree 线又有 FlagOS 官方线时，先在 STATUS.md 里问总组要哪条，
> 不要方向侧自行切换**（两条线的底层 SDK 可能换代，例如昆仑芯 FlagTree 线是 XRE 0.x 命名、
> FlagOS 官方线已到 **XRE 5.37.1**）。

| 优先级 | 来源 | 特征 | 已有先例 |
|---|---|---|---|
| **0（新增，最优先）** | **FlagOS 官方镜像体系**（`harbor.baai.ac.cn/flagos-{base,runtime,app}/…`） | 由 `flagos-ai/build-infra` 的 `configs.yaml` 统一生成（文档站 `flagos-ai.github.io/release-info/` 自动同步，不会漂移）；**覆盖含寒武纪在内的 14 后端**；每个 `base|runtime/<backend>.md` 明确写出**宿主驱动前置**；实测可匿名拉取 | `flagos-runtime-kunlunxin-xre5.37.1:2.2.0`（digest `sha256:0f488c7b…`）、`flagos-runtime-ascend-cann9.0.0-910c:2.2.0`（`sha256:1048d622…`）、`flagos-runtime-cambricon-neuware4.4.3:2.2.0`（`sha256:e55b420e…`） |
| **1** | **厂商官方发布镜像** | 有 registry、有可拉取 digest、厂商持续维护 | 910C 推理腿 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（digest `sha256:5cf8a2b6db8b06eb1bc7fc7d191d667aebf2b197351bdba13f776918c11ec7a7`） |
| **2** | **FlagTree 手册镜像 / 组内 harbor 镜像** | 有 digest、组内可拉；本机可能已有同名系列 | P800 `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（94.5 GB，本机已存在） |
| **3** | **本机已有镜像（无 registry）** | 靠 `docker save` 备份 + 配方重建保证可复现 | 910C 训练腿 `flagrt/ascend-operator-runtime-comm:0.1.3-…`；P800 `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` |
| **4** | 自建 | **原则不允许**——各方向只消费不自建（镜像的确定与调整由总组裁定） | — |

> 优先级 0–2 的好处是"拿到 digest 就等于拿到环境"；优先级 3 必须补齐**可重建配方**（`Dockerfile.repro` + assets）
> 或至少有 `docker save` 离线归档，否则镜像一丢就无法复现结论。
>
> ⚠️ **宿主要求是优先级 0 的硬前置，别只看 Python 包版本**：`flagos-*/<backend>.md` 里有一行
> **Host driver**（例：`cambricon-neuware4.7.2` → `6.5.48`，`cambricon-neuware4.4.3` → `6.2.15`，
> `kunlunxin-xre5.37.1` → `5.37.1`，`ascend-cann9.0.0-910c` → `26.0.rc1`）。
> **寒武纪就是被这一行卡住的**：两台测试机驱动 6.2.29，只能用 4.4.3 档，4.7.2 要 6.5.48。

---

## 3. 怎么"定"：入档 + 入锁两道门槛

镜像一到位，先**入档**（归档进仓库），再申请**入锁**（写进全组锁定基座）。两道门槛如下。

### 3.1 入档门槛（仓库内归档）

- 目录结构：`dev/images/<name>/v<N>/`，其中 `<name>` 是去 registry 前缀与 tag 的稳定短名，同一血统的多版本归其下
- 至少包含：`lock.yaml`（血统 + 内置组件版本 pin + 官方/自定义边界 + 重建缺口 + 重建验证结果）、
  `ARCHIVE.md`（离线归档清单：tar 名 / 大小 / sha256 / `docker load` 用法）
- 版本不可变：镜像有变更就新建 `v<N+1>/`，旧目录保留并在旧 `lock.yaml` 顶部加 `superseded_by`
- 可复现性状态（`repro_status`）两档：
  - 🟢 = `Dockerfile.repro` + assets 已**实机重建**，且与原镜像比对通过（`pip freeze` 逐行 / 关键 `.so` 逐字节）
  - 🟡 = 配方在手，未实机验证
  - 上游发布的镜像（无自建 Dockerfile）也可记 🟢，但含义是"digest 锁定 + 已归档 + 已验证可拉"

### 3.2 入锁门槛（成为全组结论性环境）

- `dev/images/<name>/v<N>/` 存在且含 `lock.yaml`，`repro_status` **≥ 🟡**
- 组内镜像不涉及 registry，看能否凭仓内 `Dockerfile.repro` + assets 重建
- 上游镜像可无 Dockerfile，但**须有可拉取 digest**
- **由总组裁定**：统一基座的确定 / 调整 / 发布由总组负责，各方向只消费不自建；
  方向侧的动作是**把诉求写进本方向 STATUS.md**，由总组收拢裁定
- 结论性验证纪律：**只在锁定镜像内做**；日常调试容器的结果不作为验收依据

### 3.3 提给总组的入锁材料（建议固定四项）

1. 镜像 tag + **digest**（`docker image inspect` 的 `RepoDigests`）
2. 内置关键组件版本（python / torch / 厂商栈 / 通信库 / vLLM）
3. 本方向的验证结果（smoke + conformance 13+6 + 两条腿的量化数字）
4. 可复现性状态与获取方式（可拉 / `docker save` 包在哪 / 有无重建配方）

---

## 4. 实操五步（我们方向照这个走）

```text
① 可行性核对（只读，30 分钟）
   设备可见 + 厂商 torch 可导入 + 原型可跑 → 见 §1 的两条判据

② 环境普查（半天）
   卡数/显存、内存/CPU、数据盘与权限、网络与拓扑、镜像落盘位置、共享程度
   → 产出一份 ENV_REPORT（P800 的做法可复用）

③ 归档（半天）
   建 dev/images/<name>/v1/，写 lock.yaml + ARCHIVE.md；镜像实体 docker save 落盘（不入仓库）

④ 验证（1 天）
   smoke（组件自检）→ conformance 13 例 + 推理 6 例 → 训练腿 2 卡 → 推理腿前向 + 服务化 → 错误闭环

⑤ 提诉求（当天）
   STATUS.md「基座与约束 / 阻塞与需要协调」登记；把 §3.3 四项材料给总组 → 总组裁定后写入 stack.lock
```

---

## 5. 两实例镜像现状（先例对照，含一处缺口）

| | 910C 训练腿 | 910C 推理腿 | P800 |
|---|---|---|---|
| tag | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` |
| 来源 | 组内（BAAI 内部 CANN 9.0.0 底座 + FlagOS 组件：Torch-FL / FlagGems / FlagCX） | **华为昇腾官方**发布物 | 内置标签显示 `description: xvllm_ubuntu2204_torch29 环境`、**`maintainer: huangyun <huangyun07@kunlunxin.com>`** ⇒ 底座由**昆仑芯**提供，外层按 BAAI·FlagTree 的 xpu3.6 线命名 |
| digest | 无 registry（靠重建 + `docker save`） | `sha256:5cf8a2b6…` | **`sha256:cd53efa40eb7ddc49c2ad76a9bfbd252572c5fb01bd10d02cffbf667c34a1975`** |
| 大小 | — | — | 磁盘 **107 GB**（镜像层 38.3 GB，创建于 2026-08-12）；官方 `-base` 磁盘 **94.2 GB**（镜像层 33.8 GB） |
| 已归档 | ✅ `dev/images/ascend-train-comm/v1` | ✅ `dev/images/ascend-infer-vllm/v1` | ❌ **未归档** |
| 已入锁 | ✅ `lock.train` | ✅ `lock.infer` | ❌ **未入锁** |
| 结论性验证依据 | ✅ | ✅ | ⚠️ 目前 P800 的结论是在**未入锁**的镜像上取得的 |
| 官方推荐对应物 | 官方镜像不含 FlagCX（训练腿必需） | 华为官方 `vllm-ascend`（vLLM 同 0.20.2） | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（33.8 GB，digest `sha256:ea6d797a…`） |
| 等价性验证 | — | — | ✅ **已完成（2026-09-20）**：在官方 `-base` 上重跑 conformance 13+6（逐用例一致）、smoke 42/0、训练腿/推理腿/服务化全 PASS、KL3 挂死一致重现 ⇒ 两镜像结论等价 |

**⇒ 这是我们当前的明确缺口**：P800 镜像在位、有 digest、两条腿已跑通，但**既没归档也没入锁**。
补这个缺口的成本很低（归档半天 + 一次 docker save），但收益明确——否则"P800 的结论性验证"在纪律上站不住
（而 910C 的结论是有锁定基座背书的）。

**⇒ 2026-09-20 补充：官方 `-base` 可直接作为入锁候选，但配方必须含补齐步骤。**
官方 `-base` 开箱**不含 `triton`**（`...-base-ssh` 变体也没有），vLLM 服务化路径会断在
`vllm_fl → flag_gems → triton`，报 `Failed to infer device type`。按官方手册 1.2 节执行：

```bash
python3 -m pip uninstall -y triton          # 反复执行至彻底卸载
python3.10 -m pip install flagtree===0.7.0rc3+xpu3.6 \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
```

实测源可达（HTTP 200）、wheel 3.3 GB、约 2.5 分钟装完，装后 `triton 3.6.0`
（与现用 `flaggems-main-dev` 变体**版本号完全一致**），`flag_gems` / `vllm_fl` 均可导入，
服务化随即跑通（`SERVE_LEG_PASS 10/10`、区分度 0.4102 与现用一致）。

---

## 6. 第三家芯片（寒武纪）的镜像选择建议

> **2026-09-22 已定案，替换原下表**：寒武纪**不需要走寒武纪官方渠道申请凭据** ——
> FlagOS 官方在 BAAI Harbor 上已有寒武纪三代镜像（`flagos-base` / `flagos-runtime` / `flagos-app`）
> 与 FlagGems 周测镜像，**实测可匿名拉取**（Registry v2 匿名 token 取 manifest 成功，已取得 digest）。
> 唯一分叉点是**档位由宿主驱动决定**：

| 档位 | Python | torch / torch-mlu / triton | 官方标注宿主驱动前置 | 我们（实测 v6.2.29） |
|---|---|---|---|---|
| `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` | 3.10 | 2.7.1+cpu / 1.29.2 / 3.2.0+mlu1.7.2 | **6.2.15** | ✅ **同 6.2.x 线，选它** |
| `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.7.2:2.2.0` | 3.12 | 2.11.0+cpu / 1.33.1 / 3.4.0+mlu2.1.1 | **6.5.48** | ❌ 不满足（要升宿主驱动） |

**三步判据**：

| 步 | 动作 | 判据 |
|---|---|---|
| 1 | **先查宿主驱动**，再按驱动选档；用 `harbor.baai.ac.cn/flagos-runtime/…-cambricon-neuware4.x.x` | 宿主驱动 ≥ 该档 `base/<backend>.md` 的 **Host driver** 行 |
| 2 | 起容器后先跑 §1 两条判据 | 设备可见 + `torch_mlu` 可导入；`current_platform` 不是 `UnspecifiedPlatform` |
| 3 | 归档 + 验证 + 提诉求 | 同 §4 |

**寒武纪的 vLLM 支持形态**：官方已有 `flagos-app/vllm0.24.0-cambricon-neuware4.7.2:2.2.0-0.3.0rc2.post2`
等应用镜像 ⇒ **推理腿预计"一条命令起服务"**（与昇腾同类），但**「厂商移植版 vs 社区版 + 插件」
仍未落实** —— 落地时解开镜像确认即可，不必问厂商。

**⚠️ 两个仍待实测的点（不要凭版本号下结论）**：① 驱动 6.2.29 能否跑标注 6.2.15 的 `neuware4.4.3`
（同 minor 线属**推断**）；② 带卡机能否出网拉 `harbor.baai.ac.cn`。

---

## 7. 可直接复用的判据清单

| # | 判据 | 检查方式 | 通过标准 |
|---|---|---|---|
| 1 | 设备可见 | `torch.cuda.device_count()` / `torch.npu.device_count()` / `torch.mlu.device_count()` | > 0 |
| 2 | 厂商 torch 栈可导入 | `import torch_npu` / `import torch_mlu`；昆仑芯 `import torch` 后即走 `cuda` 命名空间 | 无异常 |
| 3 | 我们的原型可跑 | `python runtime/smoke_runtime.py --backend <name>` | 全绿（昇腾 37/37、昆仑芯 42/0 为先例） |
| 4 | conformance | `python runtime/conformance/runner.py --backend <name>`（含 `--cases infer_cases`） | 13 例 + 6 例全绿，或如实 stub-skip 并说明缺口 |
| 5 | 推理腿 vLLM | `from vllm.platforms import current_platform`；起服务后 `curl /v1/models` | 平台非 `Unspecified`；服务 200 |
| 6 | 训练腿通信 | 分布式后端探测（`nccl` / `xccl` / `kccl` / `flagcx`）→ 三类对照（all_reduce / all_gather / P2P） | 至少一条可用路径；三类结果全对 |
| 7 | 可复现 | `docker image inspect` 的 `RepoDigests`；或 `dev/images/` 内的 `Dockerfile.repro` | 有 digest 或可重建配方 |
| 8 | 可归档 | 镜像实体能 `docker save` | 落盘成功且记录 sha256 |

---

## 8. 三条纪律（来自实测教训）

1. **两条腿不需要同时跑**：910C 上有"带卡容器并发上限 3"的硬限制（4 个并发时 `acl.init()` 返 500000、
   `torch.npu.device_count()=0`），所以约定串行执行、跑完一条停掉再起下一条。
2. **共享机要先看卡再动手**：P800 是共享机，曾把"卡被他人占用"误判为"通信库适配缺陷"，
   换到空闲卡后三类通信全通过——**用卡前先看占用，并在记录里写明用了哪张卡**。
3. **停服务要连子进程一起清**：`vllm serve` 被杀主进程后，`VLLM::EngineCore` 会残留并持续占卡
   （实测卡被占 73850 MiB / 96 GiB，导致下次启动直接报显存不足）。
