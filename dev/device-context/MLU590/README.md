# 寒武纪 MLU590 · 第三实例看板

> 分支：`kistich/device-context` ｜ 更新：**2026-09-28** ｜ 负责人：Kistich（hliu553）
> **本目录 = 第三实例（寒武纪 思元 MLU590）的芯片专属资产**；
> 芯片无关的规范与原型在 `../prototype/`，前两实例在 `../910C/`、`../P800/`。
> 接入方法见 **《新芯片接入手册》** `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 0. 状态

> ✅ **2026-10-08 补齐轮 m1 已完成**（网络恢复后一次窗口跑完工作包四组，**20 项全绿**）——
> **台账 D1 收尾**：本实例是三家**唯一**声明 `stream_priority_control`（**能设置**）的实例，
> **本轮已在本层新接线（三段校验 + 强制回读校验）下真机跑过并通过**：
> `STREAM_PRIORITY_API_PASS` **7/7**（区间 `(0,-3)`；请求 `0`/`-3` **回读与请求一致**；
> 越界 `-4`/`1` ⇒ `ValueError`；厂商路径参数**保留**）；职责审计 **K6**（端点请求必须回读一致）
> **首次被真正行使**。⚠️ 同时**更正一处预测**：本实例**并不能**行使 **L5**（建流走 torch 侧
> ⇒ 流非「本层拥有」⇒ `release_stream` 恒 `False`）⇒ 已登记台账 **E2**。
> 🔴 **2026-10-08 第十二轮（同日）再次更正与修复**：上面那条「不能行使 L5」已**通过改口径解决** ——
> 本实例的整条优先级路径改走**厂商 CNRT C API**（与 kunlun 同构）⇒ 产生「本层拥有」的流
> ⇒ **L5 经统一面真机行使并通过**（`RuntimeError: 该流已被 release_stream() 释放`），
> 释放探针 **R3c/R3d 首次真行使**。⚠️ 更要紧的是：本实例原先的「能设置」**是空转判据撑起的假象**
> —— 回读读的是 `torch.mlu.Stream.priority`（**构造参数的回显**），而**设备侧** C API 读到的是
> 4 / 3 / 1 ⇒ 契约 §1.9 第 ⑥ 条在 MLU590 **永远不可能 FAIL**。同轮还修掉**长期登记未修**的
> **id 复用越权销毁**（缺陷台账第 33 条）。详见
> **`../prototype/docs/STREAM_PRIORITY_READBACK_FIX_20261008.md`**。
> 本轮另暴露并修复 **5 处缺陷**（含职责审计 J 域把委派方的「不适用」压成 `OK`）。
> 全部实测数字与证据：`docs/CAMBRICON_MLU_M1_RERUN_20261008.md`。
> 详见 [`../prototype/docs/STREAM_PRIORITY_API_20260930.md`](../prototype/docs/STREAM_PRIORITY_API_20260930.md) §4.5 与
> [`../prototype/docs/OPEN_ITEMS_AUDIT_20260929.md`](../prototype/docs/OPEN_ITEMS_AUDIT_20260929.md) 第八轮段。

> 状态：✅ **接入完成 · 12 项判定全部通过**（2026-09-28 真机复跑）——
> 离线自检 **39/0/0** / 对称性 **5/0** / 冒烟 **46/0** / **conformance 13/13 + 6/6** /
> 多流 16 项（语义 8/8 · 图捕获 4/4 · 配额 3/3）/ 训练腿 **6/6** / **推理腿前向 13/13（另 1 项如实跳过）** /
> **服务化 `SERVE_STANDARD_PASS`** / 错误闭环 **5/0/0**。
> **09-22 因两台主机 SSH 超时未做的「推理腿与服务化」2 项已于 09-28 补齐**，
> 第三实例自此与前两实例**同口径、同判据、同证据规范** ⇒ 已并入
> [`../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md`](../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md)。
>
> ✅ **职责响应审计（39 项 sub-part）真机完成**（09-28 网络恢复后补跑，用卡 0）：
> **`DUTY_RESPONSE_PASS` 36 OK / 0 FAIL / 3 SKIP** —— 3 项 SKIP 均为「**如实不具备**」：
> `elapsed_time`（`CNRT error: failed to call the driver-api function`，契约外可选能力）·
> `D5` / `F1`（本机**无数字错误码** ⇒ 未声明 `error_map`，分级走 `message_hint`，已由 conformance F1 覆盖）。
> ⚠️ **判据数变更（2026-09-29）**：离线自检**新增 2 条判据**（**决策字段**不得受外来码表影响；
> `recover_device()["state"]` 取值域）⇒ 现行判据数 **cambricon 43**、**kunlun 43**、**ascend 38**
> （+2 / +2 / +1，均含非空转验证）。**本文档其余计数为历史批次值，按纪律保留不改写。**
>
> ⭐ **跨实例复验（2026-09-29，本实例发现并修复 1 处层内缺陷）**：按同一套判据复跑，首轮即暴露
> 「**参数类错误的措辞等价类漏网**」—— 设备序号越界时 CNRT 原文是 **`CNRT error: invalid argument.`**，
> 落不到 L2，只能兜底 **`L3_EXECUTION`（`replay`）**，而契约期望 **`L2_PARAM`（`raise`）**
> ⇒ **对一个永久性参数错误反复重放（动作反了）**。已把 L2 规则改为**按等价类覆盖**，
> 并给离线自检 **+2 条判据**（用两家真机原文）且做**非空转验证**（回退规则 ⇒ 恰好该条 FAIL，44/1）。
> 修复后 **10 项全绿**：离线自检 **45/0/0** · 对称性 **5/0** · 冒烟 **46/0** · conformance **13/13 + 6/6** ·
> 职责审计 **36/0/3** · 错误闭环 **5/0/0** · **等价性 5/6 → 6/6** · 多流 **8/8** + 配额 **3/3**。
> ⇒ 报告 [`docs/CAMBRICON_MLU_REGRESS_AFTER_FIX_20260929.md`](docs/CAMBRICON_MLU_REGRESS_AFTER_FIX_20260929.md)；
> 缺陷台账见 [`../prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md`](../prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md) 第 16 条。
>
> 离线契约自检 **41/0/0**（09-28 批次值）· 跨后端对称性 **5/0**；三处补做（`recover_device` 补 `state`、四态命名对齐、
> `sync_timeout` 别名）均**真机验证生效**。⇒
> [`../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md`](../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md)
>
> 本轮验证报告：**[`docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md`](docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md)**
> ｜接入方案与真机手册：**[`docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md`](docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md)**

| 阶段 | 状态 | 结果 |
|---|---|---|
| 阶段 0 · 环境普查 | ✅ **完成（09-22）** | 两台测试机：各 **8 × MLU590-M9（96 GB/卡）**、128 核 / 2 TB 内存、11T 数据盘挂在 `/srv`；**验收模型已在共享 HF 缓存**（`Qwen3-Embedding-0.6B` 快照 `97b0c614…`，只读复用） |
| 阶段 0b · 镜像渠道与定档 | ✅ **完成（09-22）** | 定档 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（digest `sha256:e55b420e…`）；**实测可匿名拉取** |
| 阶段 0c · **环境开通 + 起容器** | ✅ **完成（09-22）** | `hliu553` 入 `docker` 组、`/srv/hliu553` 可写；拉定档镜像（**digest 实测与定档一致**）并起容器 `dc-mlu590-hliu553` |
| 阶段 1 · 接入（`backends/cambricon/`） | ✅ **完成（09-22）** | 13 抽象 + `build()` + `supports()` 如实声明 + `known_issues()`；**离线自检 39/0（按当前原型复跑；落地时为 34/0）、smoke 42/0**（09-28 复跑 **46/0**）；能力声明已按真机证据更新 |
| 阶段 2 · conformance | ✅ **完成（09-22）** | **13/13 + 6/6 全绿（`CONFORMANCE_PASS`）** —— 接入完成的判定线已达成 |
| 阶段 3 · **多流 16 项基线** | ✅ **完成（09-22）** | 多流 **15 通过 / 1 不适用 / 0 不支持**；探针 **`STREAM_SEMANTICS_PASS 8/8`**（双卡，含 S-13）、图捕获（09-22 旧探针口径 **5/5**；09-28 按**现行口径**＝契约内 **4/4** + 1 项契约外**观察项**容忍）、S-16 配额 **2000 流 3/3** ⇒ 报告 `docs/CAMBRICON_MLU_STREAM_BASELINE_16_20260922.md` |
| 阶段 4 · **训练腿** | ✅ **完成（09-22）** | 2 卡 DDP + **`cncl`** + 三类通信对照：**`TRAIN_LEG_PASS 6/6`**、loss **15.4498 → 11.1479**（50 步，无 NaN）、**2957.8 tok/s**（卡 0,2；**09-28 复跑 3015.3 / 3017.5 tok/s，卡 6,7**）。⚠️ CNCL 未加载 `libibverbs`/`libmlx5` ⇒ 走 **MLU_LINK 片间互联**，**非 RDMA**（已如实标注） |
| 阶段 5 · **错误闭环** | ✅ **完成（09-22）** | 四类注入 **`ERROR_RECOVERY_LOOP_PASS`（闭环 5 / 跳过 0 / 失败 0）**，记录自带 `expectation`/`expect_matched` |
| 阶段 6 · **推理腿前向** | ✅ **完成（09-28）** | **`INFER_LEG_PASS 13/13`**（另 1 项 `vendor_code_map` **如实跳过**——CNRT 抛错误名而非数字码）：dim **1024**、**41.08 句/s**、p50 **72.89 ms**、区分度 **0.6391**（`dev=mlu:0`） |
| 阶段 7 · **推理腿服务化** | ✅ **完成（09-28）** | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：就绪 **150 s**、维度 **1024**、范数 **1.000001**、冒烟耗时 **42 s**。⚠️ **须用 vLLM 应用镜像容器**（运行时镜像不含 vLLM，见 §3） |
| 阶段 8 · 收敛 | ✅ **完成（09-28）** | 验证报告 + 三芯片验收矩阵已产出；本轮另修 **1 处工具问题**（冒烟超时硬编码 60 s ⇒ 假失败）并解决 **1 项入口缺口**（服务化改用官方应用镜像）；既有审计见 `../prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md` |

**预期收益**：`device_type="mlu"` 是**第三种设备命名空间**（前两种为 `npu` / `cuda`），
是接口约定修订建议**第 1 条（`device_type` 与 `vendor` 分离）的首次真实验证场景**。

> ⚠️ **一条原预期已被实测否定（09-22）**：本文档原写「寒武纪有真正的厂商错误码体系（CNRT），
> **预期可做出比 P800 更完整的 `error_map`**」—— **实测证明该预期不成立**：
> CNRT 抛出的**不是数字码而是错误名**（`RuntimeError: CNRT error: invalid argument.`），
> OOM 亦是标准 PyTorch 文案（`OutOfMemoryError: MLU out of memory…`）⇒ **无可建码表的数字码**。
> ⇒ `error_map` **如实不声明**（现已从「未验证不声明」升级为「已确认不具备」）；
> 分级由 `message_hint` 覆盖：形状错 → L2、OOM → L1，conformance f1 已通过。
> **方法论**：这正是「先如实不声明，拿到证据再声明」的价值 —— 若当初照着推测写一张码表，就是编造。

---

## 1. 环境速查（先看这里）

| 项 | `Mlu-1`（10.1.1.21） | `Mlu-2`（10.1.1.22） |
|---|---|---|
| 主机名 | `tza-0a06-ai01-em9` | `tza-0a06-ai02-em9` |
| 系统 | Ubuntu 20.04.6 LTS / kernel 5.15.0-139 | 同 |
| CPU / 内存 | 128 核 / 2004 GiB | 同 |
| 加速卡 | **8 × MLU590-M9**，驱动 v6.2.29 / 固件 v1.5.0，**98304 MiB/卡** | 同 |
| 卡占用（09-22 探测） | 卡 0 被他人占 33.6 GB；**卡 1–7 空闲** | **8 张全空闲** |
| 设备节点 | `/dev/cambricon_dev{0..7}`、`cambricon_ctl`、`cambricon_gdr`、`cambricon_ipcm{0..7}` | 同 |
| 厂商工具 | `cnmon`（CNMON v6.2.29） | 同 |
| 数据盘 | **`/dev/sda` 11T → `/srv`**（余 **6.7T**） | **`/dev/sda` 11T → `/srv`**（余 **9.6T**） |
| docker 数据 | **`/var/lib/docker` →符号链接→ `/srv/var/lib/docker`**（镜像本就在 11T 盘） | 同 |
| 宿主 Python | 3.8.10（**无 conda、无 NeuWare**）⇒ MLU 栈走**容器** | 同 |
| SSH | `ssh Mlu-1` / `ssh Mlu-2`（**公钥免密已通**） | — |
| 两机共享目录 | ❌ 无（数据需分别放） | — |

**环境报告（含逐项依据与原始日志索引）**：`docs/CAMBRICON_MLU_ENV_REPORT_20260922.md`

---

## 2. 六条关键认知（实测得出，接入前必读）

1. **`/srv/hliu553` 建不了**：`/srv` 属主 `root:root 755`，实测 `mkdir: Permission denied`；
   我们虽在 `sudo` 组但 **sudo 需密码** ⇒ 必须由 root 开通（见 §4）。**✅ 09-22 已开通**
2. **docker 镜像数据无需搬**：`/var/lib/docker` 就是 `/srv/var/lib/docker` 的符号链接，
   镜像与容器层本已落在 11T 盘上 ⇒ **不要改 `daemon.json` 的 `data-root`**（改动要重启 docker，风险大于收益）。
3. **不在 `docker` 组**（组员：`gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu`）
   ⇒ 当时**无法使用 docker CLI**，而验证流程全部在带卡容器内 ⇒ 曾与第 1 条并列的硬阻塞。**✅ 09-22 已加入 `docker` 组**
4. **宿主没有 NeuWare**（无 `/usr/local/neuware`）、也没有 MLU 版 Python 栈
   ⇒ MLU 软件栈**必然走容器**；`daemon.json` 已配寒武纪私有仓（`docker.cambricon.com` 等）。
5. **两机是 K8s 节点**（`kubelet` + `containerd` + `docker` 三服务 active，`crictl`/`nerdctl` 在位）
   ⇒ 容器可能有"docker CLI"与"k8s"两条路径，需与管理员确认走哪条。
6. **`/srv/data/` 是他人资产**（`x-benchmark` 负载）：内含 **521 个 HF 模型**（`hf_cache/`，**只读可复用**）
   与 314 个模型 venv 的构建日志。**只读、不写、不删。**

---

## 3. 环境版本组合（⚠️ 本节的版本号**不是**我们的目标档，见下）

`/srv/data/build_base_venv.sh`（他人资产，只读）给出了"寒武纪 MLU 基础 venv"的确切组合：

```text
python     3.12（注释："neuware472 需 py3.12"）
torch      2.11.0+cpu
torch-mlu  1.33.1+torch2.11.0
torch-mlu-ops 1.12.1+torch2.11.0
triton     3.4.0+mlu2.1.1          ← triton-mlu 变体
私有源     https://resource.flagos.net/repository/flagos-pypi-cambricon/simple
通用源     https://mirrors.aliyun.com/pypi/simple
NEUWARE_HOME  /usr/local/neuware
FlagGems   拉 master 源码 → /opt/FlagGems（editable, --no-deps）
```

> **⚠️ 重要更正（2026-09-22）**：该组合属 **`cambricon-neuware4.7.2` 档**
> （脚本注释原文即 "neuware472 需 py3.12"），而该档官方标注**宿主驱动前置 6.5.48**，
> 我们两台机器实测 **v6.2.29** ⇒ **不满足**。
> **本方向已定档走 `neuware4.4.3` 档**，实际组合为
> **py3.10 / torch 2.7.1+cpu / torch-mlu 1.29.2+torch2.7.1 / torch-mlu-ops 1.8.0 / triton 3.2.0+mlu1.7.2**
> （官方标注驱动前置 6.2.15，与我们同 6.2.x 线）。
> 本节的**价值仍在于**：证明依赖获取是「厂商私有源 + 通用源混合」，不是裸 `pip install torch-mlu`。
> 依据：`docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.1。

✅ **已实机复核（09-22 起）**：`import torch_mlu` → `torch.mlu.device_count() == 8`；
实际栈为 **py3.10.20 / torch 2.7.1+cpu / torch_mlu 1.29.2+torch2.7.1 / triton 3.2.0+mlu1.7.2**，**与 `neuware4.4.3` 档一致**；且**运行时镜像与 vLLM 应用镜像的栈逐项相同**（见下）。

**镜像从哪来 → 见 `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md`**（09-22 调研 + **同日下午更正**）：
- ❌ **FlagTree 没有寒武纪 User Manual / 推荐镜像**（wiki 26 页无 cambricon 条目；寒武纪只存在于编译器侧 `triton_v3.2.x` 分支）
- ✅ **但 FlagOS 官方在 BAAI Harbor 上已有寒武纪镜像**（`flagos-base` / `flagos-runtime` / `flagos-app` 共 12 个仓
  + FlagGems 周测 2 个仓），且**实测可匿名拉取**（Registry v2 匿名 token 取 manifest 成功，digest 已取得）
  ⇒ **不需要寒武纪私仓凭据**
- ⭐ **档位按宿主驱动选，不按 torch 版本选**（实测驱动 **v6.2.29**，落 6.2.x 线）：

| 档位 | py | torch / torch-mlu / triton | 官方标注宿主驱动 | 我们 |
|---|---|---|---|---|
| **`harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`** | 3.10 | 2.7.1+cpu / 1.29.2 / 3.2.0+mlu1.7.2 | **6.2.15** | ✅ **已定档（2026-09-22）：先走它解除阻塞** |
| `…/flagos-runtime-cambricon-neuware4.7.2:2.2.0` | 3.12 | 2.11.0+cpu / 1.33.1 / 3.4.0+mlu2.1.1 | **6.5.48** | ❌ 需升宿主驱动；列为**上报预案**（见下） |

**定档理由**：4.4.3 的驱动前置 6.2.15 与我们实测 **v6.2.29 同属 6.2.x 线** ⇒ 起容器风险最低、可立即推进；
4.7.2 需动**两台共享机的宿主驱动**（影响他人），且非当前瓶颈。
**代价如实标注**：4.4.3 是旧档（torch-mlu 1.29.2 vs 1.33.1 / py3.10 vs 3.12）；
本方向对 torch-mlu 小版本不敏感，故代价可接受。

⚠️ **驱动升级（→ 6.5.48 走 4.7.2）为预案、非当前诉求**：**不凭版本号要求升级，只凭证据要求升级**；
只有当原型接入 / 设备上下文验证出现「疑与旧档强相关、且已排除我方五域」的不可解问题时才启动，
且上报须写清**问题**与**原因**（四条门槛 + 模板见
`docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.2）。
**纪律**：设备上下文结论必须标注取得时所处的档位（同 P800 的「KL3 未设置条件下取得」）。

- ✅ **推理形态（09-28 实测定稿）**：服务化改用 FlagOS 官方**应用镜像**
  **`flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`**（vLLM **0.20.2**，
  digest `sha256:f568f23cf29b2…` **与 harbor 登记一致**，2.91 GiB，**实测可匿名拉取**）；
  镜像内栈与运行时镜像**逐项一致**（py3.10.20 / torch 2.7.1+cpu / torch_mlu 1.29.2）⇒ **同档可比**。
  起容器脚本 `start_container_mlu590_vllm.sh`（**与原脚本只差镜像一行**）。
  另一档 `vllm0.24.0-cambricon-neuware4.4.3:2.2.0-0.3.0rc2.post2` 列为候选（vLLM 版本更高，**未采用**）
- ✅ **原三项「仍待实测」已全部定论**：① 驱动 **6.2.29 可跑**标 6.2.15 的 4.4.3（同 6.2.x 线，实测正常）；
  ② 带卡机**可出网**拉 `harbor.baai.ac.cn`（runtime 与应用镜像均拉通，digest 一致）；
  ③ 应用镜像内为**厂商移植版 vLLM**（`vllm_fl.dispatch.manager` 算子分发日志 + triton `mlu` backend）

---

## 4. 阻塞与需要协调的事项

| # | 事项 | 需要谁 | 状态 |
|---|---|---|---|
| **1** | 建 `/srv/hliu553` 并 chown 给 `hliu553` | root / 机器管理员 | ✅ **09-22 已开通**（两台） |
| **2** | 把 `hliu553` 加入 `docker` 组 | root / 机器管理员 | ✅ **09-22 已开通**（两台；Docker 25.0.3） |
| **3** | **镜像获取** → ✅ **09-22 已定档并已拉取**（digest 实测与定档一致）：走 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（digest `sha256:e55b420e…`）；FlagOS 官方仓**实测可匿名拉取**，不需要私仓凭据。4.7.2 档（需驱动 6.5.48）列为**上报预案**、非当前诉求 | 本方向自定 | 🟢 **已定档** |
| **4** | **起带卡容器** → ✅ **09-22 已起**：`dc-mlu590-hliu553`（18 个设备节点；起容器脚本 `/srv/hliu553/start_container_mlu590.sh`） | — | 🟢 **已完成** |
| 5 | 两机**无共享目录**：数据需分别放置；若需共享需另配 NFS | 管理员（可选） | ⚪ 已知 |

root 执行命令（两台各一次）：

```bash
sudo mkdir -p /srv/hliu553 && sudo chown -R hliu553:hliu553 /srv/hliu553 && sudo chmod 750 /srv/hliu553
sudo usermod -aG docker hliu553
```

---

## 5. 目录内容

| 路径 | 内容 |
|---|---|
| `docs/CAMBRICON_MLU_ENV_REPORT_20260922.md` | **环境报告（第 0 步）**：两机并列明细 · docker 数据盘归属的证据链 · 版本组合 · 开通需求 · 探测边界 |
| `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` | **镜像渠道调研 + 更正 + 定档**：§0 更正段（FlagOS 官方 BAAI Harbor 已有寒武纪三代镜像、实测可匿名拉取）· §0.1 **定档 `neuware4.4.3`** · §0.2 **驱动升级上报预案**（四条门槛 + 上报模板）· FlagTree 无寒武纪手册（26 页证据）· 三私仓实测 |
| `docs/CAMBRICON_MLU_STREAM_BASELINE_16_20260922.md` | ⭐ **多流 Stream 验收基线 16 项逐项比对报告**：16 项 MLU590 结论（**15 通过 / 1 不适用 / 0 不支持**）· **三实例逐项对照**（唯一差异 = S-12 流优先级，**与 P800 相反**）· S-7 图捕获 5/5 与 S-16 配额 2000 流 · 证据形态差异（含 **CNCL 走 MLU_LINK 非 RDMA**）· 复现命令 · 未覆盖项 |
| `docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md` | ⭐ **接入方案 + 真机执行手册**：进度表 · 厂商栈判别（预期路径 C）· 已完成的代码层动作与能力声明理由 · 本地验证（离线自检 35/0）· **A1–A10 真机执行序列（含确切命令）** · 验收清单 13 项当前状态 · 风险与应对 · 职责边界 |
| `docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md` | ⭐ **推理腿（前向 + 服务化）验收报告**：12 项判定全通过 · 两条要点（**运行时镜像不含 vLLM ⇒ 用官方应用镜像**；**冒烟超时硬编码 60 s ⇒ 假失败**，含 PRE_FIX 原样留档）· 三实例对照与差异解释 · 边界与未覆盖 · 复现命令 |
| ⭐ `docs/DUTY_RESPONSE_AUDIT_MLU590_20260928.md` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **36 OK / 0 FAIL / 3 SKIP**（SKIP 均如实不具备）；`recover_device` 的 `state` 补做**真机验证生效** |
| ⭐ `docs/CAMBRICON_MLU_REGRESS_AFTER_FIX_20260929.md` | **修复后全套回归 + 一处层内缺陷的发现与修复**：工作包 A 等价性 **5/6 → 6/6**；L2 文案等价类的现象/根因/危害/修法；2 条新判据与**非空转验证**（回退规则 ⇒ 44/1）；三处修复的定向验证；未跑项如实登记 |
| `probes/accept_*_20260928.*` | **本轮 12 项判定的原始证据**（命名与两实例同规范）：离线自检 · 冒烟 · conformance 13+6 · 多流三项 · 训练腿 · 推理腿前向 · 服务化（含 `PRE_FIX` 失败留档与 vLLM 服务端日志）· 错误闭环 |
| ⭐ `probes/duty_audit_cambricon_20260928.json` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **36 OK / 0 FAIL / 3 SKIP**（3 项 SKIP 均为如实不具备）；离线自检 41/0/0 · 对称性 5/0 |
| `probes/preflight_env_mlu1_20260922.log` | **Mlu-1 环境普查原始日志**（`preflight_env.sh` 首跑产出） |
| `probes/preflight_env_mlu2_20260922.log` | **Mlu-2 环境普查原始日志** |
| `probes/.gitignore` | `!*.log` 例外（否则根 `.gitignore` 的 `*.log` 会让证据静默不入库） |
| ⭐ `probes/regress_*_cambricon_20260929.*` + `probes/exp_divergence_cost_cambricon_20260929.*`（共 18 份） | **09-29 修复后全套回归原始证据**：离线 **45/0/0** · 对称性 5/0 · 冒烟 46/0 · conformance 13+6 · 职责审计 36/0/3 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `probes/exp_divergence_cost_cambricon_PRE_FIX_20260929.json` | **修复前**同一实验的原始证据（**5/6，`S1` DIFF + 期望未达标**）—— 缺陷发现的**第一手证据**，原样留档 |

---

## 6. 已完成 / 后续

**已完成（09-28 收口）**：第三实例 **12 项判定全部通过**（见 §0 状态块与阶段表），
接入路径闭环＝`环境普查 → 镜像定档 → 后端落地 → 离线自检 → conformance 13+6 → 多流 16 项 → 训练腿 → 推理腿两形态 → 错误闭环`。

**资产复用（三实例同构，换芯片只改 `DC_BACKEND`）**：
`../prototype/scripts/preflight_env.sh`（环境普查）· `../prototype/scripts/backend_offline_check.py`（离线契约自检，无设备可跑）·
`../prototype/probes/probe_stream_semantics_full.py` 等三个探针（多流 16 项）· `../prototype/runtime/conformance/`（判据集）·
`../prototype/scripts/serve_standard.sh`（服务启动；**寒武纪需先起 vLLM 应用镜像容器**）。

**后续（不阻塞发布）**：
1. **多卡 TP / 关闭 `--enforce-eager`** 的服务化形态未跑（本次为 `TP=1` + eager）。
2. **CNCL 的 RDMA 路径未验证**（当前走 `MLU_LINK` 片间互联，训练数据仅代表单机 2 卡）。
3. 档位升级（→ `neuware4.7.2` / 驱动 6.5.48）仍为**上报预案**，非当前诉求（见 §3）。
