# 寒武纪 MLU590 · 第三实例看板

> 分支：`kistich/device-context` ｜ 更新：**2026-09-28** ｜ 负责人：Kistich（hliu553）
> **本目录 = 第三实例（寒武纪 思元 MLU590）的芯片专属资产**；
> 芯片无关的规范与原型在 `../prototype/`，前两实例在 `../910C/`、`../P800/`。
> 接入方法见 **《新芯片接入手册》** `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 0. 状态

> ✅ **当前状态（2026-10-08）**：**接入完成 · 12 项判定全部通过**；**m1 补齐轮 20 项全绿**
> （离线 **97/0/0** · 冒烟 46/0 · conformance 13+6 · **职责 `DUTY_RESPONSE_PASS` 62/0/16** ·
> 非空转 **26/13/0** · 训练腿 6/6 · 推理腿 13/13 · 服务化 `SERVE_STANDARD_PASS` + `SERVE_LEG_PASS` 10/10）。
> 数字与证据：`docs/CAMBRICON_MLU_M1_RERUN_20261008.md` **L10–L46**（§0）。
>
> ⭐ **本实例的三个「唯一」**（三家对照时信息量最大）：
> ① **唯一**声明 `stream_priority_control` **且区间非单点**（`(0,-3)`）⇒ **唯一能出「调度效果」结论**的实例；
> ② **唯一**经**统一面**行使契约 §1.10 **L5**（流所有权/释放）的实例 —— 整条优先级路径走**厂商 CNRT C API**
>    （与 kunlun 同构）⇒ 能产生「**本层拥有**的流」；
> ③ **唯一**暴露过「**回读判据空转**」的实例（原读 `torch.mlu.Stream.priority` = **构造参数回显**，
>    设备侧其实是 3/4）⇒ 已改为**厂商 C API 口径**，见
>    `../prototype/docs/STREAM_PRIORITY_READBACK_FIX_20261008.md` **L59–L100**。
>
> ⭐ **调度效果结论（D2 · 分场景，不得合并）**：**同时就绪 ⇒ 无实质效果**（两流并行完成，中位差
> **0.227 ms** = 工作量级 14.185 ms 的 **1.6%**）；**有排队争用 ⇒ 有实质效果**（取值 0 两种提交顺序各
> **8/8**：反超 **+6.24** / 大胜 **+7.15 ms**，同取值同顺序对照 **−1.84 ms** ⇒ 净效应 **5.3–8.1 ms**）
> ⇒ **可以**用优先级表达「插队到已在运行的低优先级流之前」，**不可以**指望「同时就绪时按优先级排序完成」。
> ⚠️ 数字**只对本栈**（驱动 6.2.29 / `neuware4.4.3`）且为**人为构造的单卡争用**。
> 详见 `../prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54 / L94–L122**。
>
> **「不适用」自证（2026-10-08）**：本实例**没有**「不适用」结论 —— 三家中的另两家才有
> （910C 未声明 `control`、P800 单档）；两家均通过自证审计，见
> `../prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md`。
>
> **接入期状态（09-28 收口 · 历史）**：12 项判定全部通过（离线 39/0/0 · 冒烟 46/0 · conformance 13+6 ·
> 多流 16 项 · 训练腿 6/6 · 推理腿前向 13/13 · 服务化 · 错误闭环 5/0/0）；09-22 因两机 SSH 超时未做的
> 推理腿与服务化 2 项已于 09-28 补齐 ⇒ 与前两实例**同口径、同判据、同证据规范**（已并入
> `../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md`）。

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

## 3. 环境版本组合（⚠️ 版本号由**宿主驱动**决定，不按 torch 版本选）

**一句话**：实测宿主驱动 **v6.2.29**（6.2.x 线）⇒ 定档
`flagos-runtime-cambricon-neuware4.4.3:2.2.0`；`neuware4.7.2` 档（torch-mlu 1.33.1）**要求驱动 6.5.48**，
列为**上报预案、非当前诉求**（**不凭版本号要求升级**）。

| 项 | 值 | 说明 |
|---|---|---|
| 宿主驱动 | **v6.2.29**（固件 v1.5.0） | `cnmon` 实测；**选档第一判据** |
| 定档镜像 | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（digest `sha256:e55b420e…`） | **实测可匿名拉取**（无需私仓凭据） |
| 实机栈 | py3.10.20 / torch 2.7.1+cpu / torch_mlu 1.29.2 / triton 3.2.0+mlu1.7.2（`torch.mlu.device_count()==8`） | 容器内实测，与 4.4.3 档一致 |
| 服务化镜像 | `flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`（vLLM 0.20.2，digest `sha256:f568f23c…`） | **运行时镜像不含 vLLM** ⇒ 必须用应用镜像；栈与运行时镜像逐项相同 ⇒ 同档可比 |
| ⚠️ 已作废的一条 | 他人脚本 `/srv/data/build_base_venv.sh` 给的 `torch 2.11.0 + torch-mlu 1.33.1` | 属 **4.7.2 档**（注释原文 "neuware472 需 py3.12"）⇒ **不是**我们的档位；其价值仅在于证明依赖获取 = **厂商私有源 + 通用源混合** |

> 完整调研 / 档位对照 / 私仓实测 / 驱动升级四条门槛与上报模板：
> `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` **L10–L51**（§0 更正 + §0.1 定档）·
> **L63–L109**（§0.2 升级预案 + §0.3 纪律）· **L154–L224**（寒武纪官方渠道实测）。
> ⚠️ **纪律**：设备上下文结论必须标注取得时所处的档位（同 P800 的「KL3 未设置条件下取得」）。

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
| 证据入口 | **`probes/`** —— 逐份说明见 [`docs/EVIDENCE_INDEX_MLU590.md`](docs/EVIDENCE_INDEX_MLU590.md)；当前结论入口：`probes/accept_*_20260928.*` · `probes/duty_audit_cambricon_*.json` · `probes/m1_*_20261008*` · `probes/regress_*_cambricon_20260929.*` |

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
