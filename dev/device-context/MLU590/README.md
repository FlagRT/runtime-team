# 寒武纪 MLU590 · 第三实例看板

> 分支：`kistich/device-context` ｜ 更新：2026-09-22 ｜ 负责人：Kistich（hliu553）
> **本目录 = 第三实例（寒武纪 思元 MLU590）的芯片专属资产**；
> 芯片无关的规范与原型在 `../prototype/`，前两实例在 `../910C/`、`../P800/`。
> 接入方法见 **《新芯片接入手册》** `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 0. 状态

> 状态：✅ **接入阶段完成** —— 环境打通 ✅ / 镜像定档 ✅ / 后端落地 ✅ / **conformance 13/13 + 6/6 全绿**（2026-09-22 真机）；
> 剩余：**推理腿（前向 + 服务化）**未做（前置已就绪，无阻塞）。
> 完整方案与真机执行手册：**[`docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md`](docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md)**

| 阶段 | 状态 | 结果 |
|---|---|---|
| 阶段 0 · 环境普查 | ✅ **完成（09-22）** | 两台测试机：各 **8 × MLU590-M9（96 GB/卡）**、128 核 / 2 TB 内存、11T 数据盘挂在 `/srv`；**验收模型已在共享 HF 缓存**（`Qwen3-Embedding-0.6B` 快照 `97b0c614…`，只读复用） |
| 阶段 0b · 镜像渠道与定档 | ✅ **完成（09-22）** | 定档 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`（digest `sha256:e55b420e…`）；**实测可匿名拉取** |
| 阶段 0c · **环境开通 + 起容器** | ✅ **完成（09-22）** | `hliu553` 入 `docker` 组、`/srv/hliu553` 可写；拉定档镜像（**digest 实测与定档一致**）并起容器 `dc-mlu590-hliu553` |
| 阶段 1 · 接入（`backends/cambricon/`） | ✅ **完成（09-22）** | 13 抽象 + `build()` + `supports()` 如实声明 + `known_issues()`；**离线自检 39/0（按当前原型复跑；落地时为 34/0）、smoke 42/0**；能力声明已按真机证据更新 |
| 阶段 2 · conformance | ✅ **完成（09-22）** | **13/13 + 6/6 全绿（`CONFORMANCE_PASS`）** —— 接入完成的判定线已达成 |
| 阶段 3 · **多流 16 项基线** | ✅ **完成（09-22）** | **15 通过 / 1 不适用 / 0 不支持**；探针 **`STREAM_SEMANTICS_PASS 8/8`**（双卡，含 S-13）、图捕获 **5/5**、S-16 配额 **2000 流 3/3** ⇒ 报告 `docs/CAMBRICON_MLU_STREAM_BASELINE_16_20260922.md` |
| 阶段 4 · **训练腿** | ✅ **完成（09-22）** | 2 卡 DDP + **`cncl`** + 三类通信对照：**`TRAIN_LEG_PASS 6/6`**、loss **15.4498 → 11.1479**（50 步，无 NaN）、**2957.8 tok/s**（卡 0,2）。⚠️ CNCL 未加载 `libibverbs`/`libmlx5` ⇒ 走 **MLU_LINK 片间互联**，**非 RDMA**（已如实标注） |
| 阶段 5 · **错误闭环** | ✅ **完成（09-22）** | 四类注入 **`ERROR_RECOVERY_LOOP_PASS`（闭环 5 / 跳过 0 / 失败 0）**，记录自带 `expectation`/`expect_matched` |
| 阶段 6 · 收敛 | 🔄 进行中 | 接入方案 + 16 项基线已产出；本轮另挖出 **3 个跨后端缺陷 + 2 处证据污染**（含 910C 侧），见 `../prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md` |

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
   我们虽在 `sudo` 组但 **sudo 需密码** ⇒ 必须由 root 开通（见 §4）。
2. **docker 镜像数据无需搬**：`/var/lib/docker` 就是 `/srv/var/lib/docker` 的符号链接，
   镜像与容器层本已落在 11T 盘上 ⇒ **不要改 `daemon.json` 的 `data-root`**（改动要重启 docker，风险大于收益）。
3. **不在 `docker` 组**（组员：`gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu`）
   ⇒ 当前**无法使用 docker CLI**，而我们的验证流程全部在带卡容器内 ⇒ 这是与第 1 条并列的硬阻塞。
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

⚠️ **待实机复核**（进容器后）：`import torch_mlu` → `torch.mlu.device_count() == 8`，
并核对实际栈版本是否与 `neuware4.4.3` 档一致。

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

- ✅ **推理形态与昇腾同类**：寒武纪有厂商移植版 vLLM（官方开源 `Cambricon/vllm-mlu`）；
  官方应用镜像 `flagos-app/vllm0.24.0-cambricon-neuware4.x.x:2.2.0-0.3.0rc2.post2` 亦已存在
- ⚠️ **仍需实测（不臆断）**：① 驱动 6.2.29 能否跑标 6.2.15 的 4.4.3；② 带卡机能否出网拉 `harbor.baai.ac.cn`；
  ③ 应用镜像内 vLLM 是厂商移植版还是社区版 + 插件

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
| `docs/`（后续） | 根因核对、阶段验证报告（对齐 `../P800/docs/` 体例） |
| `probes/preflight_env_mlu1_20260922.log` | **Mlu-1 环境普查原始日志**（`preflight_env.sh` 首跑产出） |
| `probes/preflight_env_mlu2_20260922.log` | **Mlu-2 环境普查原始日志** |
| `probes/.gitignore` | `!*.log` 例外（否则根 `.gitignore` 的 `*.log` 会让证据静默不入库） |

---

## 6. 下一步

**已完成（代码层）**：`backends/cambricon/` 已落地（13 抽象 + `build()` + `supports()` 如实声明 +
`known_issues()`），并新增**离线契约自检**工具（当前原型 **39/0** 通过）。

**剩余（真机；权限已开通，剩推理腿两形态）** —— 按 [`docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md`](docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md) §5 的 A1–A10 执行：

```text
① 拿到 root 开通（§4 第 1、2 条）→ 复核 /srv/hliu553 与 docker 组
② docker pull 定档镜像 → 起带卡容器（A1，参数已给全）
③ 环境普查复跑（A2）→ 厂商栈判别（A3）→ 镜像就绪 5 条判据（A4）
④ 容器内离线自检 + smoke + conformance 13 + 6（A5）
⑤ ⚠️ 先探测集合通信后端名（A5b）→ 多流 16 项（A6）→ 训练腿 2 卡（A7）
⑥ 推理腿前向 + 服务化（A8）→ 错误闭环（A9）→ 归档回填（A10）
⑦ 产出并入接入手册 SOP + 接口约定修订建议
```

**复用现有资产**：`../prototype/scripts/preflight_env.sh`（环境普查）· `../prototype/scripts/backend_offline_check.py`（**离线契约自检，无设备可用**）· `../prototype/probes/probe_stream_semantics_full.py`（多流 16 项探针，后端无关 V2）· `../prototype/scripts/serve_standard.sh`（服务启动，`DC_BACKEND=cambricon`）· `../prototype/runtime/conformance/`（判据集，三后端共用同一套）
