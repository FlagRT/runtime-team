# 寒武纪 MLU590 · 第三实例看板

> 分支：`kistich/device-context` ｜ 更新：2026-09-22 ｜ 负责人：Kistich（hliu553）
> **本目录 = 第三实例（寒武纪 思元 MLU590）的芯片专属资产**；
> 芯片无关的规范与原型在 `../prototype/`，前两实例在 `../910C/`、`../P800/`。
> 接入方法见 **《新芯片接入手册》** `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`。

---

## 0. 状态

> 状态：🔄 **进行中**（**第 0 步环境普查已完成 ✅；接入动作待环境开通** —— 见 §4）

| 阶段 | 状态 | 结果 |
|---|---|---|
| 阶段 0 · 环境普查 | ✅ **完成（09-22）** | 两台测试机：各 **8 × MLU590-M9（96 GB/卡）**、128 核 / 2 TB 内存、11T 数据盘挂在 `/srv`；**唯一阻塞是 root 权限** |
| 阶段 1 · 接入（`backends/cambricon/`） | ⛔ 待环境开通 | 目标：13 抽象 + `build()` + `supports()` + `known_issues()`；**单芯片 ≤5 人天** |
| 阶段 2 · conformance | ⏳ | 13 例 + 推理 6 例（全绿或如实 stub-skip） |
| 阶段 3 · 多流 16 项基线 | ⏳ | 探针 `../prototype/probes/probe_stream_semantics_full.py`（后端无关 V2） |
| 阶段 4 · 训练腿 | ⏳ | 2 卡 DDP + FlagCX(CNCL) + 三类通信对照 |
| 阶段 5 · 收敛 | ⏳ | 产出并入接入手册 SOP + 接口修订建议 |

**预期收益**：`device_type="mlu"` 是**第三种设备命名空间**（前两种为 `npu` / `cuda`），
是接口约定修订建议**第 1 条（`device_type` 与 `vendor` 分离）的首次真实验证场景**；
且寒武纪有**真正的厂商错误码体系（CNRT）**，预期可做出比 P800 更完整的 `error_map`。

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

## 3. 环境版本组合（由他人脚本间接确认，待实机复核）

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

⚠️ **待实机复核**（进容器后）：`import torch_mlu` → `torch.mlu.device_count() == 8`。

**镜像从哪来 → 见 `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md`**（09-22 调研 + **同日下午更正**）：
- ❌ **FlagTree 没有寒武纪 User Manual / 推荐镜像**（wiki 26 页无 cambricon 条目；寒武纪只存在于编译器侧 `triton_v3.2.x` 分支）
- ✅ **但 FlagOS 官方在 BAAI Harbor 上已有寒武纪镜像**（`flagos-base` / `flagos-runtime` / `flagos-app` 共 12 个仓
  + FlagGems 周测 2 个仓），且**实测可匿名拉取**（Registry v2 匿名 token 取 manifest 成功，digest 已取得）
  ⇒ **不需要寒武纪私仓凭据**
- ⭐ **档位按宿主驱动选，不按 torch 版本选**（实测驱动 **v6.2.29**，落 6.2.x 线）：

  | 档位 | py | torch / torch-mlu / triton | 官方标注宿主驱动 | 我们 |
  |---|---|---|---|---|
  | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` | 3.10 | 2.7.1+cpu / 1.29.2 / 3.2.0+mlu1.7.2 | **6.2.15** | ✅ **选它** |
  | `…/flagos-runtime-cambricon-neuware4.7.2:2.2.0` | 3.12 | 2.11.0+cpu / 1.33.1 / 3.4.0+mlu2.1.1 | **6.5.48** | ❌ 需升宿主驱动 |

- ✅ **推理形态与昇腾同类**：寒武纪有厂商移植版 vLLM（官方开源 `Cambricon/vllm-mlu`）；
  官方应用镜像 `flagos-app/vllm0.24.0-cambricon-neuware4.x.x:2.2.0-0.3.0rc2.post2` 亦已存在
- ⚠️ **仍需实测（不臆断）**：① 驱动 6.2.29 能否跑标 6.2.15 的 4.4.3；② 带卡机能否出网拉 `harbor.baai.ac.cn`；
  ③ 应用镜像内 vLLM 是厂商移植版还是社区版 + 插件

---

## 4. 阻塞与需要协调的事项

| # | 事项 | 需要谁 | 状态 |
|---|---|---|---|
| **1** | 建 **`/srv/hliu553`** 并 chown 给 `hliu553`（两台） | root / 机器管理员 | 🔴 **阻塞** |
| **2** | 把 `hliu553` 加入 **`docker` 组**（两台，需重新登录） | root / 机器管理员 | 🔴 **阻塞** |
| **3** | ~~**镜像获取**~~ → **已解除**（2026-09-22）：FlagOS 官方 BAAI Harbor 已有寒武纪三代镜像 + 周测镜像，**实测可匿名拉取**（digest 已取得），**不需要私仓凭据**。**现在待裁定的是「走哪一档」**：驱动 6.2.29 → `neuware4.4.3`（标 6.2.15）；`neuware4.7.2` 标 **6.5.48**，需先升宿主驱动 | 总组 / 管理员 | 🟢 **降级为档位裁定**（见 `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0） |
| 4 | 两机**无共享目录**：数据需分别放置；若需共享需另配 NFS | 管理员（可选） | ⚪ 已知 |

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
| `docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` | **镜像获取渠道调研**：FlagTree 无寒武纪手册（26 页证据）· 官方渠道与三私仓实测（DNS/端口/401/鉴权类型）· 官方镜像命名规律 · 目标版本档 · 申请清单 · 三家实例获取路径对照 |
| `docs/`（后续） | 接入方案、根因核对、阶段验证报告（对齐 `../P800/docs/` 体例） |
| `probes/preflight_env_mlu1_20260922.log` | **Mlu-1 环境普查原始日志**（`preflight_env.sh` 首跑产出） |
| `probes/preflight_env_mlu2_20260922.log` | **Mlu-2 环境普查原始日志** |
| `probes/.gitignore` | `!*.log` 例外（否则根 `.gitignore` 的 `*.log` 会让证据静默不入库） |

---

## 6. 下一步

```text
① 拿到 root 开通（§4 第 1、2 条）→ 复核 /srv/hliu553 与 docker 组
② 确认镜像（§4 第 3 条）→ 起第一个带卡容器
③ 容器内确认：torch_mlu 可导入 + torch.mlu.device_count() == 8
④ 落实 backends/cambricon/（name="cambricon" / device_type="mlu" / vendor="cambricon"）
⑤ conformance 13 + 6 → 多流 16 项逐项比对 → 训练腿 2 卡 DDP（FlagCX + CNCL）
⑥ 产出并入接入手册 SOP + 接口约定修订建议
```

**复用现有资产**：`../prototype/scripts/preflight_env.sh`（环境普查）· `../prototype/probes/probe_stream_semantics_full.py`（多流 16 项探针，后端无关 V2）· `../prototype/scripts/serve_standard.sh`（服务启动）· `../prototype/runtime/conformance/`（判据集，三后端共用同一套）
