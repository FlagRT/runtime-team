# 昆仑芯 P800 适配工作方案（2026-09-14）

> 目标：把 910C 上已验证的**设备上下文 + 多流 Stream** 能力适配到昆仑芯 P800，
> **基于既有统一原型接入**（`prototype/runtime/backends/` 插件机制），不另起炉灶。
> 状态：**连接已打通**（见 §7.0）；**实际运行部分因账号权限不足暂停**（见 §7.1）。
> 环境基线实测数据见 [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md)。

---

## 1. 我方职责边界（先把"我们负责什么"说清）

### 1.1 我方负责（对应 910C 已交付的能力）

| 域 | 内容 | 昆仑芯需实现的对应项 |
|---|---|---|
| **设备抽象** | `device_count` / `set_device` / `memory_stats` / `probe_device` | XPU 设备枚举（`xpu-smi` / SDK 接口）、显存查询、探活 |
| **多流 Stream** | `create_stream` / `create_event` / `current_stream` / `stream_context` / `synchronize` / `synchronize_stream` / `wait_event_host`，及 16 项流语义子项 | XPU Stream/Event 封装；**有界同步**、跨流可见性、错误隔离分层需逐项实测 |
| **错误码翻译** | `translate_error`：错误 → L1–L4 分级 + `disposition` | 昆仑芯错误码需**新建映射表**（910C 的 108 条不适用） |
| **状态恢复** | `recover_device(probe/real/hybrid)` | XPU 是否提供设备级重置原语，待实测 |
| **插件机制（我们起头）** | `RuntimeBackend` 抽象基类 + 注册表 + 统一 conformance | 新增 `kunlun` backend；**新增芯片 = 实现 backend + 跑通 conformance** |

**接入契约**：实现 13 个 `@abstractmethod`（上表五域）+ 可选 `stream_priority_range()`；
经 `supports()` 如实声明能力边界（不支持项不伪造，conformance 会如实跳过）。

> **已对着代码核实（2026-09-14）**：`prototype/runtime/backends/base.py` 实测**恰好 13 个 `@abstractmethod`**，
> 与上表五域一一对应 —— 设备抽象 4（`device_count` / `set_device` / `memory_stats` / `probe_device`）、
> 多流 Stream 7（`create_stream` / `create_event` / `current_stream` / `synchronize` /
> `stream_context` / `synchronize_stream` / `wait_event_host`）、
> 错误码翻译 1（`translate_error`）、状态恢复 1（`recover_device`）。
> 且 `backends/registry.py` 的 `_KNOWN_BACKENDS` **已预留 `("ascend", "flagos", "kunlun")`** ——
> 昆仑芯的接入位在原型设计时已留好，本次属**填位**而非改造。
> conformance 侧现状：`cases.py`（13 例）、`infer_cases.py`（6 例），
> 已有结果集 `conformance_runtime_ascend.json`（13）、`conformance_runtime_infer.json`（6）、`conf_proto_flagos_13.json`（13）。

### 1.2 不属我方（识别到即**对外提交**，不自行适配）

| 缺失项类型 | 归属子方向 | 我方提供什么 |
|---|---|---|
| 算子缺失/精度问题（FlagGems、flagtree 算子） | 算子适配方向 | 复现脚本 + 失败用例 + 参数 |
| 集合通信（xccl / flagcx 在 XPU 上的实现） | 分布式方向 | 通信对照失败记录（哪类操作、什么错误） |
| 显存池 / 缓存管理 | 显存与缓存管理方向 | 显存峰值与碎片观测数据 |
| 任务调度 / 执行引擎 | 任务调度方向 | 调度行为异常记录 |
| 监控诊断 / 多级恢复编排 | 监控诊断方向 | 设备侧分级与恢复原语结果 |
| 精度与性能分析（Oracle-Device / Profiling） | 精度与性能方向 | 统一口径的性能与精度原始数据 |
| 镜像 / 基座（驱动、SDK 版本、镜像发布） | 总组 / 镜像负责方 | 环境信息汇总 + 复现条件 |

### 1.3 归属判定规则（跑训练时逐条套用）

1. 失败点落在 `RuntimeBackend` 五域内 → **我方适配**；
2. 在五域之外 → **对外提交**，提交单须含：现象、最小复现、错误原文、初步定位证据；
3. 无法判定 → 先记入缺失项清单，在下次周会（或每周三 STATUS.md）提请定位。

---

## 2. 接入方式（复用既有原型，不另起炉灶）

```
prototype/runtime/backends/
├── ascend/     # 910C 推理腿（torch_npu）——已验证 13/13 + 6/6
├── flagos/     # 910C 训练腿（torch_fl）——已验证 13/13
└── kunlun/     # 【本次新增】昆仑芯 XPU
```

- **代码复用**：`errors.py`（统一错误对象与分级）、`recovery.py`（恢复五段式）、
  `device_state.py`（四态）、`api/stream.py`（Stream/Event 统一封装）**直接复用**，
  昆仑芯只需实现设备与流事件的原生映射。
- **conformance 复用**：`runtime/conformance/runner.py --backend kunlun` 直接跑同一套
  13 例 + 推理 6 例，作为接入完备度的度量器（910C 上 FlagOS 后端从零到 13/13 即先例）。
- **不修改**既有后端与统一 API（插件式，互不影响）。

---

## 3. 分布式训练与推理的验证要求

### 3.1 验证矩阵（沿用 910C 口径，可量化）

| 层级 | 项目 | 判据 |
|---|---|---|
| 组件自检 | `smoke_runtime.py` | 通过项 / 总项（910C: 37/37） |
| 接口完备度 | conformance 13 例 + 推理 6 例 | N/N（跨后端可跑） |
| 多流语义 | 16 项流语义子项 | 逐项 ✅/如实跳过（**有界同步/跨流可见性/错误隔离分层**为重点） |
| 训练腿 | 最小分布式训练 | 通信对照项数（all_reduce / all_gather / P2P 各 1 组）、loss 单调下降、无 NaN、吞吐（tok/s） |
| 推理腿 | 单卡推理（前向或服务化） | 向量维度/无 NaN/语义区分度、吞吐与 p50 时延 |
| 错误闭环 | 注入 → 分级 → 恢复 | 闭环项数 / 总项（910C: 推理腿 5、训练腿 4） |

### 3.2 需要一并实测的环境约束（对标 910C 踩过的坑）

#### 3.2.1 已完成实测的部分（2026-09-14，只读探测）

| 项 | 910C 基线 | P800 实测 | 影响 |
|---|---|---|---|
| 驱动/SDK 版本 | CANN 9.0 | 宿主 `xpu-smi` **5.0.21.47** / XPU-RT 5.0.21；内核模块 `kunlun` 5.0.21 | 版本已锁定，可写入约束清单 |
| 设备可见性 | `ASCEND_RT_VISIBLE_DEVICES` | `/dev/xpu0..7` 全局 `crw-rw-rw-`；**未见等价环境变量**（`/etc/profile.d` 无 XPU 配置） | 待容器内确认 XPU 侧变量名 |
| 卡间互联 | HCCL / RoCE | **XPU0-3（NUMA0）、XPU4-7（NUMA1）组内 XL 私有链路；跨组 SYS** | 训练腿规模设计：**优先组内配对**可避开跨 NUMA |
| 网卡与卡的亲和 | — | **NIC0-3 ↔ XPU0-3、NIC4-7 ↔ XPU4-7 均为 PIX**，200 Gb RoCE，`PORT_ACTIVE` | 多卡通信路径干净，优于 910C 侧条件 |
| 网卡直访显存能力 | 910C 待查（HIXL 是否等价 GDR） | **`kunlun_peermem` 模块已加载** | 昆仑芯侧存在类 GDR 原语 → **纳入跨芯片原语调研** |
| 显存容量 | 910C 64 GB | **96 GB × 8** | 已验证模型规模无压力 |
| 宿主机内存 | — | **1.5 TiB**（无 swap） | 充裕 |
| CPU | — | 2× EPYC 9K84 = **384 线程**，2 NUMA | 充裕 |
| 机器共享程度 | 独占 | **25 人在线、22 容器 shim、272 僵尸进程** | **共享机**，容器须可识别、可回收 |
| **权限** | hliu553 可用 docker | **不在 `docker` 组；`sudo` 需密码** | ❌ 阻塞（§7.1 B1） |
| **镜像落盘** | 数据盘 11 T | `/var/lib/docker` **已 bind mount 到 `/data1`（5.8 T NVMe），剩 1.5 T** | ✅ **充足**（初版曾误判为阻塞） |
| **工作目录** | 数据盘可写 | `/data1`、`/data2` 顶层**均不可写** | ❌ 阻塞（§7.1 B2） |

#### 3.2.2 仍待实测（必须在容器启动后回答）

1. **带卡容器并发上限**：910C 为 3（超限 `acl.init()` 返 500000）。昆仑芯是否有同类限制？**待实测并记录**；
2. **有界同步能力**：910C 有（pyACL `synchronize_stream_with_timeout`），FlagOS 无 —— 昆仑芯待实测；
3. XPU 侧「可见设备」环境变量名与语义；
4. 容器内 `xpu-smi` 版本（指引称容器内为 515.58，属正常分层，需实测确认）。


---

## 4. 执行步骤（任务 2 → 任务 3）与当前进度（截至 2026-09-14 11:00）

| # | 步骤 | 状态 | 说明 |
|---|---|---|---|
| 1 | **环境信息汇总**（任务 1） | ✅ **已完成** | 见 [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md) |
| 2 | **起容器 + 装组件**（任务 2） | ✅ **已完成** | 权限已解（docker 组 + `/data2/hliu553`），容器 `hliu553-device-context-p800` 运行中 |
| 2a | └ 镜像获取 | ✅ **无需拉取** | 本机镜像库已有 `flaggems-main-dev:202608`（38.3 GB）与 `ubuntu22.04:202606-base`（34.7 GB）→ 手册的 59.9 GB pull / 32 GB load **全部跳过** |
| 2b | └ 容器启动 | ✅ **已完成** | 用本机**已跑通容器的等价配置**（非 privileged / bridge / `--shm-size=64g` / `/dev/xpu0..7`+`xpuctrl`+`fuse`），挂 `/data2/hliu553:/workspace` |
| 2c | └ flagtree | ✅ **xpu3.6 已满足** | 镜像内含 **`flagtree 0.6.1+xpu3.6`**；手册最新为 `0.7.0rc1+xpu3.6` → **升级列为可选单变量实验**，非阻塞（见 §8） |
| 2d | └ FlagGems | ✅ **已满足** | 镜像内含 `flag_gems 5.3.4.post1.dev12`，**算子级实测通过**（`add` max diff = 0.0）；源码已在容器内 `/env/FlagGems`（github clone），**无需联网** |
| 2e | └ **五域基线实测** | ✅ **已完成** | 见 [`KUNLUN_P800_BASELINE_PROBE_20260914.md`](KUNLUN_P800_BASELINE_PROBE_20260914.md)；识别出 **1 缺陷 + 1 约束 + 1 设计依据** |
| 3 | **接入原型**：新增 `kunlun` 后端 → 跑 `smoke_runtime.py` 与 conformance | ⏳ **待方案确认** | 遵循「方案确认后再实现」，**尚未动代码** |
| 4 | **最小分布式训练**：跑通并记录失败点 → 按 §1.3 判定归属 | ⏳ 待执行（**前置已解除**） | 这是**识别缺失项**的主手段；可立即启动 |
| 5 | **产出**：适配结果 + 缺失项清单（含归属）+ STATUS.md 更新 + 对外提交单 | 🔄 **进行中** | 缺失项清单已出 3 条（§3），对外提交单 2 张待起草 |

> **关于「先简单运行分布式训练以识别缺失项」**：步骤 2 已打通（容器运行中、镜像与组件均就绪），
> 该步骤的**前置已解除，可立即启动**。本报告已先通过**单卡五域探针**取到第一批缺失项（§3），
> 分布式训练将补充**多卡与集合通信**维度的缺失项。


---

## 5. 交付物清单

| 交付物 | 位置 |
|---|---|
| 昆仑芯后端实现 | `prototype/runtime/backends/kunlun/` |
| **探针脚本与原始结果** | `dev/device-context/probes/kunlun/`（`dc_probe_p800.py`、`dc_probe_isolated.py` + 两份 json） |
| **基线实测报告** | `prototype/docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` |
| conformance 结果 | `prototype/runtime/conformance/kunlun_*.json` |
| 适配记录（含缺失项与归属） | `prototype/docs/KUNLUN_ADAPT_RECORD_<date>.md` |
| 本方向状态更新 | `dev/device-context/STATUS.md`（每周三） |
| 对外提交单（非我方项） | 按各子方向渠道 |

---

## 6. 尚未确认、需在实测中回答的问题

| # | 问题 | 状态 |
|---|---|---|
| 1 | 昆仑芯是否提供**设备级重置/重建**原语（影响 `recover_device` 的 real 模式） | ⏳ 待实测 |
| 2 | XPU Stream/Event 是否具备**跨流依赖**与**有界等待**能力 | ⏳ 待实测 |
| 3 | XPU 错误码体系（用于新建 `translate_error` 映射表） | ⏳ 待实测 |
| 4 | 单机多卡规模与卡间通信方式 | ✅ **已回答**：单机 8× P800（96 GB/卡）；组内 XL 私有链路、跨组 SYS；另有 8× 200 G RoCE，NIC 与卡 PIX 直连，`kunlun_peermem` 已加载 |
| 5 | 是否同样存在"带卡容器并发上限" | ⏳ 待实测（910C 为 3，超限 `acl.init()` 返 500000）；若存在需提总组入 v1 |


---

## 7. 阻塞

### 7.0 连接阻塞：✅ 已解决（2026-09-14 09:39）

**根因是端口，不是 VPN/防火墙/沙箱。**

| 项 | 内容 |
|---|---|
| 真实连接方式 | `ssh hliu553@43.180.254.67 -p 26008`（**非标准端口**） |
| 之前失败原因 | `~/.ssh/config` 的 `Host P800` **缺 `Port` 行** → `ssh P800` 默认走 22 → `Connection refused`；且此前只探测了 22/2222/2022/8022/10022/80/443 等常见端口，恰好未试 26008 |
| 佐证 | TCP 实测 `43.180.254.67:26008 → connect_ex=0 (OK)`；`:22 → connect_ex=61 (FAIL)` |
| 已修复 | `~/.ssh/config` 的 `Host P800` 增加 `Port 26008` 并加注释；`ssh -G P800` 验证生效 |
| 免密登录 | 已由使用者执行 `ssh-copy-id -i ~/.ssh/id_ed25519.pub P800`，之后可 `BatchMode` 免密自动化 |
| 经验教训 | **连不上的第一件事是核对端口与实际生效配置（`ssh -G <host>`），而不是怀疑网络策略** |

### 7.1 资源阻塞：❌ **当前阻塞任务 2**（截至 2026-09-14 10:00）

XPU / 内存 / CPU / **镜像落盘空间**均已具备，瓶颈**只在账号权限**。详见环境报告的 §5.3、§9–§10。

| # | 阻塞项 | 现状 | 需要的动作 |
|---|---|---|---|
| **B1** | **无 docker 权限** | `hliu553` 不在 `docker` 组（组内仅有 `xliu969`、`daizijian`）；`docker ps` → `permission denied`；`/var/run/docker.sock` 为 `srw-rw---- root docker`；`sudo` 需密码 | `sudo usermod -aG docker hliu553`，之后**新开 SSH 会话**生效 |
| **B2** | **无自有工作目录** | `/data1`、`/data2` 顶层均 `root:root 755`，`hliu553` 不可写；`/data` 是根分区上的普通目录（非挂载点） | `sudo mkdir -p /data2/hliu553 && sudo chown hliu553:hliu553 /data2/hliu553` |

**两项落实后任务 2 即可启动。**

> **已撤销的阻塞项（初版误判，保留记录）**
> 初版曾列 **「B2 无镜像落盘空间」**，理由是「docker data-root 在只剩 2.9 GB 的根分区」——
> **该判断错误**。`findmnt -T /var/lib/docker` 显示：
> `/var/lib/docker → /dev/nvme0n1p1[/xianghuang/docker-data/docker]`，
> 即 **该机器早已把 docker 数据目录 bind mount 到 5.8 TB 数据盘上，尚有 1.5 TB 可用**。
> 误因：拿 `du -xhd1 /` 的可读合计（11 G）与 `df` 的已用（91 G）求差并归因给 docker，
> 而 `du -x` 遇跨文件系统即停止，`/var/lib/docker` 本就不该出现其中。

### 7.2 需要提请注意的既有资产与版本差异

- 本机已有 **`/data1/dinghaisong/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04.202606-base.tar.gz`**
  （33.5 GB，**全局可读**，gzip 校验通过）→ 可省去 59.9 GB 联网 pull；
  但版本为 **`202606-base`**，官方手册当前给出 **`202608-base`**，
  **使用前须确认 `202606` 是否满足 xpu3.6 要求，不可默认等同**。
- 共享模型缓存 `/data1/dinghaisong/hf_cache`（1.7 T，可读）中已含
  **Qwen3-Embedding-0.6B（1.2 G，实测可读）** —— 正是原型验收模型，
  训练腿/推理腿可直接复用，无需重新下载。
- **`/data1/songchao` 为 `drwxrwxrwx`（他人目录但全局可写）**，技术上可作为临时落脚点，
  **但不建议占用他人目录**；如确需使用须先向该目录属主说明。

---

## 8. 可选实验（非阻塞）：flagtree 0.6.1 → 0.7.0rc1

**现状**：镜像 `flaggems-main-dev:202608` 内含 **`flagtree 0.6.1+xpu3.6`**（xpu3.6 后端已满足），
且已实测**算子级可用**（FlagGems `add` max diff = 0.0）。官方手册的免源码安装行给出的是
**`flagtree===0.7.0rc1+xpu3.6`**。

**为什么不立即升级**：镜像的 `flagtree + flag_gems + torch` 是一套**被验证过的组合**，
贸然升级会把「接入验证」与「版本升级」两个变量混在一起，违反单变量原则。

**升级实验设计（需要时再跑，同样单变量）**

```bash
# 与基线隔离：先在容器内快照现状，再升级，复跑同一探针对比
python3 -m pip freeze > /workspace/pin_before_$(date +%Y%m%d_%H%M).txt
python3 -m pip uninstall -y triton          # 反复执行至卸净
python3 -m pip install flagtree===0.7.0rc1+xpu3.6 \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
# 复跑同一探针，逐项对比 /workspace/dc_probe_isolated_result.json
```

**判据**：① 8 卡仍可见；② FlagGems 算子 max diff 仍为 0；③ **流优先级缺陷是否消失**（§3.1）；
④ 无新增报错。任一不满足即回滚镜像（容器可重建，回滚成本为零）。

**来源**：官方手册 [User manual for xpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu)；
网络实测 `resource.flagos.net` 与 `pypi.tuna.tsinghua.edu.cn` 在容器内**可达（HTTP 200）**，
但 **`github.com` 不可达**（超时）→ 故 FlagGems **不重新 clone**，改用容器内已有的
`/env/FlagGems`（本就来自 `git clone https://github.com/flagos-ai/FlagGems`，HEAD `73c5aff1`）。
