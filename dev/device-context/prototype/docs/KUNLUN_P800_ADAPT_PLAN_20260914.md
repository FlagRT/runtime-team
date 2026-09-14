# 昆仑芯 P800 适配工作方案（2026-09-14）

> 目标：把 910C 上已验证的**设备上下文 + 多流 Stream** 能力适配到昆仑芯 P800，
> **基于既有统一原型接入**（`prototype/runtime/backends/` 插件机制），不另起炉灶。
> 状态：方案已完成；**实际运行部分待 P800 可访问后执行**（见 §7 阻塞）。

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

- **带卡容器并发上限**：910C 为 3（超限 `acl.init()` 返 500000）。昆仑芯是否有同类限制？**待实测并记录**。
- **设备可见性机制**：`ASCEND_RT_VISIBLE_DEVICES` 对应物（XPU 侧环境变量）；
- **有界同步能力**：910C 有（pyACL `synchronize_stream_with_timeout`），FlagOS 无 —— 昆仑芯待实测；
- **驱动/SDK 版本锁定**：宿主机 `xpu-smi` 与容器内版本差异（指引显示宿主机 5.0.21.47 / 容器 515.58，属正常分层）。

---

## 4. 执行步骤（任务 2 → 任务 3）

1. **环境信息汇总**（任务 1，阻塞中）：内存与占用、专用数据盘、CPU 架构、XPU 设备与驱动、Docker 可用性；
2. **起容器 + 装组件**（任务 2）：
   - 镜像：`harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（pull 59.9GB / load 包 32GB）
   - 启动参数按指引（`--privileged`、`--device=/dev/xpu0..7`、`/dev/xpuctrl`、`--shm-size=256g`、挂 `/data` `/home`）
   - flagtree：`pip install flagtree===0.7.0rc1+xpu3.6 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple`（**先卸干净 triton**）
   - FlagGems：源码 `pip install .[kunlunxin] -i https://pypi.tuna.tsinghua.edu.cn/simple`
3. **接入原型**：新增 `kunlun` 后端 → 跑 `smoke_runtime.py` 与 conformance（先看接口完备度）；
4. **最小分布式训练**：跑起来并记录失败点 → 按 §1.3 规则逐个判定归属；
5. **产出**：适配结果 + 缺失项清单（含归属）+ STATUS.md 更新 + 对外提交单。

---

## 5. 交付物清单

| 交付物 | 位置 |
|---|---|
| 昆仑芯后端实现 | `prototype/runtime/backends/kunlun/` |
| conformance 结果 | `prototype/runtime/conformance/kunlun_*.json` |
| 适配记录（含缺失项与归属） | `prototype/docs/KUNLUN_ADAPT_RECORD_<date>.md` |
| 本方向状态更新 | `dev/device-context/STATUS.md`（每周三） |
| 对外提交单（非我方项） | 按各子方向渠道 |

---

## 6. 尚未确认、需在实测中回答的问题

1. 昆仑芯是否提供**设备级重置/重建**原语（影响 `recover_device` 的 real 模式）；
2. XPU Stream/Event 是否具备**跨流依赖**与**有界等待**能力；
3. XPU 错误码体系（用于新建 `translate_error` 映射表）；
4. 单机多卡规模与卡间通信方式（用于训练腿验证设计）；
5. 是否同样存在"带卡容器并发上限"（影响使用规则，若存在需提总组入 v1）。

---

## 7. 阻塞（截至 2026-09-14 09:30）

**P800（43.180.254.67）SSH 22 端口 `Connection refused`**：
- DNS/路由正常：`ping` 通（123ms，0% 丢包）
- TCP 层面：22、2222、2022、8022、10022、80、443 均不可达（`refused`）
- `~/.ssh/config` 中 P800 配置存在（`HostName 43.180.254.67`、`User hliu553`，未指定端口与密钥）

推断为三类之一：① 主机 SSH 服务未启动；② 云安全组/防火墙未放行来源 IP；③ 需经跳板或内网访问。
**任务 1/2 需待访问方式确认后执行**；任务 3 的方案部分（本文）已完成。
