# 寒武纪 MLU590 接入方案（第三个芯片实例）

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文（device-context）
> 性质：**接入工作方案 + 真机执行手册**（本实例专属，不迁移）
> 依据：《新芯片接入手册》`../../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`（8 步流程）
> 　　　《运行时层接口约定》`../../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2（Backend 插件接入规范）
> 定位：**按接入规范新建实例** —— 迁移的是**规范与方法**；910C / P800 的**实现与结论不迁移**。

---

## 0. 当前进度（结论先行）

**⭐ 2026-09-22：真机环境已打通并完成接入验证 —— conformance 13/13 + 6/6 全绿。**

| 手册步骤 | 内容 | 状态 |
|---|---|---|
| — | 环境普查 + 镜像渠道调研与**定档** | ✅ **09-22** —— `CAMBRICON_MLU_ENV_REPORT_20260922.md` / `…IMAGE_CHANNEL…` |
| — | **环境开通**（`sudo` + `docker` 组 + `/srv/hliu553`） | ✅ **09-22 打通**（`hliu553` 已入 `docker` 组；Docker 25.0.3） |
| A1 | 拉定档镜像 + 起带卡容器 | ✅ **09-22** —— digest 实测 `sha256:e55b420e…` **与定档记录一致** |
| A2–A4 | 厂商栈判别 + 镜像就绪 5 条判据 | ✅ **09-22** —— `torch_mlu` 可导入、`torch.mlu.device_count() = 8` |
| 第 3 步 | **实现 backend（13 抽象）** | ✅ **09-22** —— `prototype/runtime/backends/cambricon/` |
| 第 3.5 步 | **离线契约自检** | ✅ **35/0**（声明 `stream_priority` 后为 **34/0** —— 那条"未声明则须返回 None"的检查被如实跳过，非失败） |
| A5 | smoke 自检 | ✅ **42 通过 / 0 失败** |
| 第 4 步 | **conformance 13 例 + 推理 6 例** | ✅ **13/13 + 6/6 全绿（`CONFORMANCE_PASS`）** |
| A5b | 集合通信后端名探测 | ✅ **`cncl`**（2 进程 `all_reduce` 结果正确） |
| 第 5 步 | 两条腿（2 卡训练 / 单卡推理 + 服务化） | ⏳ 下一步（前置已全部就绪） |
| 第 6 步 | 错误闭环（四类注入） | ⏳ |
| 第 7 步 | 多流 16 项基线逐项比对 | ⏳ |
| 第 8 步 | 证据归档 + 验收清单 13 项 | 🟡 证据已入库 4 份 |

**⇒ 接入完成的判定线（手册第 5 步 = conformance 13+6）已达成。**
剩余为两条腿 / 错误闭环 / 多流 16 项，**前置全部就绪、无阻塞**。

### 0.1 真机实测结果汇总（2026-09-22，证据 `probes/A*_20260922.log`）

| 项 | 实测结果 | 对实现/声明的影响 |
|---|---|---|
| Python / OS | **3.10.20 / Ubuntu 22.04.5** | 与 `neuware4.4.3` 档文档一致 |
| torch / torch_mlu | **2.7.1+cpu / 1.29.2+torch2.7.1** | 与定档一致 |
| 设备可见 | `torch.mlu.device_count() = 8`，MLU590-M9，**94.8 GiB/卡** | 判据 1+2 通过 |
| 显存查询 | `mem_get_info(ordinal)` **可用**（(free,total) 同 CUDA 语义） | 后端取值路径 ① 即可，**降级路径 ③ 未被触发** |
| **未 record 的 `Event.query()`** | **原生返回 `True`（误报）** | ⇒ 适配层 E3 修正**是必需的**，不是防御性冗余 |
| **`Stream.synchronize(timeout_ms=…)`** | **不接受**（`TypeError`） | ⇒ 「有界」只能是**超时上报**语义（同昆仑芯口径，已如实标注） |
| `priority_range()` | **`(0, -3)`**，可用且**不崩** | ⇒ 已**声明** `stream_priority`（昆仑芯同款 API 会触发 PyTorch 断言，二者相反） |
| 图捕获 | 入口 `torch.mlu.MLUGraph` + `torch.mlu.graph`；**实测 5/5 成功** | ⇒ 已**声明** `graph_capture` |
| 设备级重置原语 | `reset*/destroy*/reinit*` **全是内存统计类** | ⇒ `recovery_real` **已确认不具备**（从"未验证"升级为结论） |
| 厂商错误码 | **不透出为数字码**：`CNRT error: invalid argument.`；OOM 为 `OutOfMemoryError: MLU out of memory…` | ⇒ `error_map` **如实不声明**；分级由 message_hint 覆盖（形状错→L2、OOM→L1，f1 已过） |
| 集合通信后端名 | **`cncl`**（`cpu:gloo,mlu:cncl` 亦可） | ⇒ 训练腿 `DC_DIST_BT=cncl` |
| 选卡变量 | `MLU_VISIBLE_DEVICES=2` → `device_count()` 由 8 变 1 ✅ | 统一启动脚本的 cambricon 分支写法得到验证 |
| `import triton` | **必须先 `import torch_mlu`**，否则 torch 的 device-backend 自动加载失败 | ⇒ 记为环境坑（`known_issues()` 已收录），**与 P800 的"缺 triton"不同因** |
| vLLM | 运行时层镜像**不含 vLLM** | ⇒ 推理腿服务化须用 `flagos-app/vllm*-cambricon-*` 应用镜像 |

**⚠️ 一处自我更正（方法论）**：集合通信探测首轮我把 `all_reduce` 判为"错误"（sum=12.0），
**是我判据公式写错了**——2×2 张量 × 两 rank（值 1 与 2）求和后每元素 = 3，
**总和应为 3×4 个元素 = 12.0**，即结果**本来就是正确的**。已更正，未把错误判断留在结论里。

---

## 1. 环境与阻塞（第 0 步结论摘要）

| 项 | 实测值 |
|---|---|
| 机器 | `Mlu-1` = 10.1.1.21（`tza-0a06-ai01-em9`）、`Mlu-2` = 10.1.1.22（`tza-0a06-ai02-em9`），SSH 免密可用 |
| 加速卡 | 各 **8 × MLU590-M9**，单卡 **96 GB**，健康 `Good` |
| 宿主驱动 / 固件 | **v6.2.29** / v1.5.0；`cnmon` = CNMON v6.2.29（宿主工具） |
| MLU 软件栈 | 宿主**无 `/usr/local/neuware`** ⇒ 软件栈必须走**容器镜像** |
| 数据盘 | `/srv` **11 T**（09-22 实测：`3.3 T 已用 / 6.7 T 可用`，33%）；docker 数据目录本就是 `→ /srv/var/lib/docker` 的符号链接 |
| 共享用户 | 同机他人：`gpfs, liangfan1, daizijian, huangxiang, qiyiyan, leihuhu` ⇒ **共享机用卡纪律照 P800 办** |
| **验收模型** | ✅ **共享 HF 缓存里已有**，无需下载：<br>`/srv/data/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`<br>（`drwxr-xr-x root root` 全局可读，**只读复用，不得写入**；与另两实例**同一模型** ⇒ 结果可比） |

**硬阻塞（需 root / 管理员，两台各一次）**：

```bash
sudo mkdir -p /srv/hliu553 && sudo chown -R hliu553:hliu553 /srv/hliu553 && sudo chmod 750 /srv/hliu553
sudo usermod -aG docker hliu553        # 执行后需重新登录 SSH 生效
```

**镜像定档**（已定，无需再申请）：

```text
harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0
digest sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2   (2.5 GiB)
栈：py3.10 / torch 2.7.1+cpu / torch-mlu 1.29.2+torch2.7.1 / torch-mlu-ops 1.8.0 / triton 3.2.0+mlu1.7.2
官方标注宿主驱动前置 6.2.15 —— 我们实测 v6.2.29，**同 6.2.x 线**
```

> 定档理由与另一档（`neuware4.7.2`，需宿主驱动 **6.5.48**）的处置见
> `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.1 定档决策 / §0.2 驱动升级上报预案。

---

## 2. 厂商栈判别（第 1 步，已给出**预期**与判别命令）

**✅ 实测结论（2026-09-22）：判定为「路径 C —— 厂商私有命名空间（PrivateUse1）」。**
证据：`torch_mlu` 导入 OK 且 `torch.mlu.device_count() = 8`；
`torch_npu` / `torch_fl` 均 `ModuleNotFoundError`（预期）；
`torch.cuda.is_available() = False`、`torch.cuda.device_count() = 0`（确认不是复用 cuda 命名空间）。
证据文件：`probes/A3A4_stack_probe_20260922.log` §B。

**原预测路径（现已被实测确认）：C —— 厂商私有命名空间（PrivateUse1）**
`torch_mlu` 把 MLU 注册为 PyTorch 的 PrivateUse1，设备串前缀 **`mlu`**：

```python
import torch, torch_mlu
torch.mlu.device_count()      # → 8
torch.device("mlu:0")         # → 有效设备串
```

故本后端 `name = "cambricon"`（厂商标识）、`device_type = "mlu"`（设备串前缀）——**两者不同名是刻意的**（后端名按厂商、设备串按命名空间），与前两家同一约定。

⚠️ **PrivateUse1 是进程级单例** ⇒ 本后端**不得与 `torch_npu` / `torch_fl` 同进程混用**。

**进容器第一件事就是跑这几条判别命令**（手册 §2 原文，逐条记录输出）：

```bash
python3 -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "import torch.npu;  print('npu:',  torch.npu.device_count())"   # 预期 ImportError（寒武纪无此栈）
python3 -c "import torch.mlu;  print('mlu:',  torch.mlu.device_count())"   # ← 预期 8
python3 -c "import torch.xpu;  print('xpu:',  torch.xpu.device_count())"   # 预期失败
python3 -c "from vllm.platforms import current_platform; print(current_platform)"
```

**最后一条决定推理腿形态**：打印具体平台 → 可直接起服务；打印 `UnspecifiedPlatform`
→ 认不出设备，必须另有厂商移植版 vLLM 或平台插件（寒武纪已知有厂商移植版 `Cambricon/vllm-mlu`，
但**镜像内是否已装、形态是哪一种，未验证**）。

---

## 3. 已完成的接入动作（第 3 步，代码层）

### 3.1 新增后端：`prototype/runtime/backends/cambricon/`

| 文件 | 内容 |
|---|---|
| `__init__.py` | 导出 `CambriconBackend` / `CambriconEventAdapter` / `build` |
| `backend.py` | **13 个抽象方法** + `build()` 工厂 + `supports()` 如实声明 + `known_issues()` + `info()` + `device_state()`；内含 `CambriconEventAdapter`（事件语义适配） |

**五域覆盖**：设备（count / set_device / memory_stats / probe）/ 内存 / 流-事件（create / current / context / 有界同步 / 有界主机等待）/ 错误翻译 / 状态恢复。

### 3.2 能力声明（**如实声明，未验证的一律不声明**）

| 能力 | 声明 | 原因 |
|---|---|---|
| `device` `memory` `stream` `event` `multidevice` | ✅ | 实现齐备；8 卡为实测环境事实 |
| `bounded_sync` | ✅ | 主机侧 `wait_host` 真有界（轮询实现）；流同步为**超时上报**语义（同前两家口径，已如实标注） |
| `recovery_probe` | ✅ | 探针级探活 |
| `device_state` | ✅ | 复用芯片无关的四态机（进程内状态机，不依赖厂商原语） |
| `error_map` | ❌ **已实测确认不具备** | 错误码**不透出为数字码**：实测 CNRT 给错误名（`CNRT error: invalid argument.`）、OOM 给 `OutOfMemoryError: MLU out of memory…` ⇒ **无可建码表的数字码**。分级走 `message_hint` 已覆盖（形状错→L2、OOM→L1，conformance f1 通过）；若底层骨架意外给出 `code_map` 一律如实降级标注 |
| `recovery_real` | ❌ **已实测确认不具备** | `torch.mlu` 下 `reset*/destroy*/reinit*` **全部是内存统计类**（`reset_peak_memory_stats` 等）⇒ 无设备级重置原语。**不写猜的重建序列**（写了会制造"看起来支持"的假象） |
| `graph_capture` | ✅ **已声明（09-22 实测）** | 入口为 `torch.mlu.MLUGraph` + `torch.mlu.graph`；**实测 `GRAPH_CAPTURE` 5/5 成功** |
| `stream_priority` | ✅ **已声明（09-22 实测）** | `priority_range()` 实测返回 **`(0, -3)`**，可用且**不崩**（昆仑芯同款 API 会触发 PyTorch `INTERNAL ASSERT`，二者相反）。⚠️ 上游**未拦截非法优先级**（`priority=99` 不报错）⇒ 本层不透传非法值；优先级**实际调度效果未单独验证** |

### 3.3 框架侧改动（都是"登记"，未改任何接口签名）

| 文件 | 改动 |
|---|---|
| `runtime/backends/registry.py` | `_KNOWN_BACKENDS` 加入 `"cambricon"`（自动发现的登记点） |
| `runtime/smoke_runtime.py` | 第 [6] 节真实后端自检的挑选顺序加入 `cambricon`（依赖缺失会如实 SKIP 后继续试下一家） |
| `runtime/__init__.py` | docstring 补上后端名示例 |
| `runtime/proto/proto_train_leg.py` | **寒武纪不给 `DC_DIST_BT` 就报错退出**（见下） |
| `scripts/serve_standard.sh` | 新增 `cambricon` 分支 + `card_snapshot` 支持 `cnmon` / `torch.mlu` 查卡降级 |
| `docs/SERVICE_STARTUP_STANDARD_20260920.md` | 登记 `cambricon` 分支（标注"尚未真机验证"） |

**为什么训练腿要"不给就报错"**：集合通信后端名**同一套代码跨芯片就不同**
（910C = `flagos`、P800 = `flagcx`，手册 §9 坑 3 已实证），寒武纪**不可类推**。
而 `proto_train_leg.py` 的兜底是 `gloo` —— 那会**静默退化为纯 CPU 集合通信**：
训练脚本照样跑完、loss 照样下降，但**设备侧通信根本没被验证**。
这类"看起来通过"的结果比失败更糟，故显式拦住并给出探测方法。

### 3.4 后端内的 `known_issues()`（已实测的环境约束，非厂商缺陷）

| id | 级别 | 内容 |
|---|---|---|
| `MLU-DRIVER-TIER-CONSTRAINT` | info | 宿主驱动 v6.2.29（6.2.x 线）⇒ 只能用 `neuware4.4.3` 档；`neuware4.7.2` 要求 6.5.48 |
| `MLU-HOST-NO-NEUWARE` | info | 宿主无 `/usr/local/neuware` ⇒ 一切验证必须在带卡容器内 |

> **本实例当前没有任何厂商缺陷结论**（尚未进容器）。容器内实测后按手册 §9 模板补条目。
> 清单为空也要如实为空 —— 不凑数。

---

## 4. 验证记录（本地离线 + 真机 A5）

### 4.0 ⭐ 真机 A5 结果（2026-09-22，本节最重要）

| 项 | 结果 | 证据 |
|---|---|---|
| 离线契约自检（容器内） | **34 通过 / 0 失败** | `probes/A5_verify_20260922.log` §④ |
| smoke（真实后端通用自检） | **42 通过 / 0 失败**；选中后端 `cambricon (device_type=mlu)`，`device_count=8` | 同上 §⑤ |
| **conformance 13 例** | **13/13 通过 · `CONFORMANCE_PASS`** | 同上 §⑥ |
| **conformance 推理 6 例** | **6/6 通过 · `CONFORMANCE_PASS`** | 同上 §⑦ |
| 集合通信后端名 | **`cncl`**（2 进程 `all_reduce` 结果正确，见 §0.1 的自我更正） | `probes/A6b_graph_dist_20260922.log` |
| 图捕获 | **5/5 成功** | 同上 |
| 能力声明实测核对 | `priority_range()=(0,-3)`；`graph_capture`/`stream_priority`=True；`error_map`/`recovery_real`=False | 同上 |

> 13 例逐项（全部 PASS）：e1 事件 record/wait、e2 未 record 主机侧超时、e2 边界、e3 未 record query、
> f1 统一错误对象（`L2_PARAM` / `graded_by=message_hint`）、r 恢复契约、s1 流内顺序（相对误差 0.00e+00）、
> s2 显式依赖、s3 结果可见性、s4 显式传递、t1 pinned 异步拷贝、t2 在途保护、t3 拓扑路径。
> 推理 6 例：i1 上下文、i2 多轮前向一致（误差 0.00e+00）、i3 KV 跨流可见性、i4 D2H 回传、
> i5 长驻 20 轮无 NaN/Inf、i6 流水线依赖链（末轮 rel_err=1.03e-07）。

### 4.1 本地静态检查与回归（接入期，无设备）

| 项 | 结果 |
|---|---|
| 语法编译（7 个改动/新增 py 文件） | ✅ 全部通过 |
| `bash -n scripts/serve_standard.sh` | ✅ 通过 |
| 本机 `smoke_runtime.py`（无 torch 环境） | ✅ **28 通过 / 0 失败**；4 个真实后端如实 SKIP ⇒ **新增后端未造成回归** |
| `registry.discover()` 是否被新后端打断 | ✅ 不打断：cambricon 模块可导入（依赖是**延迟导入**），缺 `torch_mlu` 时只在调用期报错并被 discover 容错跳过 |

### 4.2 ⭐ 新增：**离线契约自检**（`prototype/scripts/backend_offline_check.py`）

**动机**：手册第 4 步（写 backend）与第 5 步（跑 conformance）之间有一段空档 ——
代码写完但机器没到位。这段时间最容易犯的是**实现层面的错**（抽象方法没实现全、
有界同步其实没上界、事件语义没修、错误翻译冒充码表、能力声明与实现不一致），
**这些不需要真实芯片就能查出来**。

```bash
python3 prototype/scripts/backend_offline_check.py --backend cambricon
# → 离线自检结果: 35 通过 / 0 失败
```

用 stub 把 `torch.mlu` 命名空间"空跑"一遍，8 组 35 条判据，逐条对应《接口约定》
或 conformance 用例的同口径判据（代码内注明 F1 / E2-v2 / E3 / R1-R5）：

| 组 | 覆盖 | 关键判据 |
|---|---|---|
| ① 发现/实例化 | ABC 强制 13 抽象方法齐全 | 实例化即失败于缺方法 |
| ② 设备域 | count / set_device / memory_stats / probe | 结构 `{total_mb,used_mb,free_mb}`、**用完还原当前设备** |
| ③ 流-事件 + 有界同步 | 真超时 / 已完成不误判 | 未完成 + `timeout_ms=0` → 必抛 `TimeoutError`；已完成 → 正常返回；实测 **200 ms 按时返回** |
| ④ 事件语义 | E3 / E2-v2 | **stub 刻意做成"未 record 也返回 True"的坏实现**，验证适配层确实修正为 `False` |
| ⑤ 错误翻译 | F1 三投影 + 诚实性 | 形状错 → `L2_PARAM`；未声明 `error_map` ⇒ `mapped` 必须为 `False` |
| ⑥ 恢复/设备状态 | R1-R5 | `recover_device` 返回 dict；`real` 如实说明不支持 |
| ⑦ 能力自洽 | `info().capabilities` ↔ `supports()` | 无不一致项 |
| ⑧ 缺厂商扩展 | **不得静默降级** | 抛 `RuntimeError` 且文案给出下一步动作 |

> ⚠️ **结论边界（勿外推）**：本自查**只证明实现逻辑与契约形态**，
> **不能替代 conformance**，也**不能证明后端在真机上能用**。脚本末行会重复这句。
> 该工具已回写进《新芯片接入手册》**§4.4**（新芯片接入的可复用资产）。

---

## 5. 真机执行手册（**A1–A5b 已于 2026-09-22 执行完成**，A6–A10 待执行）

> **实际执行信息（供复现）**：
> · 机器 **Mlu-1**（10.1.1.21 / `tza-0a06-ai01-em9`），数据目录 `/srv/hliu553`
> · 镜像 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0`，
>   实测 digest `sha256:e55b420ee98e0fdef6c18a27b633d67b988ecef81a52a0fef5a0c6636c91d5c2`（**与定档记录一致**）
> · 容器名 `dc-mlu590-hliu553`（起容器脚本 `/srv/hliu553/start_container_mlu590.sh`，传入 18 个设备节点）
> · 用卡：宿主**卡 0 被他人占（37748 MiB / 96%）**、卡 1 有残留 ⇒ **单卡用 2、双卡用 2,3**
>   （`MLU_VISIBLE_DEVICES=2` 实测把 `device_count()` 从 8 变 1）

> 纪律：**每一步的原始输出都要落盘归档**（`probes/`，`.log` 已加 `!*.log` 例外）。
> **共享机用卡**：先 `cnmon` 看占用，挑**空闲**卡并在记录里写明用了哪张。

### A1 拉镜像 + 起容器 ✅ 已完成

```bash
IMG=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0
docker pull "$IMG"
docker image inspect "$IMG" --format '{{index .RepoDigests 0}}'   # 核对 digest 是否 = e55b420e…

CT=dc-mlu590-hliu553
docker run -dit --name "$CT" \
  --net=host --shm-size=64g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/cambricon_dev0 --device=/dev/cambricon_dev1 \
  --device=/dev/cambricon_dev2 --device=/dev/cambricon_dev3 \
  --device=/dev/cambricon_dev4 --device=/dev/cambricon_dev5 \
  --device=/dev/cambricon_dev6 --device=/dev/cambricon_dev7 \
  --device=/dev/cambricon_ctl \
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro \
  -v /srv/data/hf_cache:/hf_cache:ro \
  -v /srv/hliu553:/work -w /work \
  -v /etc/localtime:/etc/localtime:ro \
  "$IMG" bash
docker exec -it "$CT" bash
```

**参数依据（不是编的）**：设备节点来自官方 `base/cambricon-neuware4.4.3.md` 给的
`--device /dev/cambricon_dev0 --device /dev/cambricon_ctl`；
`-v /usr/bin/cnmon` 与 `--cap-add=SYS_PTRACE --shm-size` 来自 FlagGems 官方周测配置
`FlagGems/.github/configs/weekly/MLU590-M9DE.yml`（该配置跑的正是 **MLU590-M9** 这台卡型）。
⚠️ 寒武纪**无容器 toolkit**（不像昇腾 `Ascend-docker-runtime`）⇒ 直接给设备节点，不用 `--runtime`。

### A2 环境普查（第 0 步复跑，作正式证据）🟡 待补（第 0 步报告已覆盖）

```bash
OUT=/work/preflight bash /work/prototype/scripts/preflight_env.sh
```

### A3 厂商栈判别 ✅ 已完成（`probes/A3A4_stack_probe_20260922.log`）

### A4 镜像就绪 5 条判据 ✅ 已完成（判据 1+2 通过；5 见 `known_issues` 的 triton 条目）

```bash
python3 -c "import torch, torch_mlu; print('devs', torch.mlu.device_count())"      # 判据 1+2：预期 8
python3 -c "import triton; print('triton', triton.__version__)"                    # 判据 5（对照 P800 的坑）
python3 -c "from vllm.platforms import current_platform; print(current_platform)"  # 判据 4：推理腿形态
python3 -c "import torch.distributed as d; print('dist ok')"                       # 判据 3 的前置
cnmon | head -30                                                                   # 看卡与占用
```

### A5 放入原型 + 运行时代码层验证 ✅ 已完成（离线自检 34/0 · smoke 42/0 · conformance 13/13 + 6/6）

```bash
# 方式一（有出网）：git clone 分支
cd /work && git clone -b kistich/device-context git@github.com:FlagRT/runtime-team.git rt
# 方式二（无出网）：从本机拷
#   scp -r dev/device-context/prototype Mlu-1:/srv/hliu553/

cd /work/rt/dev/device-context/prototype
python3 scripts/backend_offline_check.py --backend cambricon     # 期望 35/0（与本地一致）
python3 runtime/smoke_runtime.py --backend cambricon             # 接入自检
python3 runtime/conformance/runner.py --backend cambricon                       # 13 例
python3 runtime/conformance/runner.py --backend cambricon --cases infer_cases    # 推理 6 例
```

**这一步是接入完成的判定线**：13/13 + 6/6 全绿，或未支持项有**如实 stub-skip 说明**。

### A5b 集合通信后端名探测 ✅ 已完成 → **`cncl`**

```bash
python3 - <<'PY'
import torch, torch.distributed as dist
print("mlu:", torch.mlu.device_count())
for bt in ("cncl", "cpu:gloo,mlu:cncl", "flagcx", "cpu:gloo,mlu:flagcx"):
    try:
        import os; os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29511")
        dist.init_process_group(bt, rank=0, world_size=1, timeout=__import__("datetime").timedelta(seconds=30))
        print("OK  ", bt); dist.destroy_process_group()
    except Exception as e:
        print("FAIL", bt, type(e).__name__, str(e)[:80])
PY
```

把可用者写进 `DC_DIST_BT`（形如 `cpu:gloo,mlu:<backend>`）后再跑训练腿；
**探测结果回填本文档与 `runtime/backends/cambricon/backend.py` 顶部「未实测清单」第 10 条**。

### A6 多流 16 项基线 ⏳ 待执行

```bash
DC_BACKEND=cambricon python3 prototype/probes/probe_stream_semantics_full.py   # 期望 STREAM_SEMANTICS_PASS 8/8
```

### A7 训练腿（2 卡，50 步；门禁档）⏳ 待执行（`DC_DIST_BT=cncl`，用卡 2,3）

```bash
DC_BACKEND=cambricon MLU_VISIBLE_DEVICES=0,1 \
DC_DIST_BT=<A5b 探测结果> \
DC_ROOT=/work/rt/dev/device-context \
DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
DC_OUT_DIR=/work/scratch MAX_STEPS=50 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py
```

判据：两 rank 均 `TRAIN_LEG_PASS 6/6`，loss 正常下降。
参考锚点：910C `15.4497 → 11.15` / 2117 tok/s；P800 `15.4488 → 11.1481` / 3482 tok/s。
**50 步不是随便定的**（挂死类缺陷往往在第 n 次通信才出现）。

### A8 推理腿（前向 + 服务化）⏳ 待执行（服务化需 `flagos-app` 应用镜像，见 §0.1）

```bash
# 前向
DC_BACKEND=cambricon MLU_VISIBLE_DEVICES=0 \
DC_MODEL=<同上 snapshot 路径> python3 runtime/proto/proto_infer_leg.py
# 服务化（统一入口，不要自建脚本）
DC_BACKEND=cambricon DEV=0 STOP_AFTER=1 \
MODEL=<同上 snapshot 路径> bash prototype/scripts/serve_standard.sh
```

判据：维度 **1024**、语义区分度、吞吐与 p50/p90、真实异常注入 → `L2_PARAM/raise` 且业务继续；
服务化另加 **超长输入 → HTTP 400 → L2_PARAM/raise** 与**同卡共存**。
参考锚点：910C 区分度 0.4123 / 108 句·s⁻¹；P800 0.4102 / 30.70 句·s⁻¹。
⚠️ 停机必须连 `EngineCore` 子进程一起 `kill -9`（P800 实测残留占卡 73850 MiB）。

### A9 错误闭环（四类注入）⏳ 待执行

```bash
DC_BACKEND=cambricon python3 runtime/proto/proto_error_recovery_loop.py
```

判据：四类注入（参数/资源/执行/致命）分级与处置正确、业务继续。
⚠️ 调用纪律：传**完整原始异常/服务错误消息，不得截断**（截断会把参数错退化成无意义重放）。

### A10 归档 + 回填 🟡 证据已入库 4 份，余待补

- 原始日志/JSON → `MLU590/probes/`（`.log` 已有 `!*.log` 例外；**写"已入库"前必须 `git ls-files` 实测确认**）
- 回填：本文档 §0 进度表、§6 验收清单、`backend.py` 的「未实测清单」与 `_capabilities`
- 更新 `MLU590/README.md`、主看板、`STATUS.md`

---

## 6. 验收清单（手册 §8 的 13 项，逐项标当前状态）

| # | 项 | 判据 | 当前状态（2026-09-22） |
|---|---|---|---|
| 1 | 环境打通 | 连得上 / 有权限 / 有可写目录 / 会挑空闲卡 | ✅ **已打通**（`docker` 组 + `/srv/hliu553`；Docker 25.0.3；共享机用卡纪律照办） |
| 2 | 厂商栈判别 | 明确落 A/B/C/D 哪条路径并记录证据 | ✅ **路径 C（PrivateUse1 / `mlu`）**，证据 `A3A4_stack_probe` §B |
| 3 | 镜像就绪 | 5 条判据全过（含依赖链完整自检） | ✅ 判据 1、2、4 通过；判据 5「依赖链」以 `known_issues` 的 **triton 导入顺序**条目如实登记（本方向路径不依赖 triton） |
| 4 | backend 落地 | 13 抽象 + `build()` + `supports()` 如实声明 | ✅ 完成，且**能力声明已按真机证据更新**（新增 `graph_capture`/`stream_priority`；`error_map`/`recovery_real` 由"未验证不声明"升为"已确认不具备"） |
| 5 | conformance 13 例 | 全绿或如实 stub-skip | ✅ **13/13 全绿 · `CONFORMANCE_PASS`** |
| 6 | conformance 推理 6 例 | 全绿 | ✅ **6/6 全绿 · `CONFORMANCE_PASS`** |
| 7 | smoke 自检 | 全通过 / 0 失败 | ✅ **42 通过 / 0 失败**（选中 `cambricon`，`device_count=8`） |
| 8 | 训练腿 | 两 rank `TRAIN_LEG_PASS 6/6` | ⏳ 待执行（前置已就绪：`DC_DIST_BT=cncl`、空闲卡 2,3） |
| 9 | 推理腿（前向） | 维度 / 范数 / 区分度 / 时延 / 异常分级 | ⏳ 待执行 |
| 10 | 推理腿（服务化） | 含超长输入防御与同卡共存 | ⏳ 待执行（**需 `flagos-app/vllm*-cambricon-*` 应用镜像**——运行时层镜像不含 vLLM） |
| 11 | 错误闭环 | 四类注入分级处置正确、业务继续 | ⏳ 待执行 |
| 12 | 已知问题如实声明 | `known_issues()` 结构化 + 开跑前告警 | ✅ 已实现，**4 条**：驱动档位约束 / 宿主无 NeuWare / **triton 导入顺序** / **运行时镜像不含 vLLM**；**仍无厂商缺陷结论**（不凑数） |
| 13 | 证据归档 | 原始日志/JSON 入版本库 + `git ls-files` 实测 | 🟡 本轮 **4 份**证据已入 `MLU590/probes/`（待 `git ls-files` 复核） |

---

## 7. 风险与应对

| # | 风险 | 应对 |
|---|---|---|
| 1 | ~~`torch.mlu` API 形态与预期不符~~ → ✅ **已澄清**：`mem_get_info(ordinal)` 可用、`Stream.synchronize` 确实不收 timeout（后端的多条兜底链中，显存降级路径未被触发，超时上报语义正好落在预期分支上） | 无需再应对；结论已回填 backend 与本文档 |
| 2 | **推理腿形态仍未知**（运行时层镜像**不含 vLLM** ⇒ `current_platform` 跑不了） | 改用 `flagos-app/vllm0.24.0-cambricon-neuware4.4.3`（或 0.20.2 变体）应用镜像后再判定；`serve_standard.sh` 的 cambricon 分支仍**刻意不预设任何厂商专用环境变量** |
| 3 | ~~集合通信后端名不同~~ → ✅ **已探测：`cncl`**（2 进程 `all_reduce` 结果正确） | 「不给 `DC_DIST_BT` 就报错」的拦截保留（防静默退化）；训练腿按 `DC_DIST_BT=cncl` 跑 |
| 4 | **共享机误用他人占用的卡** | P800 曾因此撤销一个"缺陷"结论 ⇒ 用卡前 `cnmon` 挑空闲卡、记录用卡、异常先换卡复测 |
| 5 | **老档位（4.4.3）相关问题** | 不得凭版本号升级宿主驱动；走 `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.2 的四条门槛 + 上报模板 |
| 6 | 停机残留占卡 | `serve_standard.sh` 已含 `pkill -9` EngineCore 清理 |
| 7 | 结论被外推 | 所有产出标注取得时的**档位/条件**（同 P800 的「KL3 未设置条件下取得」纪律）；本次已标注：**`neuware4.4.3` / py3.10 / torch 2.7.1 / torch-mlu 1.29.2 档**，用卡 2（单卡）；共享机卡 0 有他人负载，**吞吐类指标不可与独占环境直接对比** |

---

## 8. 不做什么（职责边界）

| 不做 | 归属 |
|---|---|
| 生产级性能调优 | 精度 / 调优方向 |
| 算子实现、通信库优化、显存池调优、调度策略 | 各自方向 |
| 擅自改上游口径 / 动公共资产 | 由总组裁定 |
| 规模化验收（Megatron-LM-FL / vllm-plugin-FL 适配） | 框架适配方向 |

**问题路由**：接入过程暴露的问题按"设备因素归设备"分流 —— 属五域内的先修；
算子/通信/显存/调度/性能一律对外提交。
