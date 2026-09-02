# 设备执行上下文职责 × 分布式推理 · 工作与验证映射（验收依据）

> 状态：📋 映射定稿（2026-08-31）｜ **2026-09-02 对标校准** ｜ 作者：Kistich ｜ 路线：A 线（torch_npu，不走 torch_fl）
> 用途：**本方向在分布式推理上的验收依据**——职责拆解 → 逐项 × 推理场景验证点 → 验收标准；后续按本文档验收
> 关联：DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md（阶段计划）、INFERENCE_P0_P1_RUN_20260831.md（执行记录）、
> INFERENCE_P3_SERVE_STATE_ERROR_20260901.md（P3 执行）、ACL_ERROR_MAP_20260901.md（错误码映射）
>
> **校准摘要（2026-09-02 对标核查）**：
> - A1/A2/A4-A11 共 10 项为 ✅；**A3（双缓冲）为 ⚠️ 部分达标**——功能通过、性能未达，是本方向**唯一实质缺口**
> - 修正 3 项状态滞后（D3/D4/D7 由 🔄 校准为 ✅）、1 项描述过时（D10 已同步 9-01 成果）
> - 澄清「验收标准 vs 用例判定」的差异（见 §4.1），避免"看起来达标、实际没达"的假象

---

## 0. 职责原文与拆解原则

**职责原文**：*封装不同芯片Runtime接口，统一设备、内存和执行句柄及生命周期管理。统一 Stream 语义、Host/Device 异步传输、页锁定内存与双缓冲流水线、同步语义、错误码翻译和设备状态恢复。*

拆解为 **11 个职责子项**，每项给出：推理场景验证点、验收标准、复用资产、当前状态。

| # | 职责子项 | 出处 |
|---|---|---|
| D1 | 封装不同芯片 Runtime 接口 | 首句 |
| D2 | 统一设备句柄 + 生命周期管理 | 首句 |
| D3 | 统一内存句柄 + 生命周期 | 首句 |
| D4 | 统一执行句柄 | 首句 |
| D5 | 统一 Stream 语义 | 次句 |
| D6 | Host/Device 异步传输 | 次句 |
| D7 | 页锁定内存 | 次句 |
| D8 | 双缓冲流水线 | 次句 |
| D9 | 同步语义 | 次句 |
| D10 | 错误码翻译 | 次句 |
| D11 | 设备状态恢复 | 次句 |

---

## 1. 推理场景阶段（验证载体）

| 阶段 | 场景 | 载体 |
|---|---|---|
| P0 | dense 单卡离线推理 | dense_infer_qwen_1_5b_npu.py |
| P1 | 单卡推理机制 + 双缓冲流水线 | conformance/infer_cases.py + test_double_buffer_pipeline.py |
| P2 | TP=2/4 多卡推理 | TP 推理脚本（A 线参数化，参照 B 线阶段 4） |
| P3 | 服务化（vllm serve） | serve + 状态/错误探针 |

---

## 2. 职责 × 推理验证矩阵（核心验收表）

| # | 职责子项 | 推理验证点 | 验收标准 | 复用资产 | 当前状态 |
|---|---|---|---|---|---|
| D1 | 封装 Runtime 接口 | vllm-ascend + torch_npu 在 910C 的设备接入 | P0 dense 推理输出正确（DENSE_INFER_PASS） | dense_infer 脚本 | ✅ 2026-09-01 PASS（19.5 tok/s）；⚠️ vllm-plugin-FL 不适用，见坑 A5 |
| D2 | 设备句柄+生命周期 | 推理加载/卸载设备初始化；EngineCore 子进程设备句柄；TP rank 设备枚举 | P0 加载 OK + P2 多卡枚举 + P3 子进程句柄 | conformance i1 + 设备枚举探针 + probe_enginecore_device_ctx | ✅ i1 PASS + P0 加载 OK + **A8 子进程句柄 PASS**（spawn pid/ppid + `/dev/davinci_manager` fd + 124 处设备映射 + RSS 5952MB） |
| D3 | 内存句柄+生命周期 | KV cache 分配；权重加载显存；D2H logits 缓冲；长驻显存不泄漏 | KV 分配访问正确 + 长驻 20 轮无泄漏 | conformance i3/i4/i5 | ✅ 2026-08-31：i3 KV 跨流可见性 / i4 D2H 采样回传 / i5 长驻 20 轮无 NaN-Inf，均 PASS（2026-09-02 校准：原标 🔄 属状态滞后） |
| D4 | 执行句柄 | 推理前向执行通道 | 多轮前向正确（i2） | conformance i2 | ✅ 2026-08-31：i2 多轮前向同流顺序一致（误差 0.00e+00）PASS（2026-09-02 校准：原标 🔄 属状态滞后） |
| D5 | Stream 语义 | 多流并发（H2D/计算/D2H）；TP 通信流绑定；graph capture 流语义 | 双缓冲重叠可观测（P1）+ TP 流绑定正确（P2） | S1-S4 + i6 + 双缓冲探针 + test_tp_comm_sync | ✅ i6 PASS + **TP 流绑定 4/4 PASS**（多流真并发已由 kernel 时间线证实） |
| D6 | Host/Device 异步传输 | prompt H2D 异步；logits D2H 回传；KV offload（可选） | ① 功能：异步拷贝数据一致 ② 性能：与计算重叠 | T1/T2 + i4 + 双缓冲 | ⚠️ **部分达标**：① 功能 ✅（i4 D2H 回传 PASS、T1/T2 在 conformance 13/13 内）② 性能 ❌（与计算重叠依赖 D8，当前重叠率为负，未达） |
| D7 | 页锁定内存 | pin_memory 在推理传输路径 | pin_memory 传输一致性 + 预热纪律 | T1 + 双缓冲探针 | ✅ 2026-08-31：T1 pinned→device non_blocking 拷贝数据一致 PASS；双缓冲探针全程使用 pin_memory（2026-09-02 校准：原标 🔄 属状态滞后） |
| D8 | 双缓冲流水线 | **多流+Event 流水线真实现 + 重叠测量** | ① 功能：多流+Event 数据一致 ② 性能：重叠率可观测为正 | test_double_buffer_pipeline.py + breakdown + L1 profiler | ⚠️ **部分达标（本方向唯一实质缺口）**：① 功能 ✅ 多流真并发 + 数据一致（Level1 kernel 时间线 + i6 双重验证）② 性能 ❌ **重叠率 -101.6% ~ -1306%（负）**，瓶颈 = `EVENT_WAIT` 3.37ms/12 次。优化方向「减少同步点」而非提升并发（并发能力已证实无问题）。详见 §5 与 O2 |
| D9 | 同步语义 | TP 通信同步（A 线重验 B2：flagcx 异步无同步→NaN）；wait_host 有界等待 | TP 通信无 NaN + E2 语义 | E1-E3 + test_tp_comm_sync（双卡 FlagCX） | ✅ **TP_COMM_PASS 4/4**：B2 类问题在 A 线不存在 |
| D10 | 错误码翻译 | 推理路径 ACL 错误捕获翻译 | 错误码→L1-L4 翻译正确 + 分级来源可观测 | F1 + errors.py + probe_acl_107015 + gen/audit_error_map | ✅ 2026-09-01 **ACL_107015_PASS**：真实错误注入 + A/B 单变量对照（根因 = stream 未 subscribe 即 launch callback）。**107015 已定级 L2_PARAM**（实测裁决，非规则建议）。错误码映射覆盖率 **0.8% → 64.8%（103/159）**，新增 F5 可观测（`mapped`/`graded_by`）区分确定分级与保守兜底。详见 `docs/ACL_ERROR_MAP_20260901.md` |
| D11 | 设备状态恢复 | 长驻服务四态监控 + 五段式恢复 | 四态查询可用 + 注入错误可恢复 | R1-R5 + device_state + recovery | ✅ 2026-09-01 **DEVICE_STATE_RECOVERY_PASS 8/8**：含 L4 完整 R1→R5（captured→isolated→recovered→replay_ready） |

---

## 3. 阶段覆盖矩阵（每个阶段验证哪些职责）

| 阶段 | 覆盖职责 | 出口标准 |
|---|---|---|
| P0 | D1、D2 | dense 单卡推理 PASS（DENSE_INFER_PASS） |
| P1 | D3、D4、D5、D6、D7、D8 | conformance infer 全过 + 双缓冲 DBUF2_PASS（含重叠） |
| P2 | D5、D9 | TP=2 输出正确、无 NaN；flagcx 同步语义结论明确 |
| P3 | D2、D10、D11 | serve 长稳 + 四态 + 错误可恢复 |

---

## 4. 验收清单（Checklist，最终验收依据）

> 验收 = 下表全部 ✅。**当前 10 项 ✅ + A3 一项 ⚠️ 部分达标**（A3 的功能项已过、性能项未达，转 O2）。状态列随执行更新。

| # | 验收项 | 对应职责 | 状态 |
|---|---|---|---|
| A1 | dense 单卡离线推理输出正确（Qwen2.5-1.5B） | D1/D2 | ✅ 2026-09-01：DENSE_INFER_PASS，19.5 tok/s（vllm-ascend 镜像） |
| A2 | conformance 推理版 6/6 PASS | D2/D3/D4/D5/D6/D7/D8 | ✅ 2026-08-31 |
| A3 | 双缓冲流水线：① 数据一致 + 多流真并发 ② 重叠率为正 | D8/D5/D6/D7 | ⚠️ **部分达标**：① ✅ 数据一致（i6 rel_err 2.06e-07）+ 多流真并发（Level1 kernel 时间线证实 copy 与 matmul 重叠）② ❌ **未达**：重叠率 **-101.6% ~ -1306%**（流水线比串行更慢），瓶颈 = `EVENT_WAIT` 3.37ms/12 次 → 转 O2 |
| A4 | 双缓冲重叠深挖结论（同步退化 vs 流切换开销定位） | D8/D5 | ✅ 2026-08-31：三排除 + 瓶颈=EVENT_WAIT（3.37ms/12 次），非并发能力问题 |
| A5 | TP=2 推理输出与 TP=1 一致（或记录确定性分叉） | D5/D9 | ✅ 2026-09-01：Qwen3-4B TP=1/2/4 **greedy 逐字一致 4/4**（TP_COMPARE_PASS），无 NaN；B1 bool×int / B2 异步无同步两类缺陷 A 线均不存在（详见 INFERENCE_QWEN3_TP_COMPARE_20260901.md） |
| A6 | A 线重验 B2：flagcx TP 通信无 NaN（同步语义证据） | D9 | ✅ 2026-08-31：TP_COMM_PASS 4/4，B2 类问题在 A 线不存在 |
| A7 | TP 通信流绑定正确（collective 与当前流） | D5 | ✅ 2026-08-31：all_reduce 后立即设备侧消费正确（got=6.0） |
| A8 | EngineCore 子进程设备句柄/上下文可用 | D2 | ✅ 2026-09-01：`ENGINECORE_CTX_PASS` —— spawn 子进程 pid=8421(ppid=8402) 持有 `/dev/davinci_manager` fd、124 处设备内存映射、RSS 5952MB，功能请求 0.23s 正常 |
| A9 | 推理路径错误码翻译正确（注入 ACL 错误） | D10 | ✅ 2026-09-01：`ACL_107015_PASS` —— 真实错误 107015 注入成功，A/B 对照证实根因为"stream 未 subscribe 即 launch callback" |
| A10 | 服务化四态监控 + 五段式恢复 | D11 | ✅ 2026-09-01：`DEVICE_STATE_RECOVERY_PASS 8/8` —— 四态可查、DEGRADED 转换、L4 完整 R1-R5、L3 不重建、ISOLATED→AVAILABLE、服务续跑 |
| A11 | 结果与 trap 归档（每阶段执行记录） | 通用 | ✅ P0/P1/P2/P3 全部归档（`INFERENCE_P0_P1_RUN_20260831` + `INFERENCE_QWEN3_TP_COMPARE_20260901` + `INFERENCE_P3_SERVE_STATE_ERROR_20260901` + `ACL_ERROR_MAP_20260901`） |

### 4.1 验收标准 vs 用例判定的对齐说明（2026-09-02 新增）

**问题**：A3 在 conformance i6 用例中判为 PASS，但按本文档 §2 的验收标准（"重叠率可观测为正"）并未达标。
差异根源在于两者判定口径不同：

| 口径 | 判定内容 | 对重叠率的处置 |
|---|---|---|
| **conformance i6 用例** | `ok = 数据正确性`（末轮 rel_err ≤ 1e-6） | 仅作**观测输出**，不参与 PASS/FAIL 判定 |
| **本文档验收标准** | 功能（数据一致）+ **性能（重叠率为正）** | 性能项为**硬标准** |

**为何不把重叠率改成用例硬断言**：当前重叠率为负属**性能缺口**而非功能缺陷，
若加成硬断言会让 conformance 回归从 6/6 变成 5/6，掩盖"功能其实是对的"这一事实，
也会让后续每次回归因性能波动而红。故维持用例只验功能，性能缺口单列跟踪（O2）。

**纪律**：凡"功能 + 性能"复合验收项，文档须拆成两条分别标注（如 A3/D6/D8 已按此拆分），
不得用"机制通过"笼统替代性能达标。

---

## 5. 双缓冲深挖（A4）专项设计

**问题**：DBUF2_PARTIAL，pipe 0.074s vs serial 0.005s（慢 14 倍），重叠率 -1306%。

**分段计时探针**（定位"同步退化" vs "流切换开销"）：
1. 纯传输：6 × `copy_(host_pinned, non_blocking=True)` 到固定流，墙钟（无计算）
2. 纯计算：6 × `buf@buf.sum()` 计算流，墙钟（无传输）
3. 事件链开销：6 × record/wait 空操作，墙钟
4. 大张量（2048²）复测完整流水线：计算时间 >> 同步开销时是否出现重叠
5. 结论判定：
   - 若 1 明显 > 同步拷贝理论时间 → **copy_ non_blocking 退化同步**（torch_npu 缺口）
   - 若 3 显著（每事件 ~ms 级）→ **事件链 CANN 开销主导**
   - 若 4 重叠出现 → **同步开销主导，增大计算可掩盖**（流水线设计时注意粒度）

**结论影响**：决定 torch_npu 多流方案的实现路径（低层 aclrtMemcpyAsync 封装 vs 接受同步开销 vs 事件链优化）。

---

## 6. 环境与依赖（P0 前置）

- A 线 venv：`/root/venv-infer-a`（torch 2.11.0+cu130 + torch_npu 2.11.0 + vllm 0.20.2）✅ 已装好
- 坑 A4：结论性测试在官方发布镜像内（本容器已按 A 线原则）
- vllm-plugin-FL：clone FlagRT/vllm-plugin-FL → pip install -e

## 7. 环境结论（2026-09-01 实测更新）：A 线走 vllm-ascend 官方镜像

**历史判断（8-31）**：vllm-plugin-FL 无 ascend 后端 + vLLM 官方无 ascend platform → 判定 P0/P2 被环境阻塞。

**实测结论（9-01）：阻塞已解除**。改用华为官方 **vllm-ascend 镜像**后 P0 dense 与 A5（TP=1/2/4）全部跑通。**vllm-plugin-FL 路线在 A 线弃用**。

| 项 | 结论 |
|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（容器内 py3.11 自带环境，**不复用 raid venv**：ABI 不匹配） |
| 容器 | `flagos-infer-910c`（host 网络 + 512G shm + 16 NPU + raid 挂载） |
| 模型 | Qwen3-4B（对齐同事昇腾线基准，**P3 全程用它**）/ Qwen2.5-1.5B（P0 历史基线） |
| 环境变量 | `DO_NOT_TRACK=1`；**禁设 `VLLM_PLUGINS=fl`**（坑 A5） |

### 新增坑（9-01 实测，务必遵守）

| # | 坑 | 现象 | 解法/纪律 |
|---|---|---|---|
| **A5** | **`VLLM_PLUGINS=fl` 破坏 A 线 platform 选择** | 设置后 `current_platform.device_type` 为空 → `RuntimeError: Device string must not be empty`（vllm-plugin-FL 是 torch_fl 原生栈插件，与 torch_npu 栈冲突）。**P0 跑通时该变量实为"未设置"** | A 线（torch_npu + vllm_ascend）**禁止设置 `VLLM_PLUGINS=fl`** |
| **A6** | **随机采样下 TP 逐字对比必然发散** | 默认 SamplingParams（temperature=1.0）下 TP=1/2 输出前 7-14 个 token 后发散（0/4 一致，语义均正常）——TP=2 的 HCCL all_reduce 浮点累加顺序差异被随机采样放大 | **TP 数值等价对照必须用 greedy（temperature=0）**：argmax 对微扰鲁棒 → 4/4 一致。同事 B 线"TP=1/2 逐字一致"同理以 greedy 为前提 |

---

## 8. 下一步（2026-09-02 对标核查后）

| 编号 | 工作项 | 目标 | 状态 |
|---|---|---|---|
| **O1** | **校准映射文档**（本文档） | 修正状态滞后（D3/D4/D7）、同步 D10 成果、拆分 A3/D6/D8 的功能与性能标准、补 §4.1 对齐说明 | ✅ 已完成（2026-09-02） |
| **O2** | **D8 双缓冲「减少同步点」优化** | 让重叠率由负转正，达成 A3 性能项——**这是 D1-D11 唯一实质缺口**（训练侧同为该缺口，10/11） | ⬜ 待启动 |
| **O3** | PR 到 `dev-1.0` | 描述已备（`docs/PR_DEV_1_0_20260901.md`） | ⏸ 暂不发起（用户决定） |

### O2 设计要点（待细化）

瓶颈已定位：`EVENT_WAIT` **3.37ms / 12 次**（单次最高 391us ≈ 39 个 matmul）。
多流**真并发**已由 Level1 kernel 时间线证实 → **不是并发能力问题，优化方向是减少同步点**。

候选方案（待实测对比）：

| 方案 | 思路 | 风险 |
|---|---|---|
| 批量提交替代逐批事件链 | 一次提交 N 批，把 12 次 EVENT_WAIT 摊薄为 1 次 | 需改流水线结构 |
| 用 stream 间依赖替代显式 event | 减少 record/wait 调用点 | torch_npu 流依赖 API 完备性待验 |
| 增大批次摊薄同步成本 | 计算粒度 >> 同步开销时重叠自然显现（§5 第 4 点已验证此方向有效） | 不改变单位同步成本 |

**实验纪律**：最小变更 + 单变量隔离，先用 §5 的分段计时探针复现基线，再逐方案对比重叠率。
训练侧（`DEVICE_CONTEXT_TRAINING_MAPPING_20260831.md` 10/11）回补可复用同一结论。
