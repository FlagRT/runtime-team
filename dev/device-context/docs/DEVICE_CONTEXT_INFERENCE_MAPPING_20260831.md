# 设备执行上下文职责 × 分布式推理 · 工作与验证映射（验收依据）

> 状态：📋 映射定稿（2026-08-31）｜ 作者：Kistich ｜ 路线：A 线（torch_npu，不走 torch_fl）
> 用途：**本方向在分布式推理上的验收依据**——职责拆解 → 逐项 × 推理场景验证点 → 验收标准；后续按本文档验收
> 关联：DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md（阶段计划）、INFERENCE_P0_P1_RUN_20260831.md（执行记录）

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
| D3 | 内存句柄+生命周期 | KV cache 分配；权重加载显存；D2H logits 缓冲；长驻显存不泄漏 | KV 分配访问正确 + 长驻 20 轮无泄漏 | conformance i3/i4/i5 | 🔄 i3/i4/i5 已 PASS |
| D4 | 执行句柄 | 推理前向执行通道 | 多轮前向正确（i2） | conformance i2 | 🔄 i2 已 PASS |
| D5 | Stream 语义 | 多流并发（H2D/计算/D2H）；TP 通信流绑定；graph capture 流语义 | 双缓冲重叠可观测（P1）+ TP 流绑定正确（P2） | S1-S4 + i6 + 双缓冲探针 + test_tp_comm_sync | ✅ i6 PASS + **TP 流绑定 4/4 PASS**（多流真并发已由 kernel 时间线证实） |
| D6 | Host/Device 异步传输 | prompt H2D 异步；logits D2H 回传；KV offload（可选） | 异步拷贝数据一致 + 与计算重叠 | T1/T2 + i4 + 双缓冲 | 🔄 i4 已 PASS；重叠待 D8 |
| D7 | 页锁定内存 | pin_memory 在推理传输路径 | pin_memory 传输一致性 + 预热纪律 | T1 + 双缓冲探针 | 🔄 已用（双缓冲探针） |
| D8 | 双缓冲流水线 | **多流+Event 流水线真实现 + 重叠测量** | DBUF2_PASS：数据一致 + 重叠率可观测为正 | test_double_buffer_pipeline.py + breakdown + L1 profiler | ✅ **机制通过**（多流真并发已证实）；⚠️ 性能瓶颈 = EVENT_WAIT 开销（3.37ms/12 次），优化方向"减少同步点" |
| D9 | 同步语义 | TP 通信同步（A 线重验 B2：flagcx 异步无同步→NaN）；wait_host 有界等待 | TP 通信无 NaN + E2 语义 | E1-E3 + test_tp_comm_sync（双卡 FlagCX） | ✅ **TP_COMM_PASS 4/4**：B2 类问题在 A 线不存在 |
| D10 | 错误码翻译 | 推理路径 ACL 错误捕获翻译 | 错误码→L1-L4 翻译正确 | F1 + errors.py + probe_acl_107015 | ✅ 2026-09-01 **ACL_107015_PASS**：真实错误注入 + A/B 单变量对照 + 翻译链路验证（详见 P3 执行记录）；⚠️ 分级待定：107015 现落 L3_EXECUTION，建议改 L2_PARAM |
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

> 验收 = 下表全部 ✅。状态列随执行更新。

| # | 验收项 | 对应职责 | 状态 |
|---|---|---|---|
| A1 | dense 单卡离线推理输出正确（Qwen2.5-1.5B） | D1/D2 | ✅ 2026-09-01：DENSE_INFER_PASS，19.5 tok/s（vllm-ascend 镜像） |
| A2 | conformance 推理版 6/6 PASS | D2/D3/D4/D5/D6/D7/D8 | ✅ 2026-08-31 |
| A3 | 双缓冲流水线数据一致 + 重叠可观测（DBUF2_PASS） | D8/D5/D6/D7 | ✅ 机制通过（多流真并发由 kernel 时间线证实）；⚠️ 重叠率受 EVENT_WAIT 开销压制 |
| A4 | 双缓冲重叠深挖结论（同步退化 vs 流切换开销定位） | D8/D5 | ✅ 2026-08-31：三排除 + 瓶颈=EVENT_WAIT（3.37ms/12 次），非并发能力问题 |
| A5 | TP=2 推理输出与 TP=1 一致（或记录确定性分叉） | D5/D9 | ✅ 2026-09-01：Qwen3-4B TP=1/2/4 **greedy 逐字一致 4/4**（TP_COMPARE_PASS），无 NaN；B1 bool×int / B2 异步无同步两类缺陷 A 线均不存在（详见 INFERENCE_QWEN3_TP_COMPARE_20260901.md） |
| A6 | A 线重验 B2：flagcx TP 通信无 NaN（同步语义证据） | D9 | ✅ 2026-08-31：TP_COMM_PASS 4/4，B2 类问题在 A 线不存在 |
| A7 | TP 通信流绑定正确（collective 与当前流） | D5 | ✅ 2026-08-31：all_reduce 后立即设备侧消费正确（got=6.0） |
| A8 | EngineCore 子进程设备句柄/上下文可用 | D2 | ✅ 2026-09-01：`ENGINECORE_CTX_PASS` —— spawn 子进程 pid=8421(ppid=8402) 持有 `/dev/davinci_manager` fd、124 处设备内存映射、RSS 5952MB，功能请求 0.23s 正常 |
| A9 | 推理路径错误码翻译正确（注入 ACL 错误） | D10 | ✅ 2026-09-01：`ACL_107015_PASS` —— 真实错误 107015 注入成功，A/B 对照证实根因为"stream 未 subscribe 即 launch callback" |
| A10 | 服务化四态监控 + 五段式恢复 | D11 | ✅ 2026-09-01：`DEVICE_STATE_RECOVERY_PASS 8/8` —— 四态可查、DEGRADED 转换、L4 完整 R1-R5、L3 不重建、ISOLATED→AVAILABLE、服务续跑 |
| A11 | 结果与 trap 归档（每阶段执行记录） | 通用 | ✅ P0/P1/P2/P3 全部归档（INFERENCE_P0_P1_RUN_20260831 + INFERENCE_QWEN3_TP_COMPARE_20260901 + INFERENCE_P3_SERVE_STATE_ERROR_20260901） |

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
