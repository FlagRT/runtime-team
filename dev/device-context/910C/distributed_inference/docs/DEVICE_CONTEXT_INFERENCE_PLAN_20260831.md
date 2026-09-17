# 同构 910C 分布式推理 × 设备上下文/Stream 职责 · 项目实现方案

> 状态：📋 方案（2026-08-31）｜ 作者：Kistich ｜ 路线：A 线（torch_npu + vllm-plugin-FL + FlagCX，**不走 torch_fl**）
> 目标：在昇腾 910C 同构环境下，验证"设备执行上下文"职责（设备初始化/上下文/执行队列/Stream-Event/设备间同步/Host-Device 传输/错误捕获/状态恢复）在**分布式推理**场景的完备性
> 参照：同构分布式训练方案（conformance 13/13 + 六探针 6/6 + 双卡 DDP 闭环 4245 tok/s）已定稿

---

## 0. TL;DR

**本周主线**：把设备上下文/Stream 职责从"训练已闭环"推进到"推理可验证"。分四阶段：环境就绪（dense 单卡）→ 单卡推理职责验证（含双缓冲流水线真实现）→ TP 多卡推理（A 线重验 B 线两个根因）→ 服务化（设备状态监控/错误恢复）。**先避坑（历史探索已踩 9 个），再验证，最后才是性能**。

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0 | 环境就绪 + dense 单卡离线推理 | Qwen2.5-1.5B 离线推理输出正确 |
| P1 | 单卡推理 × 设备上下文 + 双缓冲流水线真实现 | conformance 推理版 + 双缓冲重叠可观测 |
| P2 | TP=2/4 推理 × Stream/Event（A 线重验 B 线根因） | TP=2 与 TP=1 输出一致（或记录确定性分叉） |
| P3 | 服务化 × 状态监控/错误恢复 | serve 长稳 + 四态可用 + 注入错误可恢复 |

---

## 1. 历史探索的坑（先避坑，已踩实锤）

### 1.1 B 线推理插件 TP 验证的坑（xliu969，已归档，A 线需重验）

| # | 坑 | 现象 | 解法 | A 线（torch_npu）是否需要重验 |
|---|---|---|---|---|
| B1 | **flagos 设备 bool×int 类型提升错误** | TP=2 embedding 查表全查 row1 → 生成 "!!!!!!" | 显式 `.to(torch.int64)`（vllm-plugin-FL vocab_parallel_embedding） | ✅ 需重验（torch_npu 类型语义可能不同） |
| B2 | **flagcx 后端异步返回无同步 → NaN** | fallback `dist.all_reduce` 异步返回，下游消费未就绪数据 | 补 `torch.npu.synchronize()` | ✅ **核心职责重验**（Stream/Event 同步语义） |
| B3 | 探针打错路径 | 打在 KV-store 广播路径，误判"通信从未发生" | 打真实路径：`GroupCoordinator.all_reduce` → `device_communicator` | ✅ 探针设计参考 |
| B4 | flagos 设备 `.sum().item()` 数值不可靠 | flag_gems sum 算子问题，跨 rank 对比错乱 | `.cpu()` 后计算或 `.tolist()` | ✅ 数值对比纪律 |
| B5 | 多进程日志交错 | 把 rank0 输入配到 rank1 输出 | 按 rank 分别 grep/配对 | ✅ 通用 |
| B6 | TP=4 浮点分叉 vs 同步 bug 混淆 | 1/4 prompt 前缀一致后分叉（确定性） | 判定标准：同步缺陷=随机/NaN；浮点分叉=确定性 | ✅ 判定纪律 |
| B7 | `scatter_.src: backend not registered` | 多 prompt decode 批量路径挂点 | flagos_boot.py CPU fallback 注册 aten::scatter_ | ✅ 需重验（torch_npu 是否内置） |

### 1.2 A 线 dense 推理的坑（memory 方向实测）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| A1 | **首次 attention 极慢（13+ 分钟）** | vLLM 加载 24s 快，第一个请求 prefill 卡在 attention kernel 首次初始化（疑似 flag_gems/flagtune autotune + event-timing 回退） | **基准必须先跑短请求预热**；AICore 91% 忙、无新编译、cache 不增长为特征 |
| A2 | **EngineCore 是 spawn 子进程** | 主进程读不到子进程 stats；`docker exec` 被杀时子进程残留占卡 | 画像依赖设备级 HBM 采样（aclrtGetMemInfo/npu-smi）+ vLLM 日志；重跑前 `pkill -f "VLLM::EngineCore"` |
| A3 | vLLM usage 上报报错 | 容器内解析 cpuinfo 失败 | 设 `DO_NOT_TRACK=1` |
| A4 | **triton 版本偏差** | dev 容器 triton 3.6.0 ≠ 官方发布镜像 3.0.0 → FlagGems GEMM SIGABRT | **结论性测试一律在官方发布镜像内做** |
| A5 | **`VLLM_PLUGINS=fl` 破坏 A 线 platform 选择**（9-01 实测） | 设置后 `current_platform.device_type` 为空 → `RuntimeError: Device string must not be empty`。vllm-plugin-FL 是 torch_fl 原生栈插件，与 torch_npu 栈冲突。**P0 跑通时该变量实为"未设置"** | A 线**禁止设置 `VLLM_PLUGINS=fl`**；昇腾平台由 vllm-ascend 镜像内置提供（见第 2 节环境修正） |
| A6 | **随机采样下 TP 逐字对比必然发散**（9-01 实测） | 默认 SamplingParams（temperature=1.0）下 TP=1/2 输出前 7-14 个 token 后发散（0/4 一致，语义均正常）——TP=2 的 HCCL all_reduce 浮点累加顺序差异被随机采样放大 | **TP 数值等价对照必须用 greedy（temperature=0）**；同事 B 线"逐字一致"结论同理以 greedy 为前提 |

### 1.3 本方向已沉淀（直接复用）

- **conformance 13/13**（S1-S4/E1-E3/T1-T3/F1/R1-R5）+ npu_events 适配层（未 record 误报修正、wait_host 有界等待）+ errors 三维翻译 + device_state 四态 + recovery 五段式
- **六探针 6/6** + 结果 JSON（results_aline_20260826/）
- **flagcx 插件安装方法**（setup_flagcx_plugin.sh + flagcx_plugin_setup.md，幂等）
- **训练闭环**：Qwen2.5-1.5B 双卡 DDP 2481 步 loss 1.9501 / 4245 tok/s（flagcx）；训练脚本参数化 BACKEND/MAX_STEPS/PROFILE
- **A 线 stream 语义修复**：getStreamByIndex(0) 返回当前流、guardImpl+dlsym 取 torch_npu 当前流（FlagCX dev-1.0 基线已含）
- **双缓冲探针**：test_double_buffer.py 数据正确性 DBUF_PASS（重叠待做）
- **FlagCX 缺陷修复记录**：FLAGCX_CORE_DEFECT_FIXES（P2/P6/P7）

---

## 2. 环境配置（复用同构训练容器）

| 项 | 配置 | 来源/说明 |
|---|---|---|
| 设备 | 910C（10.120.72.27，2×Ascend 910 64G） | 同训练 |
| 容器 | `flagos-infer-910c`（host 网络 + 512G shm + 16 NPU + raid 挂载，见 `scripts/start_infer_container.sh`） | 9-01 实测；DrvMng 名额：多卡测试前 `docker ps` 清点 |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（华为官方 vllm-ascend；**不用 dev 容器做结论性测试**，坑 A4） | 9-01 实测跑通；**vllm-plugin-FL 路线弃用**（坑 A5） |
| Python/框架 | 容器内 py3.11 自带环境（**不复用 raid venv**：镜像 py3.11 vs raid venv py3.12，ABI 不匹配） | 9-01 实测 |
| vLLM | vLLM 0.20.2 + vllm-ascend 内置昇腾后端；**禁设 `VLLM_PLUGINS=fl`** | 坑 A5 |
| FlagCX | dev-1.0 基线（四层根因修复 + stream 语义修复已含） | 同训练；推理 A 线实际走 HCCL |
| 通信 | HCCL（torch_npu 官方，TP=2/4 实测正常）/ FlagCX（插件，P2 对照用） | 双后端对照 |
| 模型 | **Qwen3-4B**（对齐同事昇腾线基准，P2/**P3 全程用它**）/ Qwen2.5-1.5B（P0 历史基线） | 9-01 已下载至 `/mnt/raid/hliu553/models/Qwen3-4B` |
| 数据 | 少量 prompt 集（4-8 条，greedy 对照） | 推理验证不需大数据 |
| 环境变量 | `DO_NOT_TRACK=1`、首次请求预热、EngineCore 残留清理、**禁 `VLLM_PLUGINS=fl`** | 坑 A1/A2/A3/A5 |

**统一约定**：
- 所有脚本带 `BACKEND`（flagcx/hccl）与 `TP_SIZE` 参数化，复用训练脚本模式
- 输出 JSON 结果到 `results_inference_20260831/`，判定标准沿用 conformance 相对容差

---

## 3. 阶段拆解（编号化下一步）

### P0 环境就绪 + dense 单卡离线推理（0.5 天）

1. 容器内验证：`torch_npu` import、16 卡枚举、vLLM 0.20.2 + vllm-plugin-FL 插件激活（`Op 'attention_backend' using 'vendor.ascend'` 类日志）
2. Qwen2.5-1.5B 单卡离线推理（offline，几行 prompt）：输出正确 + 记录加载时间
3. 预热纪律：先跑 1 个短请求，再计时（坑 A1）
4. **验收**：输出文本合理、无 NaN/乱码；记录 baseline（latency、tok/s）

### P1 单卡推理 × 设备上下文职责（1-1.5 天）

1. **conformance 推理版**：在现有 13 用例基础上跑单卡推理场景（设备初始化/上下文/Stream/Event 在推理加载-前向-采样路径上的表现）；`--backend npu` 不变，新增 `--scenario infer` 开关
2. **双缓冲流水线真实现**：升级 test_double_buffer.py（多流 H2D/计算/D2H + Event 依赖链 + 重叠度测量，参照图 5-12）
3. **推理路径探针**：设备预热（pin_memory 依赖）、KV 分配后事件语义
4. **验收**：conformance 推理版 PASS + 双缓冲重叠可观测（timeline 显示传输与计算重叠，非串行）

### P2 TP=2/4 推理 × Stream/Event（2 天，本周核心）

1. TP=2 推理脚本：`VLLM_FL_TP=2` 参数化（沿用成员阶段 4 经验，改 A 线 torch_npu + flagcx）
2. **A 线重验 B1/B2**：
   - B1 bool×int：TP=2 输出是否仍退化（torch_npu 类型语义）
   - B2 flagcx 异步无同步：TP 通信后数据可见性（我们的 stream 语义修复是否已覆盖推理路径）——**这正是 Stream/Event 职责的验收点**
3. TP 通信 × conformance：S3/S4（跨流显式依赖）在 TP allreduce 路径上的验证；E2 有界等待
4. TP=4 扩展（可选）：链路闭环 + 确定性分叉判定（坑 B6）
5. **验收**：TP=2 与 TP=1 输出一致（或记录确定性分叉）；无 NaN；flagcx 同步语义结论明确

### P3 服务化 × 状态监控/错误恢复（1-1.5 天）

1. `vllm serve` 在线服务：长驻运行；**EngineCore spawn 子进程的设备上下文**（坑 A2）——设备初始化/上下文在子进程的验证
2. **DeviceState 四态监控**：推理服务运行中查询设备状态（AVAILABLE/DEGRADED/ISOLATED），压力注入（并发请求/大 KV）观察状态转换
3. **错误捕获/恢复**：注入 ACL 错误（如 107015 stream callback）→ 五段式恢复在长驻服务的应用
4. **验收**：serve 长稳 + 状态查询可用 + 注入错误可捕获可恢复

---

## 4. 设备上下文职责 × 推理场景映射（验收对照表）

| 职责项 | 训练已验证 | 推理验证点（本方案） | 阶段 |
|---|---|---|---|
| 设备初始化/上下文创建 | ✅ | 推理加载路径 + EngineCore 子进程 | P0/P3 |
| 执行队列 | ✅ S1-S4 | 推理多请求队列（vLLM scheduler） | P1 |
| Stream/Event | ✅ E1-E3 | TP 通信流绑定 + 双缓冲多流 + graph capture | P1/P2 |
| 设备间同步 | ✅ 双卡 DDP | TP=2 allreduce 数据可见性（重验 B2） | P2 |
| Host↔Device 传输 | ✅ T1-T3 | KV offload / D2H 采样回传（双缓冲流水线） | P1 |
| 错误捕获 | ✅ F1 | 推理 ACL 错误注入 | P3 |
| 状态恢复 | ✅ R1-R5 | 长驻服务四态 + 恢复 | P3 |
| 双缓冲流水线 | ⚠️ 数据正确性 | **重叠可观测（真实现）** | P1 |

---

## 5. 风险与开放问题

| # | 风险/问题 | 影响 | 对策 |
|---|---|---|---|
| 1 | vLLM 0.20.2 + torch_npu 2.10.0 组合未在官方镜像验证过 | P0 可能卡环境 | 先跑 memory 方向已验证的 dense 组合；不行则对齐 memory 方向的 venv |
| 2 | TP 推理在 A 线是否有新挂点（如 scatter_ 类） | P2 阻塞 | 坑 B7 预判；torch_npu 原生算子覆盖更全，大概率无 |
| 3 | flagcx 同步语义在推理路径是否已修复 | P2 核心结论 | 若未修复 → 扩 A 线 stream patch（复用训练经验）；若已修复 → 记录证据 |
| 4 | DrvMng 容器名额（≈3） | 多卡测试冲突 | 多卡前 docker ps 清点；容器用完即停 |
| 5 | 双缓冲重叠测量方法（框架层可观测性） | P1 验收 | npu_events 轮询 + 墙钟分段时间线；必要时 device 级采样 |

---

## 6. 下一步（编号）

1. **[P0]** 容器启动 + venv + vLLM/vllm-plugin-FL 验证 + dense 单卡推理（用户执行，AI 提供命令与判定）
2. **[P1]** conformance 推理版（--scenario infer）+ 双缓冲流水线真实现脚本
3. **[P2]** TP=2 推理脚本（A 线参数化）+ B1/B2 重验探针
4. **[P3]** serve 长稳 + 四态监控 + 错误注入探针
5. 每阶段结果 JSON 归档 `results_inference_20260831/`，复盘记录坑（延续调试 trap 清单习惯）
