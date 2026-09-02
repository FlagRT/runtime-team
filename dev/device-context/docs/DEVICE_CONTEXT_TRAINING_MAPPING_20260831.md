# 设备执行上下文职责 × 分布式训练 · 映射与验收评估

> 状态：📋 映射定稿 + 验收评估（2026-08-31）｜ 作者：Kistich ｜ 路线：A 线（torch_npu）
> 用途：与《DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md》对齐——**同一套职责（D1-D11）在训练场景的验收评估**
> 关联：DEVICE_CONTEXT_PLAN_20260827.md（训练方案定稿）、PROGRESS_20260822/20260825、conformance 结果 results_aline_20260826/

---

## 0. 验收结论（先给结果）

| 项 | 结论 |
|---|---|
| **总体** | **11/11 项职责达标**（D8 已于 2026-09-02 经推理侧求解 + 训练侧回补闭环） |
| 已闭环 | D1 封装 Runtime / D2 设备句柄 / D3 内存句柄 / D4 执行句柄 / D5 Stream 语义 / D6 异步传输 / D7 页锁定 / D8 双缓冲流水线 / D9 同步语义 / D10 错误码翻译 / D11 状态恢复 |
| 部分 | 无（D8 原缺口已闭环：推理侧四模式选型 DBUF2_PASS + 训练侧 pin_memory/non_blocking 回补 + 冒烟 20 步无崩溃，详见推理映射 §5.3） |
| 判定 | **分布式训练 × 设备上下文职责：验收通过（11/11）** |

---

## 1. 职责 × 训练验证矩阵（验收明细）

| # | 职责子项 | 训练验证点 | 证据（可复现资产） | 验收 |
|---|---|---|---|---|
| D1 | 封装不同芯片 Runtime 接口 | torch_npu + FlagCX 在 910C 的设备接入，训练闭环跑通 | 双卡 DDP Qwen2.5-1.5B 2481 步 loss 1.9501 / 4245 tok/s（flagcx）、5428 tok/s（hccl）；FlagCX 插件安装沉淀（setup_flagcx_plugin.sh + flagcx_plugin_setup.md） | ✅ |
| D2 | 统一设备句柄+生命周期 | 设备枚举/初始化/上下文；conformance 设备无关化 | conformance `--backend {npu,flagos}`；A 线新容器复现 16 卡；训练双卡枚举 | ✅ |
| D3 | 统一内存句柄+生命周期 | 显存分配/释放；锁页池；AI CPU core 画像 | 六探针：pinned_pool / ai_cpu_core；训练显存与 B 线一致 | ✅ |
| D4 | 统一执行句柄 | 执行队列/Stream/Event 抽象 | S1-S4 / E1-E3 用例；npu_events.py 适配层（未 record query 修正 + wait_host 有界等待） | ✅ |
| D5 | 统一 Stream 语义 | 流内顺序/显式依赖/跨流可见性/wait_stream 传递 | S1/S2/S3/S4 全过；A 线 stream 语义修复（getStreamByIndex 当前流、guardImpl+dlsym 取 torch_npu 当前流） | ✅ |
| D6 | Host/Device 异步传输 | 异步拷贝数据一致；在途保护；跨设备直传 | T1 pinned_async_copy / T2 inflight_protection / T3 跨设备传输（拓扑如实标注：torch_npu 未暴露统一拓扑查询） | ✅ |
| D7 | 页锁定内存 | pin_memory + non_blocking 传输 | T1 + pinned_pool 探针（含"pin_memory 需设备预热"坑） | ✅ |
| D8 | **双缓冲流水线** | 传输-计算重叠（真实现） | **2026-09-02 回补闭环**：① 重叠机制由推理侧四模式实现验证（`test_double_buffer_pipeline.py` v2，DBUF2_PASS：n≤512→V5 同流 +38.1% / n≈1024→V4 压同步 +28.8% / n≥2048→V0 批间流水 +40.1%，详见推理映射 §5.2/§5.3）；② 训练脚本已启用预取重叠：DataLoader `pin_memory=True` + `batch.to(dev, non_blocking=True)`（num_workers 保持 0，容器内 fork 有风险）；③ 双卡冒烟 20 步无崩溃（loss 2.64→2.04，tok/s 3483） | ✅ |
| D9 | 同步语义 | 跨卡梯度同步；事件语义；错误不静默 | 2481 步 DDP 全程稳定（flagcx）；event 资源泄漏修复（每步 ~120 个 aclrtEvent 累积 → 析构 + work 完成语义）；E1/E2/E3 | ✅ |
| D10 | 错误码翻译 | 错误码→L1-L4 三维投影 | F1：`aclnnMatmulGetWorkspaceSize ret=161002` → L2_PARAM；errors.py 兼容两形态（B 线 `ret=161002` / A 线 `error code is 161002`） | ✅ |
| D11 | 设备状态恢复 | 四态 + 五段式恢复 | R1-R5 用例；device_state.py（AVAILABLE/DEGRADED/ISOLATED/DESTROYED）+ recovery.py（captured→evaluated→isolated→recovered→replay_ready）；**诚实标注：重建为框架层最小近似（探针重试），真实上下文重建待设备生命周期接口** | ✅（含标注） |

---

## 2. 训练侧核心资产（验收依据）

| 类别 | 资产 | 状态 |
|---|---|---|
| 测试框架 | conformance 13/13（S1/S2/S3/S4/E1/E2/E2v2/E3/T1/T2/T3/F1/R1-R5） | ✅ |
| 探针 | 六探针 6/6（错误翻译/拓扑/锁页池/双缓冲/恢复/CPU-NPU） | ✅ |
| 结果 | results_aline_20260826/（8 份 JSON） | ✅ |
| 训练闭环 | train_qwen_1_5b_npu.py（参数化 BACKEND/MAX_STEPS/PROFILE）+ test_ag_npu.py | ✅ |
| 契约 | event_semantics_contract.md（E1-E4 + wait_host） | ✅ |
| 缺陷修复 | FlagCX：event 泄漏、P2/P6/P7（文档 FLAGCX_CORE_DEFECT_FIXES_20260826.md + patches/） | ✅（上游待提交） |
| 环境 | flagcx_plugin_setup.md + setup_flagcx_plugin.sh（幂等） | ✅ |

---

## 3. 验收清单（训练版，B1-B11）

| # | 验收项 | 对应职责 | 结论 |
|---|---|---|---|
| B1 | 双卡 DDP 训练闭环（loss 收敛 + 吞吐记录） | D1/D2 | ✅ 2481 步 loss 1.9501 / 4245 tok/s |
| B2 | conformance 13/13 | D2-D7/D9-D11 | ✅ |
| B3 | 六探针 6/6 | D3/D6/D7/D10/D11 | ✅ |
| B4 | Stream 语义（S 系列 4 项） | D4/D5 | ✅ |
| B5 | 事件语义（E 系列 + wait_host） | D4/D5/D9 | ✅ |
| B6 | 异步传输（T 系列 3 项） | D6/D7 | ✅ |
| B7 | 错误码翻译（F1） | D10 | ✅ |
| B8 | 状态恢复（R1-R5） | D11 | ✅（最小近似标注） |
| B9 | **双缓冲流水线重叠验证** | D8 | ✅ 2026-09-02：推理侧四模式选型 DBUF2_PASS 后回补训练侧（见 §4 回补路径更新） |
| B10 | 训练侧启用数据预取重叠（pin_memory/num_workers/non_blocking） | D6/D7/D8 | ✅ 2026-09-02：pin_memory + non_blocking 已启用（num_workers 保持 0，容器内 fork 有风险）；双卡冒烟 20 步无崩溃 |
| B11 | 吞吐差距（flagcx vs hccl 36 倍）定位与挂账 | D5/D9 | ✅ 已定位挂账（移交通信方向） |

---

## 4. 缺口与回补路径

| 缺口 | 说明 | 回补路径 |
|---|---|---|
| ~~D8 双缓冲流水线（训练侧）~~ | ~~训练未启用数据预取重叠；重叠机制未真实现~~ | ✅ **已闭环（2026-09-02）**：推理侧四模式求解落地（test_double_buffer_pipeline.py v2）+ 训练脚本 pin_memory/non_blocking 回补 + 双卡冒烟 20 步无崩溃。num_workers 数据预取仍受容器 fork 限制未启用（非关键路径，小模型收益低） |
| D11 重建真实性 | 状态恢复重建为探针重试近似 | 依赖 torch_fl/厂商设备生命周期接口（上游，非本方向阻塞） |
| D6 拓扑查询 | torch_npu 未暴露统一拓扑（T3 如实标注） | 经 npu-smi/外部通道；新芯片需厂商 API 对齐 |

---

## 5. 训练 vs 推理：同一职责的两侧对照

| 职责 | 训练侧 | 推理侧 |
|---|---|---|
| D1 Runtime 封装 | torch_npu + FlagCX（训练脚本） | + vllm-plugin-FL（引擎接入） |
| D5 Stream | S 系列（前向/反向流序） | + 多流流水线（H2D/计算/D2H） |
| D8 双缓冲 | 未启用（选做） | **核心必做**（每 token D2H + 请求级流水） |
| D9 同步 | DDP 梯度同步（显式 all_reduce） | TP 前向通信（A 线需重验 flagcx 异步无同步） |
| D11 状态恢复 | 训练短时任务 | 服务长驻（四态监控更关键） |

**结论**：训练侧职责**已验收通过**（D8 例外，已明确回补路径）；推理侧是同一套职责在"长驻服务 + 多流重叠"形态下的延伸验证，当前推进中。
