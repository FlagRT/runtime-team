# 推理 P0/P1 首轮执行记录（2026-08-31）

> 执行人：Kistich（AI 代跑容器验证）｜ 环境：910C flagos-hliu553-dev-910c，/root/tf-venv-integration（torch_npu 2.10.0，16 卡可见）
> 路线：A 线（torch_npu，不走 torch_fl）｜ 对应方案：DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md

## 一、结论速览

| 项 | 结果 | 判定 |
|---|---|---|
| conformance 推理用例（infer_cases 6 例） | **6/6 PASS**（i1 加载上下文 / i2 多轮前向流序 / i3 KV 跨流可见性 / i4 D2H 采样 / i5 长驻状态 / i6 流水线依赖链） | ✅ CONFORMANCE_PASS |
| 双缓冲流水线升级版（多流+Event） | **DBUF2_PARTIAL**：数据一致 ✅，重叠率 **-1306%**（pipe 0.0736s vs serial 0.0052s） | ⚠️ 重叠未发生，需深挖 |
| dense 单卡推理（P0） | 未跑（vLLM 未安装、workspace 无 vllm-plugin-FL） | ⏳ 待环境准备 |

## 二、conformance 推理版（infer_cases.py，6/6 PASS）

| 用例 | 验证点 | 结果 |
|---|---|---|
| i1_device_context_after_load | 推理加载后设备上下文可用（权重/输入/前向） | PASS（logits.shape=(4,64)） |
| i2_infer_forward_stream_order | 多轮前向同流顺序、固定输入确定性 | PASS（逐轮误差 0.00e+00） |
| i3_kv_buffer_visibility | KV 模拟缓冲跨流可见性（写入流→事件→计算流） | PASS（读回全 8） |
| i4_d2h_sample_transfer | D2H 采样回传（logits.cpu()→topk） | PASS |
| i5_longrun_device_state | 长驻 20 轮无 NaN/Inf 漂移 | PASS |
| i6_pipeline_overlap | 双缓冲流水线事件依赖链数据正确 | PASS（rel_err=3.09e-07） |

**修正记录（首跑 2 个 FAIL → 修正后全过）**：
- i2：原实现每轮生成随机输入，无可比性（用例 bug）→ 改固定输入，验证确定性
- i6：原断言拿 hosts[0] 对比末轮（b=1）输出（断言 bug）+ 相对容差 → 按轮对应 + rel_err<1e-3

## 三、双缓冲流水线（test_double_buffer_pipeline.py，DBUF2_PARTIAL）

### 现象
- 数据正确 ✅（6 批交替传输+计算+回传一致）
- **重叠未发生**：pipeline 0.0736s 反而比串行 0.0052s 慢 **14 倍**（重叠率 -1306%）
- 事件依赖链语义成立（ev_h2d→计算流 wait、ev_calc→回传流 wait 均生效，数据正确证明无竞争）

### 分析（待验证的假设）
1. **copy_ non_blocking 可能退化为同步**：torch_npu 的 `tensor.copy_(host, non_blocking=True)` 若内部走同步拷贝路径，双缓冲"传输与计算重叠"的基础就不成立 → 每轮传输阻塞
2. **`with torch.npu.stream(s)` 切换开销/语义**：上下文管理器可能未真正把 copy_ 调度到指定流（或每次切换有隐式同步）
3. **事件 record/wait 的 CANN 开销**：每轮 2 次事件同步（~12ms/轮？），512² 计算本身 ~1ms，同步开销主导
4. **CANN 低层行为**：aclrtMemcpyAsync 在 NPU 上的 H2D 是否真异步待确认（CUDA 的 pin_memory+DMA 在昇腾未必等价）

### 下一步（P1 深挖探针）
1. 分段计时：纯 6×copy_（non_blocking）耗时 / 纯 6×matmul / 事件链开销——分离传输/计算/同步成本
2. 对比 aclrtMemcpyAsync（CANN 低层）vs torch copy_ 的异步性
3. 大张量（2048²）重复实验：若计算时间 >> 同步开销时重叠出现，则证实"同步开销主导"而非"异步机制缺失"
4. 结果如实入档（若 torch_npu 多流异步存在真实缺口 → 记为 A 线 Stream 语义缺口，可能关联 FlagCX/上游）

## 四、环境缺口（P0 dense 推理前置）

- vLLM 未安装（/root/tf-venv-integration 无 vllm）、workspace 无 vllm-plugin-FL clone
- P0 前置：① 容器内 clone vllm-plugin-FL；② 装 vLLM 0.20.2（官方发布镜像内，坑 A4）；③ 用 dense_infer_qwen_1_5b_npu.py 验证
- 注意：当前 tf-venv-integration 是 B 线 venv（torch 2.10.0+cpu + torch_npu），vLLM 依赖 torch 的 GPU 版——可能需要新 venv 或确认 vLLM 兼容性

## 五、资产（results_inference_20260831/）

- conformance_ascend_infer_result.json（6/6）
- double_buffer_pipeline_result.json（DBUF2_PARTIAL）
- 脚本：conformance/infer_cases.py、ascend_regression/test_double_buffer_pipeline.py、inference/dense_infer_qwen_1_5b_npu.py
