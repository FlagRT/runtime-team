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

## 三之二、A4 深挖：双缓冲重叠分段定位（test_double_buffer_breakdown.py，已执行）

### 实测数据（910C，torch_npu 2.10.0）

| 场景 | 结果 | 判读 |
|---|---|---|
| S1 H2D non_blocking（当前流） | 入队 0.19ms / 完成 0.92ms（**入队占比 20.7%**） | ✅ **异步正常**（非同步退化） |
| S2 H2D 同步拷贝基线 | 1.46ms（> 异步完成 0.92ms） | ✅ 异步确实有效 |
| S3 H2D 专用流 | 完成 0.47ms（与当前流相当） | ✅ 无显著流切换开销 |
| S4 计算（当前流，首次） | **77.56ms** | ⚠️ **首次算子初始化开销** |
| S5 计算（专用流，稳态） | 0.72ms | 稳态计算（S4/S5 差 107 倍） |
| S6 事件链 6×(record+wait) | 0.85ms → **单次 0.142ms** | ✅ 开销小，非主导 |
| S7 完整流水线 512² | 重叠率：**-66.0% / +17.7%**（两次运行） | ⚠️ 噪声主导，不可靠 |
| S8 完整流水线 2048² | 重叠率 -1.8% | 接近打平 |
| S9 完整流水线 4096²×10（预热后） | 重叠率 **+1.7%**（pipe 30.52ms / serial 31.03ms） | ⚠️ 重叠收益微弱 |

### 结论（三排除 + 一待验证）

1. ✅ **排除"copy_ non_blocking 退化同步"**——S1 入队占比仅 20.7%，S2 同步基线明显更慢，异步机制正常
2. ✅ **排除"事件链开销主导"**——单次 record+wait 仅 0.142ms
3. ✅ **排除"流切换开销"**——S3 与当前流无显著差异
4. ⚠️ **真实情况：重叠收益微弱 + 小负载测量噪声大**——512² 两次运行 -66%~+17.7% 抖动；大负载 4096²×10 仅 +1.7%
5. 🔍 **待验证推测**：CANN/昇腾多流并发能力可能受限（硬件队列或流调度），或每批 D2H 同步点阻断流水线 → 需 torch.profiler / ACL profiling 看**多流时间线**确认是否真并发

### 附带硬发现（推理预热纪律的直接证据）

**首次算子开销 107 倍**：S4 首次 512² matmul 77.56ms vs S5 稳态 0.72ms。
这是历史坑 A1（"首次 attention 13+ 分钟"）在 matmul 层的实测印证——**所有推理基准必须先预热**，否则第一次测量完全被算子初始化污染。

### 下一步（A4 收尾）
1. torch.profiler 抓多流时间线（确认 CANN 是否真并发）——决定性证据
2. 简化流水线：去掉每批 D2H（真实推理 D2H 只是小张量 logits，非 64MB 矩阵），减少同步点后复测
3. 若确认昇腾多流重叠收益有限 → 改为"单流效率优化 + 批量合并"策略，结论入档（D8 职责结论）

## 四、环境缺口（P0 dense 推理前置）

- vLLM 未安装（/root/tf-venv-integration 无 vllm）、workspace 无 vllm-plugin-FL clone
- P0 前置：① 容器内 clone vllm-plugin-FL；② 装 vLLM 0.20.2（官方发布镜像内，坑 A4）；③ 用 dense_infer_qwen_1_5b_npu.py 验证
- 注意：当前 tf-venv-integration 是 B 线 venv（torch 2.10.0+cpu + torch_npu），vLLM 依赖 torch 的 GPU 版——可能需要新 venv 或确认 vLLM 兼容性

## 五、资产（results_inference_20260831/）

- conformance_ascend_infer_result.json（6/6）
- double_buffer_pipeline_result.json（DBUF2_PARTIAL）
- double_buffer_breakdown_result.json（A4 分段定位：三排除 + 重叠微弱 + 首次算子 107 倍）
- kernel_timeline_analysis.json（**A4 决定性证据：多流真并发 + EVENT_WAIT 开销 3.37ms**）
- stream_profiler_l1_result.json（Level1 kernel trace 元信息）
- 脚本：conformance/infer_cases.py、ascend_regression/test_double_buffer_pipeline.py、
  ascend_regression/test_double_buffer_breakdown.py、ascend_regression/test_stream_profiler_l1.py、
  ascend_regression/analyze_l1_timeline.py、inference/dense_infer_qwen_1_5b_npu.py

---

## 三之三、A4 收尾：kernel 级时间线决定性证据（已拿到）

### 采集方式（关键：必须用昇腾专用 profiler + Level1）

| 尝试 | 结果 |
|---|---|
| `torch.profiler(PrivateUse1)` | ❌ 只有 `cpu_op`（126 个），无 device kernel |
| `torch_npu.profiler` 默认 level | ⚠️ 只有 `cpu_op` + `enqueue/dequeue`（队列瞬间事件，非执行窗口） |
| **`torch_npu.profiler` + `ProfilerLevel.Level1`** | ✅ **209 个 kernel 级事件**（`aclnnMatmul_MatMulCommon_MatMulV2` 等，带真实 dur） |

> 注：torch_npu.profiler 无 `key_averages()`（API 与 torch.profiler 不同）；trace 导出为 list 格式（非 dict.traceEvents）。

### 决定性数据

| 指标 | 数值 |
|---|---|
| kernel 事件 | 209（copy 18 / matmul 14 / reduce 15 / zero 6 / event 26 / other 130） |
| **copy ∩ matmul 重叠对** | **5 处**（例：copy 348.37us ∥ matmul 10.85us → **重叠 10.75us**） |
| **EVENT_WAIT 同步开销** | **12 次累计 3365.31us（3.37ms），最大单次 391.59us** |
| 时间线跨度 / kernel 累计 | 116.87ms / 16.11ms |

### 最终结论（D8 双缓冲流水线）

1. ✅ **昇腾多流真并发成立**——H2D（DMA）与 matmul（AI Core）在 kernel 时间线上有 5 处实重叠，matmul 完全落在 copy 的 DMA 窗口内执行
2. ✅ **异步传输机制正常**——S1 入队占比 20.7%，非同步退化
3. 🔥 **性能瓶颈 = 事件等待开销，不是并发能力**——12 次 `EVENT_WAIT` 累计 **3.37ms**、最大单次 **391.59us**；而单次 matmul 仅 ~10us → **一次事件等待 ≈ 39 个 matmul 的时间**
4. **"流水线比串行慢"的真因**：每批 2 个事件同步点在 CANN 上成本高，小工作负载下同步成本远超并发收益

### D8 职责验收结论（可入档）

- **机制层面：通过**——异步传输、多流并发、事件依赖语义、数据一致性全部验证成立
- **性能层面：需优化**——瓶颈在事件同步点开销，优化方向是**减少同步频率**（批量合并 / 每 N 批一次同步 / streamWaitEvent 替代跨流 event），**不是放弃多流**
- **训练侧回补同理**：按"减少事件点 + 大粒度批量"策略实施

---

## 四之二、P0 dense 推理跑通（2026-09-01，DENSE_INFER_PASS）

### 结果

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-1.5B（`/mnt/raid/hliu553/models/`，raid） |
| 环境 | vllm-ascend 镜像自带：**Python 3.11 + torch 2.10.0+cpu + torch_npu 2.10.0 + vllm 0.20.2 + vllm_ascend** |
| 加载 | 39.0s |
| 推理 | 6.21s / 121 tokens / **19.5 tok/s** |
| 判定 | **DENSE_INFER_PASS**（4 条 prompt 输出均正确、无 NaN/乱码） |

输出样例：
```
"Hello, my name is"      → " Matthew Bais, currently a third year PhD student..."
"The capital of France is"→ " one of the most popular destinations in Europe..."
"2+2="                   → "____ (1)2×2=4 (2)2+2=4;;故答案为:4"
"Python is a"            → " leading server-side and web-development language..."
```

### 跑通路径（关键，供复现）

1. **用 vllm-ascend 官方镜像起容器**：`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（脚本 `scripts/start_infer_container.sh`）
2. **腾 DrvMng 名额**：起容器后 `devices 0` + `DrvMngGetConsoleLogLevel failed` → 停掉其他占 NPU 的容器（本轮停了 `flagos-hliu553-dev-910c`）后 `restart` 即恢复（**16 卡可见**）
3. **数据全在 raid**：模型 `/mnt/raid/hliu553/models/`、脚本/结果在 `/mnt/raid/hliu553/runtime-team/...`、`TMPDIR=/mnt/raid/hliu553/tmp`
4. 运行：`python3 dense_infer_qwen_1_5b_npu.py --model <raid路径> --preheat --max-tokens 32`

### 踩坑记录（本轮新增）

| 坑 | 现象 | 解法 |
|---|---|---|
| 官方 vLLM 无 ascend platform | `Device string must not be empty` | 用 vllm-ascend 镜像（不是 pip 装官方 vllm） |
| 源码编译 vllm-ascend | 缺 `regex`（系统 python 也要装）→ 缺 `triton-ascend==3.2.1`（不在 PyPI）→ CANN ops prepare 失败（`/mc2/...` 路径缺失） | **放弃源码编译**，用官方镜像 |
| Python ABI 不匹配 | 镜像 Python 3.11 vs 我们 raid venv 3.12 → 提取的 `vllm_ascend` .so 不可用 | 直接用镜像自带环境（3.11） |
| DrvMng 名额 | 新容器 `devices 0` | 停掉一个占 NPU 的容器后 restart |

### 通用纪律（本轮沉淀）
1. **所有基准必须预热**：首次算子开销 107 倍（512² matmul 77.56ms vs 稳态 0.72ms）
2. **profiler 必须用 Level1**：默认配置采集不到 kernel（只有入队/出队瞬间事件）
3. **torch_npu.profiler ≠ torch.profiler**：无 key_averages，trace 为 list 格式
