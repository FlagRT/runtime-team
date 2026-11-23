# NVIDIA A6000 × 昇腾 910C × 昆仑芯 P800 调度/组批数据汇总

> （迁自 FlagScheduler `reports/SUMMARY.md`，2026-11-23；后续在本分支维护）
>
> 数据分目录（按平台）：
> - [`platforms/nvidia-a6000/`](platforms/nvidia-a6000/)：NVIDIA A6000（来源为验证记录，原始产物在 GPU 机器上）
> - [`platforms/ascend-910c/`](platforms/ascend-910c/)：昇腾 910C（本轮真机，原始 JSON 在 `data/`）
> - [`platforms/kunlunxin-p800/`](platforms/kunlunxin-p800/)：昆仑芯 P800（2026-11-10 真机，原始 JSON 在 `data/`；
>   数值缺陷根因排查全记录见 [`investigation/INVESTIGATION_20261117.md`](platforms/kunlunxin-p800/investigation/INVESTIGATION_20261117.md)）

## 0. P800 平台缺陷速览（2026-11-17 定案，详见排查记录）

| | 缺陷一：prefix-caching 命中 | 缺陷二：跨 prefill 调度步 |
| --- | --- | --- |
| 触发 | 缓存开（vllm 0.11 默认）+ 长文本 KV 命中 | 单次调用总 token > 16384 |
| 性质 | 命中重算最后 token 位置/偏移扭曲 | 后续步序列块表指向无效内存，从垃圾数据计算 |
| 波及 | embedding + 生成（贪心解码第 1 个 token 起分叉） | embedding 后续步全坏；生成 16/16 全坏（解码跨步使步 1 也坏） |
| 规避 | `enable_prefix_caching=False` | 单次调用 ≤16384 tok（外层组批预算可钳制） |

- 映射契约（结果-请求对应）经交叉余弦矩阵与顺序交换实验证明**始终正确**
- 定位：昆仑芯算子层的**寻址/元数据分支**（主路径算术逐位正确），非 vLLM 调度/模型/本仓代码
- 安全包络实测：关缓存 + 单次 ≤16384 tok 时 64 条任意混长批逐位正确
- 本仓 P800 数值列（B/C vs A cos 为负）全部由此两缺陷造成；性能数字（吞吐/延迟）测量路径不受影响仍有效

## 1. 可比性先说清楚

| 对比项 | 是否可直接横比 | 说明 |
| --- | --- | --- |
| A 直连 vs B32 统一入口（embedding） | ✅ | 三边都是 vllm-embed、Qwen3-Embedding、192 条 mixed、闭环比吞吐 |
| 分桶 C（embedding） | ❌ | NVIDIA 的 C 是“自定义 scheduler 插件（继承原生）”，910C/P800 的 C 是“长度分桶”，不是同一个变量 |
| NVIDIA 的 +61% 组批收益 | ⚠️ 仅方向参考 | 来自 torch-predict 300 条，与 embedding 不是同一后端/负载 |
| 绝对吞吐/延迟 | ❌ | 硬件代际、vLLM 版本（0.21.0 / 0.20.2 / 0.11.0+kunlun 插件）、eager/编译、prefix caching 都不同 |
| P800 数值一致性 | ❌ 平台缺陷 | vllm-kunlun 0.11.0 长短混批 pooling 输出错误（见 `platforms/kunlunxin-p800/investigation/README.md` §3），B/C vs A 的 cos 为负不能按“批组成漂移”解读 |

## 2. 吞吐趋势：方向相反

| 平台 | A 直连 | B batch=32 | Δ vs A |
| --- | --- | --- | --- |
| NVIDIA A6000（192 条） | 467.6 req/s | **561.3 req/s** | **+20%** |
| 910C（192 条，token 预算 32768） | **110.5 req/s** | 89.9 req/s | **-19%** |
| 910C（token 预算 131072，B128=2 次调用） | 110.5 req/s | 96.1 req/s | **-13%** |
| P800（192 条，token 预算 32768） | **87.6 req/s** | 82.1 req/s | **-6%** |

- NVIDIA：batch=8 时 -12%（调用开销主导），放到 32 反超直连 +20%。
- 910C：所有切档都低于直连；外批次越少越好（6 次 -19%、2 次 -13%），单次直连最优。
- P800：所有切档同样低于直连（-5%~-14%），幅度比 910C 小；B64/B128 同样被 32768 token 预算钳成 6 批，档位未拉开。
- **结论：910C 与 P800 在实测范围内都不存在“外层组批提升吞吐”的收益；NVIDIA 的 +20% 不能迁移。**

## 3. 延迟趋势：方向一致

| 平台 | A p50 | 最优分桶/小批 p50 | 变化 |
| --- | --- | --- | --- |
| NVIDIA A6000 | 410.5 ms | 206 ms（B32）/ 233 ms（C） | -50% / -43% |
| 910C | 1737.7 ms | 528–555 ms（C64/C32） | **-68% ~ -70%** |
| P800 | 2192.2 ms | **381–384 ms（C32/C64）** | **-82% ~ -83%（三平台最大降幅）** |

- 两边都验证了同一件事：**把大批拆成更小、长度更一致的外批次，单请求 p50 明显下降**。
- 代价不同：NVIDIA 在 batch=32 时能同时拿到吞吐与延迟；910C 只能拿到延迟，吞吐下降 13%–29%。
- 910C 上 C 的 padding 从 0.48–0.57 降到 0.085–0.126，但吞吐仍最低——**padding 不是 910C 的瓶颈**。

## 4. 数值一致性：口径不同，不能互相套用

| 平台 | 批组成差异 | 同组成重复调用 | 可支撑的结论 |
| --- | --- | --- | --- |
| NVIDIA | fp32 max 3.9e-6；fp16 max ~6e-2 | — | fp32 下“批组成不改变输出” |
| 910C | B/C vs A max abs 3.0–4.1e-3，cos 0.9997–0.9998 | A 跨轮完全一致（0.0） | 只能按 ID 映射验收，不能断言逐元素一致 |
| P800 | B/C vs A cos **为负**（平台缺陷，非漂移） | A 跨轮完全一致（0.0）；同长度批 cos=1.0 | 只能按 ID 映射验收；**根因已定位（2026-11-17）**：prefix-caching 命中 + 跨 prefill 步两个独立缺陷，规避=关缓存且单次调用 ≤16384 tok，见 `platforms/kunlunxin-p800/investigation/README.md` §3 |

910C 上“同组成重复调用确定、批组成变化引入 ~3e-3 差异”是本轮新增事实。此前 32 请求脚本里 legacy vs scheduled 逐元素 0/32，源于批组成不同 + 默认开 prefix caching，不是跨调用随机漂移。
NVIDIA 的 native vs custom scheduler 对照为 max abs 2.83e-3、cos 偏差 1.28e-4，与 910C 的 3e-3 量级接近，但两者测的不是同一个变量。

## 5. 结构性差异与解释

1. **910C 的外层切批是串行开销**：单次直连已经把 192 条交给 vLLM 内部连续组批；外层再切只是把一次调用拆成多次串行调用（6 次 6.7 s、2 次 5.7 s、1 次 4.7 s 量级），没有重叠收益。
2. **NVIDIA 的优势来自外层批次的“调用摊薄 + 分桶局部性”**：batch=8 有固定调用开销，32 时被摊薄，同时分桶减少 padding，因此反超。
3. **进程内并发不可行**：`--pipelined-workers 2` 在 910C 上挂死（线程 `S`、AICore 0%），vLLM 同步 `LLM.embed` 非线程安全。若要让 910C 拿到“多批并发”，必须换 `AsyncLLMEngine` 或服务化部署（NVIDIA 侧同样未做该实验）。
4. **决策层不是瓶颈**：NVIDIA 微基准 submit 2.3–2.6 µs/请求、派发 1.9–3.5 µs/请求（<0.3%）；910C 的 execution p50 为 82 ms，差距在设备执行侧。

## 6. 限制

- 910C 为单卡单实例；matrix 第一轮 3 轮、第二轮 2 轮；`enforce_eager=True`、未开 prefix caching。
- NVIDIA 数据来自 `docs/flagscheduler_验证记录.md` 的历史记录；原始产物（`artifacts/nvidia_gpu3*`）不在本仓，无法重新解析。
- 两边样本量、预热控制和运行轮次不同；绝对数字（如 467.6 vs 110.5 req/s）不能作为硬件性能结论。
- 910C 的 B64/B128 在 32768 token 预算下被钳成与 B32 相同的批次组成，只有第二轮 131072 才拉开档位。
- P800（2026-11-10）单卡 xpu2、进程内引擎（`VLLM_ENABLE_V1_MULTIPROCESSING=0`，见 `platforms/kunlunxin-p800/investigation/README.md` §4）；数值一致性受平台混批 pooling 缺陷阻塞。

## 7. 复现

NVIDIA（GPU 机器，仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=3 python benchmarks/bench_embedding_batching.py \
  --executor vllm --model /path/to/Qwen3-Embedding-0.6B \
  --scenario mixed --num 192 --out-dir artifacts/nvidia_abc
```

910C（容器内，`ASCEND_RT_VISIBLE_DEVICES=0`）：

```bash
python benchmarks/bench_matrix_compare.py --executor vllm \
  --model /models/Qwen3-Embedding-0.6B --num 192 --rounds 3 \
  --batches 32,64,128 --max-input-tokens 2048 --enforce-eager \
  --out-dir artifacts/npu_matrix

# 拉开 batch 档位（token 预算加倍，只跑 B/C）
python benchmarks/bench_matrix_compare.py --executor vllm \
  --model /models/Qwen3-Embedding-0.6B --num 192 --rounds 2 \
  --batches 32,128 --modes B,C --max-total-tokens 131072 \
  --enforce-eager --out-dir artifacts/npu_matrix_budget
```

P800（容器 `xianyiyuan-vllm-kunlun` 内，`XPU_VISIBLE_DEVICES=0`）：

```bash
XPU_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 VLLM_PLUGINS=kunlun \
VLLM_ENABLE_V1_MULTIPROCESSING=0 python benchmarks/bench_matrix_compare.py \
  --model /work/models/qwen3-embed-0.6b-xpu --enforce-eager \
  --out-dir benchmarks/results/kunlunxin-p800
```

## 8. 建议的下一步

1. 910C 上补 batch=8/16，确认低档位是否同样无收益（当前只测了 32/64/128）。
2. 910C 上开/关 prefix caching 对照，解释历史 0/32 的逐元素差异。
3. 用 `AsyncLLMEngine` 或独立服务做真正的多批并发，验证流水线能否把 910C 的串行开销转化为吞吐。
4. 两边统一到“同 workload、同轮次、同预热”后再固化任何跨平台结论。
5. 向昆仑芯上报 vllm-kunlun 0.11.0 两个 pooling 缺陷（根因已定位，含最小复现，见 `platforms/kunlunxin-p800/investigation/README.md` §3）：① prefix-caching 命中路径输出错误；② 单次调用跨多个 prefill 调度步时约半数序列输出错误。规避=关缓存 + 单次调用总 token ≤16384（外层组批预算可钳制）。
6. P800 上以 token 预算 131072 复测 B/C 档位（当前 32768 预算把 B64/B128 钳成与 B32 相同批次）。
