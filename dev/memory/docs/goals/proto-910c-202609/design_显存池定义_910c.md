# 910C 锁定推理镜像 · 显存池定义

> 状态：🟢 实测（2026-09-10，npu1-27 davinci-7）｜ 负责人：xliu969
> 配套：[910C-显存画像报告.md](profile_显存画像_910c.md)（分阶段画像 + A/B 原始表）
> 对应：战略文档 §3 验收标准第 4 条第 2 项（显存池定义 + 对照数据）、§5 memory 第 2 条
> 镜像：`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（py3.11.15 / vllm 0.20.2+empty / vllm_ascend 0.20.2rc1 / torch 2.10.0 / torch_npu 2.10.0 / CANN 9.0.0 / SOC ascend910_9391）
> 验收模型：`Qwen/Qwen3-Embedding-0.6B`（EMBEDDING / pooling runner，dtype bf16，输出 dim 1024）

---

## 1. 两层显存池

910C 推理腿的 HBM 由**两层分配器**叠加管理，自底向上：

```
┌─────────────────────────────────────────────────────────────┐
│  vLLM 层（逻辑池，按 gpu_memory_utilization 预算）             │
│   ├─ 权重区            1.13 GiB  （bf16，一次性，load 时定）    │
│   ├─ KV/pooling 预分配池  gmu × 空闲HBM  （一次性整块，块表管理）│
│   ├─ ACLGraph capture 区  0–0.1 GiB （enforce_eager=False 时） │
│   └─ 激活区            0.2–0.57 GiB （每次 forward，可复用）    │
├─────────────────────────────────────────────────────────────┤
│  torch_npu caching allocator 底座（物理段，PYTORCH_NPU_ALLOC_CONF）│
│   ├─ segment 缓存：freed block 不还 driver，留作复用           │
│   └─ expandable_segments / max_split_size：段增长与切分策略    │
├─────────────────────────────────────────────────────────────┤
│  driver / firmware 常驻      2.82 GiB （npu-smi idle 基线）     │
└─────────────────────────────────────────────────────────────┘
   单芯 HBM total 65536 MiB（64 GiB）；vLLM 可见 61.28 GiB（driver 预留 ~2.7 GiB 后）
```

## 2. 底座：torch_npu caching allocator

- **行为**：与 CUDA caching allocator 同构。`cudaMalloc`/`aclrtMalloc` 拿到的段（segment）切成 block 给上层；`free` 的 block **不立即还给 driver**，进 allocator 的空闲链表待复用。`empty_cache()` 才把完全空闲的段还 driver。
- **`PYTORCH_NPU_ALLOC_CONF`**（对应 CUDA 的 `PYTORCH_CUDA_ALLOC_CONF`）：
  - `expandable_segments:True` —— 段可原地扩展，减少「大小不匹配导致的段浪费」。**本负载实测无差异**（见 §4 轴 3）：KV/pooling 池是一次性整块分配，没有反复增长/回收的段，`expandable_segments` 无用武之地。对未来生成式长上下文负载才有价值。
  - `max_split_size_mb` —— 限制大段被切成小块（防碎片）。本负载未调，无必要。
- **driver 侧计数为 0**：vLLM EngineCore 是 spawn 子进程，主进程读 `torch_npu.npu.memory_allocated/reserved` 恒 0。真值只能靠**宿主 npu-smi 连续采样** + **vLLM worker 日志结算行**交叉验证（沿用 ../legacy-2.4-910c/profile_V1显存画像_910c.md §3.3 的方法学）。

## 3. vLLM 层

### 3.1 `gpu_memory_utilization`（唯一大旋钮）

vLLM 启动时：`可用HBM × gpu_memory_utilization − 权重 − 峰值激活 − non_torch − graph = KV/pooling 池字节数`。

对 embedding / pooling runner：**这个池仍然被完整分配**（走 KV-cache 代码路径），但 embedding 无 decode、不会用到 KV 逐步增长 → **它是纯余量，不是需求**。实测：

| gpu_memory_utilization | KV/pooling 池 | KV tokens | HBM 峰值 | 峰值 % | 吞吐（batch 64） |
|---|---|---|---|---|---|
| 0.4 | 23.15 GiB | 216,704 | 28072 MiB | 42.8% | 314 req/s |
| 0.9 | 53.79 GiB | 503,552 | 59689 MiB | 91.1% | 323 req/s |

→ **HBM 峰值 ≈ gpu_memory_utilization × 空闲HBM(61.3 GiB) + ~4.5 GiB 固定量**（driver 基线 + 权重 + 激活 + graph）。吞吐几乎不随 gmu 变（embedding 用不到池）。

vLLM 还打印可直接定量的替代项：`--kv-cache-memory=<bytes>`（如 gmu0.9 那条日志建议 `57490650931`≈53.5 GiB 或 `63912231936`≈59.5 GiB）。想把 embedding 服务的池压到 8–12 GiB，用这个比调 gmu 更精确。

### 3.2 pooling 预分配与上下文长度无关

`max_model_len` 8192 → 32768：`available_kv_cache_memory`（53.76 GiB）、`gpu_kv_cache_size_tokens`（503,296）**均不变**，只有 `Maximum concurrency` 61.44 → 15.36（÷4 = 32768/8192）。即池是**纯字节预算**，`max_model_len` 只改「这些字节够几路并发」的换算，不改池尺寸、不改 HBM 峰值。

### 3.3 ACLGraph capture

- `enforce_eager=False`（默认）：vLLM 对 pooling 模型走 **PIECEWISE** ACL Graph（日志：`Pooling models do not support full cudagraphs. Overriding cudagraph_mode to PIECEWISE`），按 `max_num_seqs` 生成一组 capture 尺寸（实测 35 档，1..256）。
- **显存代价：0–0.1 GiB**（vLLM 结算行 `0.1 GiB for NPU graph memory`）。实测 HBM 峰值 ACLGraph vs eager 差 ~0.2 GiB（59671 vs 59444 MiB）。
- **时间代价：+~15 s**（`torch.compile` 26 s + capture 13 s，一次性）。
- **回报：吞吐 +~8%**（317 vs 294 req/s，batch 64）。
- 与旧栈对比：旧栈生成式 graph 模式实测多占 ~7 GiB 级 KV 空间；embedding + PIECEWISE **不复现**，图工作区可忽略。

### 3.4 激活区

- 每次 `llm.embed()` forward 用的临时张量，走 caching allocator，**复用良好**：batch 8→256、seq 128→512，HBM 峰值只差 ~0.4 GiB。
- 实际峰值激活 200–570 MiB（峰值 − 加载后台阶），随 batch×seq 弱增长。vLLM 结算行固定报 `0.24 GiB for peak activation`（那是它用 max_num_seqs×max_model_len 做 profiling 的合成估计）。

## 4. 复用 / 回收策略

| 机制 | 行为（实测） |
|---|---|
| KV/pooling 池 | **一次性整块**分配（vLLM 块表），进程存活期间不涨不缩；无碎片 |
| 激活 block | caching allocator 内复用，forward 间不还 driver；峰值恒定 |
| `empty_cache()` 语义 | 把完全空闲的段还 driver（本负载无需主动调；池不释放，激活量太小无所谓） |
| 进程退出 | HBM 干净归还：59669 → 2892 MiB（idle），无泄漏 |
| 预分配 vs 按需 | 池是**预分配**（启动即吃满 gmu 预算）；激活是**按需**（但量极小且复用） |
| `pkill -f "VLLM::EngineCore"` | 杀 worker 子进程即释放该进程全部 HBM；实测 ~10 s 内 npu-smi 回 idle |

**碎片 / reserved−allocated**：worker 侧 allocator 计数取不到（子进程），但 npu-smi 间接证据——运行期 HBM 曲线全程平（无增长、无尖峰），退出即回落——说明**本负载无碎片问题**。碎片风险只存在于「反复分配/回收不同大小 block」的场景（生成式长上下文），embedding 不涉及。

## 5. A/B 对照表（batch 64 / seq≈512 / warmup 3）

| axis | point | HBM 峰值 MiB | 峰值% | KV 池 GiB | 最大并发 | NPU graph GiB | load_s | 吞吐 req/s | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| gpu-mem-util | 0.4 | 28072 | 42.8% | 23.15 | 26.45 | 0.1 | 23.2 | 314 | **大杠杆**：峰值≈gmu×空闲HBM |
| gpu-mem-util | 0.9 | 59689 | 91.1% | 53.79 | 61.47 | 0.09 | 24.9 | 323 | |
| enforce-eager | ACLGraph | 59671 | 91.1% | 53.79 | 61.47 | 0.10 | 27.5 | 317 | 显存差 0.2 GiB；时延↔吞吐权衡 |
| enforce-eager | eager | 59444 | 90.7% | 53.76 | 61.44 | 0.0 | 12.1 | 294 | load 快 15s，吞吐低 8% |
| alloc-conf | 默认 | 59551 | 90.9% | 53.79 | 61.47 | 0.1 | 25.1 | 317 | expandable 无感 |
| alloc-conf | expandable_segments:True | 59451 | 90.7% | 53.79 | 61.47 | 0.1 | 26.5 | 322 | |

原始：`benchmarks/out/ab_summary.{csv,md}` + `benchmarks/out/ab_{gmu,eager,alloc}/run_*.{json,vllm.log}`。

## 6. 给 device-context / 调度 的安全区间（embedding 服务，单卡 davinci，64 GiB HBM）

| 参数 | 推荐 | 依据 |
|---|---|---|
| `gpu_memory_utilization` | **0.35–0.45**（或 `--kv-cache-memory=8~12 GiB` 直接定量） | gmu0.4 实测 HBM 峰值 42.8%（28 GiB），留半卡富余；吞吐与 gmu0.9 无差（314 vs 323）。< 0.3 可能触发「KV 池过小」告警 |
| `max_num_seqs` | **64–128**（取 128） | batch 8→256 HBM 峰值只差 ~0.4 GiB，激活非瓶颈；ACL graph 按此生成 capture 尺寸，过大只多花 capture 时间不多占显存 |
| `max_model_len` | 按业务最长输入设（如 8192） | 不影响池尺寸 / HBM 峰值，只影响「最大并发」换算 |
| `enforce_eager` | 默认 `False`（ACLGraph）；频繁起停场景用 `True` | ACLGraph 显存代价仅 0.1 GiB、吞吐 +8%；但 load +15s |
| `PYTORCH_NPU_ALLOC_CONF` | 留默认或 `expandable_segments:True` 均可 | 本负载无差异；expandable 对未来生成式负载有价值 |

**一句话**：embedding 服务把 `gpu_memory_utilization` 降到 ~0.4 就能把单卡 HBM 占用从 91% 压到 43%、不损吞吐；`max_num_seqs` 128 足够；graph/alloc 旋钮不是显存问题。多实例共卡时按「每实例 ~28 GiB @ gmu0.4」或「~12 GiB @ `--kv-cache-memory`」估算。

## 7. 复现

见 [910C-显存画像报告.md](profile_显存画像_910c.md) §8。要点：驱动绑定 `:rw`；带卡容器并发实测上限 2（x-benchmark 在跑时）；`ASCEND_RT_VISIBLE_DEVICES=7` 钉 davinci-7（= npu-smi NPU 3 / Chip 1 / Bus 0000:93:00.0）；`DO_NOT_TRACK=1` 必设。
