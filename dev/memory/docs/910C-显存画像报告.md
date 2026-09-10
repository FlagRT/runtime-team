# 910C 显存画像报告

> 状态：🟢 **环境 + 加载阶段 + 运行阶段峰值 + A/B 已实测**（2026-09-10，npu1-27 davinci-7）｜ 起草：2026-09-10 ｜ 负责人：xliu969
> 对应：战略文档 §3 验收标准第 4 条、§5 memory 第 1 条 ｜ 权威方案：[路线A-显存与缓存管理-方案-20260822](路线A-显存与缓存管理-方案-20260822.md)
> 对照基线：[archive/V1-显存画像报告-20260817](archive/V1-显存画像报告-20260817.md)（910C 旧栈，torch_fl/Qwen3-4B，已冻结）
> 数据来源：宿主 `npu-smi` 采样 CSV `benchmarks/out/infer910c_hbm.csv` + 各 run 的 profile JSON/vLLM 日志 `benchmarks/out/infer910c_mem_*.{json,vllm.log}`。
> 探针已实测通过（此前 UNTESTED 头注失效）；`infer910c_hbm_sampler.py` 解析器针对 npu-smi 25.5.0 A3 版式修过一次（按 Phy-ID 做键）。

---

## 1. 环境（锁定推理镜像栈）

| 项 | 值 | 来源 |
|---|---|---|
| 镜像 | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`（digest `sha256:5cf8a2b6…c11ec7a7`） | dev/stack.lock.910c.v1.yaml |
| 容器名 | `flagos-proto-infer-910c` | 同上 |
| 设备后端 | npu（torch_npu），Route A 华为昇腾官方纯栈（无 FlagOS/FlagGems/FlagCX/torch_fl） | 同上 |
| Python | 3.11.15 | 容器内实测 |
| vLLM / vllm_ascend | `0.20.2+empty` / `0.20.2rc1` | 容器内实测（`+empty` 为 vllm-ascend 打包方式，vllm 本体走 vllm_ascend 插件） |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0` | 容器内实测（`+cpu` 正常，NPU 后端由 torch_npu 提供） |
| transformers | 5.5.3 | 容器内实测 |
| triton / triton_ascend | 3.5.0（pip `triton`） / 3.2.1（华为昇腾插件，import 自报 3.2.0） | 容器内实测；`triton` 3.5.0 与 stack.lock `flagtree_line`（ascend3.5 线）一致；brief 里的「3.2.0」是插件自报号 |
| CANN | 9.0.0（`ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0`） | 容器内实测 |
| 宿主 driver / npu-smi | Version 25.5.0，ascendhal 7.35.23，Innerversion V100R001C23SPC005B219 | 宿主 `/usr/local/Ascend/driver/version.info` |
| SOC_VERSION | ascend910_9391（设备名 `Ascend910_9382`） | 容器内实测 |
| 单芯 HBM total | 65536 MiB（64 GiB）；vLLM 侧可见 61.28 GiB（driver 预留 ~2.7 GiB 后） | npu-smi / vLLM worker.py 日志 |
| 验收模型 | `Qwen/Qwen3-Embedding-0.6B`（EMBEDDING / pooling runner；dtype `bfloat16`；权重 1.11 GiB checkpoint；seq_pooling_type=LAST；输出 dim 1024，L2 归一化） | vLLM 日志 + 实测向量 |
| 权重路径（宿主） | `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`（snapshot 目录，直接可用；未用 HF-hub fallback） | — |
| 关键运行参数（基准 run `gmu0.9-eager`） | gpu_memory_utilization=0.9、max_num_seqs=256、max_model_len=8192、**enforce_eager=false（ACLGraph/PIECEWISE）**、PYTORCH_NPU_ALLOC_CONF=`expandable_segments:True`、batch=64、seq_len≈512、warmup=3 | profile JSON |
| 起容器方式 | house 模式：`--device /dev/davinci0..15` + davinci_manager/devmm_svm/hisi_hdc + `-v /usr/local/Ascend/driver:...:rw` + `--network host --shm-size 512g --cap-add SYS_PTRACE`；`Runtime=ascend`（宿主默认）；workload 用 `ASCEND_RT_VISIBLE_DEVICES=7` 钉 davinci-7 | 实测 |
| davinci-7 ↔ npu-smi | davinci/phy-id 7 = npu-smi「NPU 3 / Chip 1」，Bus `0000:93:00.0`，查询 `npu-smi info -i 3 -c 1`；idle HBM 2892 MiB | 实测 |

## 2. 方法

- **设备级 HBM 采样为真值**：外挂 `probes/infer910c_hbm_sampler.py`（宿主机轮询 `npu-smi info`，解析 per-chip HBM-Usage MB + AICore%，按 `--interval` 写时间戳 CSV，`--tag` 区分 run）。
  - ⚠️ 本机 `npu-smi info` 单次约 **1.9 s**（x-benchmark 容器在跑，npu-exporter/hccn_tool 抢占），故 `--interval 0.5` 实际有效分辨率约 **2 s**。embedding 无 decode、无 KV 逐步增长，加载/峰值都是秒级台阶，2 s 分辨率够用；但 ACLGraph capture 那一段（~13 s）只有 6-7 个采样点。
- **vLLM 日志解析**：`probes/infer910c_mem_profile.py` 抓 `Available KV cache memory` / `GPU KV cache size` / `Maximum concurrency` / model weights / peak memory 等行。
- **driver 侧 torch_npu 计数仅作佐证**：`torch_npu.npu.memory_allocated / max_memory_allocated / memory_reserved / max_memory_reserved` 在 driver 进程读取 —— **预期 ≈ 0**，因 EngineCore 是 spawn 子进程，主进程读不到 worker 分配器状态（沿用 archive/V1 §3.3）。
- **必须先预热**：测量前先跑 `--warmup N` 轮短请求，避开首次 kernel 初始化长尾（旧栈曾达 437s）。
- **A/B 对照**：`probes/infer910c_ab_matrix.py` 跨 `gpu-mem-util` / `alloc-conf` / `enforce-eager` / `max-num-seqs` 轴批量跑，汇总 CSV + Markdown。
- 时间戳对齐：sampler CSV 的 `tag` 与 profile JSON 的 `tag` 同名，事后按时间窗切曲线。

## 3. 加载阶段画像

基准 run：`gmu0.9-eager`（gpu_mem_util=0.9，max_num_seqs=256，max_model_len=8192，enforce_eager=**false**/ACLGraph，alloc_conf=expandable_segments:True）。

### 3.1 HBM 分阶段（宿主 npu-smi 采样，davinci-7）

| 阶段 | HBM used | 占 65536 | 相对上一阶段 Δ | 说明 |
|---|---|---|---|---|
| idle 基线（起 vLLM 前） | 2892 MiB / 2.82 GiB | 4.4% | — | driver/firmware 常驻 |
| 权重加载后 | ~4144 MiB / 4.05 GiB | 6.3% | +1252 MiB（+1.22 GiB） | vLLM 日志：`Loading model weights took 1.1333 GB`（checkpoint 1.11 GiB，bf16） |
| engine-init 平台期（torch.compile + profiling run） | ~4163–4445 MiB / ~4.3 GiB | ~6.6% | +~0.2 GiB | 持续约 34 s，HBM 基本不动 |
| KV/pooling 池预分配后 | ~59221 MiB / 57.8 GiB | 90.4% | **+54776 MiB（+53.5 GiB）** | vLLM 日志：`Available KV cache memory: 53.78 GiB` / `GPU KV cache size: 503,424 tokens` |
| **加载完成峰值（ACLGraph capture + 首批 embed 后）** | **59669 MiB / 58.27 GiB** | **91.05%** | +448 MiB | graph 0.10 GiB + 峰值激活 0.22 GiB（vLLM 日志 worker.py） |
| vLLM 退出后 | 2892 MiB / 2.82 GiB | 4.4% | −56777 MiB | 干净释放，无泄漏 |

vLLM worker.py 结算行（`gmu0.9-eager`）：
`Free memory on device (61.13/61.28 GiB) on startup. Desired GPU memory utilization is (0.9, 55.15 GiB). Actual usage: 1.13 GiB for weights, 0.22 GiB for peak activation, 0.01 GiB for non-torch memory, 0.1 GiB for NPU graph memory. Current KV cache memory: 53.78 GiB.`
→ **加载后 HBM ≈ driver 基线 2.82 + 权重 1.13 + 激活 0.22 + non-torch 0.01 + graph 0.10 + KV/pooling 池 53.78 ≈ 58.06 GiB**，与 npu-smi 峰值 58.27 GiB 吻合（差值为分配器 rounding / 采样时刻）。

### 3.2 关键结论：embedding 也吃满 gpu_mem_util

`runner="pooling"` 的 embedding 模型**依旧按 `gpu_memory_utilization` 预分配一个巨大的 "KV cache" 池**（此处 53.78 GiB / 503,424 tokens），尽管 embedding 无 decode、不会用到 KV 逐步增长。即 **加载后 HBM ≈ gpu_mem_util × 单芯 HBM**（此处 0.9 → 91%）。这个池对 embedding 基本是浪费，是可被 `gpu_memory_utilization` 或 vLLM 建议的 `--kv-cache-memory=<bytes>` 直接压缩的部分（见 §5 / A/B）。

### 3.3 计时（profile JSON `timings` + vLLM 日志）

| 项 | 值 | 拆解 |
|---|---|---|
| `LLM()` 构造总时长 `load_s` | **70.44 s** | 权重加载 1.01 s ｜ torch.compile 26.32 s ｜ profiling/warmup run 5.04 s ｜ ACLGraph capture 13 s ｜ engine-init 小计 49.34 s（含 compile）；其余为进程/插件 import + tokenizer |
| 预热 `warmup_s`（3 轮 `llm.embed` 短串） | **4.665 s** | 对比旧栈「首次 attention 437 s」——embedding 无 decode，无该长尾 |
| 测量 `measured_s`（batch 64，seq≈512） | **0.236 s** | 吞吐 271 req/s |
| `total_s` | 84.93 s | |
| driver 侧 `torch_npu.max_memory_reserved` | **0**（预期） | EngineCore 是 spawn 子进程，driver 进程读不到；真值靠 npu-smi + 日志 |

### 3.4 embedding API surface（实测）

| 问题 | 结果 |
|---|---|
| `LLM(runner="pooling")` | ✅ 直接可用，**不需要**回退 `task="embed"`（profile JSON `llm_construction: 'runner="pooling"'`） |
| `llm.embed(list[str])` | ✅ 可用，返回 `.outputs.embedding`；**dim 1024，L2 归一化（norm=1.0），无 NaN/Inf**；cos("hello world","运行时层显存画像")≈0.37 合理 |
| `llm.encode(...)` | ❌ 裸调报 `ValueError: pooling_task required for LLM.encode`——需显式 `pooling_task=`。harness 里 `hasattr(llm,"embed")` 短路到 `embed`，不受影响；但头注写的 fallback `llm.encode()` 路径实际不通 |
| `enforce_eager=True` 路径 | ✅ 可用，日志 `Enforce eager set, disabling torch.compile and CUDAGraphs`；engine-init 仅 5.15 s（无 compile/capture） |
| vllm_ascend 特有提示 | `Pooling models do not support full cudagraphs → PIECEWISE`；`PIECEWISE compilation enabled on NPU ... only ACL Graph mode`；权重类映射 `Qwen3ForCausalLM`/`Qwen3ForEmbedding` → `Qwen3Model`。**无** pooling/embedding 相关报错 |
| 良性告警 | `Bind cpus failed in rank0: [Errno 2] No such file or directory: 'npu-smi' Skip binding cpu`——house 模式容器里没有 npu-smi CLI，只影响 CPU 亲和绑定，不影响正确性 |

## 4. 运行阶段峰值画像

sweep：batch ∈ {8, 64, 256} × seq-len ∈ {128, 512}，全部 **enforce_eager=True**（干净测激活，无 ACLGraph capture 噪声），gpu_mem_util=0.9，max_num_seqs=256，max_model_len=8192，alloc_conf=expandable_segments:True，warmup=3，concurrency=1（offline 单进程）。HBM 峰值 = 宿主 npu-smi 采样该 run tag 段的 max。idle 基线 2892 MiB。

| tag | batch | seq≈ | load_s | warmup_s | measured_s | 吞吐 req/s | HBM 峰值 MiB | 峰值 GiB | 峰值 − idle MiB | 峰值 − 加载后* MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| b8-s128-eager | 8 | 128 | 13.18 | 0.39 | 0.084 | 95 | 59298 | 57.91 | 56406 | ~186 |
| b8-s512-eager | 8 | 512 | 12.97 | 0.35 | 0.088 | 91 | 59298 | 57.91 | 56406 | ~186 |
| b64-s128-eager | 64 | 128 | 12.92 | 0.39 | 0.121 | 529 | 59439 | 58.05 | 56547 | ~327 |
| b64-s512-eager | 64 | 512 | 12.15 | 0.39 | 0.22 | 291 | 59446 | 58.05 | 56554 | ~334 |
| b256-s128-eager | 256 | 128 | 12.68 | 0.39 | 0.255 | 1004 | 59543 | 58.15 | 56651 | ~431 |
| b256-s512-eager | 256 | 512 | 12.68 | 0.38 | 0.719 | 356 | 59677 | 58.28 | 56785 | ~565 |
| b64-s512-mml32k-eager | 64 | 512 | 11.93 | 0.37 | 0.223 | 287 | 59420 | 58.03 | 56528 | ~308 |

\* 「加载后」= idle 2892 + 权重 1160 + KV/pooling 池 55050（53.76 GiB）+ non-torch 10 ≈ 59112 MiB（vLLM 结算口径）。峰值 − 加载后 ≈ **实际峰值激活**。

### 4.1 结论

1. **HBM 峰值几乎与 batch / seq-len 无关**：batch 8→256、seq 128→512，峰值 59298 → 59677 MiB，全程差 **≈379 MiB**。原因：`gpu_memory_utilization` 一次性预分配的 "KV cache"/pooling 池（53.76 GiB）压倒一切，运行期只在其上叠加极小的激活。
2. **实际峰值激活 200–570 MiB**，随 batch×seq 弱增长（b8-s128 ~186 MiB → b256-s512 ~565 MiB）。vLLM 结算行固定报 `0.24 GiB for peak activation`（那是它 profiling 用 max_num_seqs×max_model_len 的合成估计，与真实小 batch 无关）。
3. **max_model_len 8192 → 32768：预分配池不变**（`available_kv_cache_memory` 53.76 GiB、`gpu_kv_cache_size_tokens` 503,296 均不变），仅 `Maximum concurrency` 61.44 → 15.36（恰好 ÷4，= 32768/8192）。即 **pooling 预分配是纯字节预算（gpu_mem_util × 空闲 HBM），与上下文长度无关**。
4. **运行期无新增分配 / 无尖峰**：HBM 曲线是「idle → 权重台阶 → KV 池台阶 → 峰值（激活，AICore 拉到 ~50%）→ 退出即回落 idle」。全程无碎片式增长，退出后干净归还（59669 → 2892 MiB）。与旧栈 P800 eager「全程恒平」一致，但这里恒平的量级由预分配池决定。
5. eager 加载快：`load_s` ~12–13 s（vs §3 ACLGraph 基准 70 s），`warmup` ~0.39 s（3 轮）。吞吐随 batch 上升（b256-s128 达 1004 req/s）。

### 4.2 HBM 时间序列样例（b256-s512-eager，宿主 npu-smi，~2 s 分辨率）

```
21:45:10  2892 MiB   ai 0%     idle
21:45:13  4143 MiB   ai 0%     权重加载
21:45:15  4446 MiB   ai 0%     engine init
21:45:17  22818 MiB  ai 0%     KV/pooling 池分配中
21:45:19  53758 MiB  ai 0%     池接近满
21:45:20  59677 MiB  ai 51%    峰值（embed forward 计算中）
21:45:22  56588 MiB  ai 0%     teardown
21:45:26  32147 MiB  ai 0%     释放中 → 随后回 2892
```

## 5. 碎片与 reserved-vs-allocated / A/B 对照

详见 [910C-显存池定义.md](910C-显存池定义.md) 与 `benchmarks/out/ab_summary.{csv,md}`。A/B 均 batch 64 / seq≈512 / warmup 3。

| axis | point | HBM 峰值 MiB | 峰值% | KV/pooling 池 GiB | KV tokens | 最大并发 | NPU graph GiB | load_s | 吞吐 req/s |
|---|---|---|---|---|---|---|---|---|---|
| gpu-mem-util | gmu0.4 | 28072 | 42.8% | 23.15 | 216,704 | 26.45 | 0.1 | 23.2 | 314 |
| gpu-mem-util | gmu0.9 | 59689 | 91.1% | 53.79 | 503,552 | 61.47 | 0.09 | 24.9 | 323 |
| enforce-eager | ACLGraph（默认） | 59671 | 91.1% | 53.79 | 503,552 | 61.47 | 0.10 | 27.5 | 317 |
| enforce-eager | eager | 59444 | 90.7% | 53.76 | 503,296 | 61.44 | **0.0** | 12.1 | 294 |
| alloc-conf | 默认 | 59551 | 90.9% | 53.79 | 503,552 | 61.47 | 0.1 | 25.1 | 317 |
| alloc-conf | expandable_segments:True | 59451 | 90.7% | 53.79 | 503,552 | 61.47 | 0.1 | 26.5 | 322 |

### 5.1 碎片 / reserved−allocated

- **worker 侧 allocator 计数取不到**：EngineCore 是 spawn 子进程，driver 进程 `torch_npu.npu.memory_*` 恒 0（沿用 archive/V1 §3.3）。故 reserved−allocated 的直接数值本期未取得。
- **间接证据（npu-smi）**：加载后 KV/pooling 池台阶 → 峰值，Δ 仅 200–570 MiB（= 激活）；运行期 HBM 曲线全程无增长、无尖峰；进程退出后干净回落 idle（59669 → 2892 MiB）。
- **结论**：对 embedding-on-vLLM，KV/pooling 池是**一次性整块分配**（vLLM 块表），不存在增量 segment 增长 → 无可压缩的碎片；激活区极小。**碎片不是本负载的问题**。

### 5.2 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` vs 默认

- HBM 峰值 59451 vs 59551 MiB（差 ~100 MiB，采样噪声内）；KV 池、non_torch、maximum_concurrency 全相同；load 无实质差异。
- **无可测效果** —— 因为没有增量段增长可供 `expandable_segments` 优化（见 5.1）。生成式长上下文才可能显现差异。

### 5.3 ACLGraph capture 工作区占用（enforce_eager 对照）

- ACLGraph vs eager：HBM 峰值差仅 **~0.2 GiB**（59671 vs 59444 MiB），即 vLLM 结算行的 `0.1 GiB for NPU graph memory` + rounding。
- 代价不在显存而在 load：ACLGraph 需 `torch.compile` 26 s + capture 13 s，`load_s` 27.5 s（eager 12.1 s）。
- 回报：吞吐 +~8%（317 vs 294 req/s，batch 64）。
- **与旧栈对比**：旧栈生成式 graph 模式实测多占 ~7 GiB 级 KV 空间；embedding + PIECEWISE ACLGraph **不复现该问题**，图工作区可忽略。

## 6. 与旧栈基线对照

| 指标 | 910C 旧栈 V1（archive，2026-08-17，torch_fl / Qwen3-4B，生成式） | 本次（锁定镜像 vllm-ascend / Qwen3-Embedding-0.6B，embedding） | 说明 |
|---|---|---|---|
| 模型 / dtype | Qwen3-4B 生成式 dense / bf16 | Qwen3-Embedding-0.6B pooling(LAST) / bf16 | 不可直接比 |
| 单芯 HBM（工具口径） | 61.27 GiB（aclrtGetMemInfo） | 64 GiB（npu-smi）/ 61.28 GiB（vLLM 可见） | 同机型，取数工具不同 |
| 权重 | ~7.6 GiB（bf16 4B 参） | **1.13 GiB**（bf16 0.6B 参） | ×6.7 差距 |
| 加载后占用 | 31.89 GiB（24.6 s） | **~58.1 GiB @ gmu0.9**（load_s 70 s ACLGraph / 12 s eager）；**~28 GiB @ gmu0.4** | 旧栈 gmu 未记；本次占用由 gmu 决定 |
| KV / pooling 预分配 | 170,224 tokens / ~24.3 GiB（占加载后 **76%**） | **503,552 tokens / 53.79 GiB @ gmu0.9（占加载后 ~93%）**；216,704 tok / 23.15 GiB @ gmu0.4 | 见下：对 embedding 这个池是**纯浪费的余量**，不是真实需求 |
| 最大并发（工具口径） | — | 61.47×（8192 tok/req @ gmu0.9）；15.36×（32768 tok/req）；26.45×（@ gmu0.4） | pooling 也按 KV 池推导该指标 |
| 运行期激活增量 | ~5.7 GiB（峰值 ≥37.6 GiB） | **0.2–0.57 GiB**（峰值几乎 = 加载后） | ×10 差距——embedding 单次 forward，无 decode 累积 |
| 预热（首次 kernel/attention） | **437 s** | **4.7 s**（ACLGraph 基准）/ 0.4 s（eager） | 数量级改善：embedding 无 attention decode 长尾 |
| 长序列 prefill（2048×N 并发） | 22 min 未完成（旧栈 P0） | 不适用（embedding 无 prefill/decode 之分）；max_model_len 32768 仅把最大并发指标 ÷4，池尺寸不变 | 新栈 + embedding，旧 P0 不复现 |

**为何 embedding 0.6B 与旧栈本质不同**：
1. **权重**：4B→0.6B，7.6 GiB→1.13 GiB。
2. **无自回归 decode**：无 KV cache 逐步增长。vLLM 对 pooling runner 仍走 KV-cache 分配路径并按 `gpu_memory_utilization` 吃满空闲 HBM（此处 53.79 GiB / 503,552 tokens），但 embedding **永远不会用到**——它是可回收的余量，不是 P800/旧栈那种「并发 × 上下文 × 层数」的真实 KV 需求。
3. **峰值构成**：旧栈 = 权重 + 真实 KV 池（大头，会涨）+ 激活（5.7 GiB）。本次 = 权重（1.13）+ 预分配池（53.79，**恒定、不涨、按 gmu 可调**）+ 激活（0.2–0.57，几乎不涨）+ graph（0–0.1）。
4. **可调性**：旧栈 KV 池尺寸被真实负载顶着；本次把 `gpu_memory_utilization` 降到 0.4 甚至更低都不影响 embedding 正确性与吞吐（gmu0.9 vs gmu0.4 吞吐 323 vs 314 req/s，几乎无差），只是把 HBM 峰值从 91% 拉到 43%。

## 7. 结论

- **显存池定义**：见 [910C-显存池定义.md](910C-显存池定义.md)（torch_npu caching allocator 底座 + `PYTORCH_NPU_ALLOC_CONF` + vLLM 层 gpu_memory_utilization/pooling 预分配/ACLGraph capture + 复用回收 + 安全区间）。
- **峰值画像（硬指标）**：embedding-on-vLLM 的 HBM 峰值 ≈ **driver 基线 2.82 GiB + 权重 1.13 GiB + gpu_memory_utilization × 空闲 HBM（KV/pooling 池）+ 激活 0.2–0.57 GiB + ACLGraph 0–0.1 GiB**。峰值**几乎与 batch / seq-len / max_model_len 无关**，唯一大杠杆是 `gpu_memory_utilization`。
- **给 device-context / 调度 的安全区间**（embedding 服务，单卡 davinci，64 GiB HBM）：
  - `gpu_memory_utilization`：**0.35–0.45**。0.4 实测 HBM 峰值 42.8%（28 GiB），留一半 HBM 富余，吞吐与 0.9 无差异（314 vs 323 req/s）。低于 0.3 可能触发 vLLM「KV 池太小」告警，不建议。真要压到极限可用 vLLM 提示的 `--kv-cache-memory=<bytes>` 直接给池定量（如 8–12 GiB）。
  - `max_num_seqs`：**64–128**。实测 batch 8→256 HBM 峰值只差 ~0.4 GiB，激活不是瓶颈；取 128 兼顾吞吐与调度弹性。ACL graph 实际按 `max_num_seqs` 生成 capture 尺寸（此处 35 档，max 256），过大只多花 capture 时间，不多占显存。
  - `enforce_eager`：默认 ACLGraph（显存代价仅 0.1 GiB，吞吐 +8%）；对**冷启动敏感**（频繁起停）的场景用 `enforce_eager=True` 省掉 ~15 s compile+capture。
  - `PYTORCH_NPU_ALLOC_CONF`：`expandable_segments:True` 对本负载无害无益，可留作默认（对未来生成式负载有价值）。
- **A/B 对照要点**：见 §5 表 + `benchmarks/out/ab_summary.md`。一句话：**gmu 是唯一值得调的旋钮；eager/graph 是时延-吞吐权衡不是显存权衡；expandable_segments 在 embedding 上无感**。
- 备注：KV 分层 / Host 溢出验证按战略文档 §7 本期不作硬性指标；对 embedding 也无意义（池是余量不是需求）。

## 8. 原始数据位置与复现命令

- sampler CSV：`dev/memory/benchmarks/out/infer910c_hbm.csv`（tag: `gmu0.9-eager` / `b*-eager` / `phaseC`）
- profile JSON + vLLM 日志：`dev/memory/benchmarks/out/infer910c_mem_*.{json,vllm.log}`（加载基准 + Phase B sweep）
- A/B：`dev/memory/benchmarks/out/ab_{gmu,eager,alloc}/run_*.{json,vllm.log}` + 各轴 `ab_summary.*` + 合并 `dev/memory/benchmarks/out/ab_summary.{csv,md}`
- 真实 npu-smi 版式样例 + 起容器踩坑记录：`dev/memory/benchmarks/out/npu-smi-sample.txt`
- 探针：`dev/memory/probes/infer910c_{hbm_sampler,mem_profile,ab_matrix}.py`（**已实测通过**；sampler `parse_npu_smi` 针对 npu-smi 25.5.0 A3 版式修过，按 Phy-ID 做键）
- 复现：`docker run -d --name flagos-proto-infer-910c --network host --shm-size 512g --cap-add SYS_PTRACE --device /dev/davinci0..15 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw -v /mnt/raid:/mnt/raid -v <repo>:/workspace -w /workspace quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 sleep infinity`；容器内 `ASCEND_RT_VISIBLE_DEVICES=7 DO_NOT_TRACK=1 python3 dev/memory/probes/infer910c_mem_profile.py ...`；宿主同时 `python3 dev/memory/probes/infer910c_hbm_sampler.py --chips 7 ...`。
  - ⚠️ 驱动绑定须 `:rw`（house 脚本的 `:ro` 在 driver 25.5.0 报 `DrvMngGetConsoleLogLevel failed ret=4`）。
  - ⚠️ 带卡容器并发实测上限 **2**（x-benchmark 在跑时）：第 3 个容器 acl/dcmi init 报 `-8020 device is used`，与 stack.lock 写的 3 不符。
