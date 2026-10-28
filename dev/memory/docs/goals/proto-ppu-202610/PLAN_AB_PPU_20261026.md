# PLAN —— PPU ZW810E KV 卸载开关 A/B（2026-10-26 夜，PROFILE-PROTOCOL v0.3）

## 跑前声明（预注册，跑后不改）

### 五要素
- 栈：vLLM 0.24.0（FL vendor 0.24 + plugin-fl 0.3.0rc0）+ offload.ppu_offload（PpuSimpleCPUOffloadConnector）
- 芯片：PPU-ZW810E（96GB）第 7 卡，容器独占该卡
- 卡况：独占（ppu-smi 复核无他人进程）
- 冷暖：**暖跑配对**——每腿先 1 暖轮不计入，再 2 测轮取中位；S1 每轮换冷 prompt
- 池口径：GPU 池 `--kv-cache-memory-bytes 33554432`（64 块 = 1,024 tok，两腿同口径）；
  Host 池 ON 腿 `cpu_bytes_to_use=1073741824`（1 GiB）；OFF 腿无 connector（差分唯一=connector 开关）

### 固定参数
- 模型 tiny = Llama-3.2-1B-Instruct（RO 借用共享路径）；TP=1；eager 开
- max-model-len 1024；served 端口 18332

### 腿本（P→S→B 腿序为判据，跑前声明）
| 腿 | 内容 | 目标数字 | 有效判据 |
|---|---|---|---|
| P | 单条 768 tok prompt，max_tokens=1，3 轮（1 暖 2 测） | prefill TTFT 中位 | ON−OFF 差 ≤±5% 判「prefill 无代价」 |
| S1 | 单流 768 tok prompt + 256 out（≥128 才算 decode），每轮冷 prompt（seed 递增×7919），3 轮（1 暖 2 测） | TTFT + decode tok/s（usage 口径） | decode 差 ≥±5% 判显著；<128 tok 不写 decode（此处 256 满足） |
| B-ex | 并发 2 条 × (768 prompt + 256 out) ≈ 2048 tok 需求 = 池 200%，3 轮（1 暖 2 测） | 完工时间（wall） + 成功率 | 成功率必须 200%×2=2/2 才算数；失败轮弃用并记录 |

### 预注册预期（写跑前，跑后对照）
- P/S1 池内形态：ON≈OFF（P800 结论：池不满零代价）；Host(1 GiB) ≫ GPU(1024 tok) 满足
  「Host 池 > GPU 池」的 L2 条件，但池内用不到 Host。
- B-ex 超池形态：OFF 只能排队/preemption 重算；ON 走 store/load。方向假设 ON 更快或相近，
  量级未知（PPU 逐块 copy_ 无批量 DMA，store/load 带宽待实测）。
- 若 B-ex OFF 腿出现 400/调度拒绝，记录为「OFF 不可行」形态，改并发 2×(512+256)=1536（150%）。

### 判读规则（防"测了跟没测一样"）
- 任一腿成功率 <100% → 该腿数字只标"未饱和参考"，不入结论。
- 冷跑数据（未暖配对）一律不写。
- 全部数字带口径注释（哪条腿、第几轮、中位/单轮）。
