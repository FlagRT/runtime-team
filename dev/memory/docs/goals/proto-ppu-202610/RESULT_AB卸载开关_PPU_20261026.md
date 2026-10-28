# RESULT —— PPU ZW810E KV 卸载开关 A/B（2026-10-26/27 夜）

对照文档：`PLAN_AB_PPU_20261026.md`（预注册五要素与判据，跑后未改）。
原始数据：服务器 `/bmcp_lvm_fs/chenyuanxin/logs/ab_ppu_{off,on}.json`、
`serve-ab-{off,on}.log`（[PpuOffload-DMA] 计数在其中）。

## 一句话结论

**PPU 栈上 KV 卸载（PpuSimpleCPUOffloadConnector）在 tiny 模型 + 64 块小池口径下：
池内零代价（prefill/TTFT/decode 全持平），超池 200% 形态净代价 +2.1%（判据 ±5% 内）。**

## 数字表（每腿 1 暖轮不计 + 2 测轮取中位，暖跑配对口径）

| 腿 | 口径 | OFF | ON | 差 | 判据裁决 |
|---|---|---|---|---|---|
| P prefill | 647 tok prompt, max_tokens=1, 稳态轮 | 0.106 s | 0.105 s | −1% | **无代价** ✓ |
| S1 TTFT | 647 prompt + 256 out, 冷 prompt | 0.203 s | 0.203 s | 0% | 无代价 ✓ |
| S1 decode | 同上, usage 口径 | 10.34 tok/s | 10.37 tok/s | +0.3% | **无代价** ✓ |
| B-ex 完工 | 并发 2×(647+256) ≈200% 池, wall | 49.28 s | 50.30 s | **+2.1%** | 判据内，无显著代价 |

S1 三轮原始：OFF decode 10.24/10.34/10.33；ON 明细见 JSON。B-ex 三轮：OFF 49.51/49.21/49.35，
ON 49.88/50.32/50.28，成功率 6/6。

## DMA 证据（ON 腿 stderr 标记）

- store：154 批（B-ex 超池阶段触发 eager store）。
- load：0 次——两条 B 相互独立、无共享前缀，超池表现为排队而非召回；
  **recall-load 路径已于 2026-10-26 冒烟单独实锤**（store 190 块 + load 7 块×16 层），
  两份证据合起来覆盖 store/recall/load 三段。

## 三个必读口径警示（防超报）

1. **tiny 模型口径**：Llama-3.2-1B-Instruct + eager + TP1。所有数字只在「功能/开销验证」
   级别成立，不外推大模型吞吐；真模型 A/B 另测。
2. **P 腿是未饱和参考**：0.1s 级 prefill ≈6.5K tok/s，远未饱和，只支撑「无代价」方向，
   不给绝对性能结论。
3. **B-ex 的 OFF 侧形态是排队**（2 条串行 wall≈2×单流），不是 P800 expand 那种
   preemption 重算崩坏——因此 P800 的「ON 快 38%」场景在 tiny+200% 池下不存在；
   ON 的逐块 DMA 开销（~2%）是净账。

## 过程发现（记录，不进结论）

- **flagos triton 按-shape JIT 编译主导首跑**：同一服务里 prefill 同 prompt_tokens
  （647）稳态 0.1s；token 数差 1（marker seed 位数不同）即触发全量重编译 2.6-3.2s；
  当日首次编译 12.5s。OFF 腿同样存在（warm 2.75s）。→ 测 prefill 必须**固定 prompt
  token 数**再跨轮比较，否则量的是编译器不是引擎。腿设计缺陷：ON 腿需要 2 暖轮
  （JIT 拖尾比 OFF 长），本轮 P 腿中位数曾被拖尾轮污染（1.65s），以稳态轮读数为准。
- ON 腿 S1/P 稳态与 OFF 完全持平 → 逐块 copy_ 的固定开销在池内路径上不可见；
  B-ex +2.1% 即 Host 池搬运的真实单价（190+ 块逐块 DMA 摊在 50s 完工时间上）。
- 15+ 并发 device-side assert（vendor 线）本轮未触碰（峰值并发 2）。
