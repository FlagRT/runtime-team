# PPU 显存参数（gmu / max_num_seqs）实测对照（2026-11-03）

> 回答两件事：
> ① memory 方向建议「embedding 用 gmu 0.35–0.45 + `max_num_seqs=128`」在 PPU 上是否成立；
> ② v2 全量验收里服务化吞吐 91.65 句/s（vs 10-26 归档 105.74）到底是镜像问题、默认值问题还是环境问题。
> 全部为**同机、同时段、同卡（ppu1）、v2 镜像**的单变量对照；判据一律用 **宿主侧 `ppu-smi` 外部采样**（2 s）
> 与 vLLM 自报的 KV 池行，不用被测进程自报的峰值。

## 1 显存两臂（gmu 0.9 vs 0.4）

| 臂 | gmu | 外部采样峰值 | 占卡（98,304 MB） | verdict |
|---|---|---|---|---|
| H09 | **0.9**（vLLM 原默认） | **88,517 MB** | **90.0%** | `SERVE_STANDARD_PASS (ready=1 smoke=1)` |
| L04 | **0.4**（我方新默认） | **39,341 MB** | **40.0%** | `SERVE_STANDARD_PASS (ready=1 smoke=1)` |

- 两臂 verdict 完全相同、显存差 2.25 倍 ⇒ 「**embedding 的 KV 池是纯余量**」在 PPU 上成立，
  与 910C（91%→43%）、P800（93.5%→25%）同向。
- 峰值几乎精确等于 `gmu × 卡容量`（0.9×98304=88,474 / 0.4×98304=39,322）⇒ **显存只由 gmu 决定**。
- ⇒ PPU 不再是「未实测」档：`serve_standard.sh` 的 ppu 分支从「平移 0.4 + 告警」升级为「PPU 实测 0.4」。

## 2 吞吐四臂（拆分 gmu 与 max_num_seqs）

消费方腿 `runtime/proto/proto_infer_serve.py --rounds 5`，四臂**区分度全部 0.4118、全部 `SERVE_LEG_PASS | 10/10`**：

| 臂 | gmu | max_num_seqs | 吞吐 句/s | p50 |
|---|---|---|---|---|
| A2（10-26 归档口径） | 0.9 | 256（VLLM 默认） | **115.51** | 27.6 ms |
| A1 | 0.9 | 128 | 102.13 | 29.0 ms |
| B（11-03 起采用的默认） | 0.4 | 128 | 105.79 | 27.9 ms |
| C | 0.4 | 256 | 111.21 | 28.6 ms |

派生结论：

- **并发上限是吞吐主因**：同为 gmu 0.4，128→256 = **+5.1%**；gmu 0.9 时 128→256 = **+13.1%**。
- **gmu 的吞吐代价很小**：同 seqs=256，0.4→0.9 只 +3.9%；同 seqs=128 反而 −3.5%（噪声内）。
- gmu 0.4 下 vLLM 自报 `Available KV cache memory: 36.95 GiB` / `Maximum concurrency for 4,096 tokens: 84.45x`
  ⇒ **84 > 128 不成立**，`max_num_seqs=128` **确实在卡并发**，而它并不省显存（显存由 gmu 决定）。
- ⇒ 处置：**保留 gmu 默认 0.4**（显存收益 90%→40%，吞吐代价 ~4%），
  **取消 embed 形态的 `max_num_seqs=128` 默认**（改为不传 = 256，需要限流时显式给）。
  这是对 memory 建议 #1 的**部分不采纳**，依据即上表。

### 2.1 与 10-26 归档 105.74 句/s 的关系

v2 全量验收当轮测得 91.65 句/s，本目录同配置（gmu 0.4 + seqs 128）在 **ppu1** 上复测为 **105.79**，
与归档 105.74 相差 **0.05%**。⇒ 91.65 属**当轮环境**（验收跑在 ppu0，该卡被他人占用 5,533 MiB；
且共享盘 `/bmcp_lvm_fs` 同期已 100% 满），**不是镜像缺陷、也不是默认值缺陷**。

## 3 最终默认值的端到端复核

改完默认值后用容器内 `/tmp` 的脚本副本跑一次完整起停（`SERVE_FORM=embed`、`STOP_AFTER=1`）：

```
[gmu] embed 形态未显式给 GPU_MEM_UTIL ⇒ ppu 默认 0.4（可 GPU_MEM_UTIL=<x> 覆盖）
[max_num_seqs] embed 形态未显式给 ⇒ **不传**（vLLM 默认 256）；限显存请用 GPU_MEM_UTIL，限并发用 MAX_NUM_SEQS=<n>
[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)
```

⚠️ 该复核用的是送到容器 `/tmp` 的脚本副本，**不是** `/workspace` 里的那份 —— 因为共享盘满了，
无法把新脚本同步到机器上的代码副本（见 §4）。

## 4 本轮环境事实（必须一并读）

**`/bmcp_lvm_fs` 于 2026-11-03 11:29 实测 100% 满、可用 0**（`dd` 写 2 MiB 返回 `0 bytes copied`），
根盘 `/` 尚有 66 GiB。影响与处置：

- `serve_standard.sh` 的 `exec > "$LOG"` 会直接失败 ⇒ 服务「起不来/日志是空洞」，极易误判成产品缺陷；
- 容器内建的输出目录属宿主用户不可写（属主是 root）⇒ 宿主侧重定向报 `Permission denied`；
- 本轮四臂对照全部改用**容器 `/tmp`**（落根盘）+ `docker cp` 取回，属**非归档路径**，已在 §1/§2 标注；
- 未替他人清理共享盘；我方 18.4 GiB 离线包（`images-archive/…tar.gz`）是**唯一实体副本**，未动。

## 5 文件索引

| 路径 | 说明 |
|---|---|
| `H09/` · `L04/` | 两个 gmu 臂的 `serve_standard.log` / `vllm_serve.log` / pid |
| `H09_hbm.csv` · `L04_hbm.csv` | 宿主 `ppu-smi` 2 s 采样的原始曲线（峰值即由上表算出） |
| `tput_arms/*_consumer.log` | 四臂消费方腿原始输出（吞吐/区分度/verdict 原文） |
| `tput_arms/*_serve.log` | 四臂服务端日志（含 KV 池与最大并发行） |
| `tput_arms/tput_ab4.log` · `tputB.log` · `tputC.log` | 三批对照的控制台（含中途作废的臂与原因） |
| `tput_arms/final_default_serve.log` | §3 的最终默认端到端复核 |
