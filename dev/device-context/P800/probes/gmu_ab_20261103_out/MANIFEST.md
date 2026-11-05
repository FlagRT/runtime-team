# P800 · gmu A/B + 多流基线复跑（2026-11-03）

> 目的：① 为 memory 方向建议 #1/#4（embedding 服务降 `gpu_memory_utilization`）取**本实例的外部观测证据**；
> ② 为我们的新默认值（`serve_standard.sh` 的 `kunlun → 0.25`）取得实测背书；
> ③ 共享层改动后复跑多流基线（P800 侧此前未做）。

## A. gmu A/B（同卡单变量，卡 7，98304 MB）

`scripts/serve_standard.sh` · `SERVE_FORM=embed` · `STOP_AFTER=1`（起→就绪→冒烟→停机）·
显存由**宿主 `xpu-smi` 轮询采样**（2 s 间隔，外部观测，非 vLLM 自报）。

| 臂 | `GPU_MEM_UTIL` | `MAX_NUM_SEQS` | verdict | 采样峰值 | 基线 | 增量 | 占卡 |
|---|---|---|---|---|---|---|---|
| **H09** | **0.9**（vLLM 原默认） | 未传 | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`** | **91,950 MB** | 3,424 MB | **88,526 MB** | **93.5%** |
| **L025** | **0.25**（我方新默认） | **128** | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`** | **28,054 MB** | 3,424 MB | **24,630 MB** | **25.1%** |

**读法**（一句话）：两臂**判据完全相同（都 PASS）**，但显存占用一个 93.5%、一个 25.1%
⇒ 印证 memory 的结论「embedding 的 KV 池是**纯余量**，gmu 只买"用不到的并发"」，
并且在 P800 上**跨实例同向**（910C 归档 91%→43% · PPU 88% · 本机 93.5%→25.1%）。

⚠️ **93.5% 意味着什么**：98304 MB 卡上只剩约 6.3 GB 余量。本机是多用户共卡环境
（当日普查卡 0 用 28 GB、卡 4 用 45 GB、卡 5 用 1.6 GB…），gmu 0.9 的实例一旦起来，
**同卡其他容器再要显存就会被挤**，这正是 memory 报告里"画像被卡三轮"的处境。
⇒ 我方默认取 0.25（本实例 09-20 既有实测建议值），把余量留给共存。

⚠️ **口径边界**：本 A/B 只验"起得来 + 判据不变 + 显存占用差"，**未压吞吐/并发**；
"0.25 是否够用"依赖业务并发，memory 给的是 `max_num_seqs=128` 下 `0.4`(=43%) 的档位，
本实例更保守（0.25）。若要按 0.4 对齐，需另取 0.4 臂。

⚠️ **就绪时间**：H09 132 s、L025 187 s（同一轮首次 180 s 预算曾判 FAIL ⇒ 本机首次加载偏慢，
`READY_BUDGET` 建议 ≥ 300 s）。两臂均 `ready=1`，无假失败。

## B. 多流基线复跑（共享层改动后首次）

`probes/probe_stream_semantics_full.py` / `probe_graph_capture_stream_v2.py` / `probe_stream_quota.py`
（`DC_BACKEND=kunlun`，卡 0）

| 段 | verdict | rc | 对照 2026-09-20 归档 |
|---|---|---|---|
| 流语义 8 项 | **`STREAM_SEMANTICS_PASS: 8/8`** | 0 | 8/8 —— **逐项一致** |
| 图捕获（契约内） | **`GRAPH_CAPTURE_PASS: 4/4`** | 0 | 4/4 —— 一致 |
| 流数量配额（2000 流） | **`STREAM_QUOTA_PASS: 3/3`** | 0 | 3/3 —— 一致 |

⇒ 共享层改动（`_stream_ctx` 强持有 / 事件计时入口 / 两腿模型路径）**未影响** P800 多流判据。

## 文件

| 目录 | 内容 |
|---|---|
| `gmu_ab_20261103_out/` | `H09/` `L025/` 各自的 `serve_standard.log`（含 verdict 与参数回显）+ `vllm_serve.log`；`{H09,L025}_hbm.csv` 为 2 s 间隔显存采样曲线 |
| `multistream_20261103_out/` | `sem/graph/quota` 的 `.log` 与 `.rc` |

⚠️ 采样峰值汇总文件（`*_hbm_peak.txt`）因驱动脚本 bug 未落盘，峰值由 `*_hbm.csv` 重算
（`max(第2列)`），与本文件读数一致。
