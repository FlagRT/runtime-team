# MLU590 · 共享层改动后全套复跑（2026-11-03）

> 背景：10-27 之后共享层又动了三处——① `runtime/proto/model_path.py` 两腿共用（`DC_MODEL` 口径）
> ② `prototype/scripts/serve_standard.sh` 分芯片显存默认值（`gmu` / `max_num_seqs`）
> ③ 三个 PPU 驱动脚本的 run-id 目录修复。本实例需要按同口径全量复跑一遍，
> 并补齐此前因网络不可达而缺失的两项：**weakref 四家矩阵第 4 家** 与 **`DC_MODEL` 两腿口径**。
>
> 机器：`tza-0a06-ai01-em9`（宿主观角 `/srv/hliu553` ↔ 容器 `/work`）
> 容器：`dc-mlu590-hliu553`（运行时；`/flagos/bin/python3` · torch 2.7.1+cpu · torch_mlu 1.29.2 · 8 卡）
> 　　　`dc-mlu590-vllm-hliu553`（服务化；vllm 0.20.2）
> 入口：`probes/run_mlu590_acceptance_20261103.sh`（本轮新写的单入口，按 run-id 分目录）

## 1 总判

| 项 | 结果 |
|---|---|
| 单入口全量（9 段） | **`overall: PASS`（逐段 rc=0）** |
| `DC_MODEL` 两腿口径 | **`LEGPARITY_PASS` 9/0** · `\|Δloss\| = 0.000000` |
| weakref 退出期缺陷 | **未中招** 🟢（Stream/Event 回调均触发、`rc=0`）⇒ 四家矩阵补齐 |
| 服务化显存参数 | gmu 0.9 → **89.3%** / 0.4 → **39.9%**，两臂均 10/10；默认值端到端 `SERVE_STANDARD_PASS` |

## 2 单入口 9 段（与 10-27 归档逐项对照）

| 段 | 本轮读数 | 10-27 归档 | 判定 |
|---|---|---|---|
| s01 离线契约自检 | **103 通过 / 0 失败 / 0 跳过** | 103/0/0 | ✅ 逐字相同 |
| s02 跨后端对称性 | 7 通过 / 0 失败 | — | ✅ |
| s03 真机 smoke | **46 通过 / 0 失败** | 46/0 | ✅ |
| s04 conformance 13 例 | `CONFORMANCE_PASS` | PASS | ✅ |
| s05 conformance 推理 6 例 | `CONFORMANCE_PASS` | PASS | ✅ |
| s06 契约不变式 I1–I4 | `CONTRACT_INVARIANTS_PASS`（rc=0） | 4/4 rc=0 | ✅ |
| s11 图捕获 | `GRAPH_CAPTURE_PASS` **4/4** | 4/4 | ✅ |
| s12 流配额 | `STREAM_QUOTA_PASS` **3/3** | 3/3 | ✅ |
| s13 职责响应审计 | **OK 63 / FAIL 0 / SKIP 15（共 78）** | 63/0/15 | ✅ 逐字相同 |

**未在本轮重跑、但同日已覆盖的三项**（范围裁定，不等于"已证明无回归"）：
训练腿 50 步 / 推理腿 —— 由 `run_mlu590_legparity_20261027.sh` 覆盖（各跑两种 `DC_MODEL` 传法）；
服务化 —— 由 `gmu_ab_20261103_out/round2` 覆盖。

## 3 `DC_MODEL` 两腿口径复验（补齐 10-27 遗留）

| 判据 | 读数 |
|---|---|
| P1 离线口径自检（含 AST 同源守卫） | 23 / 0 / 0 |
| 推理腿 × HF 缓存根 | `INFER_LEG_PASS` **13/13** · 41.89 句/s · p50 70.54 ms · 区分度 **0.6391** |
| 训练腿 50 步 × HF 缓存根 | `TRAIN_LEG_PASS` **6/6** · loss **15.4498→11.1479** |
| 推理腿 × 普通目录 | 13/13 · 42.54 句/s · 区分度 0.6391 |
| 训练腿 × 普通目录 | 6/6 · loss **15.4498→11.1479** |
| 两腿解析同一 snapshot / 诱饵未命中 / `\|Δloss\|` | PASS / PASS / **0.000000** |

loss 与区分度与 10-27 归档**逐字相同**（11.1479 / 0.6391）⇒ 共享层改动未动读数。

## 4 weakref 退出期缺陷：MLU590 **未中招**

同一份判据脚本（与 910C / PPU / P800 完全同源），MLU 命名空间 `torch.mlu`：

| 判据 | 读数 |
|---|---|
| `torch.__version__` | 2.7.1+cpu（torch_mlu 1.29.2） |
| Stream weakref 可创建 | 是（30/30） |
| Stream / Event weakref 回调是否触发 | **True / True** |
| `Stream + weakref + 被回收` × 30 的进程退出码 | **0**（无 SIGSEGV） |

⇒ **四家矩阵最终版**：PPU 🔴 / 910C（torch_npu 2.10.0）🔴 / P800（torch 2.9.0）🟢 / **MLU590（torch 2.7.1）🟢**。
补充一条事实：MLU 侧 **weakref 是可创建的且清理正确**（不是"不支持弱引用"），说明不存在"2.8 之前一律不支持"的说法。

## 5 服务化：显存参数两轮对照

### 5.1 第一轮（`STOP_AFTER=1` 单纯验显存，卡 2）

| 臂 | gmu | 宿主 `cnmon` 外部采样峰值 | 占卡 | verdict |
|---|---|---|---|---|
| H09 | 0.9 | **87,858 MiB** | **89.4%** | `SERVE_LEG_PASS \| 10/10` |
| L04 | 0.4 | **39,274 MiB** | **40.0%** | `SERVE_LEG_PASS \| 10/10` |
| DEF | 默认 | — | — | ⚠️ **驱动误杀导致的假 FAIL，见 §6** |

### 5.2 第二轮（预热后计时，卡 2）

⚠️ 第一轮的吞吐读数（0.09 句/s、`max 86544 ms`）**作废**：`proto_infer_serve.py --rounds 5` 把
**MLU 首请求的编译开销（86 s）** 算进了吞吐。改为「先预热一次（丢弃）再计时」后：

| 臂 | gmu | 峰值 | 吞吐 | p50 / max | 区分度 | verdict |
|---|---|---|---|---|---|---|
| T09 | 0.9 | **87,762 MiB = 89.3%** | **4.66 句/s** | 644.9 / 647.2 ms | **0.4091** | 10/10 |
| T04 | 默认 **0.4** | **39,210 MiB = 39.9%** | **4.62 句/s** | 647.7 / 658.3 ms | **0.4091** | 10/10 |

派生结论：

- **显存 89.3% → 39.9%（差 2.24 倍），吞吐 4.66 → 4.62 句/s（−0.9%，p50/max 均在同一档）**
  ⇒ MLU 上 gmu 对吞吐**无代价**（该实例单请求 p50 约 645 ms，吞吐由单请求计算主导、与并发上限无关）。
  与 PPU（gmu 代价 ≤4%）、P800（两臂均 PASS）同向：**「embedding 的 KV 池是纯余量」在 MLU 同样成立**。
- 区分度 **0.4091** 与 10-14 归档 `SERVE_LEG_PASS 10/10 · 区分度 0.4091` **逐字相同**
  （注：0.4118 是 PPU 的值，两家不可混用）。
- 据此把 `serve_standard.sh` 的 `cambricon` 分支从「未实测平移 0.4 + 告警」升级为**实测 0.4**。

### 5.3 默认值端到端

```
[gmu] embed 形态未显式给 GPU_MEM_UTIL ⇒ cambricon 默认 0.4（可 GPU_MEM_UTIL=<x> 覆盖）
[max_num_seqs] embed 形态未显式给 ⇒ **不传**（vLLM 默认 256）；限显存请用 GPU_MEM_UTIL，限并发用 MAX_NUM_SEQS=<n>
[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)
```
（cambricon 已不再打「未实测」告警；停机后卡上显存回到 0 MiB、无残留进程。）

## 6 本轮踩到的两个驱动侧坑（都不是被测对象的问题）

1. **单入口脚本路径拼错**：第一版把 `P`（代码副本）拼在了 run-id 目录下
   （`$WC/dev/...` 而 `WC` 已含 `acceptance_20261103`）⇒ `cd` 失败 ⇒ **9 段全部 rc=1 且日志文件为空**。
   ⇒ 规则：**基目录 / run-id 目录 / 代码副本**三者必须分开声明，并在开头各打印一行自证。
2. **外部驱动把被测脚本「提前杀掉」⇒ 假 FAIL**：`STOP_AFTER=1` 的 `serve_standard.sh` 会**自己**做
   ready+smoke 再打 verdict；我的驱动在它自查完成前就 `pkill`，得到
   `[verdict] SERVE_STANDARD_FAIL (ready=0 smoke=0)`，而同一服务在我自己的就绪轮询里是 200。
   ⇒ 规则：**自证型脚本必须等它自己出 verdict 再停**；驱动侧的 `hard_stop` 只能在拿到 verdict 之后调。
   （与 PPU 的「判据读错目录 ⇒ 假 FAIL」同族：**假 FAIL 与假 PASS 一样危险**。）

## 7 文件索引

| 路径 | 说明 |
|---|---|
| `probes/acceptance_20261103_out/console.log` | 单入口 9 段控制台（含 `overall: PASS`） |
| `probes/acceptance_20261103_out/run/` | 各段原始日志 + `summary.tsv` + 结构化 JSON |
| `probes/legparity_20261103_out/` | 两腿口径复验（console + run/ 原始产物 + rc） |
| `probes/weakref_check_20261103_out/` | weakref 判据脚本与原始输出 |
| `probes/gmu_ab_20261103_out/round1/` · `round2/` | 服务化两轮（含采样 CSV 与消费方腿原始输出） |
| `probes/gmu_ab_20261103_out/round1_console.log` · `round2_console.log` | 两轮控制台（含作废臂与原因） |
| `probes/regress_20261103_out/` | 离线/口径/契约/smoke 四项回归的细粒度日志（与 s01–s06 同源，供逐条查） |
