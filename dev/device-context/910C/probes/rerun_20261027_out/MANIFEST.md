# 910C 共享层复跑 · 读数清单（2026-10-27）

目的：验证四处**共享层**改动在 910C 实例上零回归 ——
① `create_event(enable_timing=…)` 新入口　② `_stream_ctx` 改强持有（退出期段错误第五载体）
③ 职责审计 C13 的方向/量纲/三态　④ `proto_infer_leg` 模型路径自动探测

## 环境（实测，非推断）

| 项 | 值 |
|---|---|
| 机器 | `npu1-27`（10.120.72.27）；**RAID5 已由 IT 重建**：`/dev/md127` 现 3.5T（原 11T），我们的 `/mnt/raid/hliu553` **已不存在** |
| 数据 | 仍在 10-21 迁移快照 `/tgm_rent_data/data/hliu553/migrate_20261021/`（72G） |
| 镜像 | `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`<br>image id **`e13a15d0d02b`** ＝ 归档时的发布 id（IT 恢复后即发布物，无需 pin） |
| 容器 | `dc-rerun-1027-910c`；**只挂 `/dev/davinci6` `/dev/davinci7`**（另外挂 manager/svm/hdc）；`--shm-size=64g`；`TORCH_DEVICE_BACKEND_AUTOLOAD=0`（对齐历史基线） |
| 选卡依据 | 容器映射表：`temp-cp-npu4567` 明确占 davinci8–15 ⇒ 取 0–7 内的一对 6/7（与接入手册训练腿示例同一对）。⚠️ `mwy-dev-ascend3` 是 **privileged 且不列设备**，无法从映射表判断其用卡 |
| 栈版本 | python 3.11.15 / torch 2.10.0+cpu / torch_npu 2.10.0 / triton 3.5.1 / flagtree 3.5.1 / flag_gems 0.0.0+f7ae8e6 / vllm 0.20.2 |
| 模型 | `Qwen3-Embedding-0.6B`，容器内 `/workspace/models/`（从迁移快照只读挂载） |

## 判据读数

| 段 | 判据 | 读数 | rc |
|---|---|---|---|
| [0.5] | **模型权重预检**（本轮新增，防 nan 伪装成判据失败） | 非有限值 **0** | 0 |
| [1] | 离线契约自检 | **92 通过 / 0 失败 / 2 跳过**（跳过＝stub 能力边界） | 0 |
| [2] | smoke | **52 通过 / 0 失败** | 0 |
| [3] | 契约不变式 I1–I4 | **4/4 通过** · `CONTRACT_INVARIANTS_PASS` | **0** |
| [4] | 职责响应审计 78 项 | **OK 73 / FAIL 0 / SKIP 5** · `DUTY_RESPONSE_PASS`<br>C13 = **`elapsed_time=10.474 ms`**（落在 [5,5000] ms 带内；首轮为 10.441 ms） | 0 |
| [5] | 推理腿（前向） | **`INFER_LEG_PASS 14/14`** · 76.14 句/s · p50 39.29 ms · **区分度 0.6380** | 0 |
| [5'] | 推理腿（**不传 `DC_MODEL`**） | **`INFER_LEG_PASS 14/14`** · 78.13 句/s · 区分度 0.6380 · **`model_source=按实例布局探测`** | 0 |
| [6] | 训练腿 50 步（2 卡，容器内 0/1） | **`TRAIN_LEG_PASS 6/6`** · **loss 15.4498→11.1479** · 4340.9 / 4341.3 tok/s | 0 |

**与归档基线对照**

| 项 | 归档基线 | 2026-10-27 | 结论 |
|---|---|---|---|
| 训练腿 loss | 15.4498 → **11.1479**（`train_leg_910c_npu_20260929_rank0.json`） | 15.4498 → **11.1479** | **逐字相同** ⇒ 共享层改动零回归 |
| 训练腿判据 | 6/6 | 6/6 | 同 |
| 训练腿吞吐 | 4513.2 tok/s | 4341.3 tok/s | −3.8%（共享机争用，非回归） |
| 推理腿判据 | 14/14 | 14/14 | 同 |
| 推理腿区分度 | 0.6391 | **0.6380** | ⚠️ **不同**，见下「未解释项」 |
| 契约不变式退出码 | 归档无记录 | **rc=0**（PPU 上曾是 139） | 910C 不复现该退出期段错误 |

## ⚠️ 本轮抓到的真问题：迁移快照里的模型权重内容损坏

首次跑训练腿得到 **`TRAIN_LEG_FAIL 5/6`、loss 从第 0 步就是 `nan`**（集通信、模型加载等 5 项全过）。
按纪律先查权重，链路如下（全部可复现）：

| 步 | 动作 | 读数 |
|---|---|---|
| 1 | `transformers` 加载后逐参数扫非有限值 | 310 个张量中 **`model.embed_tokens.weight` 含 410 NaN + 1 inf**（shape `(151669, 1024)`） |
| 2 | 单卡 fp32 前向 `out.loss` | **`nan`**（逐步 backward 后 310 个张量梯度全含非有限值） |
| 3 | 同一文件两种读路径（dtfs 直读 vs 拷到本机盘）`sha256` | **一致**（`dba4a6dd…`）⇒ 是**文件内容**坏，不是读路径 |
| 4 | 同快照内独立副本 `dc_r10_20261008/models/Qwen3-Embedding-0.6B/` | **0 个非有限值**（`0437e45c…`）⇒ 有干净件 |
| 5 | 换干净权重重跑训练腿 | **6/6 PASS，loss 15.4498→11.1479** |

处置（已执行）：坏件改名留存 `model.safetensors.CORRUPT-20261027`，用 r10 副本替换；登记在
`/tgm_rent_data/data/hliu553/migrate_20261021/_manifest/MODEL_CORRUPTION_20261027.txt`。
两份差异 `cmp` 计 **657,943 字节**（不是零星 bit rot，像是某段区域被写坏）。

⚠️ **这条同时更正 10-21 迁移记录的一处措辞**：那里写「45126 文件 / 68.8GB **逐字节一致**」，
实际是「体积对账 + **抽样** sha256」⇒ 对**大文件**（模型权重）不成立。见
[`../docs/STORAGE_MIGRATION_20261021.md`](../docs/STORAGE_MIGRATION_20261021.md) 的补注。

## 未解释项（如实登记，不臆断）

**推理腿区分度 0.6380 ≠ 归档 0.6391**，且**用干净权重后仍为 0.6380** ⇒ 与上面的权重损坏**无关**。
同一指标在 PPU 上是 0.6391（与其历史逐字一致）、P800 是 0.6392。
⇒ 提示**跨机模型副本本身存在差异**（同一 `Qwen3-Embedding-0.6B` 不同副本）。**未取到三份副本的
全量哈希对账**（PPU 与 MLU590 本轮均不可达），故不下结论；记为待查：
① 三机模型副本 `model.safetensors` 全量 sha256 对账；② 若确有差异，注明各机基线指标绑定的副本哈希。

## 文件

- `offline.log` / `smoke.log` / `invariants.log` / `ci.json` — 无设备段与契约不变式
- `duty.log` / `duty.json` — 职责审计 78 项（含 C13 明细）
- `infer_leg.log` / `infer_leg.json`（显式 `DC_MODEL`）· `infer_leg_autodetect.log` / `.json`（不传 `DC_MODEL`）
- `train_leg.log` / `train_leg_result_rank{0,1}.json` — 训练腿 50 步（干净权重）
- `model_precheck.log` — 模型权重预检（0 非有限值）
- `model_corrupt_scan.log` — **坏件**扫描（411 个非有限值），留证用

复跑入口：`probes/run_910c_rerun_20261027.sh`（六段 + 模型预检，逐段记 rc 与 verdict）
