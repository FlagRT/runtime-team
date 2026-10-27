# MLU590 共享层复跑 · 读数清单（2026-10-27）

目的：验证四处**共享层**改动在 MLU590 实例上零回归 ——
① `create_event(enable_timing=…)` 新入口　② `_stream_ctx` 改强持有（退出期段错误第五载体）
③ 职责审计 C13 的方向/量纲/三态　④ `proto_infer_leg` 模型路径自动探测

## 环境（实测）

| 项 | 值 |
|---|---|
| 机器 | `tza-0a06-ai01-em9`（Mlu-1，10.1.1.21）；`/` 71%、`/srv` 6.0T 可用 |
| 容器 | `dc-mlu590-hliu553`（**自 2026-09-22 Up，复用未重启**），镜像 `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` |
| 栈 | python 3.10.20 · torch **2.7.1+cpu** · torch_mlu **1.29.2+torch2.7.1** · 8× MLU590-M9 |
| 选卡 | `MLU_VISIBLE_DEVICES`（**容器内 0 基**）；MLU **无独占卡模型**；`cnmon` 实测 8 卡 0% 且无进程 ⇒ 全空闲 |
| 路径映射 | 宿主 `/srv/hliu553` == 容器 `/work`；`/hf_cache` 亦为挂载点（宿主 `/srv/data/hf_cache`） |
| 模型 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B` → 解析到 `snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/`（**与 P800 同一 snapshot 哈希**） |
| 分布式后端 | **`DC_DIST_BT=cncl`**（必须显式给；本原型对寒武纪**拒绝兜底 gloo**——见下「踩坑 1」） |

## 判据读数

| 段 | 判据 | 读数 | rc | 与归档基线 |
|---|---|---|---|---|
| [0.5] | 模型权重预检（本轮新增） | 非有限值 **0** | 0 | 新增段；**MLU 这份副本是干净的** |
| [1] | 离线契约自检 | **103 / 0 / 0** | 0 | 归档 **97/0/0** ⇒ **+6**，逐条点名见下 |
| [2] | smoke | **46 / 0** | 0 | 与归档 46/0 **同** |
| [3] | 契约不变式 I1–I4 | **4/4** · `CONTRACT_INVARIANTS_PASS` | **0** | 与归档 4/4 同 |
| [4] | 职责响应审计 78 项 | **OK 63 / FAIL 0 / SKIP 15** · C13 = **11.381 ms** | 0 | 归档 **62/0/16**（C13 为 SKIP）⇒ **C13 SKIP→OK**，见「更正 1」 |
| [5] | 推理腿（**不传 `DC_MODEL`**） | **`INFER_LEG_PASS 13/13`** · 41.28 句/s · p50 72.63 ms · **区分度 0.6391** · 自动探测命中 HF 缓存候选 | 0 | 同判据；区分度**与归档 0.6391 逐字相同** |
| [5b] | 推理腿（显式 `DC_MODEL`） | **13/13** · 40.53 句/s · 区分度 **0.6391** | 0 | 两传法结果一致 |
| [6] | 训练腿 50 步（2 卡，容器内 0/1） | **`TRAIN_LEG_PASS 6/6`** · **loss 15.4498→11.1479** · `dist=cncl` · 2964.5 / 2966.6 tok/s | 0 | loss **与归档逐字相同**（`train_leg_result_rank{0,1}_20260922.json` 同为 15.4498→11.1479） |

**离线自检 97 → 103 的 +6 逐条点名**（全在第 `[7]` 节「能力声明自洽 / 已知问题」，全部 PASS）：

| # | 新增检查 | 来自哪次共享层改动 |
|---|---|---|
| 1 | 跨类型地址复用：`owns_stream` 必须因**类型不符**而不认 | 2026-10-10 类型复核（`tname`） |
| 2 | 跨类型地址复用：`release_stream` **不得返回 True**（不得越权销毁） | 同上 |
| 3 | 悬垂弱引用：新建登记后，使用点**不得**被旧「已释放」记录误拦 | 2026-10-14 T11 修法 |
| 4 | ① 无本层上下文时不登记 `_stream_ctx` | **2026-10-26 `_stream_ctx` 改强持有** 配套 |
| 5 | ② 「已释放」记录的持有器是**强持有** | 同上 |
| 6 | ③ 按 id 键的登记表有条数上限（`\|reg\|=4096`） | 同上 |

⇒ `97 + 6 = 103` 完全对上，不是“通过率变化”。

## 更正 1：MLU590 的 C13 不是「平台不具备计时能力」，是我们 API 缺入口

归档（2026-10-14 `t1_duty_cambricon_20261014.json`）里 C13 = **SKIP**，原因是
`RuntimeError: CNRT error: failed to call the driver-…` —— 这个**长得像平台限制**。
本轮 `create_event(enable_timing=True)` 打通后，MLU590 **能正常计时**：`elapsed_time=11.381 ms`
（另有 10.363 / 10.416 / 11.453 ms 三次复现，均落在 [5,5000] ms 带内）⇒ 该 SKIP 源于
**统一 API 没有「建可计时事件」入口**，与 910C / P800 同因，**不是寒武纪驱动缺陷**。
审计随之 `62/0/16 → 63/0/15`（C13 计入 OK，SKIP 少一项）。

## 对应 910C 的「区分度待查」：MLU 这一份给的是 0.6391

| 实例 | 模型副本 | 推理腿区分度 |
|---|---|---|
| MLU590 | `snapshots/97b0c614…`（与 P800 同哈希） | **0.6391**（与自身归档逐字相同） |
| P800 | 同一 snapshot 哈希 | 0.6392 |
| 910C | 迁移快照副本（曾损坏，已换 r10 副本） | **0.6380** |

两组事实：① P800/MLU 用**同一 snapshot 哈希**仍差 0.0001 ⇒ 该指标有 ±1e-4 级平台抖动；
② 910C 的 0.6380 偏离 **0.0011**，约为抖动的 10 倍 ⇒ 仍指向 910C 那份副本与其它两机不同源。
本轮仍**不下结论**（三份副本的全量 sha256 未对账完），记为待查。

## 未解释 / 待修项

1. ⭐ **共享层两腿对 `DC_MODEL` 传法不一致**（本轮实测暴露）：`proto_infer_leg` 有 `resolve_model`
   会把 HF 缓存根解析到 `snapshots/<hash>`，`proto_train_leg` **没有**该解析 ⇒ 同一份
   `DC_MODEL=/hf_cache/hub/models--Qwen--…` 推理腿过、训练腿报
   `ValueError: Unrecognized model in …`。本轮按历史用法（先解析 snapshot 路径）绕过，
   **未临时改共享层**（改共享层须四实例复验，不能在 parity 跑里夹带）。⇒ 待办：训练腿补同一解析，并四实例复验。
2. 上面那条「910C 区分度待查」：三机 `model.safetensors` 全量 sha256 对账。

## 踩坑（都写进脚本注释了）

1. **`DC_DIST_BT` 必须显式 `cncl`**：本原型对寒武纪拒绝兜底 `gloo`（那会静默退化成纯 CPU 集合通信，
   训练照样跑完但设备侧通信未被验证）。缺该变量时两个 rank 均 `exitcode 2` 并打印探测指引。
2. **`/hf_cache` 是容器内路径**：第一版在**宿主**上算 `ls -d $MODEL/snapshots/*/` ⇒ 空 ⇒ 回落到缓存根
   ⇒ 训练腿仍报同一个 `Unrecognized model`。改在**容器内**解析后通过。

## 文件

- `model_precheck.log` 权重预检 · `offline.log` 离线自检（103 项）· `smoke.log` · `invariants.log` / `ci.json`
- `duty.log` / `duty.json` 审计 78 项（C13 明细在内）
- `infer_leg*.log` / `.json`（自动探测 + 显式两种传法）· `train_leg.log` / `train_leg_result_rank{0,1}.json`

复跑入口：`probes/run_mlu590_rerun_20261027.sh`
