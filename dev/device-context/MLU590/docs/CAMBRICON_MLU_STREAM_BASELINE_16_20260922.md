# MLU590 · 多流 Stream 验收基线 16 项逐项比对报告

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 主机：`tza-0a06-ai01-em9`（10.1.1.21，Mlu-1）｜ 用卡：MLU0（S-13 用 MLU0,2）
> **对象**：多流 Stream **验收基线 16 项（S-1 ~ S-16）**——见 `../../prototype/docs/RUNTIME_DC_STREAM_PLAN_20260907.md` §5，
> 该基线是该文档明确要求"**新后端接入时逐项比对**"的判据。
> **目的**：把 MLU590（第三实例）对这 16 项的实测结论补齐，并与 910C（第一实例）、P800（第二实例）逐项对照。
> **写法**：结论先行 + 逐项证据自包含（现象 / 数字 / 命令）；未覆盖项如实标注。

---

## 0. 结论摘要

| 结论 | 数字 |
|---|---|
| MLU590 对 16 项的覆盖 | **15 项通过 / 1 项不适用 / 0 项不支持**（合计 16） |
| 与另两家的逐项差异 | **仅 1 项（S-12 流优先级）**：910C ✅ / **P800 ❌ 不支持** / **MLU590 ✅** |
| 探针（可真正创建流验证的 8 项） | **`STREAM_SEMANTICS_PASS 8/8`** —— 与 910C、P800 的 8/8 **逐项一致** |
| 本次新增覆盖 | **S-7 图捕获（5/5）**、**S-16 流数量配额（2000 流）** |
| 顺带产出 | ① 图捕获探针**后端无关化 V2**（原版硬编码 `torch_npu`）；② 新增 S-16 配额探针；③ 为 `cambricon` 补上 `graph_capture` / `stream_priority` 能力声明 |
| 一条反向对照 | MLU590 与 P800 在**同一个 API**（`Stream.priority_range()`）上结论**完全相反** —— 又一次实证"跨芯片不可类推" |

---

## 1. 16 项逐项结论（MLU590）

| # | 子项 | 结论 | 证据来源 | 关键数字 |
|---|---|---|---|---|
| **S-1** | 流内顺序性 | ✅ | 探针 S1（**真正创建流**） | 流内 4 个 op 按序执行 → **5.000000**（期望 5.0） |
| **S-2** | 显式依赖 / 无隐式同步 | ✅ | 探针 S2（真创建两条流） | 流A=**7.0000** 流B=**11.0000**（无串扰） |
| **S-3** | 跨流可见性（event） | ✅ | conformance `s3_result_visibility` | 写流→事件→计算流，读回**全 3 = True** |
| **S-4** | wait_stream 传递 | ✅ | conformance `s4_explicit_transfer` | 显式 `wait_stream` 传递：**b 全 10 = True** |
| **S-5** | 多流并发重叠 | ✅（功能项） | conformance `i6_pipeline_overlap` | 末轮数据正确 **rel_err=1.03e-07**；观测重叠率 −651%（**噪声区间，不作性能结论**） |
| **S-6** | 集合通信与流绑定 | ✅ | 训练腿 2 卡 DDP + 通信三类对照 | 两 rank **`TRAIN_LEG_PASS 6/6`**；loss **15.4498 → 11.1479**（50 步，无 NaN）；`all_reduce` 3.0/3.0、`all_gather` [0.0,1.0]、P2P 一致；**2957.8 tok/s（两卡合计）** |
| **S-7** | 图捕获流语义 | ✅ | 图捕获探针 V2（G1–G5 同口径） | **`GRAPH_CAPTURE_PASS 5/5`**（5 项 rel_err 全 **0.00e+00**） |
| **S-8** | 默认流 vs 命名流 + `record_stream` | ✅ | 探针 S8 | 默认流=**2.0000** / 命名流=**4.0000**；跨流 `record_stream` 后复用=**10.0000** |
| **S-9** | 流错误隔离（分层） | ✅（API 级） | 探针 S9 + 错误闭环 | 流A 注入参数类异常后流B 仍正常 → **6.0000**；设备级错误为语义推断（真实触发风险高，未做） |
| **S-10** | 流/事件生命周期与配额 | ✅ | 探针 S10 | **1000 次**创建销毁（**0.23 s**）后新流仍可用，无配额泄漏 |
| **S-11** | 跨流内存分配 | ✅ | 探针 S11 | 流A 分配内存在流B 使用（依赖后）→ **6.0000** |
| **S-12** | 流优先级 | ✅ **支持** | 探针 S12 | `priority_range()` = **`(0, -3)`**，可用且**不崩**；多流并发正确（hi=**5.0000** / lo=**8.0000**）。⚠️ 优先级**实际调度效果**未单独验证；上游**未拦截非法优先级**（`priority=99` 不报错）⇒ 本层不透传非法值 |
| **S-13** | 多设备流绑定 | ✅ | 探针 S13（两卡） | 双设备各建独立流 → **[2.0, 3.0]**（期望 [2.0, 3.0]） |
| **S-14** | 流同步超时语义 | ✅ | conformance `e2_host_timeout` | 未 record 的 `wait_host(200ms)` 返回 **False**、耗时 **200 ms**（不永久阻塞） |
| **S-15** | 跨进程流共享（IPC） | — 不适用 | 单机多卡场景，与另两家同判定 | — |
| **S-16** | 流数量配额 | ✅ | 09-22 新增配额探针 | 连续创建 **2000 个流全部成功（0.02 s）**；第 1 个流仍可执行（**192.0000**）；释放后再建新流正常（**32.0000**） |

---

## 2. 三实例逐项对照（16 项）

| # | 子项 | 910C（第一实例） | P800（第二实例） | **MLU590（第三实例）** | 判定 |
|---|---|---|---|---|---|
| S-1 | 流内顺序性 | ✅ 5.000000 | ✅ 5.000000 | ✅ 5.000000 | **三家数字相同** |
| S-2 | 显式依赖 / 无隐式同步 | ✅ A=7 B=11 | ✅ A=7 B=11 | ✅ A=7 B=11 | **三家数字相同** |
| S-3 | 跨流可见性 | ✅ | ✅ | ✅ | 一致 |
| S-4 | wait_stream 传递 | ✅ | ✅ | ✅ | 一致 |
| S-5 | 多流并发重叠 | ✅（功能项） | ✅（功能项） | ✅（功能项） | 一致（性能项三家均不作结论） |
| S-6 | 集合通信与流绑定 | ✅ DDP + TP 层内 all_reduce | ✅ DDP + 三类通信 | ✅ DDP + 三类通信 | 一致（证据形态不同，见 §4） |
| S-7 | 图捕获流语义 | ✅ 5/5 | ✅ 5/5 | ✅ **5/5** | **三家同为 5/5** |
| S-8 | 默认流 vs 命名流 | ✅ 2/4 + 10.0 | ✅ 2/4 + 10.0 | ✅ 2/4 + 10.0 | **三家数字相同** |
| S-9 | 流错误隔离 | ✅（注入 107015） | ✅（注入参数类异常，**替代**） | ✅（注入参数类异常，**替代**） | 语义一致、注入形态不同 |
| S-10 | 流/事件生命周期 | ✅ 500 次 | ✅ 1000 次（0.10 s） | ✅ 1000 次（**0.23 s**） | 一致 |
| S-11 | 跨流内存分配 | ✅ 6.0 | ✅ 6.0 | ✅ 6.0 | **三家数字相同** |
| S-12 | 流优先级 | ✅ least=7 / greatest=0 | ⚠️ **不支持**（上游缺陷） | ✅ **`(0, -3)` 可用** | ⚠️ **唯一差异：P800 与 MLU590 相反** |
| S-13 | 多设备流绑定 | ✅ [2.0, 3.0] | ✅ [2.0, 3.0] | ✅ [2.0, 3.0] | **三家数字相同** |
| S-14 | 流同步超时语义 | ✅（507046 真实触发） | ✅（有界返回 201 ms） | ✅（有界返回 200 ms） | 一致（触发形态不同） |
| S-15 | 跨进程流共享（IPC） | — 不适用 | — 不适用 | — 不适用 | 三家一致 |
| S-16 | 流数量配额 | ✅ 2000 流 | ✅ 2000 流 | ✅ **2000 流** | **三家数字相同** |

> **可比性说明**：S-1/S-2/S-8/S-11/S-13/S-16 的期望值是**确定数值**，三家逐位相同 ⇒ **可直接比**。
> S-6 的 loss / 吞吐是模型与超参的函数（三家已对齐 `MAX_STEPS=50 / BATCH=4 / SEQ=128`，同一模型），
> 属**同口径可比**；S-5/S-12 的性能面（重叠率、调度效果）**不可比也不作结论**。

---

## 3. 本次的两项新增覆盖

### 3.1 S-7 图捕获：`GRAPH_CAPTURE_PASS 5/5`

按与 910C / P800 同口径的 5 项验证（G1 capture 正确性 / G2 replay 确定性 / G3 输入更新 /
G4 捕获内流切换 / G5 显式 stream 参数）：

| # | 验证项 | MLU590 结果 |
|---|---|---|
| G1 | capture vs eager | ✅ rel_err **0.00e+00** |
| G2 | 两次 replay 逐位一致 | ✅ 差 **0.00e+00** |
| G3 | 换值后 replay 用新值 | ✅ rel_err **0.00e+00** |
| G4 | 捕获内切命名流 | ✅ rel_err **0.00e+00** |
| G5 | `graph(g, stream=s)` | ✅ rel_err **0.00e+00** |

图捕获入口实测为 **`torch.mlu.MLUGraph` + `torch.mlu.graph`**（`CUDAGraph` 名不存在）。
据此为 `cambricon` 后端**补上 `graph_capture` 能力声明**（原先如实不声明，理由是"入口未实测"）。

> ⚠️ **一条必须遵守的用法约束**：**捕获区内不得调用任何同步原语**。
> P800 首测曾因在 `with graph(g):` 内调用 `synchronize()` 而报出看起来像"厂商不支持"的错误，
> 按此写下的"不支持"结论**是错的**（已在 P800 报告中撤回，见其 §3.2）。
> 本次的 V2 探针**刻意不在捕获区内同步**。

### 3.2 S-16 流数量配额：2000 流 0.02 s，三项全过

| # | 判据 | MLU590 结果 |
|---|---|---|
| Q1 | 连续创建 2000 个流全部成功 | ✅ **2000/2000，0.02 s** |
| Q2 | 第 1 个流仍可正常执行 | ✅ 结果 **192.0000**（期望 192.0） |
| Q3 | 全部释放后再建新流正常 | ✅ 结果 **32.0000**（期望 32.0） |

---

## 4. 证据形态差异的如实说明（不要误读为"不一致"）

| # | 项 | 910C 形态 | P800 形态 | **MLU590 形态** | 说明 |
|---|---|---|---|---|---|
| S-6 | 集合通信与流绑定 | 双卡 DDP + **TP 层内 all_reduce** | 双卡 DDP + 三类通信 | 双卡 DDP + 三类通信 | MLU590 未做 TP 场景，**排他证据弱于 910C**；"梯度同步正确 + loss 正常下降"足以证明成立 |
| S-9 | 流错误隔离 | 真实注入 **ACL 107015** | 注入参数类异常（替代） | 注入参数类异常（替代） | 寒武纪的错误码**以错误名透出、非数字码**（实测 `CNRT error: invalid argument.`）⇒ 无 107015 等价物；替代方式语义同为"单次调用失败"，结果里已标注 |
| S-14 | 流同步超时 | 真实触发 **507046** 并走错误翻译 | 有界返回 201 ms | 有界返回 200 ms | 507046 是昇腾专有码；另两家验证的是"有界语义"本身 |
| S-12 | 流优先级 | 原生区间可用 | **不可用**（上游断言） | **可用**（`(0,-3)`） | 同一 API 在三家上三种结果；MLU590 侧另有一处额度：非法优先级未被上游拦截 |

### 4.1 ⚠️ 一条必须写进结论的限制：CNCL 走**片间互联**而非 RDMA

训练腿日志里 CNCL 明确告警：

```
libcncl.so.1: undefined symbol: mlx5dv_modify_qp_udp_sport
Fail to load libmlx5.so, cannot use mlx5dv_modify_qp_udp_sport to set udp port!
Failed to open libibverbs.so[.1]. If you want to use RDMA transport, you need add libibverbs.so path to LD_LIBRARY_PATH.
Failed to load libibverbs.so!
...
Clique build homogeneous all-connected MLU_LINK topo: success
```

⇒ **本次 S-6 的集合通信证据是"单机 2 卡经 MLU_LINK（片间互联）"**，**不是 RDMA**。
这不影响"集合通信与流绑定成立"的结论（通信确实经设备侧集合通信库完成、结果正确），
但**不得**把本次数据用于任何"网络/跨机通信"的性能论述。

---

## 5. 本次的资产产出

| 产出 | 说明 |
|---|---|
| **`prototype/probes/probe_graph_capture_stream_v2.py`**（新增） | 图捕获探针**后端无关 V2**：设备 API 前缀改由统一运行时给出（`runtime.current().device_type`），图对象类**动态发现**（`CUDAGraph` / `NPUGraph` / `MLUGraph` / `Graph`），判据与 V1 **完全同口径**（G1–G5）。910C 原版 `probe_graph_capture_stream.py` **保留不删**（历史归档） |
| **`prototype/probes/probe_stream_quota.py`**（新增） | S-16 流数量配额探针：S-16 既不在 8 项探针范围，也不在 conformance 用例里（S-10 查的是"创建销毁循环后新流仍可用"，不查"同时在世数量"）⇒ 独立成资产，供各实例同口径取数 |
| `cambricon` 后端 **能力补声明** | 依据本次 5/5 与 `priority_range()=(0,-3)`，补上 `graph_capture` 与 `stream_priority` |
| MLU590 证据（6 份，均在 `probes/`） | `stream_semantics_full_result_mlu590_{1card,2card}_20260922.json`、`graph_capture_stream_result_mlu590_20260922.json`、`stream_quota_result_mlu590_20260922.json`、`train_leg_result_rank{0,1}_20260922.json`，以及控制台日志 `A6_stream_20260922.log` / `A7_train_20260922.log` / `A9_S7_S16_20260922.log` |

---

## 6. 复现命令

```bash
# 容器 dc-mlu590-hliu553（镜像 flagos-runtime-cambricon-neuware4.4.3:2.2.0）
cd /work/prototype

# ① 8 项流语义探针（S-13 需要 ≥2 张可见卡，只给 1 张会如实跳过并说明原因）
MLU_VISIBLE_DEVICES=0,2 DC_BACKEND=cambricon DC_TAG=_mlu590_2card \
  DC_OUT_DIR=/work/probes/stream python3 probes/probe_stream_semantics_full.py --rounds 5 --leak-iters 1000

# ② 图捕获 5 项（S-7）
MLU_VISIBLE_DEVICES=0 DC_BACKEND=cambricon DC_TAG=_mlu590 \
  DC_OUT_DIR=/work/probes python3 probes/probe_graph_capture_stream_v2.py

# ③ 流数量配额（S-16）
MLU_VISIBLE_DEVICES=0 DC_BACKEND=cambricon DC_TAG=_mlu590 \
  DC_OUT_DIR=/work/probes python3 probes/probe_stream_quota.py

# ④ 训练腿 2 卡（S-6）—— 注意 DC_DIST_BT 必须显式给，脚本对未探测的芯片拒绝兜底
MLU_VISIBLE_DEVICES=0,2 DC_BACKEND=cambricon DC_DIST_BT=cncl DC_ROOT=/work/prototype \
DC_MODEL=$(ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ | head -1) \
DC_OUT_DIR=/work/probes/train MAX_STEPS=50 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  runtime/proto/proto_train_leg.py
```

**用卡提示**：S-13 需要至少 2 张**可见**卡（`MLU_VISIBLE_DEVICES=0,2`）；
若只给 1 张，S-13 会如实跳过并在 detail 中说明原因（不是失败）。
共享机上请先 `cnmon` 挑**完全空闲**的卡——本次卡 0/2 空闲、卡 3 被他人占 17 GB。

**模型路径提示**：`DC_MODEL` 必须指向 **snapshot 目录**（`…/snapshots/<hash>/`），
不能给 HF 缓存的 repo 根（`…/models--Qwen--Qwen3-Embedding-0.6B`）——
后者没有 `config.json`，transformers 会报
`Unrecognized model … Should have a model_type key`（本次首跑即踩到，已在脚本内改为自动取 snapshot）。

---

## 7. 未覆盖 / 待办

| # | 项 | 说明 |
|---|---|---|
| 1 | **S-5 多流并发重叠的性能判据** | 功能项（数据正确）已通过；重叠率在噪声区间波动，**不作性能结论**（与另两家同口径） |
| 2 | **S-6 的 TP 排他证据** | 只有 DDP 证据；若后续在 MLU 上做 TP 推理可补齐 |
| 3 | **S-9 设备级错误隔离** | 三家均为**语义推断未实测**（真实触发芯片级错误风险高）——既有缺口，非本次新增 |
| 4 | **S-12 优先级调度效果** | 属性能/调度方向，非本层验收项；另需补一条"非法优先级上游未拦截"的上报 |
| 5 | **跨机（多机）集合通信** | 本次仅单机 2 卡（`MLU_LINK`）；CNCL 未加载 `libibverbs`/`libmlx5` ⇒ RDMA 路径**未验证** |
| 6 | 合入节奏 | 按约定本批内容**随 10 月统一提交**（PR 前置条件：不对 910C/P800 既有结论造成回归——MLU590 侧已复验离线自检 38/0、smoke 46/0、conformance 13+6） |
