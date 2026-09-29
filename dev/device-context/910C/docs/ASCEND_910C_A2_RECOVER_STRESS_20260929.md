# 910C · A2 多卡多进程「设备级恢复（`mode="real"`）」压测（2026-09-29）

> 定位：**实测记录**。收尾 `../prototype/docs/OPEN_ITEMS_AUDIT_20260929.md` 的 **A2**
> —— 现网默认走 `mode="probe"`（进程内安全），而 **`real`（真重建：`destroyContext →
> aclrtResetDevice → setDevice → 重建`）在多 rank 并发下会不会互扰，此前没有生产级证据**
> （旧档 `distributed_inference/docs/DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md` 明写
> "real/hybrid 在生产默认启用前需压力测试验证多卡并发恢复"）。
> 边界：结论只在**本次档位/环境**（宿主 `npu1-27`、`torch_npu 2.11.0`、精简容器只挂 `davinci1,2,3`）成立；
> **未触及 P800 / MLU590，不得外推**。

---

## 0 结论先行

| 判定 | 结果 |
|---|---|
| **A2 压测** | ✅ **`MULTIPROC_REAL_RECOVER_PASS`** —— 3 rank（一卡一进程）× **30 轮**，每 rank 各当 10 次恢复者，**共 30 次真重建，零失败、零异常**（11.0 s） |
| **S1 恢复者** | 30/30 轮：`recovered=True` · 契约**五键齐全** · **context 三键取值正确**（`context_recreated=True`、`context_supported=True`、`context_count=0`）· `mode="real"` · `ordinal` 正确 · `state` 在四态取值域内 |
| **S2 隔离性** | 60/60 个"非恢复者轮次"：同输入 **digest 逐位不变**（89440）· `device_state` 仍 `available` · `context_query()` 快照逐键不变 · `probe_device=True` · 无异常 |
| **S3 重建后可继续** | 30/30：重建后 `set_device` + **再建流成功**（`Stream`）+ digest 正确（89440）+ `device_state` 回到 `available`（R4） |
| **本轮新发现** | **3 处**（台账 **第 21 / 22 / 23 条**），均**已修 + 非空转验证**；其中 1 处是"判据静默空转"类 |
| **破坏面回归** | ✅ 910C 第 6 轮 **9 项全绿**（离线 78/0/1 · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · 契约不变式 4/4 · 职责审计 39/0/0 · 错误闭环 5/0/0 · B/C 探针 7/7 · A2-P1 定向验证） |

> ⚠️ **本轮最该记住的一条**：判据失败时，**先确认"判据与被测对象看的是不是同一个世界"**
> —— 本次一条假 FAIL 的根因不是行为错，而是**共享状态机被两条导入名各加载了一份**（第 23 条）。

---

## 1 压测怎么设计的（三件套 + 判定）

脚本：`../prototype/probes/recover_multiproc_stress.py`（**芯片无关**，`--backend` 参数化，P800/MLU590 可复用）。

| 设计点 | 做法 | 为什么这样 |
|---|---|---|
| 进程模型 | `spawn` + 一 rank 一进程一卡（`mp.Barrier` 逐轮对齐） | 与真实分布式训推同构；`fork` 在 NPU 环境有风险（项目既有结论） |
| 故障注入 | 每轮**轮转**一个 rank 当"恢复者"：先用共享状态机把它置 `ISOLATED`，再调 `rt.recover_device(dev, mode="real")` | `real` 只在设备处于 ISOLATED 时才执行重建（`recovery.recover_device` 的既有约束） |
| 判定 S0 | 全轮无异常 | 防止"崩了"被读成"0 失败" |
| 判定 S1 | 恢复者返回 **五键 + context 三键**齐全且取值正确；`context_recreated` **必须为 True** | 五键 = `{ordinal, mode, recovered, state, detail}`；context 三键 = `{context_supported, context_count, context_recreated}` |
| 判定 S2 | 非恢复者 **digest 逐位不变** + 状态/上下文快照不变 + 探针可用 | 见下"为什么用 digest" |
| 判定 S3 | 重建后重新 `set_device` + 再建流 + 重算 digest 正确 + 状态回 `available` | R4 要求 |
| 轮次 | ≥ 30（本次正好 30） | 沿用审计给定的收尾条件 |

**为什么 digest 能做到"逐位相同"而不是容差近似**：算式取
`x = arange(1..64)`（fp32）与 `A = tril(ones(64,64))` ⇒ 每个中间结果都是**整数且远小于 2²⁴**
⇒ fp32 可精确表示、**与归约顺序无关**；闭式 `Σk² = 64·65·129/6 = **89440**`，
脚本用纯 Python 复算同一算式 ⇒ 同时验"**算得对**"与"**没变化**"。

**非恢复者复用同一批张量**（不重建）：若它所在进程的上下文被他人重建波及，这批张量就会失效/失真
⇒ 这比"每轮重新分配再算"是**更强**的"未受影响"证据。

---

## 2 实测结果

```
=== A2 多卡多进程 real 恢复压测：backend=ascend devices=[0, 1, 2] ranks=3 rounds=30 ===
[前置声明] ISOLATED 由 runtime/conformance/device_state.py（文档化用法）构造，公开面无此入口；非真实硬件 L4 故障。
[结果] 恢复者轮次 30（每 rank 各 ~10 次）
  PASS  S1_recoverer_real_rebuild
  PASS  S2_peers_unaffected
  PASS  S3_usable_after_rebuild
  PASS  S0_no_exception
  rank0: ok=True victim轮次=10 异常轮=[]
  rank1: ok=True victim轮次=10 异常轮=[]
  rank2: ok=True victim轮次=10 异常轮=[]
MULTIPROC_REAL_RECOVER_PASS（11.0s）
```

**恢复者一轮的逐键原样输出**（rank0 第 0 轮）：

```
keys: ['context_count','context_recreated','context_supported','detail','mode','ordinal','recovered','state']
recovered/state/mode/ordinal: True isolated real 0
context_supported/context_count/context_recreated: True 0 True
detail: rebuild_mode=real：真实重建成功（aclrtResetDevice 序列已执行）
stream_after: Stream | digest: 89440 | post_state: available
```

**非恢复者一轮的快照**（rank0，digest 全 30 轮恒为 89440）：

```json
{"state": "available",
 "context_query": {"queryable": true, "present": true, "ordinal": 0,
                   "flags": null, "managed_by": "external", "reason": ""},
 "probe": true}
```

**状态机事件**（每个 rank 进程内都能看到自己的 30 次转换）：

```json
{"ordinal": 0, "state": "available",
 "last_transition": {"from": "isolated", "to": "available",
                     "reason": "a2-stress r27: 多卡并发 real 重建", "ts": …}}
```

> ⚠️ **`state="isolated"` 与 `recovered=True` 同时出现不是矛盾**：`state` 的时点是
> **调用时（恢复前）**（本轮定死，见 §3.2），`recovered` 的语义是**设备当前可用**。
> 前者=从哪来，后者=现在能不能用；恢复**后**的状态看 `post_state` / `device_state(ordinal)`。

---

## 3 本轮新发现 3 处（均已修 + 非空转验证）

> 编号按台账 `../prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md`：**第 21 / 22 / 23 条**（§2.12–§2.14）。
> 顺序即发现顺序：**先撞上第 21 条 → 修的过程中撞上第 22 条 → 写判据时撞上第 23 条**。

### 3.1 第 21 条：派生字段用错了上游事实 ⇒ **未重建却声称已重建**

`recover_device()["context_recreated"]` 原由 `bool(mode == "real" and rec["recovered"])` **反推**；
而 `recovered` 的语义是「设备当前可用」⇒ **三条误报路径**（同一次返回里 `detail` 与它自相矛盾）：

| 场景 | `detail`（同一次返回） | `context_recreated`（修前） | 实际 |
|---|---|---|---|
| ascend：健康设备上调 `real` | `设备状态=available，无需重建，探活可用` | **`True`** | 未执行任何重建（实测 `context_query` 前后**逐键相同**） |
| kunlun：`real`（未声明 `recovery_real`） | `昆仑芯无设备级重置/重建原语 … → real 模式不支持` | **`True`** | 不可能重建 |
| cambricon：同上 | `寒武纪的设备级重置/重建原语未验证 ⇒ real 模式不支持` | **`True`** | 不可能重建 |

**修法（源头给事实，不做反推）**：`conformance/recovery.py` 在**做决策的同一处**记录实际路径
（`last_rebuild_path()` = `None` / `"probe"` / `"aclrtResetDevice"`）→ 后端经私有键 `_rebuilt` 如实回报 →
基类只认该事实：`context_recreated = bool(_rebuilt and mode in ("real","hybrid"))`，
**未回报视为未重建**（宁可不声明，不臆造）。顺带修 ascend 的 `detail`：真实重建与探针重试成功**分开说**。

### 3.2 第 22 条：字段**时点**未定义 ⇒ 同一 dict 内看着自相矛盾

契约 §1.5 只说「`state` 为设备四态之一」，**没说调用前还是调用后**。实现取的是**调用时（恢复前）**，
于是真重建成功时返回 `state="isolated"` + `recovered=True` —— 下游按"当前状态"读会以为设备**仍在隔离**。

**修法（只定口径、不改行为）**：契约 §补充 写明 `state` = **调用时（恢复前）**的状态，
并解释该组合的正确读法；离线自检加判据**钉死时点**（防再漂移）。

### 3.3 第 23 条：共享资产被**两条导入名**各加载一份 ⇒ 判据「静默空转」

`runtime/conformance/device_state.py` 既可作**扁平模块** `device_state` 导入（后端与《用法》所用），
也可作**包路径** `runtime.conformance.device_state` 导入 ⇒ Python **各执行一次文件** ⇒ 进程内**两个状态机**。
**无设备即可复现**（实测）：

```
flat is pkg                : False
flat._STATES is pkg._STATES: False
flat 视角 dev0 : isolated        # 经扁平名 set_device_state(ISOLATED)
pkg  视角 dev0 : available       # 包路径完全无感知
```

一侧置状态、另一侧**毫无感知且不报错** ⇒ 一切「先置状态、再判定」的判据都**静默空转**。
**本次即因此产出一条假 FAIL**（"state 时点漂移"），追下去才发现是"**判据与被测对象不在同一个状态机上**"。

**修法**：在两个带可变模块级状态的共享资产（`device_state.py` / `recovery.py`）尾部，
把两条导入路径指向**先加载的那一份**（`sys.modules.setdefault`），并补导父包、挂好属性
（使 `import runtime.conformance.<name> as X` 的**语句形式**也走同一对象）。
`errors.py` 无模块级可变状态，未改。

### 3.4 三条的**非空转验证**（新增判据都必须能真的失败）

| 新判据 | 非空转证据 |
|---|---|
| 未声明 `recovery_real` ⇒ `context_recreated` 必须 `False`（第 21 条） | 修前在 **kunlun / cambricon** 上**真的 FAIL**（`context_recreated=True（real 不支持却声称已重建 ⇒ 违反 I1 诚实声明）`）——**判据抓到的是真缺陷** |
| `mode="probe"` ⇒ `context_recreated` 必须 `False`（第 21 条） | 注入缺陷（把派生改成无条件 `True`）：ascend / kunlun 上**两条判据均 FAIL** |
| `state` == 调用时（恢复前）状态（第 22 条） | 注入缺陷（去掉第 23 条的单一实例收敛）：该判据**真的 FAIL**（`调用前=isolated 返回的 state='available'`） |
| 共享状态机**单一实例**（第 23 条） | 去掉修复后**真的 FAIL**（`flat is pkg=False；_STATES 同一 dict=False`），恢复后 PASS |

---

## 4 破坏面回归（910C 第 6 轮 · 9 项全绿）

破坏面 = 本轮改动的 5 个文件：`conformance/recovery.py` · `conformance/device_state.py` ·
`backends/base.py` · `backends/ascend/backend.py` · `scripts/backend_offline_check.py`
⇒ 受影响的是**「恢复 / 设备状态 / 上下文」这条链**。

| # | 判定项 | 结果 |
|---|---|---|
| 1 | 离线契约自检（ascend） | **78 / 0 / 1 跳过**（判据数 76 → 78：+2 条取值域/时点；另 +1 条单一实例） |
| 2 | 跨后端对称性 `--all` | **5 / 0** |
| 3 | 组件冒烟 | **52 / 0** |
| 4 | conformance 13 例 | **13 / 13 `CONFORMANCE_PASS`** |
| 5 | conformance 推理 6 例 | **6 / 6 `CONFORMANCE_PASS`** |
| 6 | 契约不变式 I1–I4 | **4 / 4 `CONTRACT_INVARIANTS_PASS`** |
| 7 | 职责响应审计（39 sub-part） | **39 OK / 0 FAIL / 0 SKIP** |
| 8 | 错误注入→恢复闭环 | **闭环 5 / 跳过 0 / 失败 0** |
| 9 | 工作包 B/C 真机探针 | **7 / 7 PASS**（B1/B3/B4/C1/C2/C3/C4） |
| 10 | A2-P1 定向验证（修复双向） | **`A2P1_FIX_VERIFIED`**（11 条判据全 PASS） |

**离线侧三家同时复跑**（本机，无设备）：ascend **78/0/1** · kunlun **80/0/1** · cambricon **68/0/0**。

### 4.1 免跑项（**当轮可验证**的理由）

| 项 | 不跑的理由 | 可验证形式 |
|---|---|---|
| 训练腿 / 推理腿 | 两条腿脚本**不调用** `recover_device` / `device_state` ⇒ 不在破坏面内 | `grep -cE "recover_device\|device_state" runtime/proto/proto_train_leg.py` = **0**；`proto_infer_leg.py` = **0** |
| 服务化 | `serve_standard.sh` 里 2 处命中是**注释**与 `MONITOR=1` 时的**可选清理分支**（默认 `MONITOR=0`），默认路径不触及 | `grep -nE "recover_device\|device_state" scripts/serve_standard.sh` ⇒ 只有第 47 行（注释）与第 205 行（`[ "$MONITOR" = "1" ] && …`） |

> ⚠️ 该理由**只对当轮成立**：后续任何落在 `Stream` / `allocate` / 前向路径上的改动都会让它失效。

---

## 5 环境与「跑的是当前版本」

| 核验项 | 实测 |
|---|---|
| 宿主 / 容器 | `npu1-27` · 精简容器 `dc-lean-910c-20260929`（只挂 `davinci1,2,3` + 3 个管理设备） |
| 解释器 | `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python`（`torch 2.11.0+cu130` / `torch_npu 2.11.0`，**无 torch_fl**） |
| 设备 | `torch.npu.device_count()=3`；`probe_device` 三轮均 True |
| 跑的是当前版本 | `runtime/backends/` = `{ascend, cambricon, kunlun}`（无 `flagos`）；`recovery.py` 含 `last_rebuild_path`、`base.py` 含 `_rebuilt`、`device_state.py` 含 `ALIASED`（回归脚本第 [0] 步逐项打印） |
| 同步方式 | `rsync -az --delete prototype/ → /mnt/raid/hliu553/dc_legs_20260929/prototype/` |
| 用卡 | **未停任何他人容器**；本轮只用我方精简容器，收工后已 `docker stop` |

---

## 6 结论的边界（**未**做的部分，如实登记）

| 项 | 状态 | 说明 |
|---|---|---|
| **真实硬件 L4 故障**触发这条链 | **未做** | ISOLATED 是**构造**出来的（公开面无此入口，见 §7）；真实故障（AICORE 异常）无法在健康卡上制造 |
| 重建后**不重新分配**即使用旧张量 | **未做（刻意）** | 无法预判是抛错、读出垃圾还是挂死 ⇒ 会污染整轮；如实登记为未做项 |
| `P800 / MLU590` 的 A2 | **不适用 / 未跑** | 两家**均未声明 `recovery_real`**（设备级重置原语：kunlun 实测无、cambricon 未验证）⇒ 脚本会**如实跳过**（`SKIP_UNSUPPORTED`，退出码 3，不计失败）；**不得**由 910C 外推 |
| 多机（>1 节点） | 未做 | 超出「同厂商多卡」范围裁定 |
| `context_recreated` 的**独立**佐证 | **部分** | 真重建的佐证来自 `aclrtResetDevice` 返回值 + `detail` + 状态机事件（**同源**）；本层没有"重建计数器"这类**独立可观测物**，且 `context_count` 在 ascend 恒为 0（本层不建上下文，`managed_by="external"`）⇒ 如实标注 |
| 吞吐/性能 | 不做结论 | 11.0 s / 30 轮（≈0.37 s/轮）是**共享机 + 单次取数**，只作"同档可比" |

---

## 7 一个需要你知道的接口面事实（**未擅自改**）

**公开面（`runtime`）没有把设备置为 `ISOLATED` 的入口**：实测 `recover_device(mode="real")` 在健康设备上
**永远走"无需重建"分支**（`detail` 自述"无需重建"）；`handle_error` / `set_device_state` 都只在
`runtime/conformance/` 内部，未导出。故本压测按该模块**文档化用法**（`sys.path.insert(.../conformance)`）构造隔离。

⇒ **"某卡 L4 故障 → 设备级恢复"这条链在公开面上不可触发**。这是否要补一个公开入口（例如导出
`handle_error`，或新增 `recover_device(..., force=True)`），**属接口面扩张**，已按纪律**不改、先报**，
等你裁定。

---

## 8 一键复跑

```bash
cd dev/device-context
ssh 910C 'docker start dc-lean-910c-20260929'
rsync -az --delete prototype/ 910C:/mnt/raid/hliu553/dc_legs_20260929/prototype/

# ① A2-P1 定向验证（单进程，11 条判据；先跑它当门禁）
ssh 910C 'docker exec -e DC_ROOT=/mnt/raid/hliu553/dc_legs_20260929/prototype \
  dc-lean-910c-20260929 /mnt/raid/hliu553/venvs/venv-infer-a/bin/python \
  /mnt/raid/hliu553/dc_legs_20260929/a2_p1_probe.py'

# ② A2 压测（3 rank × 30 轮）
ssh 910C 'docker exec -e DC_ROOT=/mnt/raid/hliu553/dc_legs_20260929/prototype \
  dc-lean-910c-20260929 /mnt/raid/hliu553/venvs/venv-infer-a/bin/python \
  /mnt/raid/hliu553/dc_legs_20260929/prototype/probes/recover_multiproc_stress.py \
  --backend ascend --devices 0,1,2 --rounds 30 \
  --out /mnt/raid/hliu553/dc_legs_20260929/a2_out_full'

# ③ 破坏面回归（9 项）
ssh 910C 'docker exec -d dc-lean-910c-20260929 bash -lc \
  "bash /mnt/raid/hliu553/dc_legs_20260929/run_regress_ascend_r6.sh \
   > /mnt/raid/hliu553/dc_legs_20260929/run_r6.log 2>&1"'
```

---

## 9 证据清单（`../probes/`，共 8 份）

| 文件 | 内容 |
|---|---|
| ⭐ `a2_recover_multiproc_910c_npu_20260929.json` | **A2 压测原始结果**（3 rank × 30 轮逐轮记录、digest、快照、返回 dict、状态机转换） |
| ⭐ `a2_recover_multiproc_910c_npu_20260929.log` | 同上（stdout，含四项判定与边界声明） |
| `a2p1_fix_verified_910c_npu_20260929.json` | **A2-P1 定向验证**（D1 未重建 ⇒ False / D2 真重建 ⇒ True / D3 probe ⇒ False，11 条判据） |
| `a2p1_fix_verified_910c_npu_20260929.log` | 同上（stdout） |
| `a2_p0_public_surface_910c_npu_20260929.json` | **A2-P0 前置探针**（健康设备上调 `real` 的逐键返回 + 公开面无 ISOLATED 入口的实测） |
| `r6_regress_910c_npu_20260929.log` | **第 6 轮破坏面回归汇总**（9 项 + 免跑理由的可验证形式） |
| `r6_regress_910c_npu_20260929_out/`（**16 份** log/json） | 回归逐项原始输出（离线/对称性/冒烟/conformance 13+6/契约不变式/职责审计/错误闭环/B-C 探针/A2-P1） |
| `offline_3backends_910c_npu_20260929.log` | **离线自检三家同跑**（ascend 78/0/1 · kunlun 80/0/1 · cambricon 68/0/0） |

> ⚠️ 证据命名里的 `910c` 指**芯片实例**，`npu` 才是**设备后端名**（见主看板「命名陷阱」）。
> 证据命名规范与「当前结论 = 哪一份」见 `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。
