# 910C · 两条腿 + 服务化四形态复跑（A1 + A4 收尾 · 2026-09-29）

> 定位：**实测记录**。回答 `../prototype/docs/OPEN_ITEMS_AUDIT_20260929.md` 里
> **A1（两条腿未在 B/C 之后的版本复跑）** 与 **A4（多卡 TP / 关闭 `--enforce-eager` 形态未跑）**
> 两项是否收尾。
> 边界：结论只在**当前档位/环境**成立 —— 宿主 `npu1-27`、训练/服务镜像见 §6、
> 解释器 `venvs/venv-infer-a`（仅 torch_npu）、容器只挂 `davinci1,2`（服务）与 `davinci1,2,3`（两条腿）。
> **本轮未触及 P800 / MLU590** ⇒ 本报告结论**不得外推**到另两实例。

---

## 0 结论先行

| 判定项 | 本轮结果 | 历史对照 | 判定 |
|---|---|---|---|
| **A1 · 训练腿**（2 卡 torch_npu + HCCL，50 步） | `TRAIN_LEG_PASS` **6/6** · loss **15.4498 → 11.1479** · **4513.2 tok/s**（两卡合计） | 09-22 `6/6` · 4402.3 tok/s；09-28 `6/6` · 4075.4 tok/s | ✅ **无回归** |
| **A1 · 推理腿前向**（单卡 transformers） | `INFER_LEG_PASS` **14/14** · dim 1024 · 78.84 句/s · p50 37.59 ms · 区分度 **0.6391** | 09-22 `14/14` · 77.49 句/s · p50 38.31 ms · 区分度 **0.6391** | ✅ **无回归** |
| **A1 · 服务化**（TP=1 EAGER=1 基线） | `SERVE_STANDARD_PASS` · 就绪 **30 s** · 冒烟 1 s · dim 1024 · 范数 1.000000 | 09-28 `PASS` · 35 s 就绪 · 冒烟 0 s | ✅ **无回归** |
| **A4 · TP=2 服务化** | `SERVE_STANDARD_PASS` · 就绪 **45 s** · 冒烟 1 s · dim 1024 · 范数 1.000000 | 历史**从未跑过** | ✅ **首次覆盖** |
| **A4 · 关闭 `--enforce-eager`（TP=1）** | `SERVE_STANDARD_PASS` · 就绪 **45 s** · 冒烟 0 s · dim 1024 · 范数 1.000000 | 历史**从未跑过** | ✅ **首次覆盖** |
| **A4 · TP=2 + 关闭 eager 组合** | `SERVE_STANDARD_PASS` · 就绪 **80 s** · 冒烟 0 s · dim 1024 · 范数 1.000000 | 历史**从未跑过** | ✅ **首次覆盖** |
| 本轮新发现 1 处（工具假信号） | 已修 + **非空转验证**（见 §4） | —— | ✅ 已收尾 |

> **A1 与 A4 均已收尾。** 一轮窗口内共跑 **4 个服务形态 + 1 次非空转验证 + 两条腿**，全绿、0 失败。
> **两条腿仍须串行**：训练容器（挂 `davinci1,2,3`）与服务容器（挂 `davinci1,2`）挂载集相交，
> 同一时刻只能有一个真正 `acl.init` 成功 —— 本轮**先停训练容器、再起服务容器**。

---

## 1 为什么这一轮必须做（免跑理由失效的依据）

09-29 上一份 910C 报告 `ASCEND_910C_REGRESS_AFTER_FIX_20260929.md` §5 写的免跑理由是
「本次修复**不触及**前向/服务路径 ⇒ 该腿本轮免跑」。但**该理由只对当轮成立**：

- 910C 两条腿最后一次取数是 **09-22**（训练腿 `probes/accept_train_npu_20260922/`、推理腿 `probes/proto_infer_leg_result.json`），
  服务化最后一次取数是 **09-28**；
- 此后共享层被改过：路线 B 退出（`7d05d69`）· 容错修复（`vendor_codes` / `state_token` / L2 文案等价类）·
  **工作包 B/C 接口（`runtime/backends/base.py` / `runtime/api/stream.py` / `__init__.py`）** · `context_query`（`ab80d87`）· B1 契约不变式（`6b664ce`）；
- **训推两条腿都建流、都用跨流内存保护**（`Stream` / `record_stream` / `allocate`）⇒ B/C 的改动**正落在它们的破坏面内**。

⇒ 按「按破坏面覆盖」纪律，两条腿必须用**当前版本**重跑。本报告即该轮取数。

---

## 2 两条腿实测（A1）

### 2.1 训练腿 · 2 卡分布式微调（`TRAIN_LEG_PASS 6/6`）

| 判据 | 实测 |
|---|---|
| `device_context` | rank0 绑定 `npu:0`、可见设备 **2**；统一流 `<Stream backend=ascend native=Stream>` ✅ |
| `model_load` | 加载到 `npu:0` ✅ |
| `comm_all_reduce` | 实测 **3.0** / 期望 3.0 ✅ |
| `comm_all_gather` | 收集到 `[0.0, 1.0]` ✅ |
| `comm_p2p` | rank0 接收一致 `True` ✅ |
| `loss_decrease` | **首 15.4498 → 末 11.1479**（50 步）；无 NaN `True` ✅ |

- 口径：`DC_BACKEND=ascend` · `DC_DIST_BT=hccl` · `ASCEND_RT_VISIBLE_DEVICES=0,1` · `MAX_STEPS=50 BATCH=4 SEQ=128`
- 吞吐：**4513.2 tok/s（两卡合计）**，per-rank 2256.6 tok/s，`elapsed_s=11.34`
- 两个 rank 判定一致（`rank0` / `rank1` 均为 `6/6`）

**可比性说明**：4513.2 / 4402.3 / 4075.4 三个值是**同一实例、同一模型、同超参**下的三次取数，
差异来自**共享机当轮卡占用与启动抖动**，**不构成性能结论**；跨实例数字（如 MLU590 的 3015.3）**不可直接比**。

### 2.2 推理腿前向 · 单卡（`INFER_LEG_PASS 14/14`）

| 项 | 实测 |
|---|---|
| 判据 | **14/14**（含 `vendor_code_map` —— 910C 有厂商数字码表，故比另两家多 1 项） |
| 设备 | `device_count=3`（容器可见）；`npu:0` |
| 向量 | dim **1024** · 同主题 0.7821 / 异主题 0.1430 · 范数 1.0 · 无 NaN |
| 时延/吞吐 | **78.84 句/s** · p50 **37.59 ms** · p90 39.10 ms · avg 38.05 ms（5 轮） |
| 区分度 | **0.6391**（与 09-22 完全一致） |
| 错误分级 | `L2_PARAM → raise`（`graded_by=code_map`）✅ |

> 09-22 推理腿前向同为 `14/14`、区分度同为 `0.6391` ⇒ **数值层面无变化**。
> （审计文档 A1 的收尾条件里写的是 `13/13`，那是**另两家**的口径 —— 910C 因有厂商码表为 **14/14**，本轮以实物为准。）

---

## 3 服务化四形态（A1 基线 + A4 多卡 TP / 非 eager）

统一入口 `prototype/scripts/serve_standard.sh`，`DC_BACKEND=ascend SERVE_FORM=embed
MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B STOP_AFTER=1 SMOKE_TIMEOUT=180`。

| 配置 | 就绪 | 冒烟耗时 | 维度 / 范数 | verdict |
|---|---|---|---|---|
| TP=1 · EAGER=1（A1 基线复跑） | **30 s** | 1 s | 1024 / 1.000000 | `SERVE_STANDARD_PASS` |
| **TP=2 · EAGER=1**（A4） | **45 s** | 1 s | 1024 / 1.000000 | `SERVE_STANDARD_PASS` |
| **TP=1 · EAGER=0**（A4） | **45 s** | 0 s | 1024 / 1.000000 | `SERVE_STANDARD_PASS` |
| **TP=2 · EAGER=0**（A4 组合） | **80 s** | 0 s | 1024 / 1.000000 | `SERVE_STANDARD_PASS` |

### 3.1 「TP=2 真的用了 2 卡」的证据（不是参数被吞）

来自服务端日志 `serve_vllm_910c_npu_20260929_r1_tp2_eager1.log`（**逐字**，未做任何改写）：

```
INFO 09-29 08:33:59 [multiproc_executor.py:139] DP group leader: node_rank=0, node_rank_within_dp=0, master_addr=127.0.0.1, mq_connect_ip=10.120.72.27 (local), world_size=2, local_world_size=2
INFO 09-29 08:34:12 [parallel_state.py:1402] world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:50769 backend=hccl
INFO 09-29 08:34:12 [parallel_state.py:1402] world_size=2 rank=1 local_rank=1 distributed_init_method=tcp://127.0.0.1:50769 backend=hccl
(Worker_TP0 pid=6594) INFO 09-29 08:34:12 [model_runner_v1.py:3374] Starting to load model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B...
(Worker_TP1 pid=6595) INFO 09-29 08:34:12 [model_runner_v1.py:3374] Starting to load model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B...
```

⇒ 两个 Worker、各自 `local_rank` 不同、**进程组后端 `hccl`**；容器可见设备数 = 2（只挂 `davinci1,2`）。
复现：`grep -aE "world_size=2, local_world_size=2|rank=[01] local_rank=[01]|Starting to load model" serve_vllm_910c_npu_20260929_r1_tp2_eager1.log`

### 3.2 「关闭 `--enforce-eager` 真的生效」的证据

`serve_vllm_910c_npu_20260929_r1_tp1_eager0.log`（**逐字**）：

```
WARNING 09-29 08:35:53 [vllm.py:1041] Pooling models do not support full cudagraphs. Overriding cudagraph_mode to PIECEWISE.
INFO 09-29 08:35:53 [platform.py:408] PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode
INFO 09-29 08:35:53 [utils.py:660] Calculated maximum supported batch sizes for ACL graph: 62
INFO 09-29 08:36:09 [compiler_interface.py:207] enable_npugraph_ex is enabled, which will bring graph compilation optimization.
```

同一日志的 EngineCore 初始化行是**单行超长 dataclass repr**，此处只摘字段片段（非整形引用）：
`enforce_eager=False` · `'mode': <CompilationMode.VLLM_COMPILE: 3>` · `'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>`。

**与 `EAGER=1` 的逐字段对照**（同一模型、同一脚本，只差这一个变量）：

| 配置 | 日志文件 | `enforce_eager=` | `CompilationMode.*` | `CUDAGraphMode.*` |
|---|---|---|---|---|
| EAGER=1 | `serve_vllm_910c_npu_20260929_r1_tp2_eager1.log` | `True` | （无编译模式行） | `NONE` |
| EAGER=0 | `serve_vllm_910c_npu_20260929_r1_tp1_eager0.log` | `False` | `VLLM_COMPILE` | `PIECEWISE` |

⇒ 实际走到 **ACL Graph（PIECEWISE）** 路径，不是"参数没传/被忽略"。
复现：`grep -aoE "enforce_eager=[A-Za-z]*|CompilationMode\.[A-Z_]*|CUDAGraphMode\.[A-Z_]*" <上述文件> | sort -u`

> ⚠️ **如实记录一条上游行为**：embedding（`--runner pooling`）形态下 vLLM **主动拒绝 full cudagraph**、
> 自动降为 `PIECEWISE` —— 这是框架既有约束，**不是**我们的实现选择。
> 因此本轮 `EAGER=0` 验证的是「**图捕获子集形态可用**」，**不能**声称"910C 上 full ACL graph 可用"。

### 3.3 就绪耗时如实说明

四形态就绪 30–80 s、冒烟 0–1 s；`EAGER=0` 与 `TP=2` 都更慢（图编译 / 双 rank 加载）。
**单次取数 + 共享机** ⇒ 只作"同档可比"，**不构成性能结论**。

---

## 4 本轮新发现 1 处：停机复查的**假信号**（已修 + 非空转验证）

### 4.1 现象（改动前，证据原样留档）

`serve_standard.sh` 在 `STOP_AFTER=1` 停机后紧接着打印「用卡后复查（确认已释放）」。**TP=2** 两轮实测：

| 配置 | 用卡前 dev0 | 「用卡后复查」dev0 | 差 |
|---|---|---|---|
| TP=2 · EAGER=1 | 61.12 GiB | **55.06 GiB** | **−6.06 GiB** |
| TP=2 · EAGER=0 | 61.12 GiB | **55.07 GiB** | **−6.05 GiB** |

读起来像"**没释放 / 内存泄漏**"。实际**不是**：

- 约 1 min 后重新取值 ⇒ dev0 **61.12 GiB**、dev1 **60.90 GiB**（与用卡前**逐位相同**）；
- 宿主侧 `npu-smi info -t proc-mem -i 0` 全程 **`No process in device`**；
- TP=1 两轮**没有**该现象（用卡前后均为 61.12 GiB）。

⇒ 结论：**释放有延迟**（TP=2 有 2 个 `Worker_TP` 进程被 kill，设备侧回收需要时间），
脚本"kill 完立刻读数"把**正在回落**读成了**未释放**。

### 4.2 修法（改的是**工具的观测时序**，不是判定标准）

`prototype/scripts/serve_standard.sh` 新增 4 处（共 +10 行逻辑）：

1. 新参数 `RELEASE_WAIT`（默认 **30** s；`0` = 旧行为"立即取值"）；
2. 新辅助函数 `card_free_gib()`（设备侧空闲显存合计，POSIX awk 求和，兼容容器内 mawk/gawk）；
3. 停机复查改为**轮询到回落稳定**（相邻两次采样空闲显存增幅 < 0.05 GiB 即停），并打印**实际等待秒数**；
4. 超过上限仍未稳定时打印 `⚠️` 提示 + 交叉核对入口（宿主 smi 进程视图）。

> ⛔ **没有"改判据变绿"**：`verdict` 仍只看 `ready` + `smoke`，该复查**不参与判定**；
> 改动前的失败/误导读数**原样留档**（`*_PRE_FIX_*.log` 4 份）。

### 4.3 非空转验证（证明新分支**能真的触发**）

`tp2_eager1_wait5`（同一 TP=2 配置，仅把 `RELEASE_WAIT` 压到 **5 s**）：

```
  （释放复查：等待 5s 后取值；上限 RELEASE_WAIT=5s）
  ⚠️ 释放复查：5s 内空闲显存仍在增长 —— 可能仍在回落，或确有残留
[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)
```

⇒ ⚠️ 分支**可达且真的会报**；同时 `verdict` 不受影响（仍 PASS），证明"观测"与"判定"确已解耦。

### 4.4 修后复跑（r1 · 4 形态）

| 配置 | 释放回落等待 | 用卡后 dev0 / dev1 | 与用卡前 |
|---|---|---|---|
| TP=1 · EAGER=1 | 5 s | 61.12 / — | **一致** |
| TP=2 · EAGER=1 | **10 s** | 61.12 / 60.90 | **一致** |
| TP=1 · EAGER=0 | 5 s | 61.12 / — | **一致** |
| TP=2 · EAGER=0 | **10 s** | 61.12 / 60.90 | **一致** |

⇒ 假信号消除；且 TP=2 需等 10 s、TP=1 只需 5 s —— 与新分支的设计意图一致。

### 4.5 同一族里**未**处理的一条（如实登记，不在本轮改）

服务名与形态**不联动**：`ascend` 分支的 `SERVED_NAME` 默认 `qwen3-4b`，
而本轮用 `SERVE_FORM=embed` + `Qwen3-Embedding-0.6B` 起的是 **embedding 服务**
⇒ 服务端 `/v1/models` 会报模型名 `qwen3-4b`（服务端日志 `'served_model_name': ['qwen3-4b']`）。
数值与 verdict **不受影响**，但对下游核对**有误导**。
⇒ 本轮**保持单变量**不改（一改就得把 4 个形态全部重跑），**登记为待收尾项**。

---

## 5 环境与「跑的是当前版本」核验

| 核验项 | 实测 |
|---|---|
| 分支 / HEAD | `kistich/device-context` · `1f26633`（本地 = 远端，工作区干净） |
| 后端目录 | `runtime/backends/` = `{__init__.py, ascend, base.py, cambricon, kunlun, registry.py}` —— **无 `flagos`**（路线 B 已退出） |
| 同步方式 | `rsync -az --delete prototype/ → 910C:/mnt/raid/hliu553/dc_legs_20260929/prototype/`（**不复用**旧目录 `dc_regress_20260929/`） |
| 两条腿容器 | `dc-lean-910c-20260929`，挂 `davinci1,2,3` + 3 个管理设备；`get_device_count=(3,0)` |
| 服务容器 | **新建** `dc-lean-infer-910c-20260929`，镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，**只挂 `davinci1,2`** |
| 解释器（两条腿） | `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python`（`torch 2.11.0+cu130` / `torch_npu 2.11.0` / `transformers 4.57.6`；**无 torch_fl** ⇒ 物理隔离） |
| 服务容器 torch | `torch 2.10.0+cpu` / `torch_npu 2.10.0` / `vllm 0.20.2`；`torch.npu.device_count()=2` |
| 名额前置 | 起跑前探针 `DEVICE_PROBE_PASS`：`acl.init` 成功、3 卡 free 62.6 GB / 62.4 GB / 62.6 GB |

> ⚠️ **为什么服务用"精简容器"而不是既有 `flagos-infer-910c`**：后者挂**全部 16 张** `davinci`，
> 其挂载集与任何活跃容器的挂载集**必然相交**（09-28 正是因此失败：`acl.init`=500000 +
> `Different containers share the same device`）。只挂要用的空闲卡即可与现网并存 —— 见
> `ASCEND_HOST_NAMESLOT_RULE_20260929.md`。

---

## 6 结论的边界（**未**做的部分，如实登记）

| 项 | 状态 | 理由 |
|---|---|---|
| **P800 / MLU590 的两条腿与服务化** | **未跑** | 本轮破坏面为**共享层 + `serve_standard.sh`**；`serve_standard.sh` 的改动落在三实例**共用**路径上 ⇒ 另两家**也需在各自窗口复跑**（不得由 910C 外推）。已登记为后续项 |
| `graceful shutdown`（SIGTERM 让服务自行释放） | **未验** | 本轮用脚本既有的 `kill -9` 路径（与 09-28 口径一致）；§4 的现象正来自该路径 |
| full cudagraph（非 PIECEWISE） | **不可得** | 上游对 pooling 模型主动拒绝（见 §3.2），**不是**我们未做 |
| 多机（>1 节点）分布式训推 | 未跑 | 超出「同厂商多卡」范围裁定 |
| 吞吐/时延的横向比 | **不成立** | 共享机 + 单次取数 ⇒ 只作"同档可比" |

---

## 7 一键复跑

```bash
cd dev/device-context

# 0) 起精简训练容器（只挂要用的空闲卡；先看是否需要停再决定）
ssh 910C 'docker start dc-lean-910c-20260929'
rsync -az --delete prototype/ 910C:/mnt/raid/hliu553/dc_legs_20260929/prototype/

# 1) 名额探针（不通过就别怀疑代码，先看宿主挂载集是否相交）
ssh 910C 'docker exec dc-lean-910c-20260929 \
  /mnt/raid/hliu553/venvs/venv-infer-a/bin/python \
  /mnt/raid/hliu553/dc_legs_20260929/probe_device_quota.py'

# 2) 训练腿（2 卡，50 步）
P=/mnt/raid/hliu553/dc_legs_20260929/prototype; O=/mnt/raid/hliu553/dc_legs_20260929/out
PYA=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python; C=dc-lean-910c-20260929
ssh 910C "docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e DC_BACKEND=ascend -e DC_DIST_BT=hccl \
  -e DC_ROOT=$P -e DC_MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
  -e DC_OUT_DIR=$O/train_npu -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 $C \
  bash -lc 'cd $P/runtime/proto && $PYA -m torch.distributed.run --standalone \
    --nproc_per_node=2 --master_addr=127.0.0.1 proto_train_leg.py'"

# 3) 推理腿前向（单卡）
ssh 910C "docker exec -e DC_BACKEND=ascend -e DC_ROOT=$P \
  -e DC_MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B -e DC_OUT_DIR=$O $C \
  bash -lc 'cd $P && $PYA runtime/proto/proto_infer_leg.py'"

# 4) 服务化四形态（**先停训练容器**：两条腿挂载集相交，必须串行）
ssh 910C 'docker stop dc-lean-910c-20260929'
ssh 910C 'docker run -d --name dc-lean-infer-910c-20260929 --network host --shm-size 512g \
  --device=/dev/davinci1 --device=/dev/davinci2 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /mnt/raid/hliu553:/mnt/raid/hliu553 \
  quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 sleep infinity'
# 逐配置跑（TP / DEV / EAGER 三个变量；结果看 $DC_OUT_DIR/serve_standard.log）
M=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B
for spec in "1 0 1" "2 0,1 1" "1 0 0" "2 0,1 0"; do
  set -- $spec
  ssh 910C "docker exec -e DC_BACKEND=ascend -e DEV=$2 -e MODEL=$M -e SERVE_FORM=embed \
    -e STOP_AFTER=1 -e DC_OUT_DIR=$O/serve_tp$1_eager$3_r1 -e SMOKE_TIMEOUT=180 \
    -e TP=$1 -e EAGER=$3 dc-lean-infer-910c-20260929 \
    bash -lc 'cd $P && bash scripts/serve_standard.sh'"
done
```

> ⚠️ 服务脚本内部 `exec > "$DC_OUT_DIR/serve_standard.log" 2>&1`
> ⇒ **外层重定向会拿到空文件**，收结果请读 `$DC_OUT_DIR/serve_standard.log` 与 `vllm_serve.log`。

---

## 8 证据清单（`../probes/`，共 23 份）

**A1 · 两条腿**

| 文件 | 内容 |
|---|---|
| `legs_train_910c_npu_20260929.log` | 训练腿原始日志（含 step 曲线与两条 `TRAIN_LEG_PASS`） |
| `train_leg_910c_npu_20260929_rank0.json` / `_rank1.json` | 训练腿逐判据结果（`6/6`，含 `loss_curve` 50 点） |
| `legs_infer_910c_npu_20260929.log` | 推理腿原始日志 |
| `infer_leg_910c_npu_20260929.json` | 推理腿逐判据结果（`14/14`，含 `env.capabilities` 17 项） |

**A1 + A4 · 服务化**

| 文件 | 内容 |
|---|---|
| `serve_pool_910c_npu_20260929_r1.log` | 改动后 4 形态 + 非空转验证的**汇总日志** |
| `serve_standard_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4） | 逐配置 `serve_standard` 输出（含 verdict、就绪、冒烟耗时、释放复查） |
| `serve_vllm_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4） | 逐配置服务端日志（TP/图捕获的生效证据所在） |
| `serve_standard_910c_npu_20260929_r1_nonidle_wait5.log` | **非空转验证**：`RELEASE_WAIT=5` 触发 `⚠️` 分支 |
| `serve_standard_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4） | **改动前**原样留档（含被修正的误导读数） |
| `serve_vllm_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4） | 同上（服务端） |

> 证据命名规范与「当前结论 = 哪一份」见 `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。
