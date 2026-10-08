# MLU590（寒武纪 `cambricon`）· 补齐轮 m1 验收报告（2026-10-08）

> 执行：Kistich（hliu553）｜定位：**第三家实例的首次全口径覆盖** —— 把职责审计口径补齐到
> **78 项**并首次行使 6 类新探针；同时收尾台账 **D1**。
> 依据：`MLU590_FIX_WORKPACK_20261008.md` §3 第 1–4 组。
> ⚠️ 本文件所有数字均为**本轮实测**；历史值均标注「首测」。

---

## 0 结论先行

**全轮 20 项通过，0 失败。**

| # | 项 | 本轮结果（2026-10-08） | 历史（首测） | 判定 |
|---|---|---|---|---|
| 0 | **模型完整性（步 0）** | `MODEL_INTEGRITY_PASS`（sha256 双向自证 + 310 张量零非有限值） | ⛔ 从未做 | ✅ |
| 1 | 离线契约自检（cambricon） | **88 通过 / 0 失败 / 1 跳过** | 45/0/0（09-29）· 39/0/0（09-28） | ✅（判据数变，0 失败即通过） |
| 2 | 对称性自检（`--all`） | **7 通过 / 0 失败** | 5/0（09-29） | ✅ |
| 3 | 冒烟 `smoke_runtime` | **46 通过 / 0 失败** | 46/0（09-29） | ✅ |
| 4 | conformance 13 例 | `CONFORMANCE_PASS` | 13/13 | ✅ |
| 5 | conformance 推理 6 例 | `CONFORMANCE_PASS` | 6/6 | ✅ |
| 6 | 契约不变式 I1–I4 | `CONTRACT_INVARIANTS_PASS` | ⛔ **从未跑** | ✅ |
| 7 | **职责响应审计** | **`DUTY_RESPONSE_PASS` 61 / 0 / 17（共 78 项）** | 36/0/3（09-28，**旧口径 39 项**） | ✅ |
| 7b | **非空转验证（新增）** | **`SELFCHECK_DUTY_EXT_PASS`：抓到 25 · 本机不适用 14 · 未抓到 0** | ⛔ **从未跑** | ✅ |
| 8 | 错误注入→恢复闭环 | `ERROR_RECOVERY_LOOP_PASS` 5/0/0 | 5/0/0 | ✅ |
| 9 | 公开入口定向验证 | `ENTRY_VERIFY_PASS` | ⛔ **从未跑** | ✅ |
| 10 | 根解析自检 | `ROOT_RESOLUTION_PASS` | ⛔ **从未跑** | ✅ |
| 11 | 流语义 | `STREAM_SEMANTICS_PASS` 8/8 | 8/8 | ✅ |
| 12 | 流配额（N=2000） | `STREAM_QUOTA_PASS` | 3/3 | ✅ |
| 13 | 统一 API 演示 | 跑通（rc=0） | —— | ✅ |
| 14 | **B/C 契约探针** | `PASS`（5 个能力键**全部如实不声明**，见 §4.1） | ⛔ **从未跑** | ✅ |
| 15 | **流优先级 API（台账 D1）** | **`STREAM_PRIORITY_API_PASS` 7/7** | 09-22 记录（**旧接线**） | ✅ **D1 收尾** |
| 16 | **流所有权/释放** | `STREAM_RELEASE_CONTROL_PASS`（R2/R3c/R3d 如实 SKIP） | ⛔ **从未跑** | ✅ |
| 17 | **训练腿（2 卡）** | **`TRAIN_LEG_PASS` 6/6** · `dist=cncl` · loss **15.4498→11.1479** · 3017.2/3018.8 tok/s | 6/6 · 15.4498→11.1479 · 2957.8/3015.3 tok/s | ✅ |
| 18 | **推理腿（前向）** | **`INFER_LEG_PASS` 13/13** · dim 1024 · 40.46 句/s · p50 73.72 ms · **区分度 0.6391** | 13/13 · 39.69~41.08 句/s · p50 72.89 ms · **0.6391** | ✅ |
| 19 | 多进程 `real` 压测 | **如实跳过** | —— | ⏸（本实例**已实测确认不具备** `recovery_real`） |
| 20 | **服务化 `SERVE_FORM=embed`** | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`** · 就绪 **360 s** · 维度 1024 · 范数 **1.000001** | PASS · 就绪 150 s · 维度 1024 · 范数 1.000001 | ✅ |
| 21 | **服务化消费方 `proto_infer_serve.py`** | **`SERVE_LEG_PASS` 10/10** · 维度 1024 · 范数 1.000001 · 区分度 **0.4091** | ⛔ **从未跑** | ✅ |

> **数值可比性**：`区分度 0.6391`（推理腿）与 `loss 15.4498→11.1479`（训练腿）与历史**逐位相同**
> ⇒ 既证明本次改动**没有改变数值结果**，也反证模型资产完好。
> 吞吐（3017.2 / 3064.6 / 2957.8 tok/s）与就绪耗时（360 s vs 150 s）**只作同档可比**：
> 共享机 + 单次取数 ⇒ **不作性能结论**。

---

## 1 步 0：模型资产完整性（本轮新增的前置步骤）

**为什么要做**：同日在 **910C** 上实测到「同名同源」的验收模型被**存储层静默损坏**
（`model.safetensors` 里 2 个张量含 **34396** 个非有限值；`mtime`/`ctime` 停在 2026-09-08 未变；
所在 `md127` RAID5 降级 `[4/3] [UUU_]`）。它的症状是**两条腿以数值判据失败**（`loss=nan`、
区分度 `nan`）而**不是崩溃** ⇒ 极易被读成「本层回归」。

**本机实测（2026-10-08）**：

| 检查 | 结果 |
|---|---|
| 期望 sha256（= HF blob 名） | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| `snapshots/97b0c614…/model.safetensors` 实测 sha256 | **逐字一致** |
| `blobs/0437e45c…` 实测 sha256（blob 名 = 其自身内容哈希） | **逐字一致**（snapshot 与 blob 是不同 inode 的两份拷贝 ⇒ **双向自证**） |
| 大小 / mtime / ctime | 1191586416 B / 2026-09-10 09:45:49 / 同 |
| 逐张量非有限值扫描 | 310 个张量 · 595,776,512 参数 · **含非有限值的张量数 = 0** |

⇒ **`MODEL_INTEGRITY_PASS`**：本机模型资产完好，两条腿的数值判据可信。
取证原始日志：`probes/MODEL_INTEGRITY_EVIDENCE_20261008.log`；脚本：`probes/mlu_model_integrity.py`。

> ⚠️ 本步骤**不是多余的**：它是 910C 那次教训的直接应用。共享机上的模型资产**会**在无人写入的
> 情况下损坏，而损坏的表现完全像"我们的代码坏了"。

---

## 2 台账 D1 收尾：三家唯一的「能设置」在**新接线下**通过真机复验

**D1 的原始内容**：MLU590 是三家唯一声明 `stream_priority_control` + `stream_priority_readback` 的实例，
但**该声明的依据是 2026-09-22 的记录**，而本层的**新接线**（三段校验 + 创建后强制回读）
**从未在本实例跑过**。

**本轮实测**（`probes/probe_stream_priority_api.py --backend cambricon --dev 0`）：

```
=== 流优先级统一 API 真机探针 · backend=cambricon ===
devices=8 | torch=2.7.1+cpu | device=mlu:0
capabilities: stream_priority=True control=True readback=True
  [PASS] D1 默认路径不回归           流上计算=16.0（期望 16.0）
  [PASS] D2 range 形状为 2 元组或 None   得到 (0, -3)
  [PASS] D2b 对默认流回读返回 int       回读=0
  [PASS] D3 设置 priority=-3 成功且回读一致   请求=-3 回读=-3
  [PASS] D3 设置 priority=0  成功且回读一致   请求=0  回读=0
  [PASS] D3b 越界 priority 抛 ValueError   -4: ValueError ✓；1: ValueError ✓
  [PASS] D5 厂商路径的参数保留性        直接 mlu 建流 priority=0 ⇒ 回读=0（保留）
=== STREAM_PRIORITY_API_PASS（7/7）===
```

**职责审计里的对应两条**：

```
  [OK] K6 声明 control ⇒ 区间端点请求必须回读 == 请求      端点 -3/0 请求与回读一致
  [OK] K7 stream_priority_readback() 返回值域严格 int 或 None   回读 = 0（类型合规）
```

⇒ **D1 收尾**。附带说明 **K6 也是首次被真正行使**（910C 因不声明 `control`、P800 因区间退化单点
⇒ 它俩都只能如实 SKIP；`K8` 在本机的 SKIP 理由即写明「由 K6 覆盖」）。

> ⚠️ 口径边界（不得外推）：本结论只在本栈成立（该实例 + 驱动 6.2.29 + `neuware4.4.3` 档 +
> `torch 2.7.1` / `torch_mlu 1.29.2`）。**「能设置」是接口能力，不是调度效果** ——
> 效果对照（台账 D2）仍未做。

---

## 3 ⭐ 本轮暴露并修复的 5 处缺陷

> 全部由**本次真机**发现；都不是"为了让流程变绿"而改的，而是**判据/默认值本身在说谎**。

### 3.1 职责审计的 J 域把委派方的「不适用」压成了 `OK`（口径缺陷 · 最要紧）

**现象**：同一份审计里**同一原因两种标签** ——

```
  [SKIP] H2/H3/H4/H5  未声明 memory_alloc ⇒ 该分支不适用
  [SKIP] I2/I3/I4/I5/I10  未声明 context_lifecycle ⇒ 该分支不适用
  [OK  ] J3           不适用：本后端未声明 `memory_alloc` 与 `context_lifecycle`
```

**根因**：J 域把 `contract_invariants.check_i1..i4` **委托**过来（同一份实现，避免两处漂移）。
被委托方在所需能力未声明时，按其模块 docstring 的既定约定返回
**`(True, "不适用：…")`**（"判为通过但显式注明"）。而 J 域的委托层把 `ok=True` 一律映成 `OK`
⇒ 一个**从未运行**的分支被记成「已响应」。

**为什么此前 910C / P800 没暴露**：910C 两项能力都声明、P800 声明了 `memory_alloc`
⇒ 它的 J3 永远走**真实分支**。**MLU590 是首个两项都不声明的实例**。

**危害（两层）**：
① 读者会以为「61/0/17 = 全部职责已响应」，实际 J3 的分支从未运行 —— 正是台账 E1 要消除的那类误导，
   E1 自己又造了一个同族；
② 非空转脚手架**无法**把「判据失效」与「本机不适用」分开 ⇒ 误报「未抓到 1」（假警报）。

**修法**（`只增不改`）：
- `runtime/conformance/contract_invariants.py`：把 `"不适用："` 提成**跨模块哨兵常量** `NOT_APPLICABLE`
  （两侧共用，避免文案漂移导致审计侧静默退化成「通过」），并在 docstring 里写明**消费方必须据此转 SKIP**；
- `scripts/duty_response_audit.py` 的 `_invariant()`：改成**三值翻译**
  （`True/False/None` ↔ `OK/FAIL/SKIP`），并显式识别哨兵。

**修后**：`J3` → `[SKIP]`；审计 `OK 61 / FAIL 0 / SKIP 17`；非空转 `未抓到 0`。

### 3.2 非空转脚手架新增「§0 委派翻译层自检」（修 3.1 的**非空转验证**）

修 3.1 之后，新分支（`ok is None` 与哨兵）在**声明了两项能力的实例上永不触发** ⇒ 单靠真机跑不动它。
故在 `probes/selfcheck_duty_audit_ext.py` 加一节**无需设备**的映射自检，用**受控替身**直接喂四种返回形状：

```
  [OK  ] 真实通过     委派=True / 判据真正执行且通过       → 期望 OK    实得 OK
  [OK  ] 如实跳过     委派=None / 委派方按三值约定跳过      → 期望 SKIP  实得 SKIP
  [OK  ] 如实不适用   委派=True / 不适用：未声明 …         → 期望 SKIP  实得 SKIP
  [OK  ] 真实失败     委派=False / 二次释放静默（I3 违反）  → 期望 FAIL  实得 FAIL
```

⭐ 纪律落地：**「能识别某种输入」必须真的喂过那种输入** —— 否则它只是一个永不触发的 `if`。
（`ok is None` 这条目前**没有任何被委托方会返回**，只有这一节能证明它被识别。）

### 3.3 `demo_unified.py` 与 `proto_infer_serve.py` 的 `--backend` 写死 `"ascend"` 且不读 `DC_BACKEND`

**现象**：MLU590 上跑演示 ⇒ `runtime.use("ascend")` → `info()` → `import torch_npu`
⇒ **`ModuleNotFoundError: No module named 'torch_npu'`**。
这个报错**看着像"环境缺包"，实际是"选错后端"**。

**为什么此前没暴露**：`demo_unified.py` 只在 910C 上跑过；`proto_infer_serve.py` 在 MLU590 上从未跑过。

**修法**：两处均改为 `default=os.environ.get("DC_BACKEND", "ascend")`，与
`runtime/proto/` 其余脚本及**全部探针**的既有约定一致。
另给 `demo_unified.py` 的 `info()` 加了兜底：把"厂商栈缺失"变成**可操作提示**，不再是一串误导性 traceback。

### 3.4 ⭐ `serve_standard.sh` 的 cambricon 分支默认 `MODEL` 是**宿主路径**，容器内永不可用

```
原值：MODEL=${MODEL:-/srv/hliu553/models/Qwen3-Embedding-0.6B}
```
- 容器内 `/srv/hliu553` 被挂成 **`/work`** ⇒ 该路径**在容器内永不存在**；
- 宿主上也**没有**这个目录（宿主真实位置 = `/srv/data/hf_cache/hub/…`）。

**失败形态极具误导性**：vLLM 抛
`OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/srv/hliu553/models/…'`
—— 看着像"模型 id 格式不对"，实为"这个路径不存在"。

**为什么此前没暴露**：历史复现命令**总是显式传 `MODEL`** ⇒ 默认值走不到。
（该分支注释里恰好写着"首次在 MLU 容器内跑时请把实际报错回填" —— 2026-10-08 首次触发，已回填。）

**修法**：默认值改为**容器内可见**的 HF 快照（`ls -d …/snapshots/*/`）。
本轮 [A] 段**刻意不传 `MODEL`** 以验证该默认值：`SERVE_STANDARD_PASS`，
且 `vllm_serve.log` 的 `non-default args` 里 `model` = 容器内快照路径 ⇒ **新默认值真的生效**。

### 3.5 `serve_standard.sh` 三个分支的 `MODEL` 都没有存在性校验（附）

任何"默认值失效"都会退化成一串厂商侧 traceback（见 3.4）。已在 `esac` 之后补一个**共用前置断言**：
路径为空或缺 `config.json` ⇒ 一条可操作的错误（并列出三家的宿主/容器映射口径）后 `exit 4`。
它同时覆盖了 kunlun 分支的一个潜在同类问题：其默认值给的是 HF **缓存根**而非 `snapshots/<hash>/`
（本方向手册明确「给缓存根会报 `Unrecognized model`」）。

---

## 4 取数结论与一处预测更正

### 4.1 B/C 契约探针：5 个能力键**全部如实不声明**（取数决定声明）

`probes/probe_bc_contract.py --backend cambricon` ⇒ `PASS`。逐组结果：

| 组 | 结果 |
|---|---|
| `m1` | `未声明 memory_alloc（如实不具备）` |
| `m2` | `record_stream_capability: false`，**走了保守同步**（退化计数 +1） |
| `c1` / `c2` / `c3` | `未声明 context_lifecycle（如实不具备）` |
| `c4` | `未声明 context_query（如实不具备）` |

⚠️ **口径纪律（两者不可混写）**：
- 本层对这 5 个键**如实不声明**，相关分支走**保守/退化**路径 ⇒ 这是**已实测**的；
- 但**不等于**"寒武纪硬件不具备" —— 本层**未**做裸厂商原语探测
  ⇒ 该问题的准确表述是「**本层未接线**」，而不是「**已确认不具备**」。
  （对照：`recovery_real` / `error_map` 两项才是「**已实测确认不具备**」，因为
   `torch.mlu` 下 `reset*`/`destroy*`/`reinit*` 全是内存统计类、厂商错误码不透出为数字码。）

### 4.2 ⚠️ 更正工作包 §0.1 的一处预测：**L5 在三家现役实例上都行使不了**

**原预测**（`MLU590_FIX_WORKPACK_20261008.md` §0.1，以及脚手架 docstring 里同一句话）：
> 「本实例是三家唯一会真正行使 **K6 / L5** 的……**K6**（端点请求必须回读一致）与
> **L5**（本层拥有的流释放后再用必须 `RuntimeError`）**首次被真正验证**」

**实测结果**：**K6 成立**（见 §2）；**L5 不成立** —— 它如实 SKIP：

```
  [SKIP] L5 使用已释放对象必须明确：本层拥有的流 release 后再用 ⇒ RuntimeError
         本机该路径未产生「本层拥有」的流（release=False）⇒ 使用已释放对象的场景不成立
```

**根因**：L5 的**充分条件**不只是"区间非单点"，还要求该后端真的存在
「**厂商 C API 建流 + 本层包装**」的路径。MLU590 的建流走 **torch 侧**
（`torch.mlu.Stream(priority=…)`，本家**没有**独立的句柄式 C API，与 ascend 的
`aclrtStreamGetPriority` / kunlun 的 `cuStreamGetPriority` 不同）
⇒ 该流由 torch 拥有 ⇒ `release_stream` 必须 no-op 返回 `False` ⇒ **不是「本层拥有」**。

⇒ **「区间非单点」只是 L5 的必要条件，不是充分条件。**
**三家现役实例目前都行使不了 L5**：

| 实例 | 区间 | 是否声明 `control` | L5 的可能路径 | 实际 |
|---|---|---|---|---|
| 910C | [0,7] | ❌ 不声明 | 无 | SKIP（未声明 control） |
| P800 | `(0,0)` 单点 | ✅ | C API 建流 + 包装 | SKIP（真机走**等价放行**，C API 路径不可达） |
| MLU590 | `(0,-3)` 非单点 | ✅ | torch 侧建流 ⇒ 非本层拥有 | SKIP（release=False） |

**后果（如实登记为台账 E2）**：契约 §1.10 的「**使用已释放对象必须明确报错**」这条，
目前**没有任何现役实例提供真实覆盖**（离线 stub 只覆盖「单点等价放行」分支）。
要收尾需要：① 在某个实例上打通「厂商 C API 建流 + 本层包装」并使其区间非单点；
或 ② 如实承认该条款**暂时只有离线覆盖**，并在契约里标注覆盖状态。

---

## 5 我方脚本缺陷（已修，**未为此放宽任何判据**）

| # | 缺陷 | 现象 | 处置 |
|---|---|---|---|
| 1 | 服务化编排在**宿主**上用容器路径取 `MODEL`（`ls /hf_cache/…`） | 宿主没有 `/hf_cache`（宿主是 `/srv/data/hf_cache`）⇒ 变量为空 ⇒ 退到脚本默认值 ⇒ 整轮失败 | 改为 `docker exec` 内解析 + **非空断言**；`MODEL` 概念上明确为"给容器程序的路径" |
| 2 | 模型完整性取证日志把**宿主路径**传给容器内解释器 | `FileNotFoundError`（容器里没有 `/srv/data/hf_cache`） | 日志内做显式**宿主→容器映射**并保留首版错误原文（不删证据） |
| 3 | 服务化脚本 [C] 段的「已清」是**无条件打印**的 | 首跑 `ps` 里明明还有一条 vllm（当时是 `<defunct>` 僵尸、端口已关）⇒ 那句话在说谎 | 改为**按结果判定** + 轮询等待 + 端口复验 |

> ⚠️ 第 1、2 条是**同一类**（宿主/容器路径混用），**同一天犯了两次**（另有第 3 次出现在
> 服务化脚本的首版）。已按运行手册 §3.5 的既有条目加固（见 §7）。

---

## 6 环境与前提（可复现）

| 项 | 值 |
|---|---|
| 机器 | `Mlu-1` = `tza-0a06-ai01-em9`（容器内 `/work` = 宿主 `/srv/hliu553`） |
| 容器 | `dc-mlu590-hliu553`（**运行时镜像**，本就 Up；两条腿/探针用它）· `dc-mlu590-vllm-hliu553`（**vLLM 应用镜像**，服务化用，原状 Exited ⇒ 收工已恢复 Exited） |
| 镜像 | `flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` · `flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2` |
| 档位 | 驱动 **6.2.29** ⇒ 只能 `neuware4.4.3`（py3.10.20 / torch 2.7.1 / torch_mlu 1.29.2 / vllm 0.20.2） |
| 解释器 | 容器内 `/flagos/bin/python3`（**注意：宿主上没有这个路径**） |
| 模型 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/` |
| 映射 | 宿主 `/srv/hliu553` → 容器 `/work`；宿主 `/srv/data/hf_cache` → 容器 `/hf_cache` |
| 集合通信后端 | **`cncl`**（必须显式给；`proto_train_leg.py` 对本后端**刻意不给默认值**且不给就 `exit 2`，防 `gloo` 静默退化为纯 CPU） |
| 用卡 | 当次（12:03）逐卡普查：卡 0/1 有他人进程 ⇒ 训练腿用 **2,3**、推理腿与服务化用 **2** |
| 收工核查 | 我的 vLLM 容器已停回 Exited · 无残留进程 · 端口 8100 已释放 · 他人 25 个运行容器未改动 · 8 卡回基线 |

**复现命令**（三条腿各自的入口）：

```bash
# 容器内（无多卡需求的部分 + 取数探针）
bash /work/dc_mlu_regen_20261008/probes/run_mlu590_m1_20261008.sh

# 容器内（两条腿）
bash /work/dc_mlu_regen_20261008/probes/run_mlu590_m1_legs_20261008.sh

# 宿主上（服务化；要先起 vLLM 应用镜像容器）
bash /srv/hliu553/dc_mlu_regen_20261008/probes/run_mlu590_m1_serve_20261008.sh dc-mlu590-vllm-hliu553
```

---

## 7 证据清单

| 内容 | 路径 |
|---|---|
| 第 1–2 组驱动日志 | `probes/m1_mlu590_20261008.log` |
| 第 1–2 组逐项日志/JSON（29 份） | `probes/m1_20261008_out/` |
| 两条腿驱动日志 + 结果 JSON | `probes/m1_legs_20261008.log` · `m1_20261008_out/train_mlu/train_leg_result_rank{0,1}.json` · `m1_20261008_out/proto_infer_leg_result.json` |
| 服务化驱动日志 + 四份原始日志/JSON | `probes/m1_serve_20261008.log` · `probes/m1_serve_20261008_out/{form,consumer}/` |
| **模型完整性取证原始日志**（5 段：机器/容器、文件与哈希、权重扫描、910C 对照、结论） | `probes/MODEL_INTEGRITY_EVIDENCE_20261008.log` |
| 本轮脚本（可复跑） | `probes/run_mlu590_m1_20261008.sh` · `run_mlu590_m1_legs_20261008.sh` · `run_mlu590_m1_serve_20261008.sh` · `mlu_model_integrity.py` |

---

## 8 边界与未做项

1. **档位条件**：全部结论在 `neuware4.4.3` 档取得（驱动 6.2.29）。引用须连同档位一起引。
2. **不作性能结论**：共享机 + 单次取数；吞吐与就绪耗时只作同档可比（本次就绪 360 s，历史 150 s，
   差异归因于机器负载，未做对照实验）。
3. **「能设置」≠「有效果」**：接口能力与调度效果分开写。**D2（priority 调度效果对照）仍未做**。
4. **未做的能力探测**：5 个能力键只确定了「**本层未接线**」，**未**做裸厂商原语探测
   ⇒ 不得声称"寒武纪硬件不具备"（见 §4.1）。
5. **首次全口径 ≠ 免跑**：本轮之后，任何改动共享层的提交都需按破坏面重跑本实例。
6. **910C 补跑未完成**（见 §9）。

---

## 9 ⚠️ 遗留：910C / P800 的同版本复核

本轮修改了**共享层**（`contract_invariants.py` 的哨兵常量、`duty_response_audit.py` 的 J 域三值翻译、
`serve_standard.sh` 的默认值与前置断言、两个 proto/演示脚本的 `--backend` 默认值）。

| 实例 | 复核状态 |
|---|---|
| **P800** | ✅ **已完成**（2026-10-08 13:xx）：职责审计 **67/0/11 PASS** · 非空转 **31/8/0 PASS** ⇒ 与 E1 轮**逐项一致**，无回归 |
| **910C** | ⏸ **未完成** —— 该机自 **2026-10-08 12:04:45** 起 TCP 22 持续超时（13:08:39 第 5 次重试仍不可达）⇒ **本实例结论尚未在本轮代码版本上复核** |

⇒ **待办**：910C 恢复后补跑「职责审计 + 非空转」两项（单脚本、几分钟、**不需要两条腿**）。
预期结果与 E1 轮一致（910C 两项能力都声明 ⇒ J3 仍走真实分支，`73/0/5`）。
在补跑完成前，**不得**声称 910C 已在本轮代码版本上通过。
