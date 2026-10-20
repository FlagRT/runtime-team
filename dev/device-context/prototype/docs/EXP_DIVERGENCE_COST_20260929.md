# 「分歧的业务代价」实验报告（工作包 A）

> 状态：🟡 起草（未提交）｜作者：Kistich（hliu553）｜2026-09-29
> 脚本：`probes/exp_divergence_cost.py`（度量口径写在脚本头，**先定后测**）
> 证据：`P800/probes/exp_divergence_cost_kunlun_20260929.json`
> 出口：回答一个问题 —— **「不做统一抽象」到底有多少实际代价？本层是否值得存在？**

---

## 1 结论先行

### 1.1 成本：上层实现成本被压缩 2.1 倍（这一条支持本层）

同一段上层需求（设备枚举 + 显存统计 + 有界同步 + 错误分级 + 恢复判定），两条路径各实现一次：

| 指标 | 路径 ① 统一层 | 路径 ② 直调原生 | 差异 |
|---|---|---|---|
| **M1** 厂商分支数 | **0** | **11** | 每处分支都对应一条真实厂商差异 |
| **M2** 厂商私有知识点 | **0** | **15** | 不查厂商资料就会写错的具体事实 |
| **M3** 上层代码行数 | **77** | **163** | **2.1×** |
| **M4** 直接引用的厂商 API 种类 | **0** | **17** | 完全机械计数，用于交叉核对 M1 |

M1/M2 逐条明细见 §3.2 / §3.3（自标记指标，须供审阅）。M3/M4 由脚本读自身源码机械计算。
**M1–M4 是代码属性、与机器无关** ⇒ 单次测量即覆盖三家；本机与 P800 容器内两次测量结果一致
（`0/0/77/0` 与 `11/15/163/17`），构成交叉核对。

### 1.2 等价性：6 个场景里 5 个两路径等价，1 个不一致 —— **而那一处是本层缺陷**

P800（`kunlun`）真机，逐场景独立进程：

| 场景 | 触发 | 契约期望 | 路径 ① | 路径 ② | 等价 |
|---|---|---|---|---|---|
| S0 设备枚举与显存统计 | real | 值一致 | 值一致（见 §4） | 值一致 | ✅ |
| S1 设备序号越界 | real | L2_PARAM / raise | L2_PARAM / raise | L2_PARAM / raise | ✅ |
| S2 有界同步超时 | best-effort | L3_EXECUTION / replay | L3_EXECUTION / replay | L3_EXECUTION / replay | ✅ |
| S3 显存不足 | real | L1_RESOURCE / retry | L1_RESOURCE / retry | L1_RESOURCE / retry | ✅ |
| S5 未知异常 | real | L3_EXECUTION / replay | L3_EXECUTION / replay | L3_EXECUTION / replay | ✅ |
| **S6 芯片级致命** | synthetic | L4_FATAL / device_recovery | **L4_FATAL / device_recovery** | **L3_EXECUTION / replay** | ❌ **DIFF** |

### 1.3 ★一条对我们自己的主张不利、但必须记下的观察

本层文档反复主张的价值是「**让上层不因换芯片而改变行为**」。本轮实测给出的是：

> **上层实现成本确实被大幅压缩（2.1 倍、0→11 分支、0→15 知识点）；
> 但"避免行为分歧"这半边，6 个场景里只有 1 处观测到差异 —— 而且那一处是本层做错了。**

即：**本层目前可被证据支撑的价值是「把 N 家适配成本从上层收走 + 保留可移植性」，
不是「让厂商行为一致」。** 后者需要判据与证据体系（就是那 4876 行判据机器）来兜底，
而这轮恰好证明**判据体系还漏了一处**（见 §5.1）。

**这条修正了我们对外的表述口径**：应主张「可移植性 ≠ 一致性；层的目标是让分歧**显式且可消费**」
—— 而不是「统一层让换芯片行为一致」。前者与实测相符，后者与实测不符。

---

## 2 方法

### 2.1 两条路径的界定

- **路径 ①**：只用 `runtime` 统一 API。设备串由接口给出的 `device_type` 拼（`base.py:35` 已把它定为
  接口的一部分），**区间内不得出现任何厂商命名空间 API**（M4 实测 = 0，即该约束成立）。
- **路径 ②**：不假设统一层，上层自己把每家接一遍，**允许按厂商分支**（这才是"不做抽象"的真实形态）。
  错误分级用**自带的最小码表**（4 条，刻意不完整），这是朴素实现的现实做法。

### 2.2 场景与触发方式（不伪造）

| 标记 | 含义 |
|---|---|
| `real` | 真机真实触发 |
| `best-effort` | 真实触发但依赖竞态（S2 靠极短超时）；不能触发时如实标 `ok=True`，不冒充 |
| `synthetic` | **真机不触发硬件致命错误**，用合成码消息验证分级路径 —— 只验"分级路径"，不代表"真实故障"结论 |

### 2.3 一条执行纪律（本轮实测教训，已写进脚本头）

**每个场景必须独立进程跑。** 第一版同进程连跑：`S1` 调 `bk.set_device(设备数+100)` 越界后，
**同进程后续所有设备操作都报同一个 `invalid device ordinal`** ⇒ S2/S3 被污染成同一结果，
表面"两路径一致"实为**两边同错**。修正为逐场景子进程隔离后，S2/S3 才真实触发。
（这是「一次失败的捕获会污染后续条目」的又一例证，与既有台账同源。）

---

## 3 成本度量（M1–M4）

### 3.1 汇总

| 记号 | 路径 ① | 路径 ② |
|---|---|---|
| M1 厂商分支数 | 0 | 11 |
| M2 厂商私有知识点 | 0 | 15 |
| M3 上层代码行数 | 77 | 163 |
| M4 厂商 API 种类 | 0 | 17 |

M4 明细（路径 ②，前 8 项）：`acl.rt.synchronize_stream_with_timeout` · `torch.cuda.Stream` ·
`torch.cuda.device_count` · `torch.cuda.mem_get_info` · `torch.cuda.memory_stats` ·
`torch.cuda.set_device` · `torch.cuda.stream` · `torch.mlu.Stream` …

### 3.2 M1 厂商分支逐条（11 处，供审阅）

1. 设备命名空间三家不同：`npu` / `cuda` / `mlu`
2. 枚举与绑定 API 挂在各自命名空间下，须分别调用
3. 取显存的可用 API 不同；**昆仑芯 `memory_stats()` 实测返回空 dict**，必须换路径
4. **只有昇腾把错误以数字码透出**；另两家没有数字码，提取策略必须分开写
5. OOM 文案三家不同（`NPU` / `CUDA` / `MLU` out of memory）
6. 昆仑芯/寒武纪无码表 ⇒ 未命中关键词时只能兜底，**且无法区分 L3 与 L4**
7. **只有昇腾有芯片级致命码**（507015 类）；另两家连数字码都没有 ⇒ L4 判定能力不对等
8. 同步原语不同；**只有昇腾有真超时中断原语**，另两家只能轮询上报
9. **只有昇腾有设备级重建原语**（`aclrtResetDevice` 序列）；另两家只能探活
10. 绑定设备的入口三家不同（须与枚举分支保持一致，否则两处会漂移）
11. 探活的设备串前缀三家不同

### 3.3 M2 厂商私有知识点逐条（15 处，供审阅）

其中**与"设备原语"无关、纯属厂商实现细节**的占多数，说明这些成本不是"上层业务复杂度"，
而是**厂商差异的转嫁**：

1. 设备命名空间前缀是 `npu`/`cuda`/`mlu`（不是加速器名，不能从设备型号推）
2. `torch_npu` 需先 `import` 才把 `npu` 注册到 torch（懒加载）
3. 昆仑芯走 `torch.cuda` 兼容层（XPytorch）+ torch_xray 符号重写
4. torch_mlu 同样需先 import 才注册 `mlu`
5. 昆仑芯 `torch.cuda.memory_stats()` 返回空 dict ⇒ 必须换 `mem_get_info()`+`memory_allocated()`
6. 部分 `torch_mlu` 实现的 `mem_get_info` 不接受 `ordinal` 参数
7. OOM 文案三家不同；昇腾同一现象**既有数字码 207001 又有文案**，两条都要兜
8. 参数类关键词各家不同：`invalid` / `illegal` / `out of range` / `ordinal` / `unaligned`
9. 超时类：昇腾给码 `107019/507046`，另两家只能靠 `timeout` 文案
10. pyACL 有界同步入口是 `rt.synchronize_stream_with_timeout(handle, ms)`，需底层流句柄
11. 昆仑芯 `Stream.synchronize()` 无 timeout 参数、无中断原语 ⇒ 只能「超时上报」
12. 寒武纪是否提供真中断原语**未验证** ⇒ 与昆仑芯同口径先做「超时上报」
13. 昇腾设备级重建须走 `destroyEvent→destroyStream→destroyContext→Reset→setDevice` 序列
14. 昆仑芯 `torch.cuda` 上 `reset*` 全是内存统计类，**无 reset_device / context 重建**
15. 寒武纪是否有设备级重置原语**未验证** ⇒ 不写未经验证的重建序列

> **口径说明**：路径 ② 若把"为昇腾建一份完整码表"也算进来，成本还要更高 —— 但那属**任何路线下
> 都必须付出**的厂商适配成本，不属本层价值，故未计入差异。此处只计"上层调用点"的成本差异。

---

## 4 等价性真机数据（P800 / `kunlun`）

- 环境：容器 `hliu553-device-context-p800`，`CUDA_VISIBLE_DEVICES=6`（可见设备数 = 1），
  解释器 conda `python310_torch29_cuda`，2026-09-29。
- **S0 值一致性**：两路径 `device_count=1` · `total_mb=98304` · `used_mb≈4374` · `free_mb≈93930`
  完全一致 ⇒ 路径 ① 的 `memory_stats` 归一化（含"先 set_device 再恢复"的副作用处理）与直调原生等价。
- S1/S2/S3/S5 四条：两路径 `(category, disposition, action)` 三元组一致，**且均符合契约期望**。
  **触发证据（消息原文，证明不是"两边同错"）**：
  - S1 → `CUDA error: invalid device ordinal` ⇒ L2_PARAM / raise
  - S2 → `kunlun: stream synchronize timeout after 1 ms（超时上报语义，不保证中断底层执行）`
    ⇒ L3_EXECUTION / replay
  - S3 → `CUDA out of memory. Tried to allocate 192.00 GiB. GPU 0 has a total capacity of
    96.00 GiB of which 91.73 GiB is free` ⇒ L1_RESOURCE / retry（按 2× 显存规模构造，确为真 OOM）
  - S5 → `expansion saw a wobble`（无任何厂商信息）⇒ L3_EXECUTION / replay，`graded_by=default`
- S6：见 §5.1。
- S6b（恢复调用）：两路径均五键齐全（`{ordinal, mode, recovered, state, detail}`），
  但 `state` 取值形式不同 —— 见 §5.2。

---

## 5 两处发现（均为本层缺陷，附根因与修法）

### 5.1 ★发现 1：`category` / `disposition` 没有随"诚实降级"一起降 ⇒ 非昇腾后端会误判 L4

**现象（P800 实测）**：输入同一异常消息 `device reset failed, error code is 507015`（507015 = 昇腾
AICORE 异常，本层表中为 L4_FATAL）：

| | category | disposition | mapped | graded_by |
|---|---|---|---|---|
| 路径 ①（本层） | **L4_FATAL** | **device_recovery** | False | `message_hint_unexpected` |
| 路径 ②（朴素原生） | L3_EXECUTION | replay | False | `default` |

**根因（`runtime/conformance/errors.py::translate_error` + 两个后端包装）**：

```python
# conformance/errors.py —— 码表是**昇腾 ACL 码表**，但任何后端都会用它判 category
retcode = _extract_acl_retcode(msg)
if retcode is not None and retcode in ACL_ERR_TO_CATEGORY:
    category = ACL_ERR_TO_CATEGORY[retcode]      # ← 507015 ⇒ L4_FATAL
    mapped, graded_by = True, "code_map"

# kunlun / cambricon 的包装（两家同款）—— 只降"置信度字段"，**不降 category**
degraded = graded_by == "code_map"
if degraded:
    graded_by = "message_hint_unexpected"
return FlagosError(
    category=coerce_category(getattr(fe, "category", None)) ...,   # ← ★仍是 L4_FATAL
    error_code=None if degraded else ...,
    mapped=False if degraded else ...,
    graded_by=graded_by,
    is_grade_confident=(graded_by == "message_hint"),
)
```

**为什么这是缺陷，而不是"保守选择"**：

1. 契约 §1.4 明确「**下游必须按 `disposition` 处理，禁止按错误消息字符串自行判断**」
   ⇒ 下游会照 `device_recovery` 去**拆重建备上下文**（最昂贵的动作）。
2. `errors.py` 自己的兜底规则写着「两者皆无 → 兜底 L3_EXECUTION（**不可当定论**）」。
   既然已判定"这不是本厂商的码表命中"，正确行为就是**回落到兜底 L3**；
   实测却是**升级为 L4** ⇒ 违反自身规则。
3. 台账 §11-① 记的纪律是「**降级必须整组一致**（`graded_by`/`mapped`/`error_code` 一起降）」。
   本轮证明**那张清单漏了 `category`** —— 而它恰恰是下游唯一据以行动的字段。
   ⇒ 2026-09-22 的第 6 条修复是**必要但不充分**的。

**影响面**：任何**非昇腾后端**，只要异常消息里出现形如昇腾码的数字（`error code is N` / `ret=N`），
就会被判成昇腾分级与处置。叠加第 2 点，**未命中时不会被兜底纠正**。
同时它也说明：**未声明 `error_map` 的后端，其 L1–L4 能力在行为上并不"如实不具备"** ——
声明与行为不一致（`§11-② 声明即承诺` 的反向形态）。

**修法建议**：降级分支里 `category` 一并回落（按 `message_hint` 重算，无命中则 L3），
并给 `disposition` 加判据。**修完必须做非空转验证**（注入上述消息，断言非昇腾后端不得出 L4）。

### 5.2 发现 2：`recover_device()["state"]` 不是四态规范取值，且同层存在两种形态

**现象（P800 实测）**：

| | `state` 原始值 | 是否四态规范 token |
|---|---|---|
| 路径 ①（本层） | `'DeviceState.AVAILABLE'` | ❌ |
| 路径 ②（朴素原生） | `'available'` | ✅ |

**根因**：三家 `recover_device` 都写 `"state": str(state)`；而
`conformance/device_state.py::DeviceState` 是 `enum.Enum`（**不是** `str, Enum` / `StrEnum`），
故 `str(member)` = `'DeviceState.AVAILABLE'`，`.value` 才是 `'available'`。
同一个文件里的 `DeviceStatus.snapshot()` 用的是 `self.state.value` ⇒ **同仓两套约定**。
另：`device_state()` 接口返回的是 **enum 对象**，而 `recover_device()["state"]` 返回 **str** ⇒
同一个"设备状态"概念在本层有**两个对外形态，且都不是规范 token**。

**为什么 09-28 的职责响应审计没抓到**：那次审计（E3）只判「五键齐全」= **字段名存在性**。
本轮是**取值域**问题 ⇒ 正是 §11-⑫（「契约里的字符串常量也要有判据守」）**再深一层**：
不只字段名要有判据，**字段的取值域也要有判据**。

**修法建议**：`recover_device()["state"]` 统一返回 `.value`；契约 §1.5 写明取值域
（`available/degraded/isolated/destroyed`）；审计加"取值域判据"并做非空转验证。

---

### 5.3 修复与验证（2026-09-29 当日完成）

**修法（治本，不是打补丁）**

| 缺陷 | 修法 | 为什么这样修 |
|---|---|---|
| 1 码表泄漏（`category` 未随降级回落） | `errors.translate_error()` 增加**关键字参数** `vendor_codes`（默认 `True` ⇒ 对既有调用**完全兼容**）；kunlun / cambricon 传 `vendor_codes=False`，**从源头不查外来码表**；同时删掉原先「事后逐字段降级」的写法 | 原写法只能逐个字段记得改，而**恰恰漏掉了下游唯一据以行动的 `category`**。改为源头不查 ⇒ `graded_by` / `mapped` / `error_code` / **`category`** 四者**结构上不可能不一致** |
| 2 `state` 取值域 | 新增 `backends/base.py::state_token()` + `DEVICE_STATE_TOKENS`；三家 `recover_device` 统一经它归一为 `.value` | `DeviceState` 是 `enum.Enum`（非 StrEnum），`str(member)` 给的是 `'DeviceState.AVAILABLE'`；**同模块 `DeviceStatus.snapshot()` 本就用 `.value`** ⇒ 归一为同仓唯一约定 |

**新增判据（补上原先漏掉的两个维度）**

| 判据 | 守什么 | 为什么原来漏过 |
|---|---|---|
| 离线自检：**「外来码表不得影响本后端分类」** —— 用一条**不命中任何消息关键词**的消息（`op failed, error code is 507015`）⇒ `category` 必须兜底 `L3_EXECUTION` | **决策字段** | 原有三条判据只查 `graded_by` / `mapped` / `error_code`，**全是置信度字段**，不含 `category` |
| 离线自检 + 职责审计 E3：**`recover_device()["state"]` 必须是四态规范 token** | **字段的取值域** | 原判据只判「五键存在性」= **字段名**，不判取值 |

**非空转验证（证明判据能真的失败）**：把两处缺陷注入回 kunlun ⇒ 两条判据均 **FAIL**
（`category=L4_FATAL graded_by=code_map` / `state='DeviceState.AVAILABLE'`，整轮 5 失败）；
还原后 **43 通过 / 0 失败**。

**P800 真机回归（修复后）**

| 项 | 结果 |
|---|---|
| 离线契约自检（判据数 +2） | **43 通过 / 0 失败 / 1 跳过**（两条新判据均 PASS） |
| 跨后端对称性 `--all` | **5 / 0** |
| conformance 13 + 推理 6 | **13/13 + 6/6** |
| 冒烟 | **46 / 0** |
| 错误注入→恢复闭环 | **5 / 0 / 0** —— 其中 `l4_by_code` 一条由修复前的 `L4_FATAL` 变为 **`L3_EXECUTION`**，期望核对 ✅ |
| **本实验重跑** | **等价性 6/6 一致**（S6 的 DIFF 闭合）；`state='available'`（规范 token） |

⇒ **两处缺陷均已修复并在 P800 上验证闭合；910C / MLU590 的 conformance 回归待主机可达后补**
（签名未变、`vendor_codes` 默认值保证 ascend 行为不变；但 kunlun / cambricon 的**分级行为确实变了**
（外来码消息 L4→L3），故仍按纪律把三实例回归列为**待补**，不以 P800 单点代替）。

---

## 6 本轮未取得的数据（如实登记，不补零）

| 实例 | 状态 | 说明 |
|---|---|---|
| P800 `kunlun` | ✅ 已取得 | 2026-09-29 上午 |
| 910C `ascend` | ❌ **未取得** | 2026-09-29 09:40–10:35 多次重试均 SSH 超时（同刻 P800 正常） |
| MLU590 `cambricon` | ❌ **未取得** | 同上 |

- **M1–M4 不受影响**：它们是代码属性、与机器无关，本机与 P800 两次测量一致即已覆盖三家。
- **S6 那条发现的可外推性**：根因在两家的**同款包装代码**（kunlun 与 cambricon 逐字相同）
  ⇒ 机理上必然同现，但**"必然同现"是代码级推断，不替代真机复现**；补跑后以真机为准。
- **一键补跑**（主机恢复后）：

```bash
# 910C
ASCEND_RT_VISIBLE_DEVICES=4 DC_BACKEND=ascend python3 probes/exp_divergence_cost.py \
  --backend ascend --ordinal 0 --out <DC>/910C/probes/exp_divergence_cost_ascend_2026xxxx.json
# MLU590
MLU_VISIBLE_DEVICES=0 DC_BACKEND=cambricon python3 probes/exp_divergence_cost.py \
  --backend cambricon --ordinal 0 --out <DC>/MLU590/probes/exp_divergence_cost_cambricon_2026xxxx.json
```

---

## 7 边界（不可外推）

1. `synthetic` 场景（S6）**不是**真实硬件致命错误，只验证了**分级路径**；
   "真机 L4 会怎样"本轮未测。
2. `best-effort`（S2）在昆仑芯上是**超时上报**语义，不保证中断底层执行 ⇒ 不能当作"真中断"结论。
3. M3 行数是**代码行数量级代理**，不是人天；两条路径由同一作者实现，已用 M4（机械计数）交叉核对，
   但仍不能完全排除自标记偏差 ⇒ §3.2/§3.3 已附逐条明细供审阅。
4. 结论只代表 **P800（`kunlun`）实例的档位/环境条件**（`CUDA_VISIBLE_DEVICES=6`、conda
   `python310_torch29_cuda`、2026-09-29）；另两实例未取得数据。
5. 本实验只度量「上层实现成本」与「两路径行为等价性」，**不构成性能结论**。

---

## 8 下一步

1. ~~修 §5 两处~~ ✅ **已于 2026-09-29 当日完成**（含判据 + 非空转验证 + P800 回归），详见 §5.3。
2. **补 910C / MLU590 真机数据**（主机恢复即跑）。
3. **按 §1.3 修正对外表述**：把「统一层让换芯片行为一致」改为
   「统一层收走 N 家适配成本，并让分歧显式可消费」；不再声称前者。
4. 若 §1.3 的结论成立，则**本层的价值论证重心应从"实现"转向"判据与证据"** ——
   这与论文切入点（可移植性 ≠ 一致性；让分歧显式且可消费）一致，但与本方向此前的对外话术不一致。
