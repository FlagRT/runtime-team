# 工作包 D · 流优先级「效果」与配额「真实上限」（2026-09-30 · v2 含根因核查）

> 定位：**判据 + 实测记录 + 根因核查**。对应《薄弱环节补做计划》工作包 **D**（P2）、审计清单 **B2** 项。
> 设备：**910C / ascend**（Atlas 训练系列），单卡 `davinci0`，容器 `dc-quota-probe-0930`，
> 解释器 `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python`（torch 2.11.0+cu130 / torch_npu 2.11.0 / **CANN 9.0.0**）。
>
> ⚠️ **本文 v2 更正了 v1（2026-09-30 上午）的两处结论** —— 起因是"为什么会出现这个结果"的追问 +
> 官方文档核查。更正内容集中在 §5，**v1 的原始取数证据一字未删**，仅更正**解释与标注**。
>
> **边界**：单卡、单进程、本档位；结论只在本环境成立。

---

## 0 一句话结论（v2）

| 问题 | v1 结论 | **v2（更正后）** |
|---|---|---|
| 优先级**范围** | `(7, 0, 0)`，两端都能建流 | 同上；但**方向标反了**：官方语义为 `leastPriority=7 / greatestPriority=0` ⇒ **数值越小优先级越高（0 最高、7 最低）** |
| 优先级**效果** | 「未观测到效果」 | **结论仍成立**，但根因不是"提交顺序主导"这么简单 → **两处硬性障碍**（§5.1）：<br>① **`torch.npu.Stream(priority=X)` 在插件层就把参数丢了**（回读恒 0，C API 校验）；<br>② 官方文档：**Atlas 训练系列产品上 `priority` 是「预留参数、暂不使用」（固定 0）** |
| 配额**上限** | 「到 100 万档未触及」 | ❌ **v1 取数无效**：`torch.npu.Stream()` 构造**不消耗设备流配额**（实测 torch 建 1000 条，`available_num` 纹丝不动）⇒ v1 测的是 **Python 对象数**。<br>✅ **真实上限 = 1979 条**（`acl.rt.get_stream_available_num()` = 1979，pyACL 直连恰好建满 1979 条，第 1980 条失败 `rc=207005`） |

---

## 1 判据（先写死）

### 1.1 优先级（两版探针）
| 判据 | 内容 |
|---|---|
| `P1_range_readable` | 范围可解析；**按官方语义取 `hi=min(a,b)`**（v1 取 `hi=第二项`，方向错了，见 §5） |
| `P1_create_hi/lo` | 两端各建流、各算一次并校验（证明参数没被吞） |
| **`P3_gate_high_wins`（v2 新增）** | ⭐ **两条流先一起等同一个 gate 事件 ⇒ 同时处于「排队」态，再一起放行** —— 这才符合官方机制（优先级只管"排队中"的任务） |
| `P4_naive_control` | 旧测法（低优先级先提交、已进入运行态）作对照，按官方机制**预期 0** |

### 1.2 配额
| 判据 | 内容 |
|---|---|
| `Q1/Q2/Q3`（既有，但**路径不可靠**） | 「连续建 N 个流」——⚠️ 见 §5.2：**用 `torch.npu.Stream()` 时它不反映设备配额** |
| **T3_resource_cost / 设备自报上限（v2）** | 改用 **`acl.rt.get_stream_available_num()`**（设备自报"还能建几个"）与 **pyACL 直连建到失败**，拿**精确上限** |

---

## 2 优先级：三次取数（v1 的两次 + v2 的机制修正版）

| 轮次 | 做法 | 结果 |
|---|---|---|
| v1-① | 单次 2048² matmul/流 | 两流各 0.4–1.8 ms ⇒ **未形成争用** ⇒ 取数无效 → 加重负载 |
| v1-② | 60 次链式（≈25 ms）+ 双向对照 | 「lo_first 0/8、hi_first 8/8」⇒ 记为"提交顺序主导" |
| **v2-③** | **两流同时排队（gate）** + 修正方向 | **gate 0/8、naive 0/8** ⇒ **即使同时排队，高优先级也不先完成**（见 §5.1 的根因） |

**v2-③ 的判据输出**（`prio_queued_910c_npu_20260930.json`）：

```
[P1] range_raw=(7, 0, 0) ⇒ 按官方语义 hi(最高)=0 lo(最低)=7 supports=True
[P2] hi=89440.0 lo=89440.0（期望各 89440）           ← 两端都能建流且算得对
[P3] gate 模式（两流同时排队）：高优先级先完成 0/8      ← 仍未观测到效果
[P4] naive 对照（低优先级先提交）：高优先级先完成 0/8
STREAM_PRIORITY_QUEUED_PASS: 4/4（effect_observed=False）
```

---

## 3 配额：三条路径给出三个不同答案（这才是关键）

| 路径 | `available_num` 变化 | 说明 |
|---|---|---|
| **`acl.rt.create_stream()`（pyACL 直连）** | 1979 → **1:1 递减** → 0 | 真实占用设备流资源 |
| **`torch.npu.Stream()`（构造）** | 1147 → **1147（建 1000 条不变）** | **不消耗配额**（懒分配/未注册） |
| **`torch.npu.Stream()` 后再真的用它算** | 仍需复核 | 见 §7 未做项 |

**决定性实验（pyACL 建到失败）**：

```
起始 available_num = (1979, 0)
pyACL 建流：成功 1979 条，耗时 3.82s，首次失败=rc=207005（第 1980 条）
建完后 available_num = (0, 0)
```

**⇒ 本栈真实流配额上限 = 1979 条**。与官方文档**对得上**：
> 「Atlas 训练系列产品，Stream 最大数为 **2048**」；
> 「只能显式创建 N 个 Stream，**N = Stream最大数 − 默认Stream个数 − 执行内部同步的Stream个数**」
⇒ 2048 − 1979 = **69** 个默认/内部流（默认流 1 个 + 框架/hccl 等内部同步流）。

**错误码语义**：`rc=207005` 在头文件里是 **`ACL_ERROR_RT_RESOURCE_ALLOC_FAIL`**（通用资源分配失败）
—— **不是**专门的"流数量超限"码 ⇒ 靠"建到失败"只能拿到**通用错误**，
**`get_stream_available_num()` 才是可预判的入口**（呼应"能力缺失要给可预判的保守路径"）。

---

## 4 ⚠️ v1 的两条结论为什么会"看起来通过"（两处探针缺陷）

| # | 缺陷 | 后果 |
|---|---|---|
| **D-a** | 优先级**方向标反**：把 `range[1]=7` 当"高优先级"，而官方语义 `greatestPriority` 是**数值最小**那个（0） | v1 报的"高优先级 0/8"其实说的是**最低优先级**；结论方向没错，但**表述与推论基础错了**，且掩盖了"两流都已运行/排队"的真正区别 |
| **D-b** | 用 `torch.npu.Stream()` 测配额 | 它**不消耗设备配额** ⇒ "到 100 万档未触及"是**必然结果**，与设备无关；v1 把"探针测不到"写成了"设备没有上限" |

> 两处都属于本项目台账的老家族：**「看起来通过」**（探针 PASS 但没测到该测的东西）。
> ⭐ 教训：**测"上限/效果"之前，先证明"我测的量确实在那个层次上被消耗/被使用"**
> （本轮的做法：用 `available_num` 与 C API 回读**交叉验证**）。

---

## 5 ⭐ 根因核查（用户追问"为什么" → 官方文档 + C API 回读）

### 5.1 优先级为什么不生效 —— **两层障碍，都已取到硬证据**

**障碍一：`torch.npu.Stream(priority=…)` 在插件层丢参数（实测）**

用 C API **回读**流优先级（`aclrtStreamGetPriority`；pyACL 未暴露，用 ctypes 调 `libascendcl.so`）：

| 建流路径 | 传入 → **回读** |
|---|---|
| pyACL `create_stream_with_config(priority, flag)` | `0→0`、`3→3`、`7→7` ✅ **ACL 真的存下来了** |
| `torch.npu.Stream(priority=0)` | → **0** |
| `torch.npu.Stream(priority=7)` | → **0** ⛔ **参数被丢弃** |

⇒ **ACL 层有这个能力，但 `torch_npu` 的 eager `Stream` 没把 priority 传下去**（`_C.so` 里也扫不到含
`priority` 的导出符号）。**我们（及所有走 `torch.npu.Stream(priority=…)` 的代码）传的值根本没到设备。**

**障碍二：官方文档说训练产品上该参数是"预留"**

CANN `create_stream_with_config` 的 `priority` 参数说明（官方文档原文要点）：

| 产品 | 该参数状态 |
|---|---|
| Atlas 200/300/500 推理 · **Atlas 训练系列** · Atlas A2 训练 · Atlas 200I/500 A2 | **「当前固定设置为 0，预留参数，暂不使用」** |
| **Atlas 推理系列产品** | **取值范围 [0, 7]，最多 8 级，数字越小优先级越高（0 最高、7 最低）**；超范围返回报错 |

⇒ **910C 属 Atlas 训练系列 ⇒ 即使参数传到了 ACL，文档口径也是"暂不使用"。**

⚠️ **上游口径冲突（可作上报材料）**：本机 CANN 9.0.0 的**头文件** `acl_rt.h` 对同一接口写的是
`@param priority [IN] the priority of stream, **value range: 0~7**`（**未提"预留"**），
而官方文档说训练产品上预留 ⇒ **同一接口的文档与头文件不一致**。

**另有官方明文（图模式 API）**：`npu_stream_switch(stream_tag, stream_priority=0)` 的说明是
「**当前版本为预留参数，建议取默认值 0**」。

### 5.2 P800（昆仑芯）为什么是"连范围都读不到" —— 与 910C 不同的根因

同法回读（`cuCtxGetStreamPriorityRange` / `cuStreamCreateWithPriority` / `cuStreamGetPriority`，经 `libxpucuda.so.515.58.kunlun`）：

| 项 | P800 实测 |
|---|---|
| 三个入口存在性 | **都存在**（不是"没实现"） |
| **范围查询** | **`least=0, greatest=0`** ⇒ **优先级空间退化为单点** |
| `cuStreamCreateWithPriority(flags=0, prio=-1)` | 回读 **0**（**不保留**传入值） |
| `torch.cuda.Stream(priority=-1)` | 回读 **0** |

⇒ **P800 是"设备侧没有任何可调的优先级空间"**（范围单点 + 兼容层不保留），因此**无效果可测**；
而 910C 是"范围有（0~7），但**插件层丢参数** + 文档说训练产品预留"。

⇒ 我们的 `kunlun` 后端 `stream_priority_range()` 返回 `None` **是如实**的，但**表述可以更准**：
设备其实报的是"**退化到单点 (0,0)**"，比 `None`（易被读成"接口不存在"）更接近事实。
**这一点列为改进项（§7），本轮未擅自改契约。**

### 5.3 机制层面的官方说明（解释"为什么两流测试很难测出差异"）

CANN 文档原文（Stream 管理）：
> 「高优先级 Stream 中待执行的任务将优先于低优先级 Stream 中的任务得到调度，**但不会抢占已处于运行状态的低优先级任务**」
> 「Device 在执行过程中**不会动态重新评估任务队列**」
> 「Stream 的优先级主要用于影响任务的**调度顺序**，而非强制规定严格的执行序列」
> 「Stream 的优先级**在 Device 范围内生效，而不是在 Context 范围内生效**」

⇒ 这解释了 v1 的 naive 测法为什么必然测不出（先提交者已进入运行态）；
但 v2 的 **gate 测法（两流同时排队）仍然测不出** ⇒ 说明**障碍在参数传递层面（§5.1），而不是调度时机**。

---

## 6 一键复跑

```bash
C=dc-quota-probe-0930; PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
B=/mnt/raid/hliu553/dc_legs_20260929

# ① 优先级：C API 回读（判断参数是否真的到了设备）
docker exec $C $PY -u $B/readback_priority_910c.py

# ② 优先级：按官方机制重测（两流同时排队）+ naive 对照
docker exec -e DC_BACKEND=ascend -e DC_OUT_DIR=$B/rca_out $C $PY -u \
  $B/prototype/probes/probe_stream_priority_queued.py --dev 0

# ③ 配额：设备自报可用流数 + pyACL 直连建到失败（精确上限）
docker exec $C bash -lc "MODE=pyacl CAP=4000 $PY -u $B/find_stream_quota_910c.py"

# ④ P800 对称核查（优先级回读）
ssh P800 'docker exec -e CUDA_VISIBLE_DEVICES=4 hliu553-device-context-p800 bash -lc \
  "source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && \
   python3 -u /workspace/dc_regress_20260929/readback_priority_p800.py"'
```

**证据**（`../910C/probes/`，11 份）：
`prio_readback_910c_npu_20260930.log` · `prio_queued_910c_npu_20260930.json` ·
`quota_pyacl_until_fail_910c_npu_20260930.log` · `available_num_behavior_910c_npu_20260930.log` ·
`prio_gate_and_torch_quota_910c_npu_20260930.log` · `b2_*_20260930*`（v1 原样保留）；
**P800**：`P800/probes/prio_readback_kunlun_20260930.log`；
**可复跑脚本**：`910C/probes/inspect_prio_{readback,api_surface}_910c_20260930.py` ·
`910C/probes/find_stream_quota_910c_20260930.py` · `P800/probes/inspect_prio_readback_kunlun_20260930.py` ·
`prototype/probes/probe_stream_priority_queued.py`

**官方依据（可点开核对）**：
- CANN `device_get_stream_priority_range`（Python）：https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/runtimeapi/aclpythondevg_01_1030.html
- CANN `create_stream_with_config` 的 `priority` 参数状态（"预留参数，暂不使用" / 推理系列支持 [0,7]）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/appdevgapi/aclpythondevg_01_0078.html
- CANN Stream 管理（不抢占 / 不重评估 / Device 范围生效）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/others/acldevg/runtime_doc_dev_0011.html
- CANN `aclrtCreateStream` 的 Stream 最大数（Atlas 训练系列 2048，`N = 2048 − 默认 − 内部同步`）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/runtimeapi/aclcppdevg_03_0065.html
- torchair `npu_stream_switch`（`stream_priority` 为**预留参数**）：https://www.hiascend.com/document/detail/zh/Pytorch/720/modthirdparty/torchairuseguide/torchair_00102.html

---

## 7 结论与改进项（**未擅自改契约，待裁定**）

| # | 事实 | 影响 | 建议 |
|---|---|---|---|
| 1 | 本层**声明了 `stream_priority` 能力**，但：① 统一 API 的 `create_stream()` **没有 priority 参数**；② 底层 `torch.npu.Stream()` **静默丢弃** priority；③ 文档说训练产品上该参数**预留** | ⚠️ **"声明即承诺"近似违反**：下游看到"支持 stream_priority + 范围 0~7"，会以为可调 | 二选一：**(A)** 补 `create_stream(priority=None)` 走 **pyACL `create_stream_with_config`** 路径 + **回读校验**（ACL 层确实接受）；**(B)** 如实**降级声明**（声明语义改为"仅可读范围，不可设置"）并加判据守 |
| 2 | `kunlun.stream_priority_range()` 返回 `None` | 会被读成"接口不存在"，而实测是"**范围退化到 (0,0)**" | 改为如实返回 `(0, 0)`（或带原因的结构），与"设备自报"一致 |
| 3 | 旗舰判据缺口 | 现有判据只查"范围可读"，**没有任何判据查"能不能真的设置并生效"** | 建议加**真机**判据：**声明 `stream_priority` ⇒ 必须能设置且能回读**（本报告的回读法即可，成本极低） |
| 4 | 上游口径冲突 | 头文件 `value range: 0~7` vs 官方文档「训练产品预留」 | 作为**上报材料**（本方向已有"上游内部口径冲突是最有力上报材料"的惯例） |

**未做（如实登记）**：MLU590 的优先级与配额未取数；`torch.npu.Stream()` 在**真正使用后**是否才消耗配额未复核（§3 第三行）；`1e7` 档未测（且 `available_num` 已给出真实上限，无需再测）。
