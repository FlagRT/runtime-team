# 接口约定修订建议 · 逐条状态复核（工作包 E / B3，2026-09-29）

> 定位：**状态复核记录**。对象是 `INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`（v1.1，共 **10 条**）。
> 目的：把每条从「当时的表述」推进到**今天的三态之一** —— **已修 / 未修 / 已失效**，
> 并给出**可复核的实测依据**（文件 + 行 + 命令），做到**零僵尸条目**。
>
> 复核方式：不看文档自述，**逐条实测当前代码**（`grep` / `read` / 用离线 stub 真调 `info()`）。
> 所有结论都在本文件内自包含：读者不需要打开建议文档即可判断。
>
> 结论边界：本复核**只判"代码/契约里现在是什么样"**，不重新论证每条建议本身的合理性。

---

## 0 一句话结论

**10 条里：9 条已落地（其中 2 条已泛化/加固，2 条的具体对象已失效但条款仍有效），1 条未修。**

| 三态 | 条数 | 条目 |
|---|---|---|
| ✅ **已修** | **8** | 第 2 · 3 · 4 · 5 · 7 · 8 · 9 · 10 条 |
| 🟡 **部分已修**（正文落地、一条子要求未做） | **1** | 第 6 条（smoke 未打印 `known_issues`） |
| 🔴 **未修** | **1** | 第 1 条（`vendor` 字段未加；但**原始问题已部分消解**，残留键名分叉） |
| ⚫ **已失效**（对象不存在） | 0 条整体失效 | 第 8 · 9 条的**具体修复对象 `flagos` 已随路线 B 删除**，条款本身仍有效 |

**零僵尸达成**：建议文档汇总表的「当前是否已改代码」列已回填为**今天的三态**，
第 3 / 5 条正文里标注的「未改」也已更正 —— 复核者不会再被引导去"做一条已完成的事"。

⭐ **复核过程中另抓到 2 处真实缺口并已修**（不在原 10 条范围内，属第 8 / 9 条家族的新实例）：

| 缺口 | 现象（实测） | 处置 |
|---|---|---|
| **G-1** `ascend.info()` **缺 `device_type`** | `info()["device_type"]` → **`KeyError: 'device_type'`**（kind: 另两家正常返回 `cuda` / `mlu`） | 已补 + 新增判据（真实后端也守）+ 非空转验证 |
| **G-2** `_base_info_fields()` **无任何调用方** | 三家 `info()` 各**手写一份** `capabilities` / `native_accesses` / `degradations` ⇒ 同一份清单**三个来源** | 三家改为并入基类唯一来源 + 新增判据 + 非空转验证 |

---

## 1 逐条三态总表（2026-09-29 实测）

| # | 修订项 | 三态 | 今天的实测依据（可复核） |
|---|---|---|---|
| **1** | `device_type` 与 `vendor` 分离 | 🔴 **未修**（问题部分消解） | `grep -rn "vendor" runtime/backends/base.py` ⇒ 仅 1 处注释（`参考：vendor 插件目录模式`），**无 `vendor` 类属性**；三家 `backend.py` 均无 `vendor = `；三家 `info()` 均**无 `vendor` 键**。**但**：三家 `device_type` 实测 = `npu` / `cuda` / `mlu`，**都是合法设备串前缀**（原隐患来自 `flagos` 的 `device_type="flagos"`，该后端已随路线 B 删除）⇒ "拼不出有效设备串"这条原始问题**不再存在** |
| **2** | `device_state` 纳入 Backend 契约 | ✅ **已修（并已扩展）** | `runtime/conformance/device_state.py` 四态 = `AVAILABLE / DEGRADED / ISOLATED / DESTROYED`（**`DESTROYED`**，非文档旧写的 `UNKNOWN`）；三家均按同一实现注册（`base.set_device_state` 为唯一实现）⇒ `supports()` 与调用均成立 |
| **3** | `.native` 逃生舱约束 + `record_stream` 能力位 | ✅ **已修**（工作包 B 落地） | `api/stream.py`：公开 `.native` 计数（`backend.info()["native_accesses"]`）、内部走私有 `_native_obj`（不计数）；`backends/base.py:382` 起为 `record_stream` 能力位 + 保守同步路径（`conservative_stream_sync`）+ 退化审计（`degradations`）；三家 `_capabilities` 均含 `record_stream` |
| **4** | 错误对象跨模块类归一 | ✅ **已修（并已收紧）** | `api/errors.py`：`coerce_category()` / `normalize_error()` / `translate_via_backend()` 归一兜底都在。2026-09-29 进一步**从源头**消除不一致：`translate_error(exc, location=…, vendor_codes=False)` ⇒ 未声明 `error_map` 的后端**结构上不可能**携带他厂码 |
| **5** | 不支持有界同步时的声明与降级 | ✅ **已修** | 能力键 `bounded_sync` **三家 `_CAPABILITY_KEYS` 均含**（另留历史别名 `sync_timeout`，在基类做别名归一）；`bounded_sync_scope` 字段如实体现实路径（ascend：`pyACL synchronize_*_with_timeout（真中断）`；kunlun / cambricon：`主机侧等待真有界；流同步为超时上报语义`）；契约已写 `synchronize_stream(native_stream, timeout_ms)` / `wait_event_host(native_event, timeout_ms)` 的**超时抛 `TimeoutError`** 语义 |
| **6** | `known_issues()` 纳入契约 | 🟡 **部分已修** | 已做：`base.known_issues()` 默认 `[]`；**三家均有实现**（`ascend:569` / `kunlun:590` / `cambricon:620`）；三家 `info()` 均含 `known_issues`（取 `id` 列表）；离线自检有结构判据（9 个必需字段）、职责审计有 G3。**未做**：建议条文里「校验自检（**smoke**）应打印本后端的能力声明与已知问题摘要」—— `grep -n known_issues runtime/smoke_runtime.py` **0 命中**，smoke 只在其他口径里覆盖（`offline_check` / `duty_response_audit` 有） |
| **7** | 能力降级必须「整组一致」 | ✅ **已修（并已转向源头）** | 离线判据 3 条 + smoke 3 条守 `mapped` / `graded_by` / `error_code` 三字段同降；且 2026-09-29 起共享翻译器对无码表后端**源头不产生**外来码表命中（第 ① 条修法：可推广为"凡事后修补，先问能否源头不产生"） |
| **8** | 声明即承诺 + `info()["supports"]` 按能力全集呈现 | ✅ **已修（并本轮加固）** | 三家 `info()["supports"]` 按 `_CAPABILITY_KEYS` 同源派生（实测均为 **19 键**）；`contract_invariants.CAPABILITY_ENTRYPOINTS` 提供「声明 ⇒ 有公共入口」的自动判据（I1④ 覆盖面 **14/14**，含本轮新增的 `device_state_control`）。⚠️ 原修复对象 **`flagos` 后端已删除** ⇒ 该条当时的具体修复已成历史；条款本身**有效且已泛化到三家**。⭐ 本轮加固见 §2（G-1 / G-2） |
| **9** | 验证资产的可达性与证据卫生 | ✅ **已修（并本轮续发现两处盲区）** | 已做：离线自检 `_STUBS` 注册 **3 家**（`cambricon` / `kunlun` / `ascend`）+ 显式 `[SKIP]` 机制（汇总行区分 通过/失败/跳过）+ 硬编码厂商码已清。⚠️ 原文写"扩到四家" ⇒ 随路线 B 退出变 **3 家**。⭐ 本轮**又抓到该条家族的两处盲区**（G-1 的两个成因：矩阵列取类属性、smoke 判据只对 stub 跑）⇒ 已补判据 |
| **10** | 厂商扩展懒加载：拼厂商专有字符串前必须先触碰设备 | ✅ **已修** | `conformance/runner.py` warm-up（注释里写明 2026-09-22 实测：`use()` 后 `hasattr(torch,"npu")` 为 `False`，调一次 `device_count()` 后为 `True`）+ `proto_train_leg.py` 进程组初始化前触碰设备；该纪律已写入接入手册「坑 10」 |

---

## 2 ⭐ 复核过程中新抓到的 2 处缺口（G-1 / G-2）—— 已修 + 已加判据

### 2.1 G-1：`ascend.info()` 缺 `device_type`（真实契约违约）

**现象（实测，无需设备）**：用离线 stub 加载三家真实后端，逐家读 `info()`：

```
ascend     info 键: [acl_available, acl_unavailable_reason, bounded_sync_scope, capabilities,
                     degradations, device_count, evidence_level, framework, known_issues,
                     name, native_accesses, supports, torch]          ← 13 键，无 device_type
           'device_type' in info: False
           ⛔ info['device_type'] -> KeyError: 'device_type'

kunlun     info['device_type'] = 'cuda'      ✅
cambricon  info['device_type'] = 'mlu'       ✅
```

而**契约与冒烟都要求 `info` 至少含 `name` / `device_type` / `capabilities`**
（`runtime/smoke_runtime.py:200` 的判据原文即 `info 至少含 name/device_type/capabilities（只增不改）`）
⇒ `ascend` 违反了本层自己写下的承诺。

**为什么长期没被发现（两处判据盲区，正是第 9 条家族）**：

| 盲区 | 说明 |
|---|---|
| ① 离线"能力矩阵"表 | 该表的 `device_type` 列取的是**类属性** `bk.device_type`，**不是 `info()`** ⇒ 表里 `npu` 一直显示正常 |
| ② smoke 的同名判据 | 该判据在**stub 上下文**里对 `b`（stub 后端）执行 ⇒ **真实后端从未被覆盖** |

**修法**（最小、只增）：`ascend.info()` 补 `"device_type": self.device_type` ⇒ 键数 13 → **14**（只增）。

### 2.2 G-2：`_base_info_fields()` 无调用方 ⇒ 公共字段三个来源

`backends/base.py` 提供了 `_base_info_fields()`（docstring 明写"所有后端 `info()` 都应并入的公共字段（新增能力时**一处生效**）"），
但实测 `grep -rn "_base_info_fields" runtime/ scripts/` 只命中 base.py 自身两行 ⇒ **三家都手写了一份**：

```python
# 三家 info() 里各自手写的同一段（原样重复三遍）
"capabilities": sorted(self._capabilities),
"native_accesses": self.native_accesses(),
"degradations": self.degradations(),
```

这正是第 8 条修过的形态的**残留**（第 8 条修的是 `supports` 手写第二份键名清单，这里是公共字段手写）。
**修法**：三家改为 `**self._base_info_fields(),` ⇒ 公共字段回落到**唯一来源**；行为等价（键与取值不变）。

### 2.3 新增 2 条判据（离线，不需卡）+ 非空转验证

判据加在 `scripts/backend_offline_check.py` 的 `--all`（跨后端对称性）段：

```
[FAIL] info() 含契约最小键集 name/device_type/capabilities（真实后端也守） {'ascend': ['device_type']}
[PASS] info() 均含公共字段 capabilities/native_accesses/degradations {}
```

**非空转验证**（先证明判据能 FAIL，再修）：

| 注入 | 判据实测反应 |
|---|---|
| 去掉 `ascend` 的 `device_type` 行 | `[FAIL] … {'ascend': ['device_type']}` ✅ 精确指名 |
| 把 `cambricon` 的 `**self._base_info_fields()` 换成只手写 `capabilities` | `[FAIL] … {'cambricon': ['native_accesses','degradations']}` ✅ 精确指名 |

> 判据 ⑥ 的第一次 FAIL **不是注入造出来的**，而是**真实缺陷**（修复前 `--all` 就是 `6 通过 / 1 失败`）——
> 这比注入验证更强：判据在真实代码上抓住了真问题。

**修复后**（实测）：三家 `info()` 三键齐备，`device_type` = `npu` / `cuda` / `mlu`；
`--all` = **7 通过 / 0 失败**。

---

## 3 文档卫生问题（顺手登记）

| 问题 | 现状 | 处置 |
|---|---|---|
| **节编号与物理顺序不一致** | 物理顺序是 §8 → **§10** → **§9**（第 10 节排在第 9 节之前） | 保留原编号（**编号是引用锚点**，重编号会打断既有引用）；本节登记，读者以**编号**为准 |
| **汇总表状态列停留在 09-20/22 时点** | 「当前是否已改代码」列写第 3/5 条"未改"等 | **已回填为 2026-09-29 三态**（见 §4） |
| **"未修复"字样与事实不符** | 第 3 条正文标"未改"，但工作包 B 已落地 | **已更正** |

---

## 4 对原建议文档的回填（本次已改）

`INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`：

1. 顶部加**状态横幅**：声明已于 2026-09-29 完成逐条复核，指向本文件；并注明"汇总表的『是否已改代码』列为**当日现状**，非历史"。
2. **汇总表「当前是否已改代码」列**逐条改为三态（✅ 已修 / 🟡 部分 / 🔴 未修，附一句现状）。
3. **第 3 条正文**标题后的状态标记由"未改"改为"✅ 已修（工作包 B，2026-09-29）"。
4. **第 5 条正文**"前瞻性条款/未改"更正为"✅ 已落地（能力键 `bounded_sync` 三家齐）"。
5. **第 6 条正文**如实标注「smoke 打印」这一子要求**未做**（避免"已实现"被读成"全部完成"）。

---

## 5 一键复核（逐条对应，可实测）

```bash
cd dev/device-context/prototype

# 第 1 条：应只有注释命中，无 vendor 类属性 / info 键
grep -rn "vendor" runtime/backends/base.py
for v in ascend kunlun cambricon; do printf "%s " $v; grep -nE '^[[:space:]]+device_type = "' runtime/backends/$v/backend.py; done

# 第 2 条：四态名与唯一实现
grep -nE "AVAILABLE|DEGRADED|ISOLATED|DESTROYED" runtime/conformance/device_state.py | head -4
grep -n "def set_device_state" runtime/backends/base.py

# 第 3 条：.native 审计 + record_stream 能力位（三家各应命中）
grep -nE "native_accesses|_native_obj|conservative_stream_sync" runtime/api/stream.py | head -5
for v in ascend kunlun cambricon; do printf "%s " $v; grep -c '"record_stream"' runtime/backends/$v/backend.py; done

# 第 4 条：归一化三件套
grep -nE "def coerce_category|def normalize_error" runtime/api/errors.py

# 第 5 条：bounded_sync 三家在全集内
for v in ascend kunlun cambricon; do printf "%s " $v; grep -c '"bounded_sync"' runtime/backends/$v/backend.py; done

# 第 6 条：三家实现 + info 含；smoke 未打印（应为 0）
grep -c "def known_issues" runtime/backends/*/backend.py
grep -c "known_issues" runtime/smoke_runtime.py

# 第 7 条：降级三字段判据
grep -nE "graded_by|mapped" scripts/backend_offline_check.py | head -5

# 第 8 条：supports 按全集派生 + I1④ 映射
grep -n "self._CAPABILITY_KEYS" runtime/backends/*/backend.py
grep -nE "CAPABILITY_ENTRYPOINTS|入口存在性覆盖" runtime/conformance/contract_invariants.py

# 第 9 条：stub 覆盖三家 + 显式 SKIP
python3 -c "import io;s=io.open('scripts/backend_offline_check.py',encoding='utf-8').read();i=s.index('_STUBS = {');print(s[i:i+260])"

# 第 10 条：warm-up
grep -n "device_count()" runtime/conformance/runner.py | head -3
grep -n "device_count()" runtime/proto/proto_train_leg.py | head -2

# 本轮新增判据（G-1 / G-2）：应为 2 条 PASS、0 FAIL
python3 scripts/backend_offline_check.py --all 2>&1 | grep -aE "契约最小键集|公共字段|对称性自检结果"
```

---

## 6 未收尾（如实登记，不补零）

| 项 | 状态 | 说明 |
|---|---|---|
| **第 1 条 `vendor` 字段** | 🔴 未修 | 现状：`name` 已能区分厂商（`ascend`/`kunlun`/`cambricon`），设备串前缀三家均合法。**残留**：三家 `info()` 里出现两个自发且互不一致的键 —— `kunlun.vendor_discriminator`（4 条判别字符串）/ `cambricon.device_namespace`（1 条）—— 属"同一概念多个对外形态"（台账 ⑫b 家族）。**是否补 `vendor` 需裁定**：补 = 契约只增一项、三家对齐；不补 = 现状可用但键名分叉会长期留着 |
| **第 6 条 smoke 打印** | 🟡 未做 | 建议条文要求 smoke 打印能力声明与已知问题摘要；现状由 `offline_check` / `duty_response_audit` 承担。补 = smoke 加一段打印（不影响判定） |
| ~~第 10 条遗留的 `flagos` 复验~~ | ⚫ **已自动关闭** | 路线 B 整体退出 ⇒ `backends/flagos/` 已删除，该待补项不存在（原文档 §8 已有说明，本次复核确认） |
| 结论边界 | —— | 本复核只在**当前提交**上成立；后续若再改 `info()` / 能力键，**本复核的"已修"结论需按破坏面重判**（免跑理由有时效性） |

---

## 7 同日核对：工作包 B+C 计划的三家完成矩阵（实测）

> 本节回答"**B+C 计划现在到哪一步了**"。数据来源 = 用离线 stub 逐个真调 `supports()`（不需设备）+ git 证据来源反查，
> **不引用任何文档的自述**。

### 7.1 能力面（`supports(<键>)` 实测）

| 能力键（B/C 相关） | 910C `ascend` | P800 `kunlun` | MLU590 `cambricon` |
|---|---|---|---|
| `memory_alloc`（B-1 句柄） | ✅ | ✅ | ➖ |
| `memory_alloc_stat`（B-2 第四键） | ✅ | ✅ | ➖ |
| `record_stream`（B-3 能力位） | ✅ | ✅ | ➖ |
| `.native` / `degradations` 审计（B-4） | ✅ | ✅ | ✅（芯片无关，常开） |
| `context_lifecycle`（C-1/C-2） | ✅ | ➖ | ➖ |
| `context_query`（C 的只读形态） | ✅ | ✅ | ➖ |
| `recovery_real`（三级重建之 real） | ✅ | ➖ | ➖ |
| `recovery_probe` | ✅ | ✅ | ✅ |
| `device_state_control`（2026-09-29 新增） | ✅ | ✅ | ✅ |
| **声明总数 / 能力全集** | **18 / 19** | **14 / 19** | **11 / 19** |

⚠️ **一处口径更正**：`context_query` 早先的表述是"**P800 声明，910C / MLU590 如实未声明**"；
实测 **ascend 也已声明**（由 `ab80d87`「910C 对齐 context_query」落地）⇒ 当前是 **910C + P800 两家声明**，
MLU590 未声明。相关文档（`WORKPACKAGE_BC_INTERFACE_20260929.md` §11.2）需按此更正。

### 7.2 覆盖点（谁最后跑过；之后哪些共享层改动**未**覆盖）

| 实例 | 最后一次"全套"覆盖 | 与之对应的提交 | 此后的共享层改动（未覆盖） |
|---|---|---|---|
| **910C** | **本轮 r8**（9 项全绿） | 本轮 | —— |
| **P800** | r5（含契约不变式）+ B/C 探针 r2 | `1f26633` | `7225889`（`serve_standard.sh`）· **`752551b`（6 文件）** · **`2647370`（10 文件）** · 本轮（3 家 `backend.py` + `offline_check.py`） |
| **MLU590** | r3 全套回归（**未含** B/C 探针、**未含**契约不变式） | `e1528bd` | 上述全部 + `f54d752` / `ab80d87` / `ab25fe1` / `6b664ce` / `1f26633` |

**共享层改动明细**（判"要不要复跑"的依据）：

```
7225889  1 个: scripts/serve_standard.sh
752551b  6 个: runtime/backends/base.py · runtime/backends/ascend/backend.py
               runtime/conformance/device_state.py · runtime/conformance/recovery.py
               scripts/backend_offline_check.py · probes/recover_multiproc_stress.py
2647370 10 个: runtime/__init__.py · runtime/backends/{base,ascend,kunlun,cambricon}*
               runtime/conformance/{contract_invariants,recovery}.py
               scripts/backend_offline_check.py · probes/{recover_entry_verify,recover_multiproc_stress}.py
本轮      4 个: runtime/backends/{ascend,kunlun,cambricon}/backend.py · scripts/backend_offline_check.py
```

### 7.3 结论（三句话）

1. **910C：B+C 已全部对照完成** —— B1/B2/B3/B4 + C1/C2/C3 **六项全通过**（C 是**完整生命周期**，
   含真机多上下文隔离取证）；且本轮 r8 再次全绿（见 §8）。
2. **P800：不是"修复"，是"复跑"** —— P800 的 B/C **接口本身已通过**（B1/B3/B4 三项；C 因**平台约束**
   只允许一个 primary context 且由框架自建，已如实补为**只读** `context_query`，真机 `compute_before = compute_after = 512.0`）。
   真正缺的是：**`752551b` / `2647370` / 本轮 的共享层改动从未在 P800 上跑过** ⇒ 需一次窗口复跑。
3. **MLU590：不是"修复"，是"首轮取数 + 覆盖"** —— B/C 的 4 个新能力键在 MLU590 **全部未声明**
   （属**"本轮范围内未探测"**，**不是"已确认该芯片不具备"**）；且**从未跑过** B/C 探针与契约不变式。
   需一次窗口：跑 `probe_bc_contract.py --backend cambricon` 取数 → 决定是否声明 → 再跑全套回归。

### 7.4 建议顺序（含依赖）

| 序 | 事项 | 需卡 | 理由 |
|---|---|---|---|
| 1 | **B2 工作包 D**（流优先级效果 + 配额真实上限） | 弱（单卡可做） | 审计清单里 A/B 类仅剩此项；三实例同口径取数 |
| 2 | **C2 / C3**（宿主副本陈旧 · 环境层 4 小项补记） | ❌ | 不需卡、成本低；C2 是"不可复现"的直接来源 |
| 3 | **P800 复跑**（共享层三步改动 + 探测记录） | ✅ 需窗口 | 三机不可同时可达（VPN 限制）⇒ 与 MLU590 **分别排期** |
| 4 | **MLU590 首轮**（B/C 取数 + 全套回归 + 契约不变式） | ✅ 需窗口 | 同上；且是"第 3 家"的首次 B/C 检验 |
| 5 | C4（P800 上报渠道） | —— | 需**人**给口径，非技术动作 |

---

## 8 本轮回归（910C 第 8 轮 · 覆盖 `info()` 改动）

**破坏面判定**：改动只在 `info()` 返回值（补 `device_type` + 公共字段收口）⇒ **元信息面**。
消费方反查（决定覆盖范围）：`proto_infer_leg.py:114` / `contract_invariants.py` I1 / `smoke_runtime.py` / 职责审计 G2
**读 `info()`** ⇒ 必跑（`proto_infer_leg.py` 命中 1 处、`contract_invariants.py` 命中 8 处）；
`serve_standard.sh` **不读** `info()`（实测只有 1 处命中，且该行是**以 `#` 开头的注释**，非真实调用）
⇒ **服务化免跑**（理由可复核，符合"免跑理由按破坏面重判"）。

| 判定项 | 结果 |
|---|---|
| 离线契约自检（ascend） | **83 / 0 / 1 跳过** |
| 跨后端对称性 `--all` | **7 / 0**（含本轮新增 2 条） |
| 组件冒烟（910C 真机） | **52 / 0** |
| conformance 13 + 推理 6 | **13/13 + 6/6** |
| 契约不变式 I1–I4 | **4/4**（I1④ 入口存在性覆盖 **14/14**） |
| 职责响应审计（39 sub-part） | **39 / 0 / 0** |
| B/C 契约探针 | **PASS**（六项判定同前） |
| 推理腿前向 | **`INFER_LEG_PASS 14/14`**（dim 1024 · 73.49 句/s） |
| 结论 | ✅ **无回归** |

**免跑项（当轮可验证，均为实测）**：服务化 —— `grep -E "info\(\)|device_type" scripts/serve_standard.sh`
只命中 1 处且为注释行；训练腿 —— `grep -c "\.info()" runtime/proto/proto_train_leg.py` = **0**。
⚠️ 这两条免跑理由**只对本轮改动成立**（改动面 = `info()` 返回值）；后续若改动落在执行路径内需重判。
证据：`../../910C/probes/r8_regress_910c_npu_20260929.log` + `r8_regress_910c_npu_20260929_out/`（15 份）。
