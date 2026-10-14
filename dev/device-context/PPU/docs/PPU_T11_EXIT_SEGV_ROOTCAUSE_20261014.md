# PPU 退出期段错误（T11）· 根因定位与收口

> 日期：2026-10-14 ｜ 对应执行单 `../../prototype/docs/TODO_NEXT_WEEK_20261012.md` **T11**
> 证据目录：`../probes/t11_20261014_out/`（115 项）｜ 脚本：见 §9
> 前置：本报告所有结论都来自**真机实测**（平头哥 PPU `PPU-ZW810E`，容器 `hliu553-dc-dev`，卡 `dev1`）；
> 未取得证据的一律标「未验证」。

---

## 0 结论先行

| 问题 | 结论 |
|---|---|
| **崩溃点在哪** | 解释器**退出期**、`Py_Exit → Py_FinalizeEx → … → **_PyWeakref_ClearRef**` 内；原生采集到 `si_code=1`（SEGV_MAPERR）、**`si_addr=0xab`**（垃圾指针）、RIP 在 `libpython3.12.so.1.0`。**判据与产物不受影响**（崩溃发生在判据打印、结果落盘之后） |
| **根因** | **平台侧「弱引用书签」不可信**：PPU 的 torch 流对象被回收后，其 `weakref` 悬垂（对象死、书签未置死、内存被复用）⇒ finalize 阶段清理弱引用时解引用垃圾指针。⚠️ **不含本层**的最小复现同样崩：30 条原生 `torch.cuda.Stream()` 各挂一个 `weakref.ref` 后丢弃 ⇒ `rc=139`（2/2；N=1 时不崩）。⚠️ **自审见 §10**：该变体所在脚本**也 `import runtime` 并 `runtime.use()`**（日志首行即 `backend=ppu`）⇒ 严格说只是「**不调用本层的流 API**」；**进程内完全不含本层**的纯净复现与**缺项对照**已备脚本待跑 |
| **本层的角色** | **暴露面**（不是根因）：本层为**每条流**建 `weakref`（`note_stream_created` 的 `_stream_ctx`；`release_stream` 的 `_released_streams`）⇒ 把"潜在"变成"必然"。这也解释了 10-10 的旧观察"去掉 `runtime.create_stream()` 那一行即干净" |
| **修法（本层，已真机验证）** | ① `note_stream_created`：**无本层上下文时不登记**（该条目本就无信息）；② `_released_streams` 改**强持有**；③ 两张按 `id()` 键的登记表加**条数上限**（强持有会 pin 住对象）。修后三个载体（本层最小复现 / T4 / T6）**全部 `rc=0`**，且判据读数与修前**逐字相同** |
| **平台侧是否修好** | ❌ **没有**（本层不声称、也无力修平台）。`vendor_weakref_many`（不含本层）修后**仍然 `rc=139` 2/2** ⇒ 该缺陷保留为**厂商侧诉求**，最小复现见 §5.2 |
| **剩余依赖** | 平台侧修复 = 厂商；本层已**不再受其影响**。⚠️ 本轮改的是**共享层**（`backends/base.py`）⇒ 910C / P800 / MLU590 **需按破坏面各复跑一次 78 项审计**（执行单 T1 口径） |

---

## 1 现象与既有事实（本轮之前的积累，先复述以免读错）

| # | 事实 | 出处 |
|---|---|---|
| 1 | 训练腿 `TRAIN_LEG_PASS 6/6`、推理腿 `INFER_LEG_PASS 13/13`，**两个进程都在判据通过后于解释器退出阶段 SIGSEGV** | `PPU_MODEL_AND_TWO_LEGS_20261010.md` §4.1 |
| 2 | 10-10 已做 **16 组对照**（E1–E16）：排除 CDLL 被 dlclose（E6）、引用环（E15）、建流与建组先后（E16）、模型/NCCL/P2P/训练循环（E1–E5、E7）；**`st = runtime.create_stream()` 换成 `st = None` 即干净（E13b）**、换原生 `torch.cuda.Stream()` 也干净（E14a） | 同上 §4.2 |
| 3 | 10-14 复现面扩大：T4 错误闭环、T6 调度效果探针**均**在判据通过后 `rc=139`；而**不建优先级流的 T5 语义探针 `rc=0`** | `EVIDENCE_INDEX_PPU.md` §4 |
| 4 | `PYTHONFAULTHANDLER=1` 下现场为 `Garbage-collecting` / `<no Python frame>` | 同上 |

### 1.1 ⭐ 本轮新发现的"天然对照"（把面收窄到"建流"这一件事）

逐文件核对"建流方式"后得到一组现成对照：

| 载体 | 建流方式 | 退出码 |
|---|---|---|
| `probes/probe_stream_semantics_full.py`（T5） | **只用原生** `torch.cuda.Stream()`（`dev.Stream()`），且建的流**更多、还有两流并发** | **0**（干净） |
| `probes/probe_stream_priority_sched_effect.py`（T6） | 经 `runtime.create_stream()` | **139** |
| `runtime/proto/proto_train_leg.py` / `proto_infer_leg.py` | 经 `runtime.create_stream()` | **139** |
| `runtime/proto/proto_error_recovery_loop.py`（T4） | 经 `runtime.create_stream()` | **139** |

**关键**：训练腿那条流**只被 `repr` 过一次**（`f"...统一流 {st!r}"`），**全程没有做任何流操作**
⇒ 触发条件与"怎么用流"无关，只与"**建了这条统一流**"有关 ⇒ 可以做成**几行的最小复现**，用退出码逐项二分。

---

## 2 ⭐ 原生现场：怎么在没有 gdb 的情况下拿到崩溃点

### 2.1 约束（先说清楚为什么以前停在"需外部"）

| 手段 | 可用性（容器 `hliu553-dc-dev` 实测） |
|---|---|
| `gdb` / `eu-stack` | ❌ 容器内都没有（宿主 `/usr/bin/gdb`、`/usr/bin/eu-stack` **有**，但宿主无免密 sudo、容器进程属 root ⇒ 不能从宿主 attach） |
| core dump | ❌ `core_pattern` 指向宿主 `systemd-coredump`（需 root 或 `systemd-journal` 组）；`coredumpctl` 实测"insufficient permissions" |
| 本地镜像里的调试器 | ❌ 逐一试过 9 个本地镜像（flagtree 系 / vllm-plugin-fl / torchrec / cuml / easyrec / xgb 等）**全部没有** `gdb`/`lldb`/`eu-stack` |
| ✅ **容器内有 `gcc`** | ✅ `/usr/bin/gcc`、`/usr/bin/g++`、`/usr/bin/make` |

### 2.2 办法：自装**原生** SIGSEGV 处理器（`LD_PRELOAD` + `dladdr`）

`probes/t11_segv_handler.c`（约 40 行）编译成 `.so` 后 `LD_PRELOAD` 进去，构造器里
`sigaction(SIGSEGV, SA_SIGINFO|SA_RESTART)` + **回读校验**，崩溃瞬间用 `write(2)` 输出：
`si_code` / `si_addr` / `RIP` / `RSP` + `dladdr(RIP)` 得到的**模块路径 + 模块基址 + 文件内偏移 + 最近符号**
+ `backtrace()` 的逐帧 `模块+偏移`；末尾 `_exit(139)`（与"真崩溃"同签名，便于与既有 rc 判据比对）。

⚠️ **失败路径也记下来（有教学价值）**：先用 `ctypes.CFUNCTYPE` + `sigaction` 写过一版
（`probes/t11_sigsegv_probe.py`），它**非空转自检通过**（故意 `ctypes.string_at(0)` 时能抓到 RIP 落在 libc），
但对真崩溃**一次都没触发** —— 因为该回调是 **libffi 闭包**，在解释器 finalize 阶段随模块一起被回收，
而目标崩溃恰好就发生在那个窗口里。⇒ 采集器本身必须**不依赖 Python 运行时**，故改用 C + `LD_PRELOAD`。

### 2.3 采集结果（三个载体，原文见 `t11_20261014_out/{t4_native1,t4_native2,t6_native}.log`）

```
===== T11_CRASH_CAPTURED (native) =====
signal=11 si_code=1 si_addr=0xab
RIP=0x7fb1d2456ad5 RSP=0x7ffe2fae3f40
RIP_MODULE=/usr/local/lib/libpython3.12.so.1.0
RIP_BASE=0x7fb1d2200000
RIP_OFFSET=0x256ad5
BACKTRACE_FRAMES=17
  FRAME#2 module=libpython3.12.so.1.0 offset=0x256ad5 symbol=?
  FRAME#3 module=libpython3.12.so.1.0 offset=0x2d2529 symbol=_PyWeakref_ClearRef
  FRAME#4 module=libpython3.12.so.1.0 offset=0x30e480 symbol=?
  FRAME#5 module=libpython3.12.so.1.0 offset=0x30ec74 symbol=?
  FRAME#6 module=libpython3.12.so.1.0 offset=0x302080 symbol=Py_FinalizeEx
  FRAME#7 module=libpython3.12.so.1.0 offset=0x302fec symbol=Py_Exit
```

**三个载体（T4×2、T6×1）的**故障签名一致**：`si_code=1`（SEGV_MAPERR）、`si_addr=0xab`（垃圾指针）、
RIP 均在 `libpython3.12.so.1.0` 的 **`0x2569xx`–`0x256axx` 区域**（无导出符号）。
⇒ 不是"某个探针的特殊路径"，而是**同一处**。

⚠️ **2026-10-14 自审更正（据原始日志，原文写"落在完全同一位置"——过强）**：三者的**帧链并不相同**：

| 载体 | `RIP_OFFSET` | 关键帧（由内向外） | 证据 |
|---|---|---|---|
| T4（错误闭环，两次） | `0x2569de` | RIP ← `0x23b07a` ← … ← `_PyModule_ClearDict` ← `Py_FinalizeEx` ← `Py_RunMain`（17 帧） | `t11_20261014_out/t4_native1.log` · `t4_native2.log` |
| T6（调度效果） | `0x256ad5` | RIP ← **`_PyWeakref_ClearRef`** ← `0x30e480` ← `0x30ec74` ← `Py_FinalizeEx` ← **`Py_Exit`** ← `Py_RunMain`（17 帧） | `t11_20261014_out/t6_native.log` |

正确表述 = **「同一区域 + 同一故障签名（`si_code=1`/`si_addr=0xab`），但帧链不同」**：
T6 明确经 `_PyWeakref_ClearRef`；T4 走的是**模块清理**路径（`_PyModule_ClearDict`）后落在相邻区域。
⇒ 这不推翻「弱引用悬垂」这一根因（注入对照仍是决定性证据：关掉本层的弱引用登记即不崩，见 §4），
但**不得**再写成"三者同址/同一帧"。RIP 偏移相差 `0x1F7` 字节、二者都为无符号静态函数，属同一代码区域。

**读法**：`si_addr=0xab` 属"被复用过的垃圾内存"，说明被处理的**弱引用对象本身**已被释放/被改写
（CPython 在给一个将死对象清理弱引用列表时，列表里挂着已死的书签）——与 10-14 已确证的
「平台侧 `weakref` 悬垂」（`PPU_L1_ROOTCAUSE_RECHECK_20261010.md` §4.7）**是同一个根**，
本轮把它从"内存地址复用导致误判"扩展到了"**解释器退出期直接段错误**"。

---

## 3 最小复现矩阵（`rc` 为退出码；每变体一个独立进程；轮次交替）

脚本 `probes/t11_exit_segv_matrix.py`；驱动 `probes/run_t11_ppu_20261014.sh`；原始记录 `t11_20261014_out/log/`、`rc_matrix.log`。

| 变体 | 做了什么 | rc（各轮） | 自证标记 |
|---|---|---|---|
| `native_plain` | 仅原生 `torch.cuda.Stream()` + 一次 matmul | 0 0 | BRANCH/EXIT_REACHED |
| `native_weakref_alive` | 原生流 + 一个 `weakref.ref`（N=1） | 0 0 | 同上 |
| `native_sync` / `vendor_stream_dropped` / `vendor_weakref_gc_del` | 原生流，丢弃 / 丢弃+gc / 连书签一起丢 | 0 0（各 2 轮） | 同上 |
| `ours_wrapper` | `runtime.create_stream()` 一条，**持有到退出** | 0 0 | 同上 |
| `ours_default_many` | 本层默认流 **8 条，全持有**到退出 | 0 0 | 同上 |
| `ours_prio_once` / `ours_prio_once_norelease` / `ours_prio_released_many` | 本层优先级流（C API 建流 + ExternalStream）：建 1 条并释放 / 不释放 / 建+释放 10 轮 | 0 0 | 同上 |
| `driver_only` / `ours_driver_loaded` | 只加载驱动库 `libcuda.so.1` / 加载驱动库 + 原生流 | 0 0 | 同上 |
| ⭐ `ours_default_dropped` | **本层默认流 30 条，每条建完即丢** | **139 139** | BRANCH/EXIT_REACHED/backend=ppu |
| ⭐ `vendor_weakref_many`（**不含本层**） | **原生流 30 条，每条挂一个 `weakref.ref` 后丢弃** | **139 139** | BRANCH/EXIT_REACHED/backend=ppu |
| `vendor_weakref_many_gc` | 同上，但每轮 `gc.collect()` | **139 139** | 同上 |

### 3.1 从矩阵读出的三条

1. **不是"建流"本身**：`ours_wrapper` / `ours_default_many` / `ours_prio_*` / `driver_*` 全干净
   ⇒ 之前"只要经本层建流就崩"的粗读**不成立**（那正是 10-10 停在"归属未定"的原因之一）。
2. **必要条件是"对象被回收 + 身上挂着弱引用"**：本层最小复现与**纯 torch 的 `vendor_weakref_many`
   同形同果**（都 139）⇒ 归因不依赖本层代码路径；`N=1` 时不崩 ⇒ 需要**累计多条**悬垂书签。
3. `vendor_weakref_many`（**不含本层**）崩 ⇒ **根因在平台侧**；本层只是**制造了弱引用**这一暴露面。

---

## 4 暴露面二分与修法验证（注入对照：`note_off` / `strong`）

**办法**：`PYTHONPATH=<工作目录> T11_PATCH=<模式>` 让 `sitecustomize.py`（`probes/t11_patch.py` 的副本）
在解释器启动早期给后端类打补丁 ⇒ **不改仓库代码**即可做单变量对照。
驱动 `probes/run_t11_patch_matrix_20261014.sh`；原始记录 `t11_20261014_out/patch_matrix.log`、`patch__*_r*.log`。

| 载体 | control（只注入、不改行为） | `note_off`（关掉 `_stream_ctx` 登记） | `strong`（`_id_holder` 改强持有） |
|---|---|---|---|
| `ours_default_dropped`（最小复现） | 139 139 | **0 0** | **0 0** |
| `t4_errorloop`（T4 错误闭环） | 139 139 | **0 0** | **0 0** |
| `t6_sched`（T6 调度效果） | 139 139 | **139 139** | **0 0** |

- 每轮都把三个载体跑一遍 ⇒ **轮次交替**，排除"时间漂移"被读成"变量效应"。
- **注入自证**：每行都带 `T11_PATCH_LOADED`（`patch_matrix.log` 里逐行标出）；未加载的行不得采信。
- **读法**：`note_off` 能救最小复现与 T4，但**救不了 T6** ⇒ T6 还走第二条暴露面（`release_stream` 的
  `_released_streams`）；`strong`（两处一起收）⇒ **三个载体全部干净**。

---

## 5 修法（本层）与真机验证

### 5.1 三处改动（`prototype/runtime/backends/base.py`）

| # | 改动 | 为什么这么改（而不是"多加一层判断"） |
|---|---|---|
| ① | `note_stream_created`：`current_context_id()` 为 `None` 时**不登记** | 该条目的**全部语义**就是"这条流属于哪个本层上下文"；`check_stream_usable()` 也**只在 `cid is not None`** 时才用它 ⇒ **不改变任何拦截行为**，但**不再挂弱引用** |
| ② | `release_stream` 写入 `_released_streams` 时用**强持有**（`lambda _o=native_stream: _o`），与 `_owned_streams`（10-14 已改）一致 | 该平台弱引用语义不可信；强持有同时**消除"id 可被复用"这一前提**，是**同类型地址复用**场景下唯一第二道保险 |
| ③ | 新增 `_ledger_put()` + `_LEDGER_MAX = 4096`：两张按 `id()` 键的表**超限丢最旧** | ②引入 pin 后条目只增不减（回收只可能发生在"同 id 再登记"）；长驻服务里"建流/释放流"是热路径 ⇒ 必须设上限。取舍如实写明：只牺牲**极旧条目**的诊断能力 |

**为什么这不是"绕过判据"**：判据（`check_stream_usable` / `_owned_handle`）从未改变；改的是
**本层自己的记录方式**——把"依赖弱引用语义"换成"不依赖"。项目的既有口径本来就是
「本层的义务是**不依赖弱引用语义**（平台缺陷不归本层修）」，本轮只是把这句话**落到代码上**。

### 5.2 真机验证（PPU，无任何注入；原始记录 `t11_20261014_out/after_*`）

| 载体 | 修前 | 修后 | 判据读数是否变化 |
|---|---|---|---|
| `ours_default_dropped`（本层最小复现） | 139 139 | **0 0** | — |
| `t4_errorloop` | 139 139 | **0 0** | ✅ 不变：`ERROR_RECOVERY_LOOP_PASS` 闭环 5 / 跳过 0 / 失败 0 |
| `t6_sched` | 139 139 | **0 0** | ✅ 不变：`STREAM_PRIORITY_SCHED_EFFECT_PASS`（`effect_observed=False`） |
| `vendor_weakref_many`（**不含本层**） | 139 139 | **139 139**（仍崩） | —— **本层不声称修好平台缺陷** |

**版本自证**：容器内 `runtime/backends/base.py` 的 `sha256 = 4dca3469…9641`，与本机**逐字相同**
（并在容器内 `grep -c '_ledger_put' = 6`、`grep -c 'cid is not None' = 3`）。

### 5.3 平台侧最小复现（交给厂商；**不含本层**）

```python
# 30 条原生流，每条挂一个弱引用后丢弃；解释器退出即 SIGSEGV（rc=139）
# 实测：N=30 ⇒ 2/2 崩；N=1 ⇒ 不崩；加不加 gc.collect() 都崩
import torch, weakref
keep = []
for _ in range(30):
    s = torch.cuda.Stream()
    keep.append(weakref.ref(s))
    del s
```
原生现场（§2.3）：`Py_FinalizeEx → _PyWeakref_ClearRef`、`si_code=1`、`si_addr=0xab`。
⇒ 诉求：厂商侧检查其流对象的 **`tp_dealloc` 是否调用 `PyObject_ClearWeakRefs`**（或等价机制），
使弱引用在对象回收时**被正确置死**。

---

## 6 判据（台账第 39 条）与**非空转**验证

`prototype/scripts/backend_offline_check.py` 新增 `[7b]` 段（四家通用路径，不依赖设备）：

| 判据 | 内容 | 本次结果（四家） |
|---|---|---|
| ① | 无本层上下文时**不登记** `_stream_ctx`（⇒ 不给厂商对象挂弱引用） | ✅ 四家 PASS |
| ② | `_released_streams` 的持有器是**强持有**（行为特征：外部引用消失后条目仍取回同一对象） | ✅ 三家 PASS（ascend 不声明 `stream_priority_control` ⇒ 如实跳过） |
| ③ | 按 `id()` 键的登记表有**条数上限**，且丢的是**最旧**条目 | ✅ 四家 PASS |

**非空转（反向实测）**：在 `runtime/backends/base.py` 的**临时副本**上把 ① 的守卫去掉、把 ② 改回弱引用，
用同一份判据跑 ⇒

```
[FAIL] ① 无本层上下文时不登记 `_stream_ctx` |_stream_ctx| 0 -> 1
[FAIL] ② 「已释放」记录的持有器是强持有  entry=True alive_after_del=False
离线自检结果: 116 通过 / 2 失败 / 1 跳过
```
⇒ 两条判据**都有牙齿**。修法版本：**92 / 114 / 103 / 118 通过、0 失败**
（修前 90 / 111 / 100 / 115；增量即本轮 ①②③ 与既有条目在新代码上的落点）。

---

## 7 剩余与边界（不许含糊）

| 项 | 状态 |
|---|---|
| **平台侧弱引用缺陷** | ❌ **未修**（本层无力）。最小复现见 §5.3，诉求见 §5.3 末 —— 需**厂商** |
| **本层是否仍受其影响** | ✅ **否**（三个载体真机 `rc=0`，判据读数不变）。⚠️ 但"平台缺陷仍存在"这一事实**必须一起读** |
| **T1 三家复跑** | ⬜ **必须做**：`backends/base.py` 是**共享层** ⇒ 910C / P800 / MLU590 各需一次完整 78 项职责审计（本轮唯一未闭环的破坏面项；910C 与 MLU590 网络在 10-14 上午实测为**不可达**，待恢复） |
| `rc=139` 是否还会出现在别处 | 未验证：本轮只覆盖 PPU 的四个载体（两条腿 / T4 / T6）。**不得**外推为"PPU 全部退出路径已干净" |

---

## 8 本轮踩坑（都属"看起来通过"家族，记下来）

| # | 坑 | 表现 | 处置 |
|---|---|---|---|
| 1 | `sitecustomize.py` 里 **bytes 字面量含中文** | 整份模块 `SyntaxError` ⇒ 注入**静默没生效**，而进程照样跑完（差点被读成"处理器已装但没触发"） | 一律 ASCII；并加"安装自证 + 回读校验" |
| 2 | 软链 `sitecustomize.py -> 宿主路径` | 容器内**断链** ⇒ `site.py` 静默跳过（第二次"注入没生效"） | `rm -f` 后 `cp`；并在**容器内**做存在性断言 |
| 3 | `cp -f` 覆盖软链 | 软链目标就是源文件本身 ⇒ 写回原文件、软链依旧 | 同上 |
| 4 | `sitecustomize` 里 `import runtime` | 解释器早期 `sys.path` 还没有原型根 ⇒ `T11_PATCH_FAILED`（易被读成"补丁无效"） | 显式插入 `DC_ROOT` |
| 5 | 宿主/容器**路径混用**（本轮第 **9** 次） | `--out` 传宿主路径 ⇒ 判据全过、落盘 `FileNotFoundError` ⇒ `rc=1`（把"崩"读成"没崩"） | 驱动脚本一律由 `O` 推 `HO` |
| 6 | `ln -sf` 与 `cp` 的混用顺序 | 见 2/3 | 同上 |
| 7 | `grep` 多模式 `\|` 静默 0 命中 | 判断"探针是否用本层建流"时差点漏掉关键对照 | 结论性命中一律换检索工具复核 |

---

## 9 复现命令与证据索引

```bash
# 0) 编译原生采集器（容器内，gcc 自带）
docker exec hliu553-dc-dev bash -lc \
  "cd /workspace/probes_ppu/dc_t11_20261014 && gcc -shared -fPIC -O1 -g -o t11_segv_handler.so t11_segv_handler.c -ldl"
# 1) 非空转自检：故意 SIGSEGV 必须被采集到（含模块/偏移）
docker exec -e LD_PRELOAD=/workspace/probes_ppu/dc_t11_20261014/t11_segv_handler.so \
  hliu553-dc-dev bash -lc "/usr/local/bin/python3 -c 'import ctypes; ctypes.string_at(0)'"
# 2) 最小复现矩阵（含"不含本层"的纯 torch 变体）
VARIANTS="native_plain ours_default_dropped vendor_weakref_many" ROUNDS=2 \
  bash -s < run_t11_ppu_20261014.sh          # 宿主运行（ai-server）
# 3) 暴露面二分（注入对照）
ROUNDS=2 PATCHES="control note_off strong" bash -s < run_t11_patch_matrix_20261014.sh
# 4) 修后真机验证
bash -s < run_t11_after_fix_20261014.sh
```

| 证据 | 文件 |
|---|---|
| 原生现场（三载体：**同一区域 + 同一故障签名，帧链不同**，见 §2.3 更正） | `t11_20261014_out/t4_native1.log` · `t4_native2.log` · `t6_native.log` |
| 采集器非空转自检 | `t11_20261014_out/native_selftest.log`（故意 SIGSEGV ⇒ 采到 RIP 落在 `libc`） |
| 最小复现矩阵 | `t11_20261014_out/log/*.log` · `*.rc` · `rc_matrix.log` |
| 同期阳性对照（载体确实会崩） | `t11_20261014_out/vehicle_t4.log` · `vehicle_t6.log` · `vehicle_rc.log` |
| 暴露面二分 | `t11_20261014_out/patch_matrix.log` · `patch__*_r*.log`（含 `T11_PATCH_LOADED` 自证） |
| 修后真机验证 | `t11_20261014_out/after_matrix.log` · `after_*.log` · `after_*.json` |
| 库卸载顺序旁证 | `t11_20261014_out/ldd3.47408` · `lddebug.47192`（`LD_DEBUG=libs`） |
| 脚本 | `../probes/t11_exit_segv_matrix.py` · `t11_segv_handler.c` · `t11_patch.py` · `run_t11_*.sh` · `check_t11_*.sh` |

---

## 10 自审复核（2026-10-14 晚）：本报告的四处问题与处置

> 起因：对「平台侧弱引用缺陷」这一**归属结论**做独立复核（"确保不是自己犯错"）。
> 方法：**不采信报告转述，直接读归档原始日志**。

| # | 问题 | 证据 | 处置 |
|---|---|---|---|
| **1** | **表述过强**：原文写三载体"落在**完全同一位置**" | 实为**同一区域 + 同一故障签名、帧链不同**：T4 `RIP_OFFSET=0x2569de` 经 `_PyModule_ClearDict`；T6 `0x256ad5` 经 `_PyWeakref_ClearRef`（差 `0x1F7` 字节） | ✅ **已更正 §2.3**（加帧链对照表） |
| **2** | **「不含本层」不精确** | `t11_exit_segv_matrix.py` **所有变体统一引导**：`import runtime` + `_bk = runtime.use("ppu")`；原始日志首行即 `backend=ppu`（**自证**）⇒ `vendor_weakref_many` 只是"**不调用本层流 API**"，不是"进程内无本层" | ⬜ **待补**：纯净复现脚本已备（`../probes/t11_attrib_verify_20261014.py`，末尾打印 `runtime_loaded=[...]` 自证），待联网跑 |
| **3** | **逻辑缺项**：矩阵没有「**N=30 原生流建完即丢、但不挂弱引用**」这一对照 | 现有对照只有「N=1 + 弱引用 + 持有 → 干净」与「N=30 + 弱引用 + 丢弃 → 崩」，**无法区分**触发因素是「弱引用悬垂」还是「大量流对象被回收」（后者与弱引用无关） | ⬜ **待补**：同脚本已加 4 组判别（无弱引用丢弃 / 弱引用但持有 / 弱引用但对象非流 / 弱引用但对象为张量）+ 剂量组 N=5/10 |
| **4** | ⭐ **证据完整性（最严重的一类）**：报告 §3 表格 15 个变体，**归档里只剩 3 个** | 驱动脚本原实现为**扁平 `log/` + `rm -rf log`、`rc_matrix.log` 用 `: >` 截断** ⇒ 复跑即销毁上一批：现存仅 `ours_default_dropped` / `vendor_weakref_many` / `vendor_weakref_many_gc`（各 2 轮），**其余 12 个变体（含 `native_plain`、`ours_wrapper`、`ours_prio_*`、`driver_*`）的原始日志已不可回读**；`N=1 不崩` 一行同属被覆盖批次 | ✅ **脚本已修**（按 `RUNID` 分目录 + 只追加）；⬜ **待重跑补档**（全 15 变体） |
| **4b** | 同类问题**不止本脚本**：普查得 6 个驱动脚本默认覆盖式落盘 | `run_t456_ppu_20261014.sh` / `run_t3_ppu_serve_20261014.sh` 用 **`rm -rf "$HO"`（删整个证据目录）**；`run_t11_patch_matrix` / `run_t11_after_fix` / MLU 两个 T7 脚本用 `: >` 截断 | ✅ **全部已修**（`rm -rf` → `RUNID` 子目录；`: >` → 只追加），9 个脚本 `bash -n` 通过 |

### 10.1 复核后**仍然成立**的部分（这些是决定性的，未受影响）

| 环节 | 为什么仍然成立 |
|---|---|
| **原生现场真实** | `LD_PRELOAD` C 采集器有**非空转自检**（故意 `SIGSEGV` 时 RIP 落在 `libc`，`native_selftest.log`）；三载体均采到 `si_code=1`/`si_addr=0xab`/libpython 区域 |
| **暴露面二分（决定性）** | 注入对照**轮次交替**且**每行带 `T11_PATCH_LOADED` 自证**：`control` 稳定 139 → `note_off` 救 min/T4 → `strong` 救全部（含 T6）⇒ 「本层挂的弱引用」与崩溃**因果相关**，这一条不依赖上述任何待补项 |
| **修后真机验证** | 无注入复跑：三载体 `rc=0`、`vendor_weakref_many` 仍 139（不声称修好平台）、判据读数逐字不变（`after_matrix.log`） |
| **归属方向** | 修后**平台侧形态仍崩** ⇒ 至少可确定「**不是本层能修好的东西**」；至于"弱引用是否唯一触发因素"，见问题 3（待补） |

### 10.2 待跑清单（联网即执行，判据先写死）

| 序 | 命令 | 通过/读法 |
|---|---|---|
| 1 | `VARIANTS="pure_baseline pure_stream_30_dropped pure_wref_30_dropped pure_wref_30_held pure_wref_30_plainobj_dropped" ROUNDS=2 bash -s < run_t11_attrib_verify_20261014.sh` | 全部行须带 `runtime_loaded=[]`（否则读数不能用于归属）；`pure_wref_30_dropped` 若 139 ⇒ 归属成立 |
| 2 | 同上，`pure_nowref_30_dropped` | 若**也** 139 ⇒ **§0 根因表述须改为「大量流对象被回收」这一更宽因素**（仍属平台侧，但"弱引用"不再是必要条件） |
| 3 | `VARIANTS="<全 15 变体>" ROUNDS=2 bash -s < run_t11_ppu_20261014.sh` | 补回 12 个缺失原始日志（新批次落在 `log/<runid>/`，历史批次不再被覆盖） |

