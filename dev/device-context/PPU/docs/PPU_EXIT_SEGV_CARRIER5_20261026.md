# PPU 退出期段错误 · 第五个载体（契约不变式 I3）的定位与收口

> 分支：`kistich/device-context` ｜ 实测：**2026-10-26** ｜ 机器：PPU-2（`ai-server`）容器 `hliu553-dc-dev`，卡 `ppu0`
> 证据目录：`../probes/s06_exit_segv_20261026_out/`（含自包含读数清单 `MANIFEST.md`）
> 前置：`PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`（第一至第四个载体的根因与修法）

## 0 结论先行

| 项 | 结论 |
|---|---|
| **发生了什么** | `s06` 契约不变式段：判据 `CONTRACT_INVARIANTS_PASS 4/4`，但进程退出码 **`rc=139`**（段错误）。崩溃点在 `main()` **返回之后**的解释器最终化阶段 |
| **哪个用例** | **只有 `i3_invalidated_governed`**（二次释放 / 二次销毁 / 销毁后使用其流）。i1 / i2 / i4 单独各 2 轮均 `rc=0` |
| **原生根因** | `Py_FinalizeEx → _PyWeakref_ClearRef` 解引用 **NULL**（`si_addr=(nil)`）⇒ 平台侧**弱引用书签**在对象回收后不可信（与 T11 同类机制，但**栈不同**：T11 是 `si_addr=0xab`） |
| **本层的暴露面** | `_stream_ctx` —— **本层最后一处**给厂商流对象挂 `weakref` 的登记表（T11 当时判断它"不在暴露面上"，**该判断已被本轮否证**） |
| **修法** | `_stream_ctx` 的持有器改**强持有**（与 `_owned_streams` / `_released_streams` 一致）⇒ **本层不再对平台对象建任何弱引用** |
| **验证** | 单用例 `rc=0`×2 · 整组 `rc=0`×2（4/4 PASS）· **13 段全量 13/13 段 `rc=0`、overall PASS**，判据读数与修前逐字相同 |
| **平台侧归因** | ✅ **独立成立**：进程内**完全不 import 本层**的纯 torch 复现同样 `rc=139`（4/4）。⚠️ 但**必须关掉提前 `gc.collect()`**，否则会被掩盖（见 §7.2） |

## 1 现象与最小复现

复现命令（容器内，cwd = `prototype`）：

```bash
DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=0 \
  python3 -u runtime/conformance/runner.py --backend ppu \
          --cases contract_invariants --out /tmp/s06.json ; echo rc=$?
```

实测 2/2：`=== 汇总: 4/4 通过 ===` / `结论: CONTRACT_INVARIANTS_PASS` / `结果已写入 …` 之后，
进程 `rc=139`。打开 faulthandler（`python3 -X faulthandler` 或 `PYTHONFAULTHANDLER=1`）后打印：

```
Fatal Python error: Segmentation fault
Current thread 0x00007f56fa8b1b80 (most recent call first):
  Garbage-collecting
  <no Python frame>
```

**这两行就把范围收死了**：不是判据跑挂了（4/4 全过），而是**回收/最终化阶段**炸的，
且没有任何 Python 帧 ⇒ 只能是原生侧（finalizer 或弱引用清理）。

## 2 原生现场：没有 gdb 也能拿到崩溃点

容器内**无 gdb**（`gdb` 缺），沿用 T11 的 `LD_PRELOAD` + `dladdr` 采集器
（`../probes/t11_segv_handler.c`，用时现编）。采集结果 2/2 **同签名**：

```
signal=11 si_code=128 si_addr=(nil)
RIP_MODULE=/usr/local/lib/libpython3.12.so.1.0   RIP_OFFSET=0x256ad5
  FRAME#2 libpython3.12 0x256ad5
  FRAME#3 libpython3.12 _PyWeakref_ClearRef
  FRAME#4..#6 libpython3.12
  FRAME#7 libpython3.12 Py_FinalizeEx
  FRAME#8 libpython3.12 Py_Exit
```

**读法**：崩溃在**清理弱引用**这条路径上，解引用的是 **NULL**。
⚠️ 与 T11 三个载体的 `si_addr=0xab`（被复用的垃圾内存）**不是同一个签名**
—— 按 T11 §1 的纪律「**不得外推**」，本轮把它记为**第五个载体、独立记录**，不并入前四个。

## 3 载体二分：只有 i3 能单独复现

把四个用例各切成一个独立模块（`ci_i<k>.py`）分别跑，各 2 轮、独立进程：

| 用例 | rc（#1 / #2） | 判据 |
|---|---|---|
| `i1_honest_declaration` | 0 / 0 | PASS |
| `i2_no_fabrication` | 0 / 0 | PASS |
| **`i3_invalidated_governed`** | **139 / 139** | PASS |
| `i4_degradation_observable` | 0 / 0 | PASS |

`i3` 与另外三个的唯一区别：它**显式走上下文生命周期**——
`context_create` → 在该上下文下 `create_stream` → `note_stream_created` → `context_destroy`
→ `context_destroy` 再来一次 → `check_stream_usable(该流)`。
即：**它把「厂商流对象」和「本层上下文登记」同时制造出来，并在同一轮里销毁上下文。**

## 4 暴露面与注入对照（决定性证据）

本层对厂商对象的登记表共三张，T11 已把两张改强持有，只剩 `_stream_ctx` 仍用 `weakref`
（当时的判断是"每次覆盖写会自愈，不在暴露面上"）。本轮用**注入对照**验证这个判断：

| 注入 | rc（#1 / #2） | i3 判据 | 读法 |
|---|---|---|---|
| `control`（不改行为） | 139 / 139 | PASS | 缺陷稳定复现 |
| `note_off`（`note_stream_created` 变 no-op） | 1 / 1 | **FAIL** | 免崩，但**判据退化** —— 丢了「销毁上下文后使用其流必须报错」这条拦截 ⇒ 不能这么修 |
| `strong`（持有器改强持有） | **0 / 0** | **PASS**（五项断言全中） | **既要免崩又要判据不退化，只有这一条路** |

⇒ **问题是「弱引用」这一形态，不是"登记"这件事本身。** 这也解释了为什么
`note_off` 能免崩却让判据从 PASS 掉到 FAIL：登记承担着真实语义。

## 5 修法（本层）与代价

`prototype/runtime/backends/base.py` 的 `note_stream_created`：

```python
if cid is not None:                              # 无本层上下文 ⇒ 不登记（T11 的 ①，保留）
    self._ledger_put("_stream_ctx", id(native_stream),
                     (cid, (lambda _o=native_stream: _o)))   # ⭐ 强持有（本轮改）
```

- **改的是什么**：登记值的持有器从 `weakref.ref` 改为强引用 lambda。至此
  `_owned_streams` / `_released_streams` / `_stream_ctx` 三张表**全部强持有**，
  本层不再对任何平台对象暴露弱引用。
- **为什么不是"少登记"**：见 §4 —— 少登记会让判据退化（`note_off` 组 FAIL）。
- **代价（如实）**：条目会 **pin 住 Python 包装对象**（设备级流不受影响）。
  与 `_ledger_put` 的 `_LEDGER_MAX = 4096` 容量上限配套，长驻服务不会无界增长；
  且 `release_stream` 会摘除对应条目。这与 T11 对 `_released_streams` 的取舍完全同构。
- **同时更正了 T11 的一处旧判断**（原文写"`_stream_ctx` 不在本类缺陷的暴露面上"），
  已在代码注释里改为"该判断已被 2026-10-26 实测否证"。

## 6 修后回归

| 项 | 修前 | 修后 |
|---|---|---|
| `ci_i3` 单用例 | 139 / 139 | **0 / 0**（PASS，五项断言全中） |
| 整组 `contract_invariants` | 139 / 139 | **0 / 0**（`CONTRACT_INVARIANTS_PASS 4/4`） |
| **13 段全量**（run-id `20261026_final`） | s06 段 `rc=139`，overall FAIL | **13/13 段 `rc=0`，overall PASS** |

13 段全量读数（与 10-14 基线对照，判据逐字未变）：

- 离线 `118/0/1` · 对称性 `7/0` · smoke `46/0` · conformance `13/13 + 6/6`
- 契约不变式 **`4/4` 且 `rc=0`** · 训练腿 `6/6`（loss `15.4498→11.1530`，两卡合计 ≈5.07k tok/s）
- 推理腿 `13/13`（区分度 **0.6391** 逐字同）· 服务化 `10/10`（区分度 **0.4118** 逐字同，106.65 句/s）
- 错误闭环 `5/0/0` · 多流 `8/8` · 图捕获 `4/4` · 配额 `3/3` · 调度 `PASS`+`effect_observed=False`
- 职责审计 **`73/0/5`**

## 7 对 T11 结论的两处更正（都是"看起来一样"的坑）

### 7.1 「`_stream_ctx` 不在暴露面上」→ 已否证

原文理由有两条：① 每次 `note_stream_created` 覆盖写、陈旧条目自愈；② 其判据是"上下文是否已销毁"。
但**崩溃发生时不走判据**，只走"清理弱引用"——覆盖写与否无关。⇒ 已按 §4 证据改为强持有。

### 7.2 ⚠️ **提前 `gc.collect()` 会把该缺陷整个掩盖掉**

本轮先跑的纯净复现（进程内**完全不 import 本层**）在退出前显式 `gc.collect()`，
结果 **N=30 与 N=1 各 2 轮全部 `rc=0`** —— 一度让人以为"纯平台侧无法复现"。
把这一行去掉后立刻翻案：

| 纯净复现（无本层） | 提前 `gc.collect()` | rc（#1 / #2） |
|---|---|---|
| N=30 | 是 | 0 / 0 |
| N=1 | 是 | 0 / 0 |
| **N=30** | **否** | **139 / 139** |
| **N=1** | **否** | **139 / 139** |

原生签名（`pure_nogc_native.log`）：`si_addr=(nil)` · `_PyModule_ClearDict`（模块清理阶段）。

**两条结论**：
1. **平台侧归因独立成立**——进程里根本没有本层代码，纯 `torch.cuda.Stream()` + `weakref` 即可崩
   ⇒ 厂商侧确有可复现问题（材料见 `PPU_VENDOR_WEAKREF_ISSUE_20261026.md`）。
2. **复现纪律**：凡"退出期段错误"的复现，**不得在退出前手动 `gc.collect()`**；
   否则对象被提前回收，最终化阶段无弱引用可清，缺陷隐身。
   T11 记录的"`N=1` 不崩、需累计多条"很可能也受回收时机影响（本轮不提前回收时 **N=1 也崩**）。

## 8 边界（不要读过头）

- **只证明 PPU 这一台/这一栈**。`kunlun` / `cambricon` / `ascend` 三家**未实测**；
  本轮改的是**共享层**（`base.py`），故三家下次复跑须按本报告 §6 的清单核对退出码。
- **平台侧缺陷仍归厂商**：本层的修法是"不依赖弱引用语义"的自保，**不是修平台**；
  对外**不得**写成"平台已修"。
- **不承诺消除所有退出期段错误**：本轮只覆盖第五个载体（契约不变式），
  与 T11 一样**不得外推**为"全部退出路径已干净"。
- §7.2 的纯净复现只在 `ppu0` 单卡、Python 3.12、torch 2.10.0 下验证过。

## 9 复现命令（全部可在容器内直跑）

```bash
# 0) 原生现场采集器（无 gdb 时）
gcc -shared -fPIC -O1 -g -o /tmp/t11_segv_handler.so \
    /workspace/runtime-team/dev/device-context/PPU/probes/t11_segv_handler.c -ldl

# 1) 载体（修前应 139 / 修后应 0）
DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=0 python3 -X faulthandler -u \
  runtime/conformance/runner.py --backend ppu --cases contract_invariants --out /tmp/s06.json

# 2) 单用例二分（需 PYTHONPATH 指向含 ci_i3.py 的目录）
DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<证据目录> python3 -u \
  runtime/conformance/runner.py --backend ppu --cases ci_i3 --out /tmp/i3.json

# 3) 三种注入（control / note_off / strong）
DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<证据目录> \
  INJ=strong python3 -u <证据目录>/ppu_i3_injection.py ; echo rc=$?

# 4) 纯净复现（⚠️ 不得提前 gc.collect()）
CUDA_VISIBLE_DEVICES=0 python3 -u <证据目录>/ppu_pure_nogc.py 30 ; echo rc=$?

# 5) 单入口全量
cd ../probes && RUNID=$(date +%Y%m%d_%H%M%S) bash run_acceptance_ppu.sh
```
