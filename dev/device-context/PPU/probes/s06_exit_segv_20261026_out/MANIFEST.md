# s06 退出期段错误取证 · 读数清单（PPU 真机，2026-10-26）

容器 `hliu553-dc-dev` · 卡 `ppu0`（`CUDA_VISIBLE_DEVICES=0`）· Python 3.12 · torch 2.10.0。
**退出码一律是「进程 rc」**，不是 `main()` 的返回值 —— 崩溃发生在 `main()` 返回**之后**的
解释器最终化阶段，所以日志末行常是 `main returned 0` 而进程 rc=139。

## A. 载体二分（`runner.py --cases ci_i<k>`，各 2 轮，不提前 gc）

| 变体 | rc（#1 / #2） | 判据 | 说明 |
|---|---|---|---|
| `i1_honest_declaration` | 0 / 0 | PASS | — |
| `i2_no_fabrication` | 0 / 0 | PASS | — |
| **`i3_invalidated_governed`** | **139 / 139** | PASS | **唯一能单独复现的用例** |
| `i4_degradation_observable` | 0 / 0 | PASS | — |
| 整组 `contract_invariants` | 139 / 139 | 4/4 PASS | 修前 |

原生现场（`s06_native1.log` / `s06_native2.log`，LD_PRELOAD 采集，2/2 同签名）：
`signal=11 si_code=128 si_addr=(nil)` · `_PyWeakref_ClearRef` ← `Py_FinalizeEx` ← `Py_Exit`。

## B. 注入对照（唯一变量 = 登记值持有方式，各 2 轮）

| 注入 | rc（#1 / #2） | i3 判据 | 结论 |
|---|---|---|---|
| `control`（不改） | 139 / 139 | PASS | 复现 |
| `note_off`（不登记） | 1 / 1 | **FAIL** | 免崩但**判据退化**（丢「销毁后使用其流」拦截） |
| `strong`（改强持有） | **0 / 0** | **PASS** | 既免崩又不退化 ⇒ 采用 |

## C. 纯净复现（进程内**完全不 import runtime**，脚本自证 `runtime_imported=False`）

| 变体 | 提前 `gc.collect()` | rc（#1 / #2） | 说明 |
|---|---|---|---|
| N=30 | 是 | 0 / 0 | ⚠️ **提前回收会把缺陷掩盖掉** |
| N=1 | 是 | 0 / 0 | 同上 |
| **N=30** | **否** | **139 / 139** | 平台侧可独立复现 |
| **N=1** | **否** | **139 / 139** | 连单条也崩（与 T11 的「需累计多条」不同，差在回收时机） |

纯净复现原生签名（`pure_nogc_native.log`）：`si_addr=(nil)` · `_PyModule_ClearDict`（模块清理阶段）
⇒ 与 A 组**同类不同栈**，按「不得外推」纪律分别记录。

## C2. 触发因素判别（纯 torch，`runtime_imported=False`，各 2 轮，均**不提前 gc**）

把「弱引用悬垂」与「大量对象被回收」分开 —— 三个必要条件必须**同时**成立才崩：

| 变体 | 对象 | 挂弱引用 | 保留强引用 | N | rc（#1 / #2） |
|---|---|---|---|---|---|
| `no_wr_drop` | Stream | 否 | 否 | 30 | 0 / 0 |
| `wr_hold` | Stream | 是 | **是** | 30 | 0 / 0 |
| `wr_tensor` | **Tensor** | 是 | 否 | 30 | 0 / 0 |
| **`wr_stream`** | Stream | 是 | 否 | 30 | **139 / 139** |
| **`wr_stream`** | Stream | 是 | 否 | 10 | **139 / 139** |
| **`wr_stream`** | Stream | 是 | 否 | 5 | **139 / 139** |

⇒ 必要条件＝**① 对象是 `torch.cuda.Stream` ② 其上挂了弱引用 ③ 该对象被回收**；
任一不满足即 `rc=0`。这排除了「对象数量大」「任意对象挂弱引用」「弱引用本身」三种替代解释。

## D. 修后回归

| 变体 | rc（#1 / #2） | 判据 |
|---|---|---|
| `ci_i3` 单用例 | 0 / 0 | PASS（五项断言全中） |
| 整组 `contract_invariants` | 0 / 0 | 4/4 PASS |
| 13 段全量（`--only` 全跑，run-id `20261026_final`） | 13/13 段 rc=0 | overall PASS |

## 文件索引

- `s06_native{1,2}.log` 修前原生现场 · `pure_nogc_native.log` 纯净复现原生现场
- `only_i{1..4}.r{1,2}.{log,json}` 载体二分 · `ci_i{1..4}.py` 单用例临时模块
- `inj_{control,note_off,strong}{1,2}.log` 注入对照
- `fix_i3_{1,2}.log` / `fix_all_{1,2}.log` 修后回归
- `pure_n30_r{1,2}.log`（提前 gc）/ `pure_nogc_n{30,1}_r{1,2}.log`（不提前 gc）
- `trig_{torch_only,ctx_then_del,ctx_keep}_r{1,2}.log` 触发条件收窄尝试（**均 rc=0，未复现**，如实留档）
- `ppu_*.py` 上述四类探针源码 · `t11_segv_handler.c` 原生采集器源码（`.so` 不in仓，用时现编）
