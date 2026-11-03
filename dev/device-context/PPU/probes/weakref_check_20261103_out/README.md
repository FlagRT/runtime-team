# PPU 退出期 weakref UAF · 运行时等价补丁（2026-11-03）

> 起因：`../docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md`（平台侧缺陷，我方已自保）+ 用户要求
> 「参考 910C 的修复方法在 PPU 上修」。910C 侧复核见 `../../910C/docs/ASCEND_WEAKREF_EXIT_UAF_20261103.md`。

## 0. 结论先行

| 项 | 结果 |
|---|---|
| 缺陷在 PPU（v2 镜像，torch 2.10.0）是否仍存在 | ✅ **存在**：`Stream+weakref+被回收` **rc=139/139**；Stream/Event 的 weakref **回调均不触发**；三条件矩阵与 910C 逐条同构 |
| 厂商是否已有修复版本 | ❌ **未找到**（容器内 torch 2.10.0 为厂商构建，无更新 wheel 可取；不像 torch_npu 有 `post7` 这类补丁版） |
| 能否在 PPU 上"修" | ⚠️ **不能改厂商 libtorch**（无源码），但可以**运行时等价补丁**：给类型对象换 `tp_dealloc` 包装，语义与 PR #41596 相同 |
| 补丁是否有效 | ✅ **P1 回调触发（False→True）· P2 rc 139→0（2/2）· P3 功能形态不变 · P4 我方验收读数零变化** |
| 补丁定位 | **止血**，不是厂商修复。验收期间可挂、随时撤；正式修复仍应由厂商把两行编进 libtorch |

## 1. 修法对照（为什么这样写就等价）

| | 厂商正确修法（Ascend PR #41596，torch_npu `2.10.0.post7` 已验证） | 本补丁（PPU，运行时） |
|---|---|---|
| 位置 | `THNPStream_dealloc` / `THNPEvent_dealloc`（C++，编译期） | 同两个类型的 `tp_dealloc` 槽（C 扩展，**import 期**改写） |
| 动作 | `tp_free` 前调 `PyObject_ClearWeakRefs((PyObject*)self)`；Stream 另补 `Py_CLEAR(self->context)` | 包装函数：先 `PyObject_ClearWeakRefs(self)`，再**回跳原 dealloc** |
| 效果 | 对象内存释放前把 weakref 书签清干净 ⇒ 退出期不再解引用悬空指针 | 同左 |
| 幂等性 | — | 若将来厂商修好，原实现自己也会清（`ClearWeakRefs` 对已清对象是空操作）⇒ 补丁可叠加不需版本判断 |

⚠️ **已知副作用（已登记）**：类型对象若落在只读页，补丁会 `mprotect` 成可写且**不还原**
（页面可能同时承载其他只读常量，还原风险更大）。仅进程内生效、进程退出即消失。

## 2. 判据与读数

判据文件名即证据（`out*` 子目录，rc 一律指**进程退出码**）：

| 判据 | 内容 | 修前 | 修后 | 证据 |
|---|---|---|---|---|
| **P1** | Stream/Event 的 weakref **回调是否触发**（唯一可靠信号） | `False` / `False` | **`True` / `True`** | `out/cb_stream.log` vs `out_wrfix/p1p3.log` |
| **P2** | `Stream + weakref + 被回收` 的进程 rc | **139 / 139** | **0 / 0** | `out/wr_stream_r{1,2}.rc` vs `out_wrfix/p2_r{1,2}.rc` |
| **P3** | 功能形态不变：流可用 / 事件计时可用 / 张量正确 | — | `sum=196608` 正确 · `elapsed_ms=56.800` 在带内 · `SANITY verdict=PASS` | `out_wrfix/p1p3.log` |
| **P4** | 我方验收读数零变化（补丁臂 vs 对照臂，其余完全相同） | 归档基线 | **两臂完全一致**：离线 **118/0/1** · smoke **46/0** · 契约不变式 **4/4 `CONTRACT_INVARIANTS_PASS`**、**rc=0** | `out_p4/{offline,smoke,inv}_{fix,ctl}.{log,rc}` |

判据有效性对照（防"判据自己撒谎"）：普通 Python 对象 weakref 回调 **触发**（`out/cb_ctrl.log`）⇒ 排除脚本问题；
⚠️ `ref()() is None` 在**修前修后都是 None**，不能当判据（与 910C 侧结论一致）。


## 2.1 四家矩阵（同判据，2026-11-03，供群内求证直接引用）

| 实例 | torch / 插件 | Stream weakref 回调 | rc | 判定 |
|---|---|---|---|---|
| **PPU** | 2.10.0（厂商兼容栈） | 不触发 | **139/139** | 🔴 中招（本目录补丁已止血） |
| **910C** | 2.10.0 + torch_npu 2.10.0 | 不触发 | **139/139** | 🔴 中招（`torch_npu 2.10.0.post7` 已修，A/B 实证） |
| **P800** | 2.9.0+cu129（昆仑芯） | **触发** | 0 | 🟢 未中招 |
| **MLU590** | 2.7.1 + torch_mlu 1.29.2 | — | — | ⚪ 未验证（2026-11-03 网络不可达；预判不适用，但未实测不声明） |

## 3. 复现与使用

```bash
# 判据本体（宿主上跑，脚本自己 docker exec）
bash run.sh              # 缺陷态复核（回调 + rc 矩阵）
bash run_ppu_wrfix.sh    # 编译补丁 + P1/P2/P3
bash run_p4.sh           # P4 双臂零回归（注入式，不改仓库代码）

# 手工挂补丁（验收/调试时）
gcc -O2 -shared -fPIC $(python3-config --includes) -o wrfix$(python3-config --extension-suffix) wrfix.c
DCFIX=1 PYTHONPATH=<本目录> python3 <被测脚本>     # 需同目录有 sitecustomize.py
```

## 4. 文件

| 文件 | 内容 |
|---|---|
| `wrfix.c` | 补丁源码（约 150 行，含实现说明与副作用登记） |
| `wrfix.cpython-312-x86_64-linux-gnu.so` | 编译产物（PPU 容器 py3.12 实测用） |
| `sitecustomize.py` | P4 注入式对照开关（`DCFIX=1` 打补丁 / `0` 对照臂，均打印加载自证） |
| `wrfix_after.py` / `wrfix_rc.py` | 补丁后判据（P1+P3 / P2） |
| `ppu_wr_check.py` / `ppu_wr_callback.py` / `ppu_cb_ctrl.py` | 缺陷态复核（与 910C 同判据） |
| `run.sh` / `run_ppu_wrfix.sh` / `run_p4.sh` | 三段驱动（`WH` 宿主视角 / `WC` 容器视角，避免路径混用） |
| `out/` · `out_wrfix/` · `out_p4/` | 全部原始日志与 rc |

## 5. 结论边界

- 本补丁**不改变**"平台侧缺陷未修"的事实陈述；它只证明**修法有效且对上层零影响**，可作为给厂商的**可执行诉求**。
- 我方 runtime 层**不需要**这个补丁（三张登记表已改强持有，不依赖弱引用语义）；补丁面向的是
  **任何挂弱引用的下游**（`torch.compile`、profiler、多进程 worker 清理等）。
- 补丁**不进镜像、不进 runtime 包**：属诊断/止血工具，放 `probes/`。若将来需要长期启用，
  须先过总组裁定并写进 env 版本，不在本方向自行默认加载。
