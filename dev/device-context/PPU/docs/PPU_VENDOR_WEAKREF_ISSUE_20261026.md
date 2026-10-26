# 【问题反馈】PPU 栈：`torch.cuda.Stream` 的 `weakref` 在解释器退出期触发段错误（可最小复现）

> 提交方：FlagOS 运行时层（device-context 方向） ｜ 日期：**2026-10-26** ｜ 状态：**平台侧复现，我方已自保**
> 我方内部取证全文：`PPU_EXIT_SEGV_CARRIER5_20261026.md` · `PPU_T11_EXIT_SEGV_ROOTCAUSE_20261014.md`

## 一、一句话

在 PPU 栈上，**给 `torch.cuda.Stream` 对象挂 `weakref`、随后丢弃该流**，进程正常退出时会在
**解释器最终化阶段**发生 `SIGSEGV`（`rc=139`）。判据/业务代码此时已全部执行成功。
**该复现不使用我方框架的任何代码**，因此可直接作为平台侧问题定位。

## 二、环境

| 项 | 值 |
|---|---|
| 芯片 | PPU-ZW810E（`ppu0`，显存 98304 MiB） |
| torch | **2.10.0**（`torch.cuda` 兼容栈，`torch.xpu.device_count()==0`） |
| Python | **3.12**（`/usr/local/lib/libpython3.12.so.1.0`） |
| 容器 | 镜像内 `CUDA_VISIBLE_DEVICES=0` 单卡；`NCCL_SOCKET_IFNAME=bond0`；`--shm-size 32g` |
| 复现率 | **4/4**（N=30 / N=1 各 2 轮，独立进程） |

## 三、最小复现（不含我方框架）

```python
# pure_repro.py  —— 运行： CUDA_VISIBLE_DEVICES=0 python3 -u pure_repro.py 30
import sys, weakref
N = int(sys.argv[1])
import torch

refs = []
for _ in range(N):
    s = torch.cuda.Stream()          # 原生流
    refs.append(weakref.ref(s))      # 只挂一个弱引用
    del s                            # 立刻丢弃强引用

# ⚠️ 关键：这里**不要**调用 gc.collect()（调用后缺陷会被掩盖，见第五节）
print("reached end, normal exit ...", flush=True)
```

运行结果：

| N | rc（#1 / #2） | 现象 |
|---|---|---|
| 30 | **139 / 139** | 打印完 `reached end, normal exit ...` 后在退出阶段段错误 |
| 1 | **139 / 139** | 同上（单条流即可） |

## 四、崩溃现场（我方用 `LD_PRELOAD` 采集，容器内无 gdb）

```
signal=11 si_code=128 si_addr=(nil)
RIP_MODULE=/usr/local/lib/libpython3.12.so.1.0   RIP_OFFSET=0x2569de
  FRAME#3 libpython3.12
  FRAME#4 libpython3.12
  FRAME#5 libpython3.12 _PyModule_ClearDict        ← 模块清理阶段
```

**判读**：崩溃发生在 CPython **清理对象/模块**、顺带清理**弱引用**的阶段，解引用的是 **NULL**。
指向的机制是：**PPU 侧对象在 `torch.cuda.Stream` 的 Python 包装被回收后，其 `weakref` 书签不可信**
（书签未随对象死亡置死，或指向已释放内存）。

### 四.1 我们已把「替代解释」逐一排除（判别实验，各 2 轮，均不含我方框架）

| 变体 | 对象 | 挂弱引用 | 保留强引用 | N | rc（#1 / #2） |
|---|---|---|---|---|---|
| `no_wr_drop` | Stream | 否 | 否 | 30 | 0 / 0 |
| `wr_hold` | Stream | 是 | **是** | 30 | 0 / 0 |
| `wr_tensor` | **Tensor** | 是 | 否 | 30 | 0 / 0 |
| **`wr_stream`** | Stream | 是 | 否 | 30 | **139 / 139** |
| **`wr_stream`** | Stream | 是 | 否 | 10 | **139 / 139** |
| **`wr_stream`** | Stream | 是 | 否 | 5 | **139 / 139** |

**读法**：三个条件必须**同时**满足——① 对象是 `torch.cuda.Stream`；② 其上挂了 `weakref`；③ 该对象被回收。
任一不满足即 `rc=0`。⇒ 可排除「对象数量大」「任意对象挂弱引用」「弱引用本身有副作用」三种替代解释，
**问题收敛到 Stream 包装对象的弱引用书签管理**。具体内部机制（书签未置死 / 指向已释放内存）仍需厂商确认。

对照：我方另一条更复杂的路径（创建/销毁上下文后再使用流）也会崩，但栈不同
（`_PyWeakref_ClearRef` ← `Py_FinalizeEx`，`si_addr=(nil)`）。两者我们**分开记录**，不合并成一个签名。

## 五、⚠️ 复现注意事项（我们踩过，避免贵方复现失败）

- **不要在退出前手动 `gc.collect()`**：一旦提前回收，最终化阶段已无弱引用可清理 ⇒ 表现为 `rc=0`，
  缺陷完全隐身。我们第一轮就是这样"复现失败"的，去掉那行后立刻 4/4 复现。
- 复现必须**观察进程退出码**，不能只看日志（`main()` 的返回值是 0，崩溃在其后）。
- 建议用 `python3 -X faulthandler`，可直接看到 `Garbage-collecting` / `<no Python frame>`。

## 六、想请贵方确认的三点

1. **`torch.cuda.Stream`（及 `Event`）的 Python 包装对象被回收时，平台是否会正确清理挂在它上面的
   `weakref` 书签？** 我们的现场指向"不会/会解引用空指针"。
2. **由 `cuStreamCreateWithPriority` 等 C API 创建、并搭配 `cuCtxCreate/cuCtxDestroy` 的流**，
   生命周期是否与纯 `torch.cuda.Stream()` 一致？若不一致，请给出**受支持的销毁顺序**
   （我们的载体正是"销毁上下文后再使用该流"，签名与前一种不同）。
3. 是否存在**平台推荐的"退出期不触达设备"约定**（例如建议在 `atexit` 早期停止持有流对象）？
   若有，我们按要求调整。

## 七、我方已做的规避（供参考，不影响本次反馈）

- 运行时层**不再对任何平台对象建弱引用**（三张登记表全部改强持有），
  并加了容量上限；改后该载体从 `rc=139` 变为 `rc=0`，**判据读数逐字未变**。
- 这是**用方自保**，不是修复平台：我们不要求贵方先修好才能验收，
  但希望在后续版本中消除该退出期崩溃，因为它会让**任何挂弱引用的下游**（含 `torch.compile`、
  多进程 worker 清理、profiler）在正常退出时崩溃。

## 八、附件

| 文件 | 说明 |
|---|---|
| `ppu_pure_weakref_repro.py` / `ppu_pure_nogc.py` | 本报告第三节的复现脚本（后者即"不提前回收"版） |
| `ppu_weakref_discriminate.py` / `disc_*_n*_r*.log` | 第四节判别实验的脚本与 12 份原始输出 |
| `pure_nogc_n{30,1}_r{1,2}.log` | 4 次复现的原始输出（含 `runtime_imported=False` 自证） |
| `pure_nogc_native.log` | 原生崩溃现场（第四节原文） |
| `t11_segv_handler.c` | 我们的 `LD_PRELOAD` 段错误采集器源码（可自行编译复采） |

（附件均位于 `dev/device-context/PPU/probes/s06_exit_segv_20261026_out/`）
