# 昇腾侧 weakened 退出期 UAF 复核：怎么修的、修没修成、对 PPU 意味着什么（2026-11-03）

> 触发：用户要求核对「昇腾 torch_npu 在 torch 2.10 同组合踩过同坑并已修」是否成立，
> 以便把修法搬到 PPU 的同缺陷（`../PPU/docs/PPU_VENDOR_WEAKREF_ISSUE_20261026.md`）。
> 逐条读数与原始日志：`../probes/weakref_check_20261103_out/MANIFEST.md`

## 0. 三句话

1. **成立**：910C 本机 `torch 2.10.0 + torch_npu 2.10.0` **一样崩**（`rc=139`，与 PPU 同形态）——
   这不是 PPU 独有，是 **PyTorch 2.8 起「父类加 weakref 支持、后端子类 tp_dealloc 没跟上」的生态共性缺陷**。
2. **修法只有两行 C++**（Ascend/**PR #41596**）：在 `THNPStream_dealloc` / `THNPEvent_dealloc` 里、
   `tp_free` 之前调 `PyObject_ClearWeakRefs((PyObject*)self)`；Stream 另补 `Py_CLEAR(self->context)`。
3. **修成了**：同一个 torch 2.10.0，只把 `torch_npu` 从 `2.10.0` 换成 **`2.10.0.post7`** ⇒
   `rc 139 → 0`、Stream/Event 的 **weakref 回调从「不触发」变「触发」**（A/B 单变量）。

## 1. 缺陷机制（PR 原文，一句话说清）

PyTorch 2.8 给 `THPStream`/`THPEvent` 父类加了 `weakreflist` 字段与 weakref 支持。
各硬件后端（CUDA/XPU/NPU/兼容层）的**子类覆写了 `tp_dealloc()`**，但 CPython 对**静态类型不会链式调用父类 `tp_dealloc`**
⇒ 父类的 `PyObject_ClearWeakRefs()` 永远不执行 ⇒ 对象死了、弱引用书签还在 → `wr_object` 悬空。
进程退出、解释器最终化（`_PyModule_ClearDict` / `_PyWeakref_ClearRef`）去清弱引用时解引用悬空指针 ⇒ **SIGSEGV**。

**关键推论**：这类缺陷**与算力芯片无关，只看"后端子类有没有补那一行"** ⇒
凡是自研/兼容 torch 的设备后端，都可能各自踩一遍。PPU 兼容层即属"没补的那一批"。

## 2. A/B 读数（本机实测，唯一变量＝torch_npu 版本）

| 臂 | torch | torch_npu | `ver_check.py` rc | Stream 回调 | Event 回调 |
|---|---|---|---|---|---|
| A | 2.10.0+cpu | **2.10.0** | **139** | **False** | **False** |
| B | 2.10.0+cpu | **2.10.0.post7** | **0** | **True** | **True** |

判别矩阵（A 臂，各 2 轮）：`wr_stream` **139/139**；`no_wr_drop` 0/0 · `wr_hold` 0/0 · `wr_tensor` 0/0
⇒ 三必要条件（Stream + weakref + 被回收）与 PPU 侧**逐条同构**。

**判据有效性**（防"判据自己撒谎"）：
- 普通 Python 对象挂 weakref + `gc.collect()` ⇒ 回调**触发**（机制正常，排除脚本问题）；
- ⚠️ `ref()() is None` 在**两臂都是 None** ⇒ 印证 Ascend 回归测试注释：
  **该写法在缺陷存在时也返回 None，不能当判据**；唯一可靠信号是**回调是否触发**。
  （我方 PPU 材料 §四.1 用的正是"rc 三必要条件"矩阵，结论不受影响；但后续判据统一升级为"回调触发"。）

## 3. 对 PPU 的可移植性（下一轮要做的）

| 事项 | 判断 |
|---|---|
| 修在哪一层 | **后端插件的 C++（`tp_dealloc`）**，不是 torch 主版本、不是我们的 runtime 层 |
| 我们能直接改吗 | ❌ 不能：PPU 的 `libtorch*` 是厂商预编译产物，无源码 |
| 我们已做的自保 | ✅ runtime 层三张登记表**全部强持有**（不依赖弱引用语义）⇒ 本层不再触发；判据读数不变 |
| 可做的等价"修复"（待 PPU 网络恢复后验证） | ① **查 PPU 栈是否已有新版含修复**（对照 torch_npu 的 post 分支做法）；② 若没有，做**运行时等价补丁**：小 C 扩展在 import 期给 Stream/Event 类型补上 `tp_dealloc` 包装（先 `PyObject_ClearWeakRefs` 再原函数），判据＝回调触发 + `rc=0` + 我方验收读数零变化；③ 同时把本 A/B 作为**可执行诉求**附进厂商材料 |
| ⚠️ 边界 | ② 属**止血**（非厂商修复），须在文档与代码里写明"待厂商修，本补丁可随时撤"；且必须先确认 PPU 类型对象可写（静态类型可能已 frozen，需 `mprotect` 解除只读）|

## 4. 顺带修正的环境事实（已写进 MANIFEST §E）

宿主机 10-24 重建设备节点后，**非 `--privileged` 容器拿不到 910C 设备**
（`acl.init rc=500000`、`device_count=0`），初始报错形态酷似"卡坏了"。
**只挂自己那两张 die + `--privileged` 即正常**（`device_count=16`）⇒ 带卡名额口径不变，改的是容器参数。
本轮已据此重建容器 `dc-probe-priv`；旧 `dc-rerun-1027-910c` 已删除。

## 5. 四家实例矩阵（同判据、同脚本，2026-11-03）

判据统一为：**① Stream 的 weakref 回调是否触发 ② 「Stream+weakref+被回收」进程 rc**。

| 实例 | torch / 后端插件 | Stream 回调 | rc | 结论 |
|---|---|---|---|---|
| **910C** | 2.10.0 + torch_npu **2.10.0** | **不触发** | **139 / 139** | 🔴 **中招**（升级到 `2.10.0.post7` 即修，见 §2） |
| **PPU** | 2.10.0（厂商 CUDA 兼容栈） | **不触发** | **139 / 139** | 🔴 **中招**（无修复版本；我方已做运行时等价补丁，见 `PPU/probes/weakref_check_20261103_out/`） |
| **P800** | 2.9.0+cu129（昆仑芯兼容栈） | **触发** | **0** | 🟢 **未中招**（后端 dealloc 已正确清理；证据 `P800/probes/weakref_check_20261103_out/`） |
| **MLU590** | 2.7.1 + torch_mlu 1.29.2 | **触发** | **0** | 🟢 **未中招**（2026-11-03 11:5x 补测，网络恢复后）。⭐ 且 weakref **可创建（30/30）且回调正确** ⇒ 此前「2.7.1 大概率不支持 weakref」的推测**已证伪**：缺陷不是「支持/不支持 weakref」，而是「父类支持 but 子类 `tp_dealloc` 未链式清理」 |

**读法**：同一判据下 **3 个可测实例里 2 个中招**，且中招的两个都是"自研/兼容 torch 后端 + torch ≥ 2.8" ⇒
印证 §1 的推论：**这是生态共性缺陷，不是某一家实现错误**。群内求证时建议各方向在自己栈上跑
`ppu_wr_callback.py`（10 行）即可自查。

⚠️ **对 910C 生产栈的提示（不主张、仅事实）**：我方环境用的 `torch_npu 2.10.0` 属中招版本；
本层因改强持有已不受影响，但**他人若在 910C 上挂 Stream 弱引用同样会崩**。
是否升级到 `post7` 属**总组基座裁定**（我方只消费 `stack.lock`），本报告只提供"官方已有修复版本"这一事实。

## 6. 结论边界

- 本报告只回答"昇腾修法 + 是否修成 + 是否同构"，**不构成 PPU 侧已修复的声明**。
- PPU 侧当前状态仍是：**平台缺陷未修、我方已自保**（`PPU_VENDOR_WEAKREF_ISSUE_20261026.md` §七）。
- 910C 生产栈是否升级到 `post7`，属**总组基座裁定**（我方只消费 `stack.lock`），本报告不主张升级；
  但可作"同缺陷已有官方修复版本"的事实依据提交总组。
