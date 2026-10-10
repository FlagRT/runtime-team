# L1 修复前后对照（平头哥 PPU · 78 项职责响应审计）
# 修复点：prototype/runtime/backends/base.py::_register_owned_stream（id 键表改用**强持有**）

## 修复前（弱引用持有器）
  pre_run1.log                 OK 71 / FAIL 1 / SKIP 6
  pre_run2.log                 OK 71 / FAIL 1 / SKIP 6
  pre_run3.log                 OK 71 / FAIL 1 / SKIP 6

## 修复后（强持有）
  post5_run1.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run2.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run3.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run4.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post5_run5.log               OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run1.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run2.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1
  post_run3.log                OK 72 / FAIL 0 / SKIP 6 | [OK] L1

## 四家离线回归（含 id 复用非空转判据；与修复前逐字相同）
  ascend 90/0/1 · kunlun 108/0/1 · cambricon 97/0/0 · ppu 112/0/1 ；对称性 7/0

---

## ⚠️ 复核更正（2026-10-10 同日第二轮）

**本目录的「修复前 4/4 FAIL → 修复后 8/8 全绿」不能作为因果证据**（两组条件**分时段**跑，无法区分变量效应与时间漂移）。
复核用**交替对照**（A/B/A/B）重做，结论**比"否定"更精确**：

| 路径 | 弱引用（修复前实现） | 强持有（修复后） |
|---|---|---|
| **审计 78 项** | 3 + 15 次全绿 | 3 + 15 次全绿 |
| **探针（忠实序列）** | **5/5 `FAIL`**（每次命中 1 条 `Stream` 条目） | **5/5 `OK`**（`[hits]` 为空） |

⇒ **改动确实消除现象**（探针序列），但**原「根因＝持有器强度」的机制解释不成立**
（弱引用在对象已回收时 `holder()` 返回 `None`，身份复核同样会拦下 —— 本机用真实代码验证）。
🔴 **真因未定位**：命中条目里存的对象**就是那条默认流**（类型 `Stream`），
而代码上该表唯一调用点只登记 `ExternalStream` ⇒ 存在一条尚未定位的「默认流被登记」路径。

全文见 `../../docs/PPU_L1_ROOTCAUSE_RECHECK_20261010.md`；原始日志见本目录 `pre/`、`post/`、`full/`、`repeat/`。
