#!/usr/bin/env python3
"""exp_divergence_cost.py — 工作包 A：「分歧的业务代价」实验（后端无关）

## 这个实验要回答什么

到现在为止我们只证明了「三家厂商语义有分歧、且我们能检出」，
**没证明「不做统一抽象就有实际损失」**。本脚本用同一段真实上层需求，
在两条路径上各实现一次，量化统一层消除的成本：

  路径 ① unified ：只用 `runtime` 统一 API（+ 由接口给出的 `device_type` 拼设备串）
  路径 ② native  ：直接调厂商原生 API（torch_npu / XPytorch / torch_mlu + acl / cnrt），
                  **允许按厂商分支** —— 这正是「不做抽象」的真实形态

## 度量口径（★先定后测，禁止事后调整★）

| 记号 | 含义 | 计数规则 |
|---|---|---|
| **M1** | 厂商分支数 | 区间内 `#V ` 标记出现次数。`#V` **只能**标在「因厂商差异而必须存在的条件分支 / 取值分支」上，且必须同行或上一行写明触发它的厂商事实 |
| **M2** | 厂商私有知识点 | `#K ` 标记次数。必须是「不查厂商资料就会写错的具体事实」（错误码值 / API 签名 / 语义边界 / 后端名 / 命名空间） |
| **M3** | 上层代码行数 | 区间内**非空且非纯注释**的物理行数（`--analyze-only` 机械统计） |
| **M4** | 厂商 API 种类 | 区间内正则抽取 `torch.(npu/mlu/cuda).*` 与 `acl.*` / `cnrt.*` **去重计数**（完全机械，可用于交叉核对 M1） |
| **M5** | 功能等价性 | 同一场景下两路径的 `(category, disposition, action)` 三元组是否一致 |

**M3/M4 由脚本读自身源码计算，不人工估算。M1/M2 是自标记指标** ——
为降低自证偏差，报告里逐条列出 M1/M2 的实际内容供审阅，并用 M4 做交叉核对。

## 执行纪律

1. **功能等价性必须先成立**，否则比较无意义 —— 两路径对同一场景必须给出同一三元组；
2. **每个场景独立进程跑**：设备类失败会污染进程后续状态（实测：`set_device(越界)` 之后
   所有设备操作都报同一个错）⇒ 绝不在同一进程里连跑多个设备场景。默认即隔离执行；
3. 不可安全真机触发的场景**不伪造**，标 `synthetic` / `best-effort`，如实计入报告；
4. **结论方向不做预设**：若统一层只减少 1–2 个分支，如实写「本层价值有限」。

用法：
  # 静态度量（不碰设备）
  python3 probes/exp_divergence_cost.py --backend kunlun --analyze-only
  # 真机跑（默认逐场景独立进程）
  DC_BACKEND=kunlun CUDA_VISIBLE_DEVICES=6 python3 probes/exp_divergence_cost.py \
      --backend kunlun --out /path/exp_divergence_cost_kunlun.json
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[1]                      # prototype/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: 契约期望值（来源：INTERFACE_CONTRACT_DC_20260908.md §1.2 / §1.4）
SCENARIOS = {
    "S0_device_enum":      ("值一致性", "device_count / total_mb 两路径应一致"),
    "S1_param_out_of_range": ("L2_PARAM + raise", "设备序号越界 ⇒ 参数类，上抛不重试"),
    "S2_bounded_sync_timeout": ("L3_EXECUTION + replay", "有界同步超时 ⇒ 执行类可重放"),
    "S3_out_of_memory":    ("L1_RESOURCE + retry", "显存不足 ⇒ 资源类，带退避重试"),
    "S5_unknown_error":    ("L3_EXECUTION + replay", "无依据兜底 + graded_by=default（最不可信）"),
    "S6_coded_error_ownership": ("按后端定：声明 error_map 者 L4，否则兜底 L3",
                                 "码表归属：只有**声明 error_map** 的后端才可用码表升级分级"),
}

#: 契约里四态的**规范取值**（`conformance/device_state.py::DeviceState` 的 `.value`）
CANONICAL_STATES = {"available", "degraded", "isolated", "destroyed"}

#: 路径 ② 自带的**最小**码表（刻意不完整 —— 这正是「不做抽象」的真实代价）
#: #K 任何路线下都得为昇腾建一份 ACL 码表；此处只列上层实际遇到的 4 条，
#:    覆盖率远低于完整 109 条 ⇒ 未命中即退化为兜底 L3（保守但有代价）
_NATIVE_MIN_CODE_TABLE = {
    107001: "L2_PARAM",     # ACL_ERROR_RT_INVALID_DEVICEID
    207001: "L1_RESOURCE",  # ACL_ERROR_RT_MEMORY_ALLOCATION (OOM)
    507015: "L4_FATAL",     # ACL_ERROR_RT_AICORE_EXCEPTION
    107019: "L3_EXECUTION",  # ACL_ERROR_RT_WAIT_TIMEOUT
}

#: 场景过滤（`--single` 用）。None = 全跑。
_ONLY = None


def _want(name: str) -> bool:
    return _ONLY is None or _ONLY == name


# ============================================================
# ===== PATH-1 BEGIN (unified · 只用统一 API) =====
# ============================================================

def path1_run(backend_name: str, ordinal: int) -> dict:
    """路径 ①：全部经统一层。**本区间内不得出现任何厂商命名空间 API。**"""
    import torch

    import runtime

    runtime.use(backend_name)
    bk = runtime.current()
    res = {}

    n = bk.device_count()
    bk.set_device(ordinal)

    # ── S0 设备枚举与显存统计 ──
    if _want("S0_device_enum"):
        try:
            ms = bk.memory_stats(ordinal)
            res["S0_device_enum"] = {
                "ok": True, "device_count": n,
                "total_mb": ms.get("total_mb"), "used_mb": ms.get("used_mb"),
                "free_mb": ms.get("free_mb"), "trigger": "real",
            }
        except BaseException as e:                              # noqa: BLE001
            res["S0_device_enum"] = {"ok": False, "msg": str(e)[:200], "trigger": "real"}

    def _classify_step(name: str, fn, trigger: str = "real") -> None:
        try:
            fn()
            res[name] = {"ok": True, "category": None, "disposition": None,
                         "action": "none", "trigger": trigger}
        except BaseException as e:                              # noqa: BLE001
            fe = bk.translate_error(e, location=f"exp:{name}")
            cat = getattr(getattr(fe, "category", None), "value", None) or str(
                getattr(fe, "category", None))
            disp = getattr(fe, "disposition", None)
            res[name] = {
                "ok": False, "category": cat, "disposition": disp, "action": disp,
                "mapped": bool(getattr(fe, "mapped", False)),
                "graded_by": getattr(fe, "graded_by", None),
                "msg": str(e)[:200], "trigger": trigger,
            }

    # ── S1 参数越界：设备序号越界 ──
    if _want("S1_param_out_of_range"):
        _classify_step("S1_param_out_of_range", lambda: bk.set_device(n + 100))

    # ── S3 显存不足：按接口给出的 device_type 拼设备串（厂商无关） ──
    if _want("S3_out_of_memory"):
        def _oom() -> None:
            dev = f"{bk.device_type}:{ordinal}"
            total_mb = bk.memory_stats(ordinal).get("total_mb") or 8192
            nelem = max(1024, int(total_mb) * 1024 * 1024 // 4 * 2)   # 约 2× 显存 ⇒ 必失败
            _ = torch.empty(nelem, dtype=torch.float32, device=dev)

        _classify_step("S3_out_of_memory", _oom)

    # ── S5 未知异常：无任何厂商信息，应兜底 L3 且 graded_by=default ──
    if _want("S5_unknown_error"):
        _classify_step("S5_unknown_error",
                       lambda: (_ for _ in ()).throw(RuntimeError("expansion saw a wobble")))

    # ── S6 芯片级致命：真机不触发硬件致命，用合成码消息验证分级路径 ──
    if _want("S6_coded_error_ownership"):
        _classify_step("S6_coded_error_ownership",
                       lambda: (_ for _ in ()).throw(
                           RuntimeError("AICORE exception, error code is 507015")),
                       trigger="synthetic")

    # ── S2 有界同步超时：在忙流上做极短超时同步（best-effort） ──
    if _want("S2_bounded_sync_timeout"):
        def _timeout() -> None:
            st = bk.create_stream()
            with bk.stream_context(st):
                a = torch.empty(2048, 2048, dtype=torch.float32,
                                device=f"{bk.device_type}:{ordinal}")
                for _ in range(200):
                    a = a @ a
            bk.synchronize_stream(st, 1)

        _classify_step("S2_bounded_sync_timeout", _timeout, trigger="best-effort")

    # ── S6b 恢复调用与返回契约（两路径都应给出五键） ──
    if _want("S6b_recover_contract"):
        try:
            rec = bk.recover_device(ordinal, mode="probe", reason="exp")
            res["S6b_recover_contract"] = {
                "ok": True, "keys": sorted(rec.keys()),
                "recovered": rec.get("recovered"), "state_raw": rec.get("state"),
                "five_keys": set(rec.keys()) >= {"ordinal", "mode", "recovered",
                                                  "state", "detail"},
                "state_is_canonical_token": rec.get("state") in CANONICAL_STATES,
                "trigger": "real",
            }
        except BaseException as e:                              # noqa: BLE001
            res["S6b_recover_contract"] = {"ok": False, "msg": str(e)[:200],
                                           "trigger": "real"}

    return res

# ===== PATH-1 END =====
# ============================================================


# ============================================================
# ===== PATH-2 BEGIN (native · 直调厂商原生，允许按厂商分支) =====
# ============================================================

def path2_run(backend_name: str, ordinal: int) -> dict:
    """路径 ②：不假设任何统一层。上层自己把每家接一遍。"""
    import torch

    VENDOR = backend_name
    res = {}

    #V 设备命名空间三家不同：昇腾 npu / 昆仑芯 cuda / 寒武纪 mlu —— 上层无法用同一个字符串
    #K torch 的设备命名空间前缀：npu / cuda / mlu（不是「加速器名」，不能从设备型号推）
    DEVNS = {"ascend": "npu", "kunlun": "cuda", "cambricon": "mlu"}
    ns = DEVNS[VENDOR]

    #V 三家枚举/绑定 API 挂在各自命名空间下，须分别调用
    if VENDOR == "ascend":
        import torch_npu                                          # noqa: F401
        #K torch_npu 需先 import 才会把 npu 命名空间注册到 torch（懒加载）
        device_count = torch.npu.device_count()
        torch.npu.set_device(ordinal)
    elif VENDOR == "kunlun":
        #K 昆仑芯走 torch.cuda 兼容层（XPytorch）+ torch_xray 符号重写
        device_count = torch.cuda.device_count()
        torch.cuda.set_device(ordinal)
    else:
        import torch_mlu                                         # noqa: F401
        #K torch_mlu 同样需先 import 才注册 mlu 命名空间
        device_count = torch.mlu.device_count()
        torch.mlu.set_device(ordinal)

    def _read_mem():
        #V 三家取显存的可用 API 不同，且昆仑芯的 memory_stats() 实测返回空 dict，必须换路径
        if VENDOR == "ascend":
            return torch.npu.mem_get_info()
        if VENDOR == "kunlun":
            #K 昆仑芯 torch.cuda.memory_stats() 返回空 dict ⇒ 必须用 mem_get_info()+memory_allocated()
            return torch.cuda.mem_get_info(ordinal)
        try:
            return torch.mlu.mem_get_info(ordinal)
            #K 部分 torch_mlu 实现的 mem_get_info 不接受 ordinal 参数
        except TypeError:
            return torch.mlu.mem_get_info()

    # ── S0 设备枚举与显存统计 ──
    if _want("S0_device_enum"):
        free, total = _read_mem()
        res["S0_device_enum"] = {
            "ok": True, "device_count": int(device_count),
            "total_mb": int(total / 1024 / 1024),
            "used_mb": int((total - free) / 1024 / 1024),
            "free_mb": int(free / 1024 / 1024), "trigger": "real",
        }

    # ── 上层自己的错误 → L1..L4 分级（统一层把这步变成了 translate_error 一行） ──
    def _native_classify(exc: BaseException):
        msg = str(exc)
        code = None
        #V 只有昇腾把错误以数字码透出；另两家没有数字码，提取策略必须分开写
        if VENDOR == "ascend":
            m = re.search(r"error code is (\d+)", msg)
            code = int(m.group(1)) if m else None
        elif VENDOR == "kunlun":
            code = None
        else:
            code = None
        if code is not None and code in _NATIVE_MIN_CODE_TABLE:
            return _NATIVE_MIN_CODE_TABLE[code], "code_map"
        low = msg.lower()
        #V 消息关键词三家文案不同（NPU out of memory / CUDA out of memory / MLU out of memory）
        #K OOM 文案在三家不同，且昇腾同一现象既有数字码 207001 又有文案，两条都要兜
        if "out of memory" in low or "memory allocation" in low:
            return "L1_RESOURCE", "message_hint"
        #K 参数类关键词各家不同：invalid / illegal / out of range / ordinal / unaligned
        if ("invalid" in low or "illegal" in low or "out of range" in low
                or "out of bound" in low or "ordinal" in low):
            return "L2_PARAM", "message_hint"
        #K 超时类关键词：昇腾给码 107019/507046，另两家只能靠 "timeout" 文案
        if "timeout" in low or "timed out" in low:
            return "L3_EXECUTION", "message_hint"
        #V 昆仑芯/寒武纪无码表 ⇒ 未命中任何关键词时只能兜底，且**无法区分 L3 与 L4**
        return "L3_EXECUTION", "default"

    def _step(name: str, fn, trigger: str = "real") -> None:
        try:
            fn()
            res[name] = {"ok": True, "category": None, "disposition": None,
                         "action": "none", "trigger": trigger}
        except BaseException as e:                              # noqa: BLE001
            cat, graded_by = _native_classify(e)
            disp = {"L1_RESOURCE": "retry", "L2_PARAM": "raise",
                    "L3_EXECUTION": "replay", "L4_FATAL": "device_recovery"}[cat]
            res[name] = {
                "ok": False, "category": cat, "disposition": disp, "action": disp,
                "mapped": graded_by == "code_map", "graded_by": graded_by,
                "msg": str(e)[:200], "trigger": trigger,
            }

    if _want("S1_param_out_of_range"):
        _step("S1_param_out_of_range", lambda: _p2_set_device(VENDOR, device_count + 100))

    if _want("S3_out_of_memory"):
        def _oom() -> None:
            free, total = _read_mem()
            nelem = max(1024, int(total) // 4 * 2)                  # 约 2× 显存 ⇒ 必失败
            _ = torch.empty(nelem, dtype=torch.float32, device=f"{ns}:{ordinal}")

        _step("S3_out_of_memory", _oom)

    if _want("S5_unknown_error"):
        _step("S5_unknown_error",
              lambda: (_ for _ in ()).throw(RuntimeError("expansion saw a wobble")))

    #V 只有昇腾有 507015 这类芯片级致命码；另两家连数字码都没有 ⇒ L4 判定能力不对等
    if _want("S6_coded_error_ownership"):
        _step("S6_coded_error_ownership",
              lambda: (_ for _ in ()).throw(
                  RuntimeError("AICORE exception, error code is 507015")),
              trigger="synthetic")

    # ── 有界同步超时：三家的实现路径完全不同 ──
    if _want("S2_bounded_sync_timeout"):
        def _timeout() -> None:
            #V 三家同步原语不同；且**只有昇腾有真超时中断原语**，另两家只能轮询上报
            if VENDOR == "ascend":
                import acl
                #K pyACL 有界同步入口是 rt.synchronize_stream_with_timeout(handle, ms)，需底层流句柄
                st = torch.npu.Stream()
                h = getattr(st, "npu_stream", None)
                with torch.npu.stream(st):
                    a = torch.empty(2048, 2048, dtype=torch.float32, device=f"{ns}:{ordinal}")
                    for _ in range(200):
                        a = a @ a
                rc = acl.rt.synchronize_stream_with_timeout(h, 1)
                if rc != 0:
                    raise TimeoutError(f"acl stream sync rc={rc}")
            elif VENDOR == "kunlun":
                #K 昆仑芯 Stream.synchronize() 无 timeout 参数、无中断原语 ⇒ 只能「超时上报」
                st = torch.cuda.Stream()
                with torch.cuda.stream(st):
                    a = torch.empty(2048, 2048, dtype=torch.float32, device=f"{ns}:{ordinal}")
                    for _ in range(200):
                        a = a @ a
                deadline = time.monotonic() + 0.001
                while not st.query():
                    if time.monotonic() > deadline:
                        raise TimeoutError("bounded sync timeout (上报语义，不保证中断底层)")
                    time.sleep(0.001)
            else:
                #K 寒武纪是否提供真中断原语未验证 ⇒ 与昆仑芯同口径先做「超时上报」
                st = torch.mlu.Stream()
                with torch.mlu.stream(st):
                    a = torch.empty(2048, 2048, dtype=torch.float32, device=f"{ns}:{ordinal}")
                    for _ in range(200):
                        a = a @ a
                deadline = time.monotonic() + 0.001
                while not st.query():
                    if time.monotonic() > deadline:
                        raise TimeoutError("bounded sync timeout (上报语义，不保证中断底层)")
                    time.sleep(0.001)

        _step("S2_bounded_sync_timeout", _timeout, trigger="best-effort")

    # ── 设备恢复：三家的能力**天然不对等**，上层必须知道这个差异 ──
    if _want("S6b_recover_contract"):
        #V 只有昇腾有设备级重建原语（aclrtResetDevice 序列）；另两家无原语 ⇒ 只能探活
        if VENDOR == "ascend":
            alive = _p2_probe(VENDOR, ns, ordinal)
            #K 昇腾设备级重建须走 aclrtResetDevice 序列（destroyEvent→destroyStream→destroyContext→Reset→setDevice）
            rec = {"ordinal": ordinal, "mode": "probe", "recovered": alive,
                   "state": "available", "detail": "probe 级：重建原语存在但本次未调 real"}
        elif VENDOR == "kunlun":
            #K 昆仑芯 torch.cuda 上 reset* 全是内存统计类，**无 reset_device / context 重建**
            alive = _p2_probe(VENDOR, ns, ordinal)
            rec = {"ordinal": ordinal, "mode": "probe", "recovered": alive,
                   "state": "available", "detail": "无设备级重建原语"}
        else:
            #K 寒武纪是否有设备级重置原语未验证 ⇒ 不写未经验证的重建序列
            alive = _p2_probe(VENDOR, ns, ordinal)
            rec = {"ordinal": ordinal, "mode": "probe", "recovered": alive,
                   "state": "available", "detail": "重建原语未验证"}
        try:
            res["S6b_recover_contract"] = {
                "ok": True, "keys": sorted(rec.keys()), "recovered": rec.get("recovered"),
                "state_raw": rec.get("state"),
                "five_keys": set(rec.keys()) >= {"ordinal", "mode", "recovered",
                                                  "state", "detail"},
                "state_is_canonical_token": rec.get("state") in CANONICAL_STATES,
                "trigger": "real",
            }
        except BaseException as e:                              # noqa: BLE001
            res["S6b_recover_contract"] = {"ok": False, "msg": str(e)[:200],
                                           "trigger": "real"}

    return res


def _p2_set_device(vendor: str, ordinal: int) -> None:
    #V 绑定设备的入口三家不同（须与上面的枚举分支保持一致，否则两处会漂移）
    import torch
    if vendor == "ascend":
        torch.npu.set_device(ordinal)
    elif vendor == "kunlun":
        torch.cuda.set_device(ordinal)
    else:
        torch.mlu.set_device(ordinal)


def _p2_probe(vendor: str, ns: str, ordinal: int) -> bool:
    #V 探活的设备串前缀三家不同（ns 由调用方按厂商给出）
    import torch
    try:
        x = torch.zeros(2, 2, device=f"{ns}:{ordinal}")
        return float(x.sum().item()) == 0.0
    except Exception:                                           # noqa: BLE001
        return False

# ===== PATH-2 END =====
# ============================================================


def analyze_source() -> dict:
    """M1–M4：读自身源码机械计数（不碰设备）。"""
    src = _HERE.read_text(encoding="utf-8")
    out = {}
    for tag in ("PATH-1", "PATH-2"):
        m = re.search(rf"^# =+ {tag} BEGIN.*?$\n(.*?)^# =+ {tag} END", src,
                      re.S | re.M)
        if not m:
            out[tag] = {"error": "区间标记未找到"}
            continue
        body = m.group(1)
        code_lines = [ln for ln in body.split("\n")
                      if ln.strip() and not ln.strip().startswith("#")]
        refs = set(re.findall(r"torch\.(?:npu|mlu|cuda)\.[A-Za-z_]+", body))
        refs |= set(re.findall(r"\b(?:acl|cnrt)\.[A-Za-z_.]+", body))
        out[tag] = {
            "M1_vendor_branches": body.count("#V "),
            "M2_vendor_facts": body.count("#K "),
            "M3_code_lines": len(code_lines),
            "M4_vendor_api_refs": sorted(refs),
            "M4_vendor_api_kinds": len(refs),
        }
    return out


#: 期望 category **按后端不同**：只有声明 `error_map` 的后端才有本厂商码表可用。
#: 2026-09-29：本场景由 `device reset failed`（会被消息规则命中也给 L4，属**巧合**）
#:   换成 `AICORE exception`（**不命中任何关键词**）⇒ category 只能来自
#:   「本厂商码表」或「兜底 L3」，从而干净地判别出**是否使用了外来码表**。
_EXPECT_BY_BACKEND = {
    "S6_coded_error_ownership": {"ascend": "L4_FATAL", "kunlun": "L3_EXECUTION",
                                 "cambricon": "L3_EXECUTION"},
}


def _expected_category(backend: str, name: str):
    return _EXPECT_BY_BACKEND.get(name, {}).get(backend)


def _compare(rows_src: dict, backend: str) -> tuple:
    rows, mismatches = [], []
    for name, (expect, why) in SCENARIOS.items():
        a, b = rows_src["path1"].get(name, {}), rows_src["path2"].get(name, {})
        if not a or not b:
            continue
        if name == "S0_device_enum":
            same = (a.get("device_count") == b.get("device_count")
                    and a.get("total_mb") and b.get("total_mb")
                    and abs(a["total_mb"] - b["total_mb"]) <= max(1, a["total_mb"] // 100))
            got1 = f"值{'一致' if same else '不一致'}"
            got2 = got1
        elif name == "S6b_recover_contract":
            same = a.get("five_keys") == b.get("five_keys")
            got1 = f"五键={a.get('five_keys')}"
            got2 = f"五键={b.get('five_keys')}"
        else:
            t1 = (a.get("category"), a.get("disposition"), a.get("action"))
            t2 = (b.get("category"), b.get("disposition"), b.get("action"))
            same = t1 == t2
            got1, got2 = str(t1[0]), str(t2[0])
        _exp_cat = _expected_category(backend, name)
        if _exp_cat:
            _matches = str(got1) == _exp_cat
        elif " + " in expect:
            _matches = expect.split(" + ")[0] in str(got1)
        else:
            _matches = None
        rows.append({"scenario": name, "expect": expect, "why": why,
                     "path1": got1, "path2": got2,
                     "equivalent": bool(same), "trigger": a.get("trigger"),
                     "expected_category": _exp_cat,
                     "path1_matches_expect": _matches})
        if not same:
            mismatches.append(name)
    return rows, mismatches


def _run_single(backend: str, ordinal: int, scenario: str) -> dict:
    """在**当前进程**只跑一个场景（供隔离模式的子进程调用）。"""
    global _ONLY
    _ONLY = scenario
    return {"path1": path1_run(backend, ordinal), "path2": path2_run(backend, ordinal)}


def main() -> int:
    ap = argparse.ArgumentParser(description="工作包 A：分歧的业务代价实验")
    ap.add_argument("--backend", required=True,
                    choices=["ascend", "kunlun", "cambricon"])
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--analyze-only", action="store_true", help="只做 M1–M4 静态度量")
    ap.add_argument("--single", default="", help="只跑一个场景（内部用，逐场景隔离执行）")
    ap.add_argument("--no-isolate", action="store_true",
                    help="不隔离（同进程连跑全部场景，**仅用于对照，结论不可信**）")
    ap.add_argument("--out", default="", help="结果 JSON 路径")
    args = ap.parse_args()

    metrics = analyze_source()

    if args.single:
        payload = _run_single(args.backend, args.ordinal, args.single)
        payload["backend"] = args.backend
        payload["scenario"] = args.single
        print("@@JSON@@" + json.dumps(payload, ensure_ascii=False))
        return 0

    if args.analyze_only:
        print("=== M1–M4 静态度量（口径见文件头，机械计数）===")
        print(_fmt_metrics(metrics))
        if args.out:
            pathlib.Path(args.out).write_text(
                json.dumps({"backend": args.backend, "metrics": metrics},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    print("=== M1–M4 静态度量（口径见文件头，机械计数）===")
    print(_fmt_metrics(metrics))

    print(f"\n=== M5 功能等价性（backend={args.backend}, ordinal={args.ordinal}）===")
    names = [n for n in SCENARIOS if n] + ["S6b_recover_contract"]
    merged = {"path1": {}, "path2": {}, "iso": {}}
    for name in names:
        if args.no_isolate:
            got = _run_single(args.backend, args.ordinal, name)
            iso = "same-process"
        else:
            proc = subprocess.run(
                [sys.executable, str(_HERE), "--backend", args.backend,
                 "--ordinal", str(args.ordinal), "--single", name],
                capture_output=True, text=True, timeout=900)
            line = next((ln for ln in proc.stdout.splitlines()
                         if ln.startswith("@@JSON@@")), "")
            if not line:
                merged["iso"][name] = f"子进程失败 rc={proc.returncode}"
                print(f"  [{name}] 子进程失败 rc={proc.returncode}: "
                      f"{(proc.stderr or '').strip()[-160:]}")
                continue
            got = json.loads(line[len("@@JSON@@"):])
            iso = "isolated"
        merged["path1"].update(got.get("path1", {}))
        merged["path2"].update(got.get("path2", {}))
        merged["iso"][name] = iso

    rows, mismatches = _compare(merged, args.backend)
    for r in rows:
        mark = "OK  " if r["equivalent"] else "DIFF"
        exp_ok = "" if r.get("path1_matches_expect") in (None, True) else "  ←期望未达标"
        print(f"  [{mark}] {r['scenario']:<26} expect={r['expect']:<28} "
              f"path1={r['path1']:<16} path2={r['path2']:<16} ({r['trigger']}){exp_ok}")
    print(f"\n  等价性：{len(rows) - len(mismatches)}/{len(rows)} 一致"
          f"{'（' + ', '.join(mismatches) + '）' if mismatches else ''}")
    print("\n  ⚠️ 注：「等价」只表示两路径给出同一三元组，**不表示该三元组符合契约期望** —— "
          "后者看上面的『期望未达标』标注。")

    payload = {
        "backend": args.backend, "ordinal": args.ordinal,
        "isolation": {k: v for k, v in merged["iso"].items()},
        "metrics": metrics, "equivalence": rows, "mismatches": mismatches,
        "path1_raw": merged["path1"], "path2_raw": merged["path2"],
        "boundary": ("本脚本只度量「上层实现成本」与「两路径行为等价性」；"
                     "不替代真机性能结论；synthetic / best-effort 场景不得当真实触发结论。"
                     "未隔离模式（--no-isolate）的结论不可信（设备类失败会污染后续场景）。"),
    }
    print("\n" + payload["boundary"])
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(f"结果已写入 {args.out}")
    return 0


def _fmt_metrics(metrics: dict) -> str:
    out = []
    for tag in ("PATH-1", "PATH-2"):
        m = metrics.get(tag, {})
        out.append(f"  {tag}: 分支(M1)={m.get('M1_vendor_branches')} "
                   f"知识(M2)={m.get('M2_vendor_facts')} "
                   f"代码行(M3)={m.get('M3_code_lines')} "
                   f"厂商API种类(M4)={m.get('M4_vendor_api_kinds')}")
        if m.get("M4_vendor_api_refs"):
            refs = m["M4_vendor_api_refs"]
            out.append(f"      M4 明细: {', '.join(refs[:8])}"
                       f"{' …' if len(refs) > 8 else ''}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
