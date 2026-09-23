#!/usr/bin/env python3
"""
设备状态恢复（conformance/recovery.py，独立版）—— 五段式恢复（捕获→评估→隔离→重建→重放）

对应设备执行上下文职责（细项21·设备状态恢复）与统一行为契约 R1-R5：
  - R1 捕获：错误归因（经 errors.translate_error 统一错误对象）
  - R2 评估：轻量探针区分 L3（可继续）与 L4（需重建），避免不必要高代价重建
  - R3 隔离：损坏上下文从调度池摘除（device_state.ISOLATED），停止派发
  - R4 重建：重建后新上下文 AVAILABLE
      · rebuild_mode="probe"  （默认）探针重试近似（兼容历史行为，进程内 torch_npu 安全）
      · rebuild_mode="real"   真实重建：CANN 官方序列 aclrtDestroyEvent→DestroyStream→
                               DestroyContext→aclrtResetDevice→setDevice→重建（2026-09-02 P1-③
                               验证 RESET_REBUILD_PASS）。⚠️ 会重置当前进程默认上下文，
                               多进程共享设备时其他进程不受影响（官方语义）；本进程内已初始化的
                               torch_npu 运行时需重新 set_device。上传仓库后需多卡多进程压力测试调优。
      · rebuild_mode="hybrid" 先探针（快路径），失败后真实重建
  - R5 重放：在途任务登记 + 重放接口（重放边界决策归上层训练支撑/检查点）

用法：
    from recovery import (
        probe_device, evaluate_device, recover_device, handle_error,
        mark_inflight, finish_inflight, inflight_snapshot, replay_tasks,
    )
    fe = handle_error(exc, ordinal=0, location="stream:0/op:x")
    # fe.category / fe.recovery_decision（captured/evaluated/isolated/recovered）
"""

import threading
import time
from typing import Dict, List, Optional

from device_state import DeviceState, query_device_state, set_device_state
from errors import ErrorCategory, FlagosError, translate_error

# R4 重建模式（2026-09-02 P1-③ 升级）
REBUILD_PROBE = "probe"      # 探针重试近似（兼容历史）
REBUILD_REAL = "real"        # CANN 官方真实重建序列
REBUILD_HYBRID = "hybrid"    # 先探针后真实


def _rebuild_real(ordinal: int) -> bool:
    """真实重建（R4-real）：CANN 官方序列（acl_rt.h）——
    destroyEvent→destroyStream→destroyContext→aclrtResetDevice→setDevice→重建。

    ⚠️ 前置：仅适用于本进程内设备上下文已损坏、需要彻底重置的场景。
    调用方需保证当前进程对设备的使用是可重建的（如 EngineCore 子进程错误后、
    独立恢复进程）。多进程共享设备时，其他进程的显式 Context/Stream 不受影响
    （acl_rt.h 注释语义：释放当前进程默认上下文）；本进程内已初始化的
    torch_npu 运行时需在重建后重新 set_device 才能继续使用。

    ⚠️ 多卡多进程压力测试前请勿在生产路径默认启用（见模块 docstring）。
    """
    try:
        import acl  # pyACL：容器内 /usr/local/Ascend/... 提供
    except ImportError:
        return False
    try:
        # 1) 显式资源按官方顺序销毁（进程内已创建的可枚举资源由调用方预清理；
        #    此处对当前默认上下文执行最小可重建序列）
        acl.init()
        acl.rt.set_device(ordinal)
        # 2) 重置设备（释放默认上下文/默认流/默认上下文下创建的所有流）
        rc = acl.rt.reset_device(ordinal)
        if rc != 0:
            return False
        # 3) 重新指定设备 + 重建默认上下文（set_device 隐式创建）
        rc = acl.rt.set_device(ordinal)
        if rc != 0:
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 探针与评估（R2）
def probe_device(ordinal: int, device: str = "npu", sync_fn=None, timeout_ms: int = 5000) -> bool:
    """轻量活性探针：在指定设备创建小张量、执行运算并同步（设备无关）。

    - device: 设备名（"npu"/"flagos"/...）；sync_fn: 该设备的主机同步原语
    - 成功：设备可用（评估为可继续）；失败（异常/超时）：活性存疑（评估为需重建）
    - 注意：探针创建小张量即隐含设备上下文可用性验证；真实上下文重建
      依赖设备生命周期接口（厂商 Runtime 层），框架层最小近似为探针重试。
    """
    import torch
    try:
        x = torch.zeros(4, 4, device=f"{device}:{ordinal}")
        y = (x + 1).sum()
        if sync_fn is not None:
            sync_fn()
        return bool(torch.isfinite(y).all())
    except Exception:
        return False


def evaluate_device(ordinal: int, reason: str = "", device: str = "npu", sync_fn=None) -> DeviceState:
    """评估设备状态（R2）：探针成功 → 设备可用；失败 → 置 ISOLATED（L4 需重建）。

    返回评估后的设备状态：
      - 探针成功：保持/恢复 AVAILABLE（若之前 DEGRADED 且探针通过，视为可用）
      - 探针失败：置 ISOLATED（触发 R3 隔离），返回 ISOLATED
    """
    ok = probe_device(ordinal, device=device, sync_fn=sync_fn)
    if ok:
        cur = query_device_state(ordinal)
        if cur == DeviceState.ISOLATED:
            # 探针通过但状态仍隔离：由重建段负责回 AVAILABLE（避免评估越权）
            return DeviceState.ISOLATED
        if cur == DeviceState.DEGRADED:
            set_device_state(ordinal, DeviceState.AVAILABLE, "evaluate: probe ok, capability restored")
            return DeviceState.AVAILABLE
        return cur
    set_device_state(ordinal, DeviceState.ISOLATED, reason or "evaluate: probe failed (L4)")
    return DeviceState.ISOLATED


# ---------------------------------------------------------------- 重建（R3/R4）
def recover_device(ordinal: int, attempts: int = 3, reason: str = "", device: str = "npu", sync_fn=None,
                   rebuild_mode: str = REBUILD_PROBE) -> bool:
    """五段式恢复的重建段（R3 隔离 + R4 重建）：ISOLATED → 重试探针/真实重建 → AVAILABLE。

    - 仅 ISOLATED 状态可重建（防止误重建可用设备）
    - rebuild_mode（2026-09-02 P1-③ 升级）：
      · "probe"（默认）：探针重试近似——重试探针成功即视为设备可用（兼容历史行为，
        进程内 torch_npu 运行时的安全默认；但"重建"语义为最小近似）
      · "real"：真实重建——CANN 官方序列 aclrtResetDevice（destroy 显式资源→reset→set_device），
        已由 probe_device_reset_rebuild.py 验证 RESET_REBUILD_PASS；⚠️ 重置当前进程默认上下文，
        多进程共享设备不受影响（官方语义），本进程 torch_npu 需重新 set_device；
        多卡多进程压力测试调优后再用于生产默认
      · "hybrid"：先探针（快路径，探测通过即恢复），失败后走真实重建
    - 成功：置 AVAILABLE（R4 保证）；失败：保持 ISOLATED
    """
    if query_device_state(ordinal) != DeviceState.ISOLATED:
        return False

    def _mark_ok(how: str, note: str = "") -> bool:
        set_device_state(ordinal, DeviceState.AVAILABLE,
                         reason or f"recover: rebuild ok via {how}{(' ' + note) if note else ''}")
        return True

    # 快路径：探针重试（probe / hybrid 共用）
    if rebuild_mode in (REBUILD_PROBE, REBUILD_HYBRID):
        for i in range(attempts):
            if probe_device(ordinal, device=device, sync_fn=sync_fn):
                return _mark_ok("probe", f"(attempt {i + 1})")
            time.sleep(0.1 * (i + 1))
        if rebuild_mode == REBUILD_PROBE:
            return False  # 纯探针模式：失败即保持 ISOLATED

    # 真实重建（real / hybrid 的兜底）
    if rebuild_mode in (REBUILD_REAL, REBUILD_HYBRID):
        if _rebuild_real(ordinal):
            return _mark_ok("aclrtResetDevice", "(real)")
        return False
    return False


# ---------------------------------------------------------------- 在途登记与重放（R5）
class InflightTask:
    """在途任务登记：错误归因（位置投影）与重放集合的数据源。"""

    __slots__ = ("op_id", "ordinal", "location", "ts")

    def __init__(self, op_id: str, ordinal: int, location: str = ""):
        self.op_id = op_id
        self.ordinal = ordinal
        self.location = location
        self.ts = time.time()

    def to_dict(self) -> Dict:
        return {"op_id": self.op_id, "ordinal": self.ordinal,
                "location": self.location, "ts": self.ts}


_INFLIGHT: Dict[str, InflightTask] = {}
_INFLIGHT_LOCK = threading.Lock()


def mark_inflight(op_id: str, ordinal: int, location: str = "") -> None:
    """登记在途任务（提交时调用；错误归因/重放的数据源）。"""
    with _INFLIGHT_LOCK:
        _INFLIGHT[op_id] = InflightTask(op_id, ordinal, location)


def finish_inflight(op_id: str) -> None:
    """任务完成/放弃时注销在途登记。"""
    with _INFLIGHT_LOCK:
        _INFLIGHT.pop(op_id, None)


def inflight_snapshot() -> List[Dict]:
    """全部在途任务快照（恢复流程/监控诊断消费）。"""
    with _INFLIGHT_LOCK:
        return [t.to_dict() for t in _INFLIGHT.values()]


def replay_tasks(ordinal: Optional[int] = None) -> List[Dict]:
    """返回需重放的任务集合（R5 重放段数据源）。

    - ordinal 为空：全部设备在途任务；指定 ordinal：该设备在途任务
    - 重放边界决策（是否整步重放、检查点周期）归上层训练支撑/检查点，
      本接口只提供"需重放任务的确定性集合"。
    """
    with _INFLIGHT_LOCK:
        tasks = list(_INFLIGHT.values())
    if ordinal is None:
        return [t.to_dict() for t in tasks]
    return [t.to_dict() for t in tasks if t.ordinal == ordinal]


# ---------------------------------------------------------------- 五段式编排（R1-R5）
def handle_error(exc: BaseException, ordinal: Optional[int] = None,
                 location: Optional[str] = None,
                 op_id: Optional[str] = None,
                 device: str = "npu", sync_fn=None) -> FlagosError:
    """五段式恢复编排入口：错误 → 统一错误对象（R1）→ 评估（R2）→
    隔离/重建（R3/R4）→ 重放数据（R5），返回带恢复决策的统一错误对象。

    recovery_decision 记录流程事件，供可观测性/监控诊断消费：
      captured / evaluated(ok) / evaluated(l4)->isolated / recovered / replay_ready
    """
    fe = translate_error(exc, location=location)
    fe.recovery_decision = {"captured": True, "steps": ["captured"]}

    if fe.category != ErrorCategory.L4_FATAL:
        # L1-L3：不触发状态恢复（L1 重试 / L2 上抛 / L3 同上下文重放）
        fe.recovery_decision["steps"].append(f"evaluated: {fe.category.name}, no device recovery")
        fe.recovery_decision["replayable"] = (fe.category in
                                              (ErrorCategory.L1_RESOURCE, ErrorCategory.L3_EXECUTION))
        return fe

    # L4：评估 → 隔离 → 重建
    if ordinal is None:
        fe.recovery_decision["steps"].append("evaluated: L4 but ordinal unknown, skip device recovery")
        return fe
    evaluated = evaluate_device(ordinal, reason=f"L4 error: {fe.root_cause[:80]}", device=device, sync_fn=sync_fn)
    fe.recovery_decision["steps"].append(f"evaluated: {evaluated.value}")
    if evaluated == DeviceState.ISOLATED:
        ok = recover_device(ordinal, reason=f"L4 recovery: {fe.root_cause[:80]}", device=device, sync_fn=sync_fn)
        fe.recovery_decision["steps"].append(f"recovered: {ok}")
        if ok:
            pending = replay_tasks(ordinal)
            fe.recovery_decision["replay_ready"] = True
            fe.recovery_decision["replay_tasks"] = pending
            fe.recovery_decision["steps"].append("replay_ready")
    return fe
