#!/usr/bin/env python3
"""
torch_fl 设备状态机（flagos/device_state.py）

对应设备执行上下文职责（细项21·设备状态恢复）与统一行为契约 R1-R5：
  - R3 隔离保证：损坏上下文从调度池摘除（状态机 ISOLATED）
  - R4 重建保证：重建后新上下文 AVAILABLE
  - R5 重放协同：状态机回 AVAILABLE 后调度器恢复派发

设备四态：AVAILABLE（正常可用）/ DEGRADED（能力受限可用）/ ISOLATED（隔离待重建）/
DESTROYED（已销毁）。转换全部产生可观测事件（记录 + 订阅通知），
上层（调度引擎）可订阅状态变化做调度调整——契约：上层收到 ISOLATED 停止派发，
收到 AVAILABLE 恢复派发。

用法：
    from torch_fl.flagos.device_state import (
        DeviceState, query_device_state, set_device_state,
        subscribe_device_state, device_states,
    )
    st = query_device_state(0)              # DeviceState.AVAILABLE
    subscribe_device_state(0, on_change)    # 状态转换回调
"""

import enum
import threading
import time
from typing import Callable, Dict, List, Optional


class DeviceState(enum.Enum):
    """设备执行上下文四态状态机。"""
    AVAILABLE = "available"   # 正常可用：调度池取用
    DEGRADED = "degraded"     # 能力受限可用：调度池按能力受限参与
    ISOLATED = "isolated"     # 隔离待重建：调度池摘除，停止派发
    DESTROYED = "destroyed"   # 已销毁：优雅退出/资源回收完成


class DeviceStatus:
    """单设备状态对象：当前状态 + 最近转换记录 + 订阅者。"""

    def __init__(self, ordinal: int):
        self.ordinal = ordinal
        self.state = DeviceState.AVAILABLE
        self.last_transition: Optional[Dict] = None
        self._listeners: List[Callable] = []
        self._lock = threading.Lock()

    def snapshot(self) -> Dict:
        """状态快照（供可观测性事件流/监控诊断消费）。"""
        return {
            "ordinal": self.ordinal,
            "state": self.state.value,
            "last_transition": self.last_transition,
        }


# 设备状态注册表：ordinal -> DeviceStatus
_STATES: Dict[int, DeviceStatus] = {}
_REGISTRY_LOCK = threading.Lock()


def _ensure(ordinal: int) -> DeviceStatus:
    """取设备状态对象，不存在则注册为 AVAILABLE（幂等）。"""
    with _REGISTRY_LOCK:
        st = _STATES.get(ordinal)
        if st is None:
            st = DeviceStatus(ordinal)
            _STATES[ordinal] = st
        return st


def query_device_state(ordinal: int) -> DeviceState:
    """查询设备状态（契约：上层对设备可用性的判断是确定的）。"""
    return _ensure(ordinal).state


def set_device_state(ordinal: int, new_state: DeviceState, reason: str = "") -> DeviceState:
    """设置设备状态并产生可观测转换事件（R 系列：转换全部产生事件）。

    - 相同状态转换：不产生事件（幂等）。
    - 转换时：记录 last_transition、通知订阅者（顺序通知，快照式参数）。
    """
    st = _ensure(ordinal)
    with st._lock:
        old = st.state
        if old == new_state:
            return old
        st.state = new_state
        st.last_transition = {
            "from": old.value,
            "to": new_state.value,
            "reason": reason,
            "ts": time.time(),
        }
        listeners = list(st._listeners)
    # 锁外通知，避免回调死锁
    for cb in listeners:
        try:
            cb(new_state, old, reason)
        except Exception:
            pass
    return new_state


def subscribe_device_state(ordinal: int, cb: Callable) -> None:
    """订阅设备状态转换事件：cb(new_state, old_state, reason)。"""
    st = _ensure(ordinal)
    with st._lock:
        st._listeners.append(cb)


def unsubscribe_device_state(ordinal: int, cb: Callable) -> None:
    st = _ensure(ordinal)
    with st._lock:
        if cb in st._listeners:
            st._listeners.remove(cb)


def device_states() -> Dict[int, Dict]:
    """全部设备状态快照（监控诊断消费）。"""
    with _REGISTRY_LOCK:
        ordinals = list(_STATES.keys())
    return {o: _STATES[o].snapshot() for o in ordinals}
