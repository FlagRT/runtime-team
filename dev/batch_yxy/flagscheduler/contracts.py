"""Versioned, framework-neutral node data contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, Sequence


def positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value, name: str, minimum=0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class Request:
    request_id: str
    binding: str
    payload: Any
    priority: int = 0
    slo_ms: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("request_id", "binding"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")
        if self.slo_ms is not None:
            finite_number(self.slo_ms, "slo_ms")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict):
            raise ValueError("request must be a JSON object")
        unknown = value.keys() - {"request_id", "binding", "payload", "priority", "slo_ms", "metadata"}
        if unknown:
            raise ValueError(f"unknown request fields: {sorted(unknown)}")
        if not {"request_id", "binding", "payload"} <= value.keys():
            raise ValueError("request_id, binding and payload are required")
        return cls(**value)


@dataclass(frozen=True)
class ExecutionInput:
    request_id: str
    payload: Any
    priority: int = 0
    deadline_mono_ns: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionOutput:
    request_id: str
    value: Any


class Backend(Protocol):
    """One instance per binding. Calls are serialized by NodeRuntime.

    Cost must use the same tokenizer/preprocessing as execute. Results carry
    IDs; completion order is irrelevant. No backend retry is implied.
    """
    def count_cost(self, payload: Any) -> int: ...
    def execute(self, inputs: Sequence[ExecutionInput]) -> Sequence[ExecutionOutput]: ...
    def close(self) -> None: ...


class Planner(Protocol):
    """Future SLO policies can implement this without changing the UI/backend.

    Planner must emit each submitted ID exactly once and preserve all budgets,
    compatibility and policy snapshots. Callbacks run on the node event loop.
    """
    def submit(self, request) -> None: ...
    def poll_ready(self, now_mono_ns: int) -> list: ...
    def next_wakeup_mono_ns(self) -> int | None: ...
    def flush(self) -> list: ...

    @property
    def pending_count(self) -> int: ...
