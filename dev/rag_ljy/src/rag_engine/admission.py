"""Event-loop-owned memory admission with conservative whole-request budgets.

Reservations are accounting, not accelerator allocations. They stay charged for
all stages and until a sample started after completion can observe retained
allocator memory. Reservations affect the free-memory budget check only; watermarks use device usage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from time import monotonic

from .memory_probe import validate_memory_stats


@dataclass(frozen=True)
class MemoryAdmissionConfig:
    enabled: bool = False
    request_budget_mib: float | None = None
    safety_margin_mib: float = 5120
    high_watermark: float = .85
    low_watermark: float = .75
    max_sample_age_s: float = 3
    allow_unverified_scope: bool = False
    estimate_requests: bool = False

    def __post_init__(self):
        def number(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

        if (not number(self.high_watermark) or not number(self.low_watermark)
                or not 0 < self.low_watermark < self.high_watermark < 1):
            raise ValueError("Require 0 < memory low watermark < high watermark < 1")
        if not number(self.safety_margin_mib) or self.safety_margin_mib < 0:
            raise ValueError("Memory safety margin must be finite and nonnegative")
        if not number(self.max_sample_age_s) or self.max_sample_age_s <= 0:
            raise ValueError("Memory sample maximum age must be finite and positive")
        if self.request_budget_mib is not None and (
                not number(self.request_budget_mib) or self.request_budget_mib <= 0):
            raise ValueError("Per-request memory budget must be finite and positive")
        if self.estimate_requests and self.request_budget_mib is not None:
            raise ValueError("Dynamic estimation cannot use a fixed per-request budget")
        if self.enabled and not self.estimate_requests and self.request_budget_mib is None:
            raise ValueError("Memory admission requires an explicit per-request budget in MiB")


class MemoryAdmissionController:
    """Call only on the API event loop, without awaiting between reserve/register.

    Completed reservations wait for a newer observation. This prevents a burst
    of fast requests from repeatedly spending the same stale free-memory value.
    Existing accepted requests always drain, even when new admission is paused.
    """

    def __init__(self, config: MemoryAdmissionConfig, *, clock=monotonic):
        self.config = config
        self._clock = clock
        self._active = {}
        self._cache_credits = {}
        self._completed = {}
        self._pressure = False
        self.accepted = 0
        self.rejected = 0
        self.last_decision = None

    def _accounting(self):
        active = sum(self._active.values())
        pending = sum(budget for budget, _ in self._completed.values())
        return {"active_reserved_mib": active, "pending_release_mib": pending,
                "reserved_mib": active + pending, "reserved_requests": len(self._active),
                "pending_release_requests": len(self._completed)}

    def _evaluate(self, sample, request_budget_mib=None):
        result = {"enabled": True, "status": "unavailable", "reason": "telemetry_unavailable",
                  **self._accounting()}
        if not isinstance(sample, dict) or sample.get("status") != "ok":
            return result
        try:
            stats = validate_memory_stats(sample.get("stats"))
            if not all(key in stats for key in ("total_mb", "used_mb", "free_mb")):
                raise ValueError("Incomplete memory statistics")
            age = sample["age_seconds"]
            started = sample["sample_started_monotonic"]
            now = self._clock()
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in (age, started)) or age < 0 or started > now:
                raise ValueError("Invalid sample timing")
            age = max(age, now - started)
        except (KeyError, TypeError, ValueError):
            result["reason"] = "invalid_telemetry"
            return result
        result["sample_age_seconds"] = round(age, 3)
        if sample.get("stale") or age > self.config.max_sample_age_s:
            result["reason"] = "stale_telemetry"
            return result
        scope = sample.get("scope")
        result["scope"] = scope
        if scope != "device" and not (
                scope == "backend_reported_unverified" and self.config.allow_unverified_scope):
            result["reason"] = "unverified_memory_scope"
            return result
        if sample.get("unit") != "MiB":
            result["reason"] = "invalid_memory_unit"
            return result

        # Only a valid read STARTED after completion may release its reservation.
        self._completed = {key: value for key, value in self._completed.items()
                           if value[1] >= started}
        total, free = stats["total_mb"], stats["free_mb"]
        used = max(stats["used_mb"], total - free)
        # Cached memory credited to one idle-start request is already in measured
        # usage. If usage falls (e.g. GC), promote that credit into its reservation
        # BEFORE anyone can spend the newly free space. Promotions never reverse.
        for key, credit in self._cache_credits.items():
            credit['promoted'] = max(credit['promoted'], min(credit['mib'], max(0, credit['baseline'] - used)))
            self._active[key] = credit['base_budget'] + credit['promoted']
        result.update(self._accounting())
        reserved = result["reserved_mib"]
        effective_used = used + reserved
        available = free - reserved - self.config.safety_margin_mib
        result.update(free_mib=free, total_mib=total, measured_used_mib=used,
                      available_after_reservations_mib=available,
                      effective_used_percent=round(100 * effective_used / total, 3),
                      measured_used_percent=round(100 * used / total, 3),
                      watermark_basis="device_measured_usage")
        if used >= self.config.high_watermark * total:
            self._pressure = True
        elif self._pressure and used <= self.config.low_watermark * total:
            self._pressure = False
        if self._pressure:
            result.update(status="paused", reason="memory_pressure")
        elif available < (request_budget_mib if request_budget_mib is not None else self.config.request_budget_mib or 0):
            result.update(status="paused", reason="request_budget")
        else:
            result.update(status="ready", reason="memory_available")
        return result

    def snapshot(self, sample):
        if not self.config.enabled:
            return {"enabled": False, "status": "disabled"}
        return {**self._evaluate(sample), "config": asdict(self.config),
                "accepted": self.accepted, "rejected": self.rejected,
                "last_decision": self.last_decision}

    def reserve(self, request_id, sample, *, request_budget_mib=None, cache_credit_mib=0):
        if request_id in self._active or request_id in self._completed:
            raise ValueError("Request already owns a memory reservation")
        budget = request_budget_mib if request_budget_mib is not None else self.config.request_budget_mib
        valid_budget = (isinstance(budget, (int, float)) and not isinstance(budget, bool)
                        and math.isfinite(budget) and budget > 0)
        if self.config.enabled and not valid_budget:
            state = {"status": "unavailable", "reason": "invalid_request_estimate"}
        else:
            state = self._evaluate(sample, budget) if self.config.enabled else {"status": "ready", "reason": "disabled"}
        valid_credit = (isinstance(cache_credit_mib, (int, float)) and not isinstance(cache_credit_mib, bool)
                        and math.isfinite(cache_credit_mib) and cache_credit_mib >= 0)
        if not valid_credit or (cache_credit_mib and (self._active or self._completed)):
            state = {'status':'unavailable', 'reason':'invalid_cache_credit'}
        accepted = state["status"] == "ready"
        if accepted:
            if self.config.enabled:
                self._active[request_id] = budget
                if cache_credit_mib:
                    self._cache_credits[request_id] = {'mib':cache_credit_mib, 'baseline':state['measured_used_mib'],
                                                      'base_budget':budget, 'promoted':0}
            self.accepted += 1
        else:
            self.rejected += 1
        decision = {"request_id": request_id, "accepted": accepted, "reason": state["reason"],
                    "decided_at": datetime.now(timezone.utc).isoformat(),
                    "sample_age_seconds": state.get("sample_age_seconds"), **self._accounting()}
        decision['request_budget_mib'] = budget if valid_budget else None
        decision['cache_credit_mib'] = cache_credit_mib if valid_credit else None
        self.last_decision = decision
        return decision

    def release(self, request_id):
        self._cache_credits.pop(request_id, None)
        budget = self._active.pop(request_id, None)
        if budget is not None:
            self._completed[request_id] = (budget, self._clock())

    def reject_unestimated(self, request_id, reason):
        """Audit fail-closed decisions made before a usable estimate exists."""
        if request_id in self._active or request_id in self._completed:
            raise ValueError('Cannot reject a request that already owns a reservation')
        self.rejected += 1
        self.last_decision = {'request_id': request_id, 'accepted': False, 'reason': reason,
                             'request_budget_mib': None,
                             'decided_at': datetime.now(timezone.utc).isoformat(), **self._accounting()}
        return self.last_decision
