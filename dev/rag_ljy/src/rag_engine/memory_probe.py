"""Cached accelerator-memory telemetry via the device-context runtime."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import math
from time import monotonic

from .device import device_selection, prepare_torch_device


def validate_memory_stats(raw):
    """Keep the runtime's MiB fields; never invent missing device-wide values."""
    if not isinstance(raw, dict):
        raise ValueError("runtime.memory_stats must return a dictionary")
    stats = {}
    for key in ("total_mb", "used_mb", "free_mb"):
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid runtime memory field: {key}")
        stats[key] = value
    if not stats or ("total_mb" in stats and stats["total_mb"] <= 0):
        raise ValueError("Runtime did not return usable memory statistics")
    if "total_mb" in stats:
        total = stats["total_mb"]
        if any(stats.get(key, 0) > total for key in ("used_mb", "free_mb")):
            raise ValueError("Reported memory exceeds device total")
        if len(stats) == 3 and abs(stats["used_mb"] + stats["free_mb"] - total) > 2:
            raise ValueError("Runtime memory totals are inconsistent")
        if "used_mb" in stats:
            stats["memory_used_percent"] = round(100 * stats["used_mb"] / total, 2)
    return stats


class DeviceMemoryProbe:
    """Bind once on the sampling thread; each read uses runtime.memory_stats."""

    def __init__(self, device="auto"):
        self.requested_device = device
        self.backend, self.prefix, self.ordinal = device_selection(device)
        self.device = device
        self._runtime = None

    def sample(self):
        started = monotonic()
        result = {
            "enabled": True, "source": "device-context.runtime.memory_stats",
            "backend": self.backend, "device": self.device, "ordinal": self.ordinal,
            "unit": "MiB", "scope": "backend_reported_unverified",
            "stats": None,
        }
        try:
            if self.backend is None:
                result["status"] = "not_applicable" if self.prefix == "cpu" else "unavailable"
                if self.prefix != "cpu":
                    result["error"] = "Select a device-context backend with DC_BACKEND for memory probing"
            else:
                if self._runtime is None:
                    _, resolved = prepare_torch_device(self.requested_device)
                    self.device = str(resolved)
                    self._runtime = importlib.import_module("runtime")
                if self._runtime.current_name() != self.backend:
                    raise RuntimeError("Device-context backend changed while memory probing was active")
                stats = validate_memory_stats(self._runtime.memory_stats(self.ordinal))
                result.update(device=self.device, stats=stats,
                              status="ok" if all(k in stats for k in ("total_mb", "used_mb", "free_mb")) else "partial")
        except Exception as error:
            result.update(status="unavailable", error_type=type(error).__name__, error=str(error)[:300])
        result.update(sampled_at=datetime.now(timezone.utc).isoformat(),
                      sample_duration_ms=round((monotonic() - started) * 1000, 3))
        return result


class DeviceMemoryMonitor:
    """One serial sampler, separate from both HTTP and model execution threads."""

    def __init__(self, device="auto", *, enabled=True, interval_s=1.0, probe=None):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("Memory probe interval must be finite and positive")
        self.enabled = enabled
        self.interval_s = interval_s
        self._probe = probe if probe is not None else DeviceMemoryProbe(device)
        self._latest = {"enabled": enabled, "status": "pending" if enabled else "disabled"}
        self._sample_time = None
        self._stop = asyncio.Event()
        self._executor = None
        self._task = None

    def start(self):
        if self.enabled and self._task is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-memory-probe")
            self._task = asyncio.create_task(self._run(), name="rag-memory-probe")

    def snapshot(self):
        snapshot = dict(self._latest)
        if self._sample_time is not None:
            age = monotonic() - self._sample_time
            snapshot.update(age_seconds=round(age, 3), stale=age > max(3 * self.interval_s, 5.0))
        return snapshot

    async def _run(self):
        while not self._stop.is_set():
            try:
                self._latest = await asyncio.get_running_loop().run_in_executor(self._executor, self._probe.sample)
            except Exception as error:
                self._latest = {"enabled": True, "status": "unavailable", "stats": None,
                                "error_type": type(error).__name__, "error": str(error)[:300]}
            self._sample_time = monotonic()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    async def close(self):
        self._stop.set()
        if self._task is not None:
            await self._task
        if self._executor is not None:
            self._executor.shutdown(wait=True)
