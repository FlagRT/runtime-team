"""Cached accelerator-memory telemetry via the device-context runtime."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib
import math
import os
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
        self._probe = probe if probe is not None else (IsolatedMemoryProbe(device) if os.environ.get('RAG_MEMORY_PROBE_PROCESS') == '1' else DeviceMemoryProbe(device))
        self._latest = {"enabled": enabled, "status": "pending" if enabled else "disabled"}
        self._sample_time = None
        self._stop = asyncio.Event()
        self._executor = None
        self._task = None
        self.on_sample = None

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
            started = monotonic()
            try:
                self._latest = await asyncio.get_running_loop().run_in_executor(self._executor, self._probe.sample)
                if not isinstance(self._latest, dict):
                    raise ValueError("Memory probe must return a dictionary")
            except Exception as error:
                self._latest = {"enabled": True, "status": "unavailable", "stats": None,
                                "error_type": type(error).__name__, "error": str(error)[:300]}
            self._latest = {**self._latest, "sample_started_monotonic": started}
            self._sample_time = started
            if self.on_sample is not None:
                self.on_sample(self.snapshot())
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
        if hasattr(self._probe, 'close'):
            self._probe.close()


def _isolated_probe_worker(connection, device):
    probe = DeviceMemoryProbe(device)
    try:
        while connection.recv() == 'sample':
            result = probe.sample()
            import os
            result['sampling_process'] = os.getpid()
            connection.send(result)
    except EOFError:
        pass
    finally:
        connection.close()


class IsolatedMemoryProbe:
    """Read the same runtime API without waiting on the model allocator's lock.

    The observer creates a device context but no tensors or pressure allocations.
    Its persistent context overhead is included in the measured idle baseline.
    """
    def __init__(self, device):
        self.device = device
        self.process = None
        self.connection = None

    def sample(self):
        if self.process is None:
            import multiprocessing
            ctx = multiprocessing.get_context('spawn')
            self.connection, child = ctx.Pipe()
            self.process = ctx.Process(target=_isolated_probe_worker, args=(child,self.device),daemon=True)
            self.process.start()
            child.close()
        if not self.process.is_alive():
            self.close()
            raise RuntimeError('Isolated memory sampler exited')
        self.connection.send('sample')
        # Import/context initialization may take longer than steady-state reads.
        if not self.connection.poll(30):
            self.close()
            raise TimeoutError('Isolated memory sampler timed out')
        return self.connection.recv()

    def close(self):
        if self.process is not None:
            try: self.connection.send('stop')
            except (OSError,EOFError): pass
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            self.connection.close()
            self.process = None
            self.connection = None
