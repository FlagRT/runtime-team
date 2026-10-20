"""Validate runtime routing, untrusted metrics and sampler failure recovery."""
import asyncio
import math
import sys
import threading
from types import SimpleNamespace

import pytest

from rag_engine import memory_probe


@pytest.mark.parametrize("raw", [None, {}, {"total_mb": 0}, {"used_mb": -1},
    {"used_mb": math.nan}, {"free_mb": math.inf}, {"used_mb": True},
    {"total_mb": 100, "free_mb": 101}, {"total_mb": 100, "used_mb": 60, "free_mb": 60}])
def test_invalid_metrics_rejected(raw):
    with pytest.raises(ValueError):
        memory_probe.validate_memory_stats(raw)


def test_device_context_used_with_correct_ordinal_and_partial_stats(monkeypatch):
    monkeypatch.setenv("DC_BACKEND", "ascend")
    calls = []
    monkeypatch.setattr(memory_probe, "prepare_torch_device", lambda d: (calls.append(("bind", d)), d))
    runtime = SimpleNamespace(current_name=lambda: "ascend", memory_stats=lambda n: (
        calls.append(("memory", n)) or {"total_mb": 100, "used_mb": 20}))
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    probe = memory_probe.DeviceMemoryProbe("npu:3")
    for _ in range(2):
        sample = probe.sample()
        assert sample["status"] == "partial"
        assert sample["scope"] == "backend_reported_unverified"
        assert sample["stats"] == {"total_mb": 100, "used_mb": 20, "memory_used_percent": 20}
        assert "free_mb" not in sample["stats"]
    assert calls == [("bind", "npu:3"), ("memory", 3), ("memory", 3)]
    runtime.current_name = lambda: "kunlun"
    assert probe.sample()["status"] == "unavailable"
    assert len(calls) == 3


def test_probe_failure_does_not_return_previous_success(monkeypatch):
    monkeypatch.setenv("DC_BACKEND", "ascend")
    monkeypatch.setattr(memory_probe, "prepare_torch_device", lambda d: (None, "npu:0"))
    runtime = SimpleNamespace(current_name=lambda: "ascend", memory_stats=lambda n: {
        "total_mb": 100, "used_mb": 20, "free_mb": 80})
    monkeypatch.setitem(sys.modules, "runtime", runtime)
    probe = memory_probe.DeviceMemoryProbe()
    assert probe.sample()["status"] == "ok"
    def fail(_):
        raise RuntimeError("driver read failed")
    runtime.memory_stats = fail
    failed = probe.sample()
    assert failed["status"] == "unavailable" and failed["stats"] is None
    assert failed["error_type"] == "RuntimeError"


def test_cpu_and_native_cuda_do_not_import_vendor_runtime(monkeypatch):
    monkeypatch.delenv("DC_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "runtime", None)
    assert memory_probe.DeviceMemoryProbe("cpu").sample()["status"] == "not_applicable"
    assert memory_probe.DeviceMemoryProbe("cuda:0").sample()["status"] == "unavailable"


def test_background_sampler_recovers_and_keeps_one_sampling_thread():
    calls = []
    def sample():
        calls.append(threading.get_ident())
        if len(calls) == 1:
            raise RuntimeError("temporary read failure")
        return {"enabled": True, "status": "ok", "stats": {"total_mb": 100}}
    async def scenario():
        monitor = memory_probe.DeviceMemoryMonitor(interval_s=.01, probe=SimpleNamespace(sample=sample))
        monitor.start()
        try:
            async with asyncio.timeout(2):
                while monitor.snapshot()["status"] != "ok":
                    await asyncio.sleep(.005)
            assert monitor.snapshot()["age_seconds"] >= 0
            assert not monitor.snapshot()["stale"]
            monitor._sample_time -= 10
            assert monitor.snapshot()["stale"]
        finally:
            await monitor.close()
        assert len(calls) >= 2 and len(set(calls)) == 1
        assert calls[0] != threading.get_ident()
    asyncio.run(scenario())


@pytest.mark.parametrize("interval", [0, -1, math.nan, math.inf])
def test_invalid_intervals(interval):
    with pytest.raises(ValueError):
        memory_probe.DeviceMemoryMonitor(interval_s=interval)
