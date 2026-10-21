"""vLLM 0.21.0 scheduler extension starting points.

These intentionally inherit native behavior. They are integration seams,
not a new scheduling algorithm or hardware certification. Develop iteration
policies here only with version-specific KV/cache/preemption regressions.
"""
from importlib.metadata import version

if version("vllm") != "0.21.0":
    raise RuntimeError("This integration targets vLLM 0.21.0; use a version-specific plugin")

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class FlagScheduler(Scheduler):
    """Native synchronous scheduling; safe starting point for custom policies."""


class FlagAsyncScheduler(AsyncScheduler):
    """Native asynchronous scheduling; select explicitly for async mode."""
