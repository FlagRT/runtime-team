"""Bounded discovery of Python environments and framework source locations."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

FRAMEWORKS = {
    "vllm": "vllm", "sglang": "sglang", "transformers": "transformers",
    "onnxruntime": "onnxruntime", "tensorflow": "tensorflow",
    "tritonclient": "tritonclient", "torchrec": "torchrec",
    "tensorrt": "tensorrt",
}
# Runs in the selected environment, outside the project cwd. Does not import
# vLLM/torch or traverse framework submodules (which could initialize CUDA).
_PROBE = r'''
import importlib.metadata as md, importlib.util, json, sys
frameworks = json.loads(sys.argv[1])
found = []
for module, distribution in frameworks.items():
    spec = importlib.util.find_spec(module)
    if spec is None:
        continue
    locations = list(spec.submodule_search_locations or [])
    try:
        version = md.version(distribution)
    except md.PackageNotFoundError:
        version = None
    found.append(dict(framework=module, version=version,
                      package_path=locations[0] if locations else spec.origin))
print(json.dumps(dict(python=sys.executable, prefix=sys.prefix, frameworks=found)))
'''


def run_probe(python: str, timeout: float = 15) -> dict:
    try:
        proc = subprocess.run(
            [python, "-I", "-B", "-c", _PROBE, json.dumps(FRAMEWORKS)],
            cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if proc.returncode:
            return {"python": python, "error": proc.stderr[-1500:].strip()}
        return json.loads(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return {"python": python, "error": str(exc)}


def python_candidates(explicit=(), roots=()) -> list[str]:
    candidates = [sys.executable, *explicit]
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(key):
            candidates.append(str(Path(os.environ[key]) / "bin/python"))
    for name in ("python", "python3"):
        if shutil.which(name):
            candidates.append(shutil.which(name))
    # Conda's registry includes environments outside the default envs directory.
    registry = Path.home() / ".conda/environments.txt"
    if registry.is_file():
        candidates.extend(str(Path(p) / "bin/python") for p in registry.read_text().splitlines() if p)
    default_roots = [Path(sys.prefix) / "envs", Path.home() / ".conda/envs"]
    for root in [*default_roots, *(Path(p) for p in roots)]:
        if not root.is_dir():
            continue
        # Only root and its direct children; never recursively scan the disk.
        candidates.append(str(root / "bin/python"))
        candidates.extend(str(p / "bin/python") for p in sorted(root.iterdir())[:256] if p.is_dir())
    unique = {}
    for candidate in candidates:
        path = Path(candidate).expanduser().absolute()
        if path.is_file():
            # Keep the environment executable spelling, not its resolved base
            # interpreter symlink: venv detection depends on the former.
            unique.setdefault(str(path), str(path))
    return list(unique.values())


def discover(pythons=(), roots=()) -> dict:
    environments = []
    for python in python_candidates(pythons, roots):
        result = run_probe(python)
        for framework in result.get("frameworks", []):
            if framework["framework"] == "vllm":
                framework["scheduler"] = inspect_vllm(Path(framework["package_path"]))
        environments.append(result)
    return {"schema_version": 1, "environments": environments}


def inspect_vllm(package: Path) -> dict:
    files = {
        "v1": ("v1/core/sched/scheduler.py", "vllm.v1.core.sched.scheduler.Scheduler"),
        "v1_async": ("v1/core/sched/async_scheduler.py", "vllm.v1.core.sched.async_scheduler.AsyncScheduler"),
        "v0": ("core/scheduler.py", "vllm.core.scheduler.Scheduler"),
    }
    result = {"candidates": [], "scheduler_cls_field": False, "fingerprints": {}}
    for kind, (relative, cls) in files.items():
        path = package / relative
        if path.is_file():
            result["candidates"].append({"kind": kind, "class": cls, "path": str(path)})
    for relative in ("config.py", "config/scheduler.py", "engine/arg_utils.py", "v1/core/sched/interface.py", *(v[0] for v in files.values())):
        path = package / relative
        if not path.is_file():
            continue
        source = path.read_bytes()
        result["fingerprints"][relative] = hashlib.sha256(source).hexdigest()
        if relative in ("config.py", "config/scheduler.py"):
            try:
                tree = ast.parse(source)
                result["scheduler_cls_field"] |= any(
                    isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                    and n.target.id == "scheduler_cls" for n in ast.walk(tree)
                )
            except SyntaxError:
                pass
    result["default_selection"] = "runtime-dependent: sync/async and engine version"
    return result


def nvidia_inventory() -> dict:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"platform": "nvidia", "available": False, "reason": "nvidia-smi not found"}
    try:
        proc = subprocess.run(
            [command, "--query-gpu=index,uuid,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode:
            return {"platform": "nvidia", "available": False, "reason": proc.stderr.strip() or proc.stdout.strip()}
        import csv
        devices = [dict(zip(("index", "uuid", "name", "driver_version", "memory_mib"),
                            (v.strip() for v in row)))
                   for row in csv.reader(proc.stdout.splitlines()) if len(row) == 5]
        return {"platform": "nvidia", "available": bool(devices), "devices": devices,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"platform": "nvidia", "available": False, "reason": str(exc)}
