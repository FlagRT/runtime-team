"""Version-pinned vLLM scheduler activation via launch configuration.

No site-packages writes, monkey patches, running-process mutation or restarts.
Plans include a native fallback and source hashes; execution rechecks them.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import re
import subprocess

from .discovery import inspect_vllm, run_probe

PROJECT_PATH = str(Path(__file__).resolve().parents[1])
_QUALNAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\Z")
_VALIDATE = r'''
import importlib, inspect, json, sys
cls_name = sys.argv[1]
module, name = cls_name.rsplit('.', 1)
cls = getattr(importlib.import_module(module), name)
from vllm.v1.core.sched.interface import SchedulerInterface
if not inspect.isclass(cls) or not issubclass(cls, SchedulerInterface):
    raise TypeError('custom scheduler must implement this installation\'s SchedulerInterface')
if inspect.isabstract(cls):
    raise TypeError('scheduler has unimplemented abstract methods')
base_sig = inspect.signature(SchedulerInterface.__init__)
params = list(base_sig.parameters.values())[1:]
kwargs = {p.name: None for p in params
          if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
inspect.signature(cls).bind(**kwargs)
print(json.dumps(dict(class_name=cls_name, signature=str(inspect.signature(cls)),
    source=inspect.getfile(cls), validation='import-and-interface-only')))
'''


def qualified_name(value: str) -> str:
    if not isinstance(value, str) or not _QUALNAME.fullmatch(value):
        raise ValueError("expected a dotted Python class name (package.module.Class)")
    return value


def plugin_environment(paths=()) -> dict:
    environment = dict(os.environ)
    # Make this checkout visible in the selected target environment and its
    # worker processes without installing or modifying that environment.
    # A checkout root contains only this project. An installed site-packages
    # root could contain incompatible dependencies for the target interpreter;
    # never inject that entire directory into another Python environment.
    entries = ([PROJECT_PATH] if (Path(PROJECT_PATH) / "pyproject.toml").is_file() else [])
    entries.extend(str(Path(p).expanduser().resolve()) for p in paths)
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Runtime state belongs to the working directory (or an explicit state
    # directory), never to an installed site-packages directory.
    cache = Path(os.environ.get("FLAGSCHEDULER_STATE_DIR", str(Path.cwd() / ".tmp"))).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    for key, child in {"TMPDIR": "", "XDG_CACHE_HOME": "cache", "HF_HOME": "huggingface",
                       "VLLM_CACHE_ROOT": "vllm", "TRITON_CACHE_DIR": "triton",
                       "TORCHINDUCTOR_CACHE_DIR": "torchinductor"}.items():
        environment[key] = str(cache / child)
    return environment


def vllm_installation(python: str) -> dict:
    environment = run_probe(python)
    if "error" in environment:
        raise ValueError(environment["error"])
    found = next((f for f in environment["frameworks"] if f["framework"] == "vllm"), None)
    if found is None:
        raise ValueError(f"vllm not installed in {python}")
    return {**found, "python": environment["python"], "prefix": environment["prefix"],
            "scheduler": inspect_vllm(Path(found["package_path"]))}


def validate_scheduler(python: str, class_name: str, paths=()) -> dict:
    qualified_name(class_name)
    proc = subprocess.run(
        [python, "-B", "-c", _VALIDATE, class_name], cwd=PROJECT_PATH,
        env=plugin_environment(paths), capture_output=True, text=True, timeout=90,
    )
    if proc.returncode:
        raise ValueError("scheduler validation failed: " + proc.stderr[-3000:].strip())
    # Frameworks may write initialization logs to stdout.
    return json.loads(proc.stdout.strip().splitlines()[-1])


def build_launch_plan(python: str, model: str, *, scheduler_class: str | None = None,
                      expected_version: str | None = None, plugin_paths=(),
                      extra_args=(), platform="nvidia", scheduler_mode="sync",
                      pin_scheduler_mode=False) -> dict:
    if not model or platform != "nvidia":
        raise ValueError("vLLM launch currently requires a model and platform=nvidia")
    install = vllm_installation(python)
    scheduler = install["scheduler"]
    if scheduler_class:
        qualified_name(scheduler_class)
        if not expected_version or install["version"] != expected_version:
            raise ValueError(f"custom scheduler requires --expect-version exactly {install['version']!r}")
        if not scheduler["scheduler_cls_field"] or not any(c["kind"] == "v1" for c in scheduler["candidates"]):
            raise ValueError("this installation has no recognized V1 scheduler_cls extension")
    # These options are owned by the plan, not by opaque pass-through args.
    reserved = ("--scheduler-cls", "--scheduler-config", "--config", "--model",
                "--async-scheduling", "--no-async-scheduling")
    if any(a.split("=", 1)[0] in reserved for a in extra_args):
        raise ValueError("extra args cannot override model, config or scheduler ownership")
    native = [install["python"], "-B", "-m", "vllm.entrypoints.cli.main", "serve", model, *extra_args]
    if scheduler_mode not in ("sync", "async"):
        raise ValueError("scheduler_mode must be sync or async")
    builtin_modes = {
        "flagscheduler.integrations.vllm_scheduler.FlagScheduler": "sync",
        "flagscheduler.integrations.vllm_scheduler.FlagAsyncScheduler": "async",
    }
    if scheduler_class in builtin_modes and builtin_modes[scheduler_class] != scheduler_mode:
        raise ValueError("built-in scheduler class and scheduler_mode do not match")
    mode_args = ["--async-scheduling" if scheduler_mode == "async" else "--no-async-scheduling"]
    if pin_scheduler_mode:
        native += mode_args
    selected = native + (["--scheduler-cls", scheduler_class] + ([] if pin_scheduler_mode else mode_args)
                         if scheduler_class else [])
    return {
        "schema_version": 1, "framework": "vllm", "platform": platform,
        "model": model, "scheduler_mode": scheduler_mode, "extra_args": list(extra_args),
        "pin_scheduler_mode": pin_scheduler_mode,
        "installation": install, "scheduler_class": scheduler_class,
        "plugin_paths": [str(Path(p).expanduser().resolve()) for p in plugin_paths],
        "argv": selected, "native_argv": native,
        "activation": "new-process-only", "validation": "static; run launch --execute for import validation",
        "notes": ["Existing framework processes are unchanged.",
                  "Custom scheduler API is private and version dependent.",
                  "No model/GPU execution has been certified by this plan."],
    }


def execute_launch_plan(plan: dict) -> int:
    if plan.get("schema_version") != 1 or plan.get("framework") != "vllm":
        raise ValueError("unsupported launch plan schema/framework")
    install = plan["installation"]
    current = vllm_installation(install["python"])
    if (current["version"], current["package_path"], current["scheduler"]["fingerprints"]) != (
        install["version"], install["package_path"], install["scheduler"]["fingerprints"]
    ):
        raise ValueError("vLLM installation changed since planning; regenerate the plan")
    rebuilt = build_launch_plan(install["python"], plan["model"],
                               scheduler_class=plan["scheduler_class"],
                               expected_version=install["version"],
                               scheduler_mode=plan["scheduler_mode"],
                               pin_scheduler_mode=plan.get("pin_scheduler_mode", False),
                               extra_args=plan["extra_args"], plugin_paths=plan["plugin_paths"])
    if plan["argv"] != rebuilt["argv"] or plan["native_argv"] != rebuilt["native_argv"]:
        raise ValueError("launch argv differs from declared plan settings")
    choice = plan.get("scheduler_choice")
    if choice:
        from .catalog import file_hash
        if (file_hash(choice["script_path"]) != choice["sha256"]
                or file_hash(choice["snapshot_path"]) != choice["sha256"]):
            raise ValueError("selected scheduler script/snapshot changed since planning")
    if plan["scheduler_class"]:
        validate_scheduler(install["python"], plan["scheduler_class"], plan["plugin_paths"])
    # Foreground process inherits terminal and environment (including
    # CUDA_VISIBLE_DEVICES); process supervision belongs to systemd/container.
    return subprocess.call(plan["argv"], env=plugin_environment(plan["plugin_paths"]))


def load_symbol(name: str):
    qualified_name(name)
    module, symbol = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), symbol)


def write_json(path: str, value: dict) -> None:
    """Create a review artifact; refuse accidental replacement of an old plan."""
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
