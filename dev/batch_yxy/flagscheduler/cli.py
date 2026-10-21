"""CLI adapter. All operations delegate to reusable application services."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from . import __version__


def emit(value, stream=None):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), file=stream or sys.stdout, flush=True)


def _target_args(parser):
    parser.add_argument("--python", required=True, help="Python executable of the target vLLM environment")


def _launch_args(parser):
    parser.add_argument("--python", help="Python executable of the target environment")
    parser.add_argument("--model", help="Local model path or repository ID")
    parser.add_argument("--scheduler-class", help="One-off dotted class override; omitted = saved selection")
    parser.add_argument("--expect-version", help="Exact vLLM version required for a custom scheduler")
    parser.add_argument("--scheduler-mode", choices=("sync", "async"), default="sync")
    parser.add_argument("--plugin-path", action="append", default=[])
    parser.add_argument("--arg", action="append", default=[], help="vLLM argument; use --arg=--flag and --arg=value")
    parser.add_argument("--output", help="Create a JSON plan, refusing to overwrite")
    parser.add_argument("--state-dir", help="Scheduler catalog directory inside batch_yxy")


def build_parser():
    parser = argparse.ArgumentParser(prog="flagscheduler", description="Node-local inference scheduling and framework control")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Find frameworks in current/Conda/explicit Python environments")
    discover.add_argument("--python", action="append", default=[])
    discover.add_argument("--env-root", action="append", default=[])
    commands.add_parser("doctor", help="Report local NVIDIA readiness without initializing CUDA")
    commands.add_parser("models", help="List the fifteen target model contracts and validation scope")
    config = commands.add_parser("config", help="Validate node configuration without loading models")
    config.add_argument("path")
    scheduler = commands.add_parser("scheduler", help="Inspect/select the installed framework scheduler")
    sub = scheduler.add_subparsers(dest="operation", required=True)
    inspect = sub.add_parser("inspect")
    _target_args(inspect)
    for operation, help_text in (("list", "List scripts with Default/Currently labels"),
                                 ("select", "Save a scheduler choice for subsequent launches"),
                                 ("add", "Register a custom Python scheduler script")):
        operation_parser = sub.add_parser(operation, help=help_text)
        _target_args(operation_parser)
        operation_parser.add_argument("--scheduler-mode", choices=("sync", "async"), default="sync")
        operation_parser.add_argument("--state-dir", help="Catalog directory inside batch_yxy")
        operation_parser.add_argument("--json", action="store_true", help="Emit catalog JSON for tools/TUI")
        if operation in ("select", "add"):
            operation_parser.add_argument("script_id", help="default, flagscheduler, or a registered ID")
        if operation == "add":
            operation_parser.add_argument("--class", dest="class_name", required=True)
            operation_parser.add_argument("--script", required=True)
            operation_parser.add_argument("--expect-version", required=True)
            operation_parser.add_argument("--plugin-path", action="append", default=[])
    validate = sub.add_parser("validate", help="Import and inspect a trusted plugin; no model instantiation")
    _target_args(validate)
    validate.add_argument("--class", dest="class_name", required=True)
    validate.add_argument("--plugin-path", action="append", default=[])
    plan = sub.add_parser("plan", help="Generate a version-pinned replacement/native launch plan")
    _launch_args(plan)
    launch = commands.add_parser("launch", help="Review a launch plan or explicitly execute it in the foreground")
    _launch_args(launch)
    launch.add_argument("--plan", help="Read a previously generated plan")
    launch.add_argument("--native", action="store_true", help="Use the native fallback from --plan")
    launch.add_argument("--execute", action="store_true", help="Start a new framework process")
    run = commands.add_parser("run", help="Execute a JSONL file through the node runtime")
    run.add_argument("--config", required=True)
    run.add_argument("--input", required=True, help="JSONL path or - for stdin")
    serve = commands.add_parser("serve", help="Run a persistent loopback node API")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    status = commands.add_parser("status", help="Read a running node's state")
    status.add_argument("--url", default="http://127.0.0.1:8765")
    submit = commands.add_parser("submit", help="Submit one request JSON file to the local node")
    submit.add_argument("--url", default="http://127.0.0.1:8765")
    submit.add_argument("--request", required=True)
    return parser


def _plan(args):
    from flagscheduler.deployment import build_launch_plan
    if not args.python or not args.model:
        raise ValueError("--python and --model are required when generating a plan")
    if args.scheduler_class is None:
        from .catalog import SchedulerCatalog
        return SchedulerCatalog(args.python, mode=args.scheduler_mode, state_dir=args.state_dir).build_plan(
            args.model, expected_version=args.expect_version, plugin_paths=args.plugin_path, extra_args=args.arg)
    return build_launch_plan(args.python, args.model, scheduler_class=args.scheduler_class,
                             expected_version=args.expect_version, plugin_paths=args.plugin_path,
                             extra_args=args.arg, scheduler_mode=args.scheduler_mode)


def render_catalog(catalog):
    print(f"vLLM {catalog['version']} | {catalog['mode']} | {catalog['python']}")
    print("Currently = saved selection for subsequent launches")
    for script in catalog["scripts"]:
        labels = " ".join(f"({label})" for label in script["labels"])
        print(f"\n{script['id']} {labels}".rstrip())
        print(f"  Class: {script['class_name']}")
        print(f"  Script: {script['script_path']}")
        print(f"  Preserved: {script['snapshot_path']}")
        if not script["source_unchanged"] or not script["snapshot_intact"]:
            print("  Changed: inspect and reselect before launch")


async def run_jsonl(config_path, input_path):
    from .config import load_config
    from .contracts import Request
    from .runtime import NodeRuntime
    runtime = NodeRuntime(load_config(config_path))
    await runtime.start()
    pending = set()
    failed = False

    async def deliver(future):
        nonlocal failed
        value = await future
        failed |= value["status"] == "failed"
        # One complete object per line, suitable for piping and streaming.
        print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)

    source = sys.stdin if input_path == "-" else Path(input_path).open(encoding="utf-8")
    try:
        line_number = 0
        while True:
            line = await asyncio.to_thread(source.readline)
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            # Bound JSONL-side outstanding tasks too, rather than rejecting a
            # large offline file because the device is busy.
            if len(pending) >= runtime.config.max_pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            try:
                request = Request.from_dict(json.loads(line))
                pending.add(asyncio.create_task(deliver(runtime.submit(request))))
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                failed = True
                print(json.dumps({"schema_version": 1, "status": "rejected", "line": line_number,
                                  "error": str(exc)}, ensure_ascii=False), flush=True)
            done = {task for task in pending if task.done()}
            for task in done:
                task.result()
            pending -= done
    finally:
        if source is not sys.stdin:
            source.close()
        await runtime.stop()
        if pending:
            await asyncio.gather(*pending)
    return 1 if failed else 0


async def serve_node(args):
    from .config import load_config
    from .runtime import NodeRuntime
    from .server import start_server
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    runtime = NodeRuntime(load_config(args.config))
    await runtime.start()
    server = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    try:
        server = await start_server(runtime, args.host, args.port)
        emit({"event": "listening", "addresses": [s.getsockname() for s in server.sockets]}, sys.stderr)
        stop_waiter = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait((stop_waiter, runtime._task), return_when=asyncio.FIRST_COMPLETED)
        if runtime._task in done:
            runtime._task.result()
        stop_waiter.cancel()
    finally:
        if server:
            server.close()
            await server.wait_closed()
        await runtime.stop()


def _local_http(url, route, payload=None):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1", "::1") or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("--url must be a loopback HTTP base URL")
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(url.rstrip("/") + route, data=data, headers={"Content-Type": "application/json"})
    # Local IPC must not pass through an environment-configured HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            return json.load(response), 0
    except urllib.error.HTTPError as exc:
        return json.load(exc), 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "discover":
            from flagscheduler.discovery import discover
            emit(discover(args.python, args.env_root))
        elif args.command == "doctor":
            from flagscheduler.discovery import nvidia_inventory
            report = nvidia_inventory()
            emit(report)
            return 0 if report["available"] else 1
        elif args.command == "models":
            from sched_engine.models.registry import all_profiles
            emit({"schema_version": 1, "models": [
                {**{key: value for key, value in asdict(p).items() if key != "token_counter"},
                 "node_support": "contract + mock; real backend requires validation"} for p in all_profiles()]})
        elif args.command == "config":
            from .config import load_config
            emit({"valid": True, "config": asdict(load_config(args.path))})
        elif args.command == "scheduler" and args.operation == "inspect":
            from flagscheduler.deployment import vllm_installation
            emit(vllm_installation(args.python))
        elif args.command == "scheduler" and args.operation in ("list", "select", "add"):
            from .catalog import SchedulerCatalog
            catalog = SchedulerCatalog(args.python, mode=args.scheduler_mode, state_dir=args.state_dir)
            if args.operation == "select":
                view = catalog.select(args.script_id)
            elif args.operation == "add":
                view = catalog.add(args.script_id, args.class_name, args.script,
                                   expect_version=args.expect_version, plugin_paths=args.plugin_path)
            else:
                view = catalog.view()
            emit(view) if args.json else render_catalog(view)
        elif args.command == "scheduler" and args.operation == "validate":
            from flagscheduler.deployment import validate_scheduler
            emit(validate_scheduler(args.python, args.class_name, args.plugin_path))
        elif args.command == "launch" or (args.command == "scheduler" and args.operation == "plan"):
            from flagscheduler.deployment import execute_launch_plan, write_json
            plan = json.loads(Path(args.plan).read_text()) if getattr(args, "plan", None) else _plan(args)
            if getattr(args, "native", False):
                if not getattr(args, "plan", None):
                    raise ValueError("--native requires --plan")
                plan["scheduler_class"] = None
                plan["argv"] = plan["native_argv"]
                plan.pop("scheduler_choice", None)
            if args.output:
                write_json(args.output, plan)
            if getattr(args, "execute", False):
                emit(plan, sys.stderr)
                return execute_launch_plan(plan)
            emit(plan)
        elif args.command == "run":
            return asyncio.run(run_jsonl(args.config, args.input))
        elif args.command == "serve":
            asyncio.run(serve_node(args))
        elif args.command in ("status", "submit"):
            payload = json.loads(Path(args.request).read_text()) if args.command == "submit" else None
            result, code = _local_http(args.url, "/v1/infer" if payload is not None else "/v1/status", payload)
            emit(result)
            return code
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit({"error": type(exc).__name__, "message": str(exc)}, sys.stderr)
        return 2
