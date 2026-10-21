import json
from pathlib import Path
from unittest.mock import patch

import pytest

from flagscheduler.cli import main
from flagscheduler.config import parse_config
from flagscheduler.contracts import Request
from flagscheduler.discovery import inspect_vllm
from flagscheduler.deployment import build_launch_plan, execute_launch_plan, write_json


@pytest.fixture
def install(tmp_path):
    package = tmp_path / "vllm"
    for name in ("config/scheduler.py", "v1/core/sched/scheduler.py", "v1/core/sched/interface.py"):
        file = package / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("class SchedulerConfig:\n    scheduler_cls: str | None = None\n" if name.startswith("config") else "# fixture\n")
    return {"framework": "vllm", "python": "/env/bin/python", "version": "0.21.0",
            "package_path": str(package), "scheduler": inspect_vllm(package)}


def test_inspection_does_not_import_framework(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('never import during discovery')")
    assert inspect_vllm(package)["candidates"] == []


def test_version_pin_custom_command_and_native_fallback(install):
    with patch("flagscheduler.deployment.vllm_installation", return_value=install):
        with pytest.raises(ValueError, match="expect-version"):
            build_launch_plan("/env/bin/python", "/weights", scheduler_class="my.plugin.Scheduler")
        plan = build_launch_plan("/env/bin/python", "/weights", scheduler_class="my.plugin.Scheduler", expected_version="0.21.0")
    assert "--scheduler-cls" in plan["argv"]
    assert "--no-async-scheduling" in plan["argv"]
    assert "--scheduler-cls" not in plan["native_argv"]


def test_sync_async_builtin_modes_must_match(install):
    with patch("flagscheduler.deployment.vllm_installation", return_value=install):
        with pytest.raises(ValueError, match="do not match"):
            build_launch_plan("/env/bin/python", "weights", expected_version="0.21.0",
                              scheduler_class="flagscheduler.integrations.vllm_scheduler.FlagAsyncScheduler")


@pytest.mark.parametrize("argument", ["--scheduler-cls=evil.Class", "--config", "--model", "--async-scheduling"])
def test_opaque_extra_flags_cannot_override_owned_settings(install, argument):
    with patch("flagscheduler.deployment.vllm_installation", return_value=install):
        with pytest.raises(ValueError):
            build_launch_plan("/env/bin/python", "weights", extra_args=[argument])


def test_source_change_and_tampered_command_are_rejected(install):
    with patch("flagscheduler.deployment.vllm_installation", return_value=install):
        plan = build_launch_plan("/env/bin/python", "weights")
        plan["argv"] = ["something-else"]
        with pytest.raises(ValueError, match="argv"):
            execute_launch_plan(plan)
        plan["installation"] = {**install, "version": "stale"}
        with pytest.raises(ValueError, match="changed"):
            execute_launch_plan(plan)


def test_artifact_write_refuses_overwrite(tmp_path):
    file = tmp_path / "plan.json"
    write_json(str(file), {"schema_version": 1})
    with pytest.raises(FileExistsError):
        write_json(str(file), {})
    assert json.loads(file.read_text())["schema_version"] == 1


@pytest.mark.parametrize("mutation", [
    {"schema_version": 2}, {"unexpected": True}, {"max_pending": True},
    {"result_history": 0}, {"bindings": []},
])
def test_invalid_config_fails_without_loading_models(mutation):
    path = Path(__file__).resolve().parents[1] / "examples/flagscheduler.mock.json"
    value = json.loads(path.read_text())
    value.update(mutation)
    with pytest.raises((ValueError, TypeError)):
        parse_config(value)


@pytest.mark.parametrize("priority,slo", [(True, None), ("1", None), (0, float("nan")), (0, -1)])
def test_request_rejects_invalid_priority_or_slo(priority, slo):
    with pytest.raises(ValueError):
        Request("r", "binding", "text", priority=priority, slo_ms=slo)


def test_cli_models_has_fifteen_contracts_and_json_output(capsys):
    assert main(["models"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert len(value["models"]) == 15
    assert all("requires validation" in m["node_support"] for m in value["models"])


def test_cli_reports_invalid_config_without_traceback(tmp_path, capsys):
    file = tmp_path / "config.json"
    file.write_text('{"schema_version": 99}')
    assert main(["config", str(file)]) == 2
    assert json.loads(capsys.readouterr().err)["error"] == "ValueError"


def test_discovery_uses_conda_registry_without_importing_frameworks(tmp_path):
    from flagscheduler.discovery import python_candidates
    home = tmp_path / "home"
    registry = home / ".conda/environments.txt"
    registry.parent.mkdir(parents=True)
    environment = tmp_path / "separate-env"
    interpreter = environment / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("# fixture interpreter")
    registry.write_text(str(environment) + "\n")
    with patch("flagscheduler.discovery.Path.home", return_value=home):
        assert str(interpreter) in python_candidates()


def test_installed_control_environment_does_not_shadow_target_dependencies(tmp_path, monkeypatch):
    from flagscheduler.deployment import plugin_environment
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("FLAGSCHEDULER_STATE_DIR", str(tmp_path / "state"))
    with patch("flagscheduler.deployment.PROJECT_PATH", str(tmp_path / "site-packages")):
        environment = plugin_environment([str(tmp_path / "plugins")])
    assert environment["PYTHONPATH"] == str(tmp_path / "plugins")
    assert environment["TMPDIR"] == str(tmp_path / "state")
