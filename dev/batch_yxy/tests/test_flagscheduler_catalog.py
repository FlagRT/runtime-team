"""Script retention, saved selection and launch/binding integration."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from flagscheduler.catalog import SchedulerCatalog
from flagscheduler.cli import main
from flagscheduler.discovery import inspect_vllm
from flagscheduler.deployment import execute_launch_plan


@pytest.fixture
def installation(tmp_path):
    package = tmp_path / "vllm"
    for path, source in {
        "config/scheduler.py": "class SchedulerConfig:\n    scheduler_cls: str | None = None\n",
        "v1/core/sched/scheduler.py": "class Scheduler:\n    pass\n",
        "v1/core/sched/async_scheduler.py": "class AsyncScheduler:\n    pass\n",
        "v1/core/sched/interface.py": "# interface fixture\n",
    }.items():
        file = package / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
    return {"framework": "vllm", "version": "0.21.0", "python": "/env/bin/python",
            "package_path": str(package), "scheduler": inspect_vllm(package)}


def catalog(installation, tmp_path, mode="sync"):
    return SchedulerCatalog(installation["python"], installation=installation,
                            state_dir=tmp_path / "state", mode=mode)


def test_default_is_preserved_and_initially_current(installation, tmp_path):
    node = catalog(installation, tmp_path)
    before = {c["path"]: Path(c["path"]).read_bytes() for c in installation["scheduler"]["candidates"]}
    view = node.view()
    native = next(e for e in view["scripts"] if e["default"])
    assert native["labels"] == ["Default", "Currently"]
    assert sum(e["current"] for e in view["scripts"]) == 1
    for source, content in before.items():
        assert Path(source).read_bytes() == content
        snapshot = node.root / "originals" / (hashlib.sha256(content).hexdigest() + ".py")
        assert snapshot.read_bytes() == content
        assert snapshot.stat().st_mode & 0o222 == 0


def test_selection_persists_and_original_can_be_selected_again(installation, tmp_path):
    node = catalog(installation, tmp_path)
    changed = node.select("flagscheduler")
    assert next(e for e in changed["scripts"] if e["id"] == "default")["labels"] == ["Default"]
    assert next(e for e in changed["scripts"] if e["id"] == "flagscheduler")["labels"] == ["Currently"]
    assert catalog(installation, tmp_path).current()["id"] == "flagscheduler"
    restored = node.select("default")
    assert next(e for e in restored["scripts"] if e["current"])["labels"] == ["Default", "Currently"]


def test_sync_and_async_choices_are_isolated(installation, tmp_path):
    sync = catalog(installation, tmp_path)
    sync.select("flagscheduler")
    asynchronous = catalog(installation, tmp_path, "async")
    assert asynchronous.current()["id"] == "default"
    assert asynchronous.current()["class_name"].endswith("AsyncScheduler")
    asynchronous.select("flagscheduler")
    assert asynchronous.current()["class_name"].endswith("FlagAsyncScheduler")
    assert sync.current()["class_name"].endswith(".FlagScheduler")


def test_python_aliases_share_selection_but_other_environments_do_not(installation, tmp_path):
    original = catalog(installation, tmp_path)
    original.select("flagscheduler")
    alias = catalog({**installation, "python": "/env/bin/python3.12"}, tmp_path)
    assert alias.path == original.path
    assert alias.current()["id"] == "flagscheduler"
    other = catalog({**installation, "python": "/different-env/bin/python"}, tmp_path)
    assert other.path != original.path
    assert other.current()["id"] == "default"


def test_installed_catalog_keeps_state_out_of_site_packages(installation, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("flagscheduler.catalog.PROJECT_PATH", str(tmp_path / "site-packages")):
        node = SchedulerCatalog(installation["python"], installation=installation)
    assert node.root == tmp_path / ".state/schedulers"
    assert node.current()["labels"] == ["Default", "Currently"]


def test_framework_change_creates_new_default_without_destroying_old_snapshots(installation, tmp_path):
    previous = catalog(installation, tmp_path)
    old_snapshot = Path(previous.current()["snapshot_path"])
    old_content = old_snapshot.read_bytes()
    previous.select("flagscheduler")
    native_path = Path(installation["scheduler"]["candidates"][0]["path"])
    native_path.write_text("# changed installation fixture\nclass Scheduler: pass\n")
    refreshed = {**installation, "scheduler": inspect_vllm(Path(installation["package_path"]))}
    current = catalog(refreshed, tmp_path)
    assert current.path != previous.path
    assert current.current()["id"] == "default"
    assert old_snapshot.read_bytes() == old_content


def test_add_custom_script_does_not_execute_it_or_change_current(installation, tmp_path):
    source = tmp_path / "custom.py"
    source.write_text("raise RuntimeError('do not import at registration')\nclass CustomScheduler: pass\n")
    node = catalog(installation, tmp_path)
    view = node.add("my-policy", "custom.CustomScheduler", str(source), expect_version="0.21.0")
    assert view["selected_id"] == "default"
    chosen = node.select("my-policy")
    entry = next(e for e in chosen["scripts"] if e["current"])
    assert entry["class_name"] == "custom.CustomScheduler"
    assert str(tmp_path) in entry["plugin_paths"]
    assert Path(entry["snapshot_path"]).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("script_id", ["default", "flagscheduler", "../overwrite", "a/b"])
def test_reserved_or_unsafe_ids_are_rejected(installation, tmp_path, script_id):
    with pytest.raises(ValueError):
        catalog(installation, tmp_path).add(script_id, "custom.CustomScheduler", "unused", expect_version="0.21.0")


def test_edited_custom_script_requires_explicit_reselection(installation, tmp_path):
    source = tmp_path / "custom.py"
    source.write_text("class CustomScheduler: pass\n")
    node = catalog(installation, tmp_path)
    node.add("custom", "custom.CustomScheduler", str(source), expect_version="0.21.0")
    node.select("custom")
    snapshot = Path(node.current()["snapshot_path"])
    original = snapshot.read_bytes()
    source.write_text("class CustomScheduler:\n    revision = 2\n")
    with pytest.raises(ValueError, match="changed"):
        node.current()
    node.select("custom")
    assert snapshot.read_bytes() == original
    assert node.current()["snapshot_path"] != str(snapshot)


def test_damaged_original_snapshot_is_never_silently_overwritten(installation, tmp_path):
    node = catalog(installation, tmp_path)
    original = node.current()
    source_bytes = Path(original["script_path"]).read_bytes()
    snapshot = Path(original["snapshot_path"])
    snapshot.chmod(0o644)
    snapshot.write_text("# damaged local snapshot\n")
    with pytest.raises(ValueError, match="snapshot"):
        node.select("default")
    assert snapshot.read_text() == "# damaged local snapshot\n"
    assert Path(original["script_path"]).read_bytes() == source_bytes


def test_saved_plan_rejects_script_edit_before_execution(installation, tmp_path):
    source = tmp_path / "custom.py"
    source.write_text("class CustomScheduler: pass\n")
    node = catalog(installation, tmp_path)
    node.add("custom", "custom.CustomScheduler", str(source), expect_version="0.21.0")
    node.select("custom")
    with patch("flagscheduler.deployment.vllm_installation", return_value=installation):
        plan = node.build_plan("/weights")
        source.write_text("class CustomScheduler:\n    revision = 2\n")
        with patch("flagscheduler.deployment.subprocess.call") as start:
            with pytest.raises(ValueError, match="changed since planning"):
                execute_launch_plan(plan)
            start.assert_not_called()


def test_selected_script_drives_launch_and_default_pins_mode(installation, tmp_path):
    node = catalog(installation, tmp_path)
    with patch("flagscheduler.deployment.vllm_installation", return_value=installation):
        native = node.build_plan("/weights")
        assert native["scheduler_class"] is None
        assert "--no-async-scheduling" in native["argv"]
        assert native["scheduler_choice"]["labels"] == ["Default", "Currently"]
        node.select("flagscheduler")
        plan = node.build_plan("/weights")
        assert plan["scheduler_class"].endswith(".FlagScheduler")
        assert plan["scheduler_choice"]["labels"] == ["Currently"]
        assert plan["native_argv"].count("--no-async-scheduling") == 1
        with patch("flagscheduler.deployment.validate_scheduler"), patch("flagscheduler.deployment.subprocess.call", return_value=0) as call:
            assert execute_launch_plan(plan) == 0
            assert call.call_args.args[0] == plan["argv"]


def test_catalog_cli_labels_selection_and_plan(installation, tmp_path, capsys):
    shared = ["--python", installation["python"], "--state-dir", str(tmp_path / "state")]
    with patch("flagscheduler.catalog.vllm_installation", return_value=installation), patch(
        "flagscheduler.deployment.vllm_installation", return_value=installation
    ):
        assert main(["scheduler", "list", *shared]) == 0
        assert "default (Default) (Currently)" in capsys.readouterr().out
        assert main(["scheduler", "select", "flagscheduler", *shared, "--json"]) == 0
        view = json.loads(capsys.readouterr().out)
        assert view["selected_id"] == "flagscheduler"
        assert main(["scheduler", "plan", *shared, "--model", "/weights"]) == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["scheduler_class"].endswith(".FlagScheduler")
        file = tmp_path / "plan.json"
        file.write_text(json.dumps(plan))
        assert main(["launch", "--plan", str(file), "--native"]) == 0
        fallback = json.loads(capsys.readouterr().out)
        assert fallback["scheduler_class"] is None
        assert "scheduler_choice" not in fallback


@pytest.mark.parametrize("selected", ["default", "flagscheduler"])
def test_vllm_binding_uses_saved_selection_and_exposes_current_marker(monkeypatch, installation, tmp_path, selected):
    from flagscheduler.backends import VllmEmbeddingBackend
    from flagscheduler.config import BindingConfig
    node = catalog(installation, tmp_path)
    node.select(selected)
    calls = []
    class FakeLLM:
        model_config = SimpleNamespace(max_model_len=2048)
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def get_tokenizer(self):
            return SimpleNamespace(encode=lambda text: [1])
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM))
    with patch("flagscheduler.catalog.SchedulerCatalog", return_value=node), patch("importlib.metadata.version", return_value="0.21.0"):
        binding = BindingConfig("qwen", "Qwen3-Embedding-0.6B", "vllm-embed", {"model_path": "/weights"}, {})
        backend = VllmEmbeddingBackend(binding)
    if selected == "flagscheduler":
        assert calls[0]["scheduler_cls"].endswith(".FlagScheduler")
    else:
        assert "scheduler_cls" not in calls[0]
    assert calls[0]["async_scheduling"] is False
    assert backend.scheduler_info["labels"] == (["Default", "Currently"] if selected == "default" else ["Currently"])
