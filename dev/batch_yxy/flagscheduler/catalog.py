"""Preserved scheduler scripts and a saved selection for subsequent launches.

Original framework files are read-only inputs. Content-addressed snapshots and
selection state live under the project's state directory. Currently describes
the saved selection, never a claim about an already-running engine process.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .deployment import PROJECT_PATH, build_launch_plan, qualified_name, vllm_installation

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SchedulerCatalog:
    """One catalog per installation fingerprint and sync/async mode.

    Framework upgrades create a new catalog using Default. Old snapshots and
    selections remain intact; custom selections are not silently carried across
    versions. Methods return JSON data so both CLI and a future TUI can use them.
    """
    def __init__(self, python: str, *, mode="sync", state_dir=None, installation=None):
        if mode not in ("sync", "async"):
            raise ValueError("scheduler mode must be sync or async")
        self.install = installation or vllm_installation(python)
        self.mode = mode
        project = Path(PROJECT_PATH).resolve()
        checkout = (project / "pyproject.toml").is_file()
        state_base = project if checkout else Path.cwd()
        self.root = Path(state_dir or state_base / ".state/schedulers").expanduser().resolve()
        if checkout and not self.root.is_relative_to(project):
            raise ValueError("scheduler state must stay inside batch_yxy")
        identity = {key: self.install[key] for key in ("package_path", "version")}
        # python/python3/python3.12 aliases within one environment share a
        # selection; distinct virtual environments still remain isolated.
        identity["environment"] = str(Path(self.install.get(
            "prefix", Path(self.install["python"]).parent.parent)).resolve())
        identity["fingerprints"] = self.install["scheduler"]["fingerprints"]
        self.key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path = self.root / self.key / mode / "catalog.json"

    def _snapshot(self, source: Path) -> dict:
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / "originals" / f"{digest}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as output:
                output.write(content)
            target.chmod(0o444)
        except FileExistsError:
            if file_hash(target) != digest:
                raise ValueError(f"preserved scheduler snapshot was modified: {target}")
        return {"script_path": str(source.resolve()), "sha256": digest,
                "snapshot_path": str(target)}

    def _save(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".catalog-", suffix=".json")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _load(self) -> dict:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("schema_version") != 1 or data.get("installation_key") != self.key:
                raise ValueError("unsupported or mismatched scheduler catalog")
            return data
        kind = "v1_async" if self.mode == "async" else "v1"
        candidates = self.install["scheduler"]["candidates"]
        native = next((c for c in candidates if c["kind"] == kind), None)
        if native is None and self.mode == "sync":
            native = next((c for c in candidates if c["kind"] == "v0"), None)
        if native is None:
            raise ValueError(f"no native {self.mode} scheduler script found")
        # Retain every discovered original script, including the other mode.
        originals = [{"kind": c["kind"], "class_name": c["class"],
                      **self._snapshot(Path(c["path"]))} for c in candidates]
        selected_original = next(c for c in originals if c["kind"] == native["kind"])
        entries = [{"id": "default", "class_name": native["class"], "origin": "native",
                    "default": True, "plugin_paths": [], "expect_version": self.install["version"],
                    **{k: selected_original[k] for k in ("script_path", "sha256", "snapshot_path")}}]
        # This supplied extension explicitly targets vLLM 0.21.0 only.
        if self.install["version"] == "0.21.0" and native["kind"] == kind:
            project_source = Path(__file__).parent / "integrations/vllm_scheduler.py"
            class_name = "FlagAsyncScheduler" if self.mode == "async" else "FlagScheduler"
            entries.append({"id": "flagscheduler", "class_name": f"flagscheduler.integrations.vllm_scheduler.{class_name}",
                            "origin": "builtin", "default": False, "plugin_paths": [],
                            "expect_version": "0.21.0", **self._snapshot(project_source)})
        data = {"schema_version": 1, "installation_key": self.key, "mode": self.mode,
                "installation": self.install, "originals": originals, "entries": entries,
                "selected_id": "default"}
        self._save(data)
        return data

    def view(self) -> dict:
        data = self._load()
        entries = []
        for entry in data["entries"]:
            current = entry["id"] == data["selected_id"]
            labels = (["Default"] if entry["default"] else []) + (["Currently"] if current else [])
            source = Path(entry["script_path"])
            entries.append({**entry, "current": current, "labels": labels,
                            "source_unchanged": source.is_file() and file_hash(source) == entry["sha256"],
                            "snapshot_intact": Path(entry["snapshot_path"]).is_file()
                            and file_hash(entry["snapshot_path"]) == entry["sha256"]})
        return {"schema_version": 1, "framework": "vllm", "python": self.install["python"],
                "version": self.install["version"], "mode": self.mode, "selected_id": data["selected_id"],
                "selection_scope": "saved selection for subsequent launches; running processes unchanged",
                "state_path": str(self.path), "scripts": entries}

    def select(self, script_id: str) -> dict:
        data = self._load()
        entry = next((e for e in data["entries"] if e["id"] == script_id), None)
        if entry is None:
            raise ValueError(f"unknown scheduler script: {script_id}")
        if entry["expect_version"] != self.install["version"]:
            raise ValueError("scheduler version does not match the selected environment")
        # Explicit reselection accepts edits to a custom script, keeping older
        # snapshots. Native baseline bytes are never refreshed in-place.
        if not entry["default"]:
            entry.update(self._snapshot(Path(entry["script_path"])))
        elif file_hash(entry["script_path"]) != entry["sha256"]:
            raise ValueError("original framework script changed; inspect the installation again")
        if file_hash(entry["snapshot_path"]) != entry["sha256"]:
            raise ValueError("preserved scheduler snapshot was modified")
        data["selected_id"] = script_id
        self._save(data)
        return self.view()

    def add(self, script_id: str, class_name: str, script: str, *, expect_version: str, plugin_paths=()) -> dict:
        if not _ID.fullmatch(script_id) or script_id in ("default", "flagscheduler"):
            raise ValueError("custom scheduler ID must use letters/numbers/_/- and cannot replace a built-in ID")
        qualified_name(class_name)
        if not expect_version or expect_version != self.install["version"]:
            raise ValueError(f"custom script requires exact --expect-version {self.install['version']!r}")
        data = self._load()
        if any(e["id"] == script_id for e in data["entries"]):
            raise ValueError("scheduler ID already exists; choose another ID or reselect it after editing")
        source = Path(script).expanduser().resolve()
        tree = ast.parse(source.read_bytes())
        module, symbol = class_name.rsplit(".", 1)
        if not any(isinstance(n, ast.ClassDef) and n.name == symbol for n in tree.body):
            raise ValueError("script must define the selected class at module scope")
        parts = module.split(".")
        import_root = source.parent
        for _ in range(len(parts) if source.name == "__init__.py" else len(parts) - 1):
            import_root = import_root.parent
        expected_source = import_root.joinpath(*parts)
        expected_source = expected_source / "__init__.py" if source.name == "__init__.py" else expected_source.with_suffix(".py")
        if expected_source.resolve() != source:
            raise ValueError("script path does not match the Python module in --class")
        data["entries"].append({"id": script_id, "class_name": class_name, "origin": "custom",
                                "default": False, "expect_version": expect_version,
                                "plugin_paths": list(dict.fromkeys([str(import_root), *(str(Path(p).resolve()) for p in plugin_paths)])),
                                **self._snapshot(source)})
        self._save(data)
        return self.view()

    def current(self) -> dict:
        view = self.view()
        entry = next(e for e in view["scripts"] if e["current"])
        if not entry["source_unchanged"] or not entry["snapshot_intact"]:
            raise ValueError("selected scheduler script/snapshot changed; inspect and select the script again")
        return {**entry, "selection_scope": view["selection_scope"]}

    def build_plan(self, model: str, *, plugin_paths=(), extra_args=(), expected_version=None) -> dict:
        entry = self.current()
        if expected_version and expected_version != entry["expect_version"]:
            raise ValueError("requested version does not match selected scheduler")
        plan = build_launch_plan(self.install["python"], model,
                                 scheduler_class=None if entry["default"] else entry["class_name"],
                                 expected_version=entry["expect_version"], scheduler_mode=self.mode,
                                 pin_scheduler_mode=True,
                                 plugin_paths=[*entry["plugin_paths"], *plugin_paths], extra_args=extra_args)
        plan["scheduler_choice"] = entry
        return plan
