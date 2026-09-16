import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("export_results", Path(__file__).parents[1] / "probes/export_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExportTests(unittest.TestCase):
    def test_acceptance_drops_paths_and_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "acceptance.json").write_text(json.dumps({"verdict": "FAIL", "returncode": -6,
                "command": ["/private/secret"], "errors": ["host private info"], "output_dir": "/secret"}))
            report = module.export(root)
            result = report["records"][0]["result"]
            self.assertEqual(result, {"returncode": -6, "verdict": "FAIL"})
            self.assertNotIn("secret", json.dumps(report))

    def test_fingerprint_drops_directory(self):
        result = module.fingerprint({"inner_name": "flagcx", "libraries": [
            {"path": "/internal/machine/libflagcx.so", "sha256": "a" * 64}]})
        self.assertEqual(result["libraries"][0]["name"], "libflagcx.so")
        self.assertNotIn("internal", json.dumps(result))

    def test_empty_results_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                module.export(Path(folder))

    def test_malformed_results_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "acceptance.json").write_text("broken JSON")
            with self.assertRaises(ValueError):
                module.export(root)
