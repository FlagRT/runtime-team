"""Torch-free control tests of metadata and CLI safety, not operator tests."""
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qwen_embedding_baseline import initialize_device, last_token_indices, tolerance


class ProbeMetadataTests(unittest.TestCase):
    def test_lazy_vendor_registers_before_device_parsing(self):
        events = []
        def count():
            events.append("register")
            return 1
        backend = SimpleNamespace(__version__="test", __file__="test-runtime",
            use=lambda name: events.append("select_backend"),
            set_device=lambda ordinal: events.append("set_device"), device_count=count)
        def parse(spec):
            self.assertEqual(events, ["select_backend", "register", "set_device"])
            events.append("parse")
            return SimpleNamespace(index=0)
        torch = SimpleNamespace(device=parse, ones=lambda *a, **kw: SimpleNamespace(cpu=lambda: None))
        with patch.dict(sys.modules, {"runtime": backend}), patch.object(sys, "path", sys.path.copy()):
            initialize_device(torch, "npu:0", "/unused", {})
        self.assertEqual(events, ["select_backend", "register", "set_device", "parse"])

    def test_device_failure_does_not_fallback_to_cpu(self):
        def fail(ordinal):
            raise RuntimeError("device unavailable")
        backend = SimpleNamespace(__version__="test", __file__="test-runtime",
            use=lambda name: None, set_device=fail, device_count=lambda: 0)
        with patch.dict(sys.modules, {"runtime": backend}), patch.object(sys, "path", sys.path.copy()):
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                initialize_device(None, "npu:0", "/unused", {})

    def test_npu_requires_runtime(self):
        with self.assertRaises(ValueError):
            initialize_device(None, "npu:0", None, {})

    def test_no_implicit_other_backend(self):
        with self.assertRaises(ValueError):
            initialize_device(None, "cuda:0", None, {})

    def test_right_padding(self):
        self.assertEqual(last_token_indices([[1, 1, 0], [1, 1, 1]]), [1, 2])

    def test_left_padding(self):
        self.assertEqual(last_token_indices([[0, 0, 1], [0, 1, 1]]), [2, 2])

    def test_all_padding_rejected(self):
        with self.assertRaises(ValueError):
            last_token_indices([[0, 0]])

    def test_empty_row_rejected(self):
        with self.assertRaises(ValueError):
            last_token_indices([[]])

    def test_tolerances(self):
        self.assertEqual([tolerance(s) for s in ("float32", "float16", "bfloat16")], [2e-5, .005, .03])

    def test_unknown_dtype_rejected(self):
        with self.assertRaises(KeyError):
            tolerance("float64")

    def test_explicit_device_required(self):
        script = Path(__file__).with_name("qwen_embedding_baseline.py")
        result = subprocess.run([sys.executable, str(script), "--model", "/missing", "--dtype", "float32", "--out", "/unused"], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"--device", result.stderr)

    def test_existing_evidence_preserved(self):
        script = Path(__file__).with_name("qwen_embedding_baseline.py")
        # Existing source is used read-only as an output collision sentinel.
        before = script.read_bytes()
        result = subprocess.run([sys.executable, str(script), "--model", "/missing", "--device", "cpu", "--dtype", "float32", "--out", str(script)], capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(script.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
