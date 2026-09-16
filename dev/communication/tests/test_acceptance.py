"""Process/result contract regression on host; does not import torch."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("acceptance", HERE.parent / "probes/run_acceptance.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)


class AcceptanceTests(unittest.TestCase):
    def execute(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            return acceptance.run([sys.executable, str(HERE / "fake_worker.py"), mode],
                                  Path(directory), iterations=2,
                                  timeout=0.5 if mode == "hang" else 5, grace=0.1)

    def test_success(self):
        self.assertEqual(self.execute("ok")["verdict"], "PASS")

    def test_nonzero_after_pass(self):
        row = self.execute("nonzero")
        self.assertTrue(row["numeric_ok"])
        self.assertEqual(row["returncode"], 7)
        self.assertEqual(row["verdict"], "FAIL")

    def test_signal_after_pass(self):
        row = self.execute("signal")
        self.assertTrue(row["numeric_ok"])
        self.assertLess(row["returncode"], 0)
        self.assertEqual(row["verdict"], "FAIL")

    def test_hang_ignoring_term(self):
        row = self.execute("hang")
        self.assertTrue(row["timed_out"])
        self.assertEqual(row["returncode"], -9)
        self.assertEqual(row["verdict"], "FAIL")

    def test_abort_after_pass(self):
        row = self.execute("abort")
        self.assertTrue(row["numeric_ok"])
        self.assertEqual(row["returncode"], -6)
        self.assertEqual(row["verdict"], "FAIL")

    def test_missing_rank(self):
        self.assertEqual(self.execute("missing")["verdict"], "FAIL")

    def test_stale_result(self):
        self.assertEqual(self.execute("stale")["verdict"], "FAIL")

    def test_wrong_rank(self):
        self.assertEqual(self.execute("wrong-rank")["verdict"], "FAIL")

    def test_inconsistent_counts(self):
        self.assertEqual(self.execute("bad-count")["verdict"], "FAIL")

    def test_missing_cleanup(self):
        row = self.execute("before-destroy")
        self.assertTrue(row["numeric_ok"])
        self.assertFalse(row["cleanup_ok"])
        self.assertEqual(row["verdict"], "FAIL")

    def test_numeric_failure(self):
        self.assertEqual(self.execute("numeric-fail")["verdict"], "FAIL")

    def test_malformed_json(self):
        self.assertEqual(self.execute("malformed")["verdict"], "FAIL")

    def test_launcher_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            row = acceptance.run([str(Path(directory) / "missing")], Path(directory))
        self.assertIsNotNone(row["launch_error"])
        self.assertEqual(row["verdict"], "FAIL")

    def test_invalid_parameters(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                acceptance.run(["unused"], Path("/tmp/unused"), timeout=timeout)


if __name__ == "__main__":
    unittest.main()
