"""Synthetic record-loader checks; no actual environment is qualified here."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from admission_records import load_records
from silu_policy import Environment


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "result.txt").write_text("synthetic test only")
        self.item = dict(environment=asdict(Environment(*(["test"] * 7))),
                         operator="silu", dtype="torch.float32", shape=[2, 3],
                         stages=["forward", "backward"], evidence="result.txt",
                         evidence_sha256=hashlib.sha256(b"synthetic test only").hexdigest(),
                         status="passed", revision=1)

    def load(self):
        path = self.root / "records.json"
        path.write_text(json.dumps({"schema": 1, "records": [self.item]}))
        return load_records(path)

    def test_valid_record(self):
        record, = self.load()
        self.assertEqual(record.shape, (2, 3))
        self.assertEqual(record.status, "passed")

    def test_changed_evidence(self):
        (self.root / "result.txt").write_text("changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_missing_evidence(self):
        self.item["evidence"] = "missing.txt"
        with self.assertRaises(ValueError):
            self.load()

    def test_escaping_evidence(self):
        self.item["evidence"] = "../outside.txt"
        with self.assertRaises(ValueError):
            self.load()

    def test_incomplete_identity(self):
        self.item["environment"].pop("compiler_build")
        with self.assertRaises(ValueError):
            self.load()

    def test_invalid_types(self):
        for field, value in (("shape", [True, 3]), ("stages", "forward"),
                             ("revision", True), ("status", "PASS")):
            old = self.item[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.item[field] = value
                self.load()
            self.item[field] = old

    def test_non_object_or_boolean_schema(self):
        path = self.root / "invalid.json"
        for document in ([], None, {"schema": True, "records": []}):
            with self.subTest(document=document), self.assertRaises(ValueError):
                path.write_text(json.dumps(document))
                load_records(path)


if __name__ == "__main__":
    unittest.main()
