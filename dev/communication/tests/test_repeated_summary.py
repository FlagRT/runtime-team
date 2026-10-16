"""Evidence mutations must never turn missing/failed execution into PASS."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from summarize_repeated import summarize


class RepeatedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'evidence'
        shutil.copytree(Path(__file__).resolve().parents[1]/'results/20261014-host', self.root)

    def change(self, file, function):
        path=self.root/file; data=json.loads(path.read_text()); function(data)
        path.write_text(json.dumps(data))

    def test_valid_records(self): self.assertEqual(summarize(self.root)['runs'],6)

    def test_nonzero_launcher_rejected(self):
        self.change('execution.json', lambda d: d['runs'][0].update(exit_code=134))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_missing_rank_rejected(self):
        (self.root/'01-manual/rank1.json').unlink()
        with self.assertRaises(FileNotFoundError): summarize(self.root)

    def test_wrong_scope_rejected(self):
        self.change('execution.json', lambda d:d.update(scope='Ascend integration'))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_wrong_call_count_rejected(self):
        self.change('01-fused/rank0.json', lambda d:d['training']['gradient_communication'].update(gradient_allreduce_calls=0))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_duplicate_trial_rejected(self):
        self.change('execution.json', lambda d:d['runs'].append(d['runs'][0]))
        with self.assertRaises(ValueError): summarize(self.root)


if __name__ == '__main__': unittest.main()
