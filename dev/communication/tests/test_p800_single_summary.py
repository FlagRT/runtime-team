"""Validate one-rank evidence boundaries; never manufacture hardware results."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from summarize_p800_single import summarize


class P800SingleSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'evidence'
        source=Path(__file__).resolve().parents[1]/'results/20261028-p800/deepfm-single-v2'
        shutil.copytree(source,self.root)

    def change(self,name,function):
        path=self.root/name;data=json.loads(path.read_text());function(data)
        path.write_text(json.dumps(data))

    def test_single_card_not_two_card_acceptance(self):
        result=summarize(self.root)
        self.assertEqual(result['world_size'],1)
        self.assertFalse(result['two_card_acceptance'])

    def test_nonzero_launcher_rejected(self):
        self.change('selection-execution.json',lambda d:d['runs'][0].update(exit_code=134))
        with self.assertRaises(ValueError):summarize(self.root)

    def test_wrong_uuid_rejected(self):
        self.change('identity.json',lambda d:d.update(visible_uuid_order=['wrong-card']))
        with self.assertRaises(ValueError):summarize(self.root)

    def test_mode_parameters_differ_rejected(self):
        self.change('fused/rank0.json',lambda d:d.update(final_parameter_sha256='different'))
        with self.assertRaises(ValueError):summarize(self.root)

    def test_wrong_world_size_rejected(self):
        self.change('manual/rank0.json',lambda d:d.update(world_size=2))
        with self.assertRaises(ValueError):summarize(self.root)


if __name__=='__main__':unittest.main()
