import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('compare_trace', Path(__file__).resolve().parents[1]/'training/compare_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TraceTests(unittest.TestCase):
    def fixture(self, root):
        for rank in range(2):
            d = root/f'rank{rank}'
            d.mkdir()
            (d/'trace.json').write_text(json.dumps({'snapshots':['00-local-grad','00-reduced-grad']}))
            np.savez(d/'00-local-grad.npz', weight=np.array([rank*2.],dtype=np.float32))
            np.savez(d/'00-reduced-grad.npz', weight=np.array([1.],dtype=np.float32))

    def test_correct_mean(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.fixture(p)
            s=module.compare(p)['snapshots']['00-reduced-grad']
            self.assertEqual(s['host_mean_max_abs'],0)
            self.assertEqual(s['rank_max_abs'],0)

    def test_equal_but_wrong_reduction_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.fixture(p)
            for r in range(2): np.savez(p/f'rank{r}/00-reduced-grad.npz',weight=np.array([2.],dtype=np.float32))
            self.assertEqual(module.compare(p)['snapshots']['00-reduced-grad']['host_mean_max_abs'],1)

    def test_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.fixture(p)
            (p/'rank1/trace.json').write_text(json.dumps({'snapshots':[]}))
            with self.assertRaises(ValueError): module.compare(p)


if __name__ == '__main__': unittest.main()
