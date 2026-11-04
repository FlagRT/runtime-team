import importlib.util
from pathlib import Path
import tempfile
import unittest
import torch

source=Path(__file__).resolve().parents[3]/'probes/qualify_public_training.py'
spec=importlib.util.spec_from_file_location('qualification_probe',source)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class AuditTests(unittest.TestCase):
    def test_save_exact_and_target_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original=module.TensorAudit(root/'native',None,0)
            original.tensor('loss',torch.tensor([1.]))
            repeat=module.TensorAudit(root/'repeat',root/'native',0)
            repeat.tensor('loss',torch.tensor([1.]))
            self.assertEqual(repeat.rows[0]['status'],'PASS')
            target=module.TensorAudit(root/'target',root/'native',0,True)
            target.tensor('loss',torch.tensor([1.00001]))
            self.assertEqual(target.rows[0]['status'],'PASS')
            self.assertEqual(list((root/'repeat').rglob('*.pt')),[])
            self.assertEqual(list((root/'target').rglob('*.pt')),[])

    def test_mismatch_nonfinite_and_missing_reference_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); original=module.TensorAudit(root/'native',None,0)
            original.tensor('loss',torch.tensor([1.]))
            repeat=module.TensorAudit(root/'repeat',root/'native',0)
            with self.assertRaises(AssertionError): repeat.tensor('loss',torch.tensor([1.01]))
            self.assertEqual(repeat.rows[-1]['status'],'FAIL')
            with self.assertRaises(AssertionError): original.tensor('bad',torch.tensor([float('nan')]))
            with self.assertRaises(FileNotFoundError): repeat.tensor('missing',torch.tensor([1.]))


if __name__=='__main__': unittest.main()
