"""CPU controls for frozen ports; do not count as PPU execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(os.environ['PPU_MODEL_INPUT']).resolve()
sys.path.insert(0,str(ROOT))
from model_probe import compare_tensor, validate_reference
from resources import require_idle

class PortControls(unittest.TestCase):
    def test_manifest(self):
        manifest=json.loads((ROOT/'inputs.json').read_text())
        for name,sha in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha,name)
    def test_compile(self):
        for f in ROOT.glob('*.py'): compile(f.read_text(),str(f),'exec')
    def test_shape(self):
        import torch
        shape=json.loads((ROOT/'port.json').read_text())['shape']
        tree=ast.parse((ROOT/'unary_probe.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='linspace']
        self.assertEqual(len(calls),2)
        for n in calls: self.assertEqual(list(torch.linspace(*[ast.literal_eval(x) for x in n.args]).reshape(shape).shape),shape)
    def test_model_tolerances_unchanged(self):
        from model_probe import MODEL_TOLERANCE
        from onerec_probe import TOLERANCE
        self.assertEqual(MODEL_TOLERANCE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(TOLERANCE,dict(atol=1e-3,rtol=1e-2))
    def test_candidate_separate_from_model_approval(self):
        from types import SimpleNamespace
        from candidate import prepare_candidate
        for a in [SimpleNamespace(mode='candidate',adapter_config='bad',records=None,reference='ref'),
                  SimpleNamespace(mode='candidate',adapter_config=None,records=None,reference='ref')]:
            with self.assertRaises(ValueError): prepare_candidate(a,{},'silu',[1,32,3072])
    def test_native_reference_identity(self):
        with self.assertRaises(ValueError): validate_reference(dict(mode='native',status='fail'),{})
        with self.assertRaises(ValueError):
            validate_reference(dict(mode='native',status='pass',environment={'backend':'kunlunxin'}),{'environment':{'backend':'ppu'}})
    def test_resource_exclusions(self):
        sample='| 1 PPU-ZW810E |\n| N/A | 59MiB / 98304MiB | 0% |\nProcesses:\n'
        self.assertEqual(require_idle(sample,1)['memory_mib'],59)
        for bad in (sample.replace('59MiB','65MiB'),sample.replace('0%','1%'),sample+'| 1 N/A N/A 123 C task 0MiB |'):
            with self.assertRaises(RuntimeError): require_idle(bad,1)
    def test_comparison_rejects_nonfinite(self):
        import torch
        with self.assertRaises(ValueError): compare_tensor(torch.tensor([float('nan')]),torch.zeros(1),dict(atol=0,rtol=0))

if __name__=='__main__': unittest.main()
