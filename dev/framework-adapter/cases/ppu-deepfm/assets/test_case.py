"""CPU controls against a frozen input bundle; no accelerator claim."""
import os
from pathlib import Path
import sys
import unittest
import torch

ROOT=Path(os.environ['PPU_DEEPFM_INPUT']).resolve()
sys.path.insert(0,str(ROOT))
from deepfm_probe import load_model,make_batch,state_hash
from model_probe import compare_tensor,validate_reference,MODEL_TOLERANCE
from resources import require_idle

class Controls(unittest.TestCase):
    def test_idle_resource_gate(self):
        sample='| 1 PPU-ZW810E |\n| N/A | 59MiB / 98304MiB | 0% |\nProcesses:\n'
        self.assertEqual(require_idle(sample,1)['memory_mib'],59)
        for bad in (sample.replace('59MiB','65MiB'),sample.replace('0%','1%'),
                    sample+'| 1 N/A N/A 123 C task 0MiB |',sample.replace('Processes:','Absent:')):
            with self.assertRaises(RuntimeError): require_idle(bad,1)
    def test_qualification_input_shape(self):
        import ast
        tree=ast.parse((ROOT/'unary_probe.py').read_text())
        linspaces=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Attribute) and node.func.attr=='linspace']
        self.assertEqual(len(linspaces),2)
        for node in linspaces:
            arguments=[ast.literal_eval(v) for v in node.args]
            self.assertEqual(tuple(torch.linspace(*arguments).reshape(32).shape),(32,))
    def setUp(self):
        torch.set_num_threads(2)
    def test_public_identity(self):
        model,_=load_model(ROOT/'model')
        self.assertEqual(sum(p.numel() for p in model.parameters()),421452)
    def test_sigmoid_shape_and_gradients(self):
        model,config=load_model(ROOT/'model')
        x,y=make_batch(config,32,0)
        out=model(x)
        self.assertEqual(tuple(out.shape),(32,))
        torch.nn.functional.binary_cross_entropy(out,y).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
    def test_initial_seed(self):
        torch.manual_seed(20261023); first,_=load_model(ROOT/'model')
        torch.manual_seed(20261023); second,_=load_model(ROOT/'model')
        self.assertEqual(state_hash(first),state_hash(second))
    def test_data_seed(self):
        _,config=load_model(ROOT/'model')
        x,y=make_batch(config,32,0); xx,yy=make_batch(config,32,0)
        self.assertTrue(torch.equal(x,xx) and torch.equal(y,yy))
        self.assertFalse(torch.equal(x,make_batch(config,32,1)[0]))
    def test_exact_native_comparison(self):
        value=torch.tensor([0.,1.])
        self.assertEqual(compare_tensor(value,value,dict(atol=0,rtol=0))['mismatches'],0)
        self.assertEqual(compare_tensor(value,value+1e-6,dict(atol=0,rtol=0))['mismatches'],2)
    def test_reference_fail_closed(self):
        with self.assertRaises(ValueError): validate_reference({'status':'fail','mode':'native'}, {})
        with self.assertRaises(ValueError):
            validate_reference({'status':'pass','mode':'native','environment':{'backend':'kunlunxin'}},
                               {'environment':{'backend':'ppu'}})
    def test_finite_and_dtype(self):
        with self.assertRaises(ValueError):
            compare_tensor(torch.tensor([float('nan')]),torch.zeros(1),MODEL_TOLERANCE)
        with self.assertRaises(ValueError):
            compare_tensor(torch.zeros(1),torch.zeros(1,dtype=torch.float64),MODEL_TOLERANCE)

if __name__=='__main__': unittest.main()
