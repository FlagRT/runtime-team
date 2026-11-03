"""CPU checks for P800 model comparison; no device/model loading."""
import copy
import unittest
import torch
from model_probe import compare_tensor, mismatch_examples, validate_reference, MODEL_TOLERANCE


class ComparisonTests(unittest.TestCase):
    def test_equal(self):
        x=torch.arange(16,dtype=torch.float32)
        self.assertEqual(compare_tensor(x,x,dict(atol=0,rtol=0)),dict(max_abs=0.,mismatches=0))

    def test_exact_fast_path_matches_original_rule(self):
        for dtype in (torch.float32,torch.float16,torch.bfloat16):
            base=torch.linspace(-3,3,2048).to(dtype)
            for delta in (0.,.001,.1):
                actual=(base.float()+delta).to(dtype)
                expected=dict(max_abs=float((actual.float()-base.float()).abs().max()),
                              mismatches=int((~torch.isclose(actual.float(),base.float(),atol=1e-3,rtol=1e-2)).sum()))
                self.assertEqual(compare_tensor(actual,base,dict(atol=1e-3,rtol=1e-2)),expected)
            for special in (float('nan'),float('inf'),float('-inf')):
                x=torch.tensor([special],dtype=dtype)
                with self.assertRaises(ValueError): compare_tensor(x,x,MODEL_TOLERANCE)

    def test_difference_and_fixed_tolerance(self):
        x=torch.ones(8)
        self.assertEqual(MODEL_TOLERANCE,dict(atol=1e-5,rtol=1e-4))
        self.assertEqual(compare_tensor(x+1e-2,x,MODEL_TOLERANCE)['mismatches'],8)
        self.assertEqual(compare_tensor(x+1e-6,x,dict(atol=0,rtol=0))['mismatches'],8)

    def test_nan_and_shape(self):
        for x,y in ((torch.tensor([float('nan')]),torch.ones(1)),(torch.ones(2),torch.ones(3)),
                    (torch.ones(2),torch.ones(2,dtype=torch.float16))):
            with self.assertRaises(ValueError): compare_tensor(x,y,MODEL_TOLERANCE)

    def test_bounded_mismatch_excerpt(self):
        reference=torch.ones(12)
        actual=reference+1e-2
        samples=mismatch_examples(actual,reference,MODEL_TOLERANCE)
        self.assertEqual(len(samples),8)
        self.assertEqual(samples[0]['flat_index'],0)
        self.assertGreater(samples[0]['abs_error'],samples[0]['allowed_error'])
        self.assertAlmostEqual(samples[0]['allowed_error'],1.1e-4)
        self.assertEqual(mismatch_examples(reference,reference,MODEL_TOLERANCE),[])

    def test_reference_identity(self):
        current=dict(environment={'chip':'P800'},adapter_version='candidate',model_files={'weight':'sha'},
                     workload={'steps':3},runtime_sources={'backend':'sha'})
        reference=dict(current,status='pass',mode='native')
        validate_reference(reference,current)
        for key in current:
            bad=copy.deepcopy(reference); bad[key]='other'
            with self.assertRaisesRegex(ValueError,key): validate_reference(bad,current)

    def test_failed_or_target_reference_rejected(self):
        for ref in ({'status':'fail','mode':'native'},{'status':'pass','mode':'on'}):
            with self.assertRaises(ValueError): validate_reference(ref,{})


if __name__=='__main__': unittest.main()
