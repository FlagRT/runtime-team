import unittest
import torch
from trace_ops import Inventory


class TraceTests(unittest.TestCase):
    def test_values_gradients_and_phases(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,1))
        x=torch.randn(2,3); oracle=model(x).sum();oracle.backward()
        expected=[p.grad.clone() for p in model.parameters()];model.zero_grad()
        trace=Inventory();trace.attach(model)
        with trace.scope('forward'): actual=model(x).sum()
        with trace.scope('backward'): actual.backward()
        trace.detach()
        self.assertTrue(torch.equal(actual,oracle))
        for p,g in zip(model.parameters(),expected):self.assertTrue(torch.equal(p.grad,g))
        rows=trace.report()['signatures']
        self.assertTrue(any(r['operator']=='aten.relu.default' and r['phase']=='forward' for r in rows))
        self.assertTrue(any(r['operator']=='aten.threshold_backward.default' and r['phase']=='backward' for r in rows))
        self.assertTrue(any(r['operator'].endswith('.linear') and r['layer']=='python' for r in rows))
        self.assertEqual(trace.stack,[])
        self.assertFalse(trace.report()['tensor_values_recorded'])

    def test_limit_fails_instead_of_truncating(self):
        trace=Inventory(limit=0)
        with self.assertRaises(RuntimeError):
            with trace.scope('forward'):torch.ones(2).relu()

    def test_exception_restores_modes(self):
        trace=Inventory()
        with self.assertRaises(ValueError):
            with trace.scope('forward'):raise ValueError('test')
        torch.ones(3).relu()
        self.assertEqual(trace.report()['signatures'],[])
        self.assertEqual(trace.phase,'outside')


if __name__=='__main__':unittest.main()
