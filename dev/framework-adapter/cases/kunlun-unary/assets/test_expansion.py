"""CPU safeguards for the additional unary cases/public DeepFM workload."""
import logging
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import torch
from probe import backward_argument, Hits, OP_SHAPES
from deepfm_probe import load_model, make_batch, state_hash


class ExpansionTests(unittest.TestCase):
    def test_saved_output_backward_contract(self):
        source,output=object(),object()
        self.assertIs(backward_argument('silu',source,output),source)
        for name in ('sigmoid','tanh'):
            self.assertIs(backward_argument(name,source,output),output)
        with self.assertRaises(ValueError): backward_argument('unknown',source,output)

    def test_hits_are_operator_specific(self):
        h=Hits('sigmoid')
        for message in ('GEMS SIGMOID FORWARD','GEMS SIGMOID BACKWARD','GEMS SILU FORWARD'):
            h.emit(logging.LogRecord('test',logging.DEBUG,'',1,message,(),None))
        self.assertEqual(dict(h.counts),{'GEMS SIGMOID FORWARD':1,'GEMS SIGMOID BACKWARD':1})
        self.assertEqual(OP_SHAPES['sigmoid'],((16,),(32,)))

    def test_unknown_model_source_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('build.py','config.py','model.py'):
                Path(d,name).write_text('raise RuntimeError("must not import")')
            with self.assertRaisesRegex(ValueError,'source identity'): load_model(d)

    def test_data_repeatable_and_in_range(self):
        config=SimpleNamespace(field_dims=[7,11,5])
        for size in (16,32):
            x,y=make_batch(config,size,0); xx,yy=make_batch(config,size,0)
            self.assertTrue(torch.equal(x,xx) and torch.equal(y,yy))
            self.assertEqual(tuple(x.shape),(size,3))
            self.assertTrue(all(bool((x[:,i]<n).all()) for i,n in enumerate(config.field_dims)))
            self.assertTrue(bool(((y==0)|(y==1)).all()))
            self.assertFalse(torch.equal(x,make_batch(config,size,1)[0]))

    def test_state_hash_includes_bn_buffers(self):
        model=torch.nn.BatchNorm1d(3)
        before=state_hash(model)
        model.num_batches_tracked.add_(1)
        self.assertNotEqual(before,state_hash(model))


if __name__=='__main__': unittest.main()
