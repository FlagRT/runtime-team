"""CPU wiring/review controls; does not establish PPU numerical qualification."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
from flagrt_adapter.adapter import FrameworkAdapter
from flagrt_adapter.policy import Environment
from qwen_review import validate_operator,validate_model

spec=importlib.util.spec_from_file_location('qwen_admission',Path(os.environ['QWEN_EMBEDDING_INPUT'])/'embedding_admission.py')
admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)

class Controls(unittest.TestCase):
    def test_exact_qwen_configuration(self):
        self.assertEqual(admission.OPERATORS['embedding']['signatures'],[dict(weight_shape=[151669,1024],
            indices_shape=[1,32],weight_dtype='torch.float32',indices_dtype='torch.int64')])
        self.assertFalse(admission.configuration('off')['enabled'])
        self.assertIsNone(admission.configuration('unknown')['workload_reviews'])
        self.assertTrue(admission.configuration('on')['require_target'])

    def test_mixed_cpu_scope(self):
        env=Environment(*(['synthetic']*7),backend='ppu',device_type='cuda')
        gems=SimpleNamespace(embedding=lambda w,i,*opts:torch.ops.aten.embedding.default(w,i,*opts),
            embedding_backward=lambda g,i,n,*opts:torch.ops.aten.embedding_backward.default(g,i,n,*opts))
        runtime=SimpleNamespace(synchronize=lambda _:None)
        for mode,reason in [('on',None),('off','experimental_disabled'),('unknown','workload_unverified')]:
            policy=SimpleNamespace(operator='embedding',reason=lambda *_,reason=reason:reason,strict=False)
            w=torch.randn(20,8,requires_grad=True)
            with FrameworkAdapter({'embedding':policy},env,runtime,gems) as scope:
                x=F.embedding(torch.arange(4).reshape(1,4),w)
                for _ in range(28):x=F.silu(x)
                x.backward(torch.ones_like(x))
            admission.check_model_routes([list(k)+[v] for k,v in scope.operator_counts.items()],mode)

    def test_route_rejects_extra_replacement(self):
        rows=[['embedding','flaggems','forward','completed',1],['embedding','flaggems','backward','completed',1],
              ['silu','native','forward','operator_disabled',28]]
        admission.check_model_routes(rows,'candidate')
        rows[-1][1]='flaggems'
        with self.assertRaises(AssertionError):admission.check_model_routes(rows,'candidate')

    def test_operator_review_rejects_wrong_shape(self):
        d=dict(status='PASS',cases=[dict(width=1024,pattern=s,repeat=n,status='pass',restored=True,
            hits={'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1})
            for s in ('unique','repeated','all_same','boundary') for n in range(3)],
            contract=dict(operator='embedding',weight_shapes=[[151669,1024]],weight_dtype='torch.float32',
                indices_shape=[1,32],indices_dtype='torch.int64',padding_idx=-1,scale_grad_by_freq=False,
                sparse=False,layout='contiguous',max_norm=None,atol=1e-5,rtol=1e-4))
        validate_operator(d);d['contract']['weight_shapes']=[[3850,16]]
        with self.assertRaises(AssertionError):validate_operator(d)

    def test_model_review_requires_two_complete_candidates(self):
        n=dict(status='pass',mode='native',parameter_tensors=310,
            comparisons=[dict(name=str(i),mismatches=0,max_abs=0.) for i in range(1863)],
            steps=[dict(status='pass',dispatcher_restored=True) for _ in range(3)],optimizer_steps=3,
            environment={},workload_identity={},initial_parameter_sha256='fixed',model_files={},runtime_sources={})
        r=dict(n,mode='native_repeat');c=dict(n,mode='candidate',qualification_only=True,
            provider_calls={'GEMS EMBEDDING FORWARD':3,'GEMS EMBEDDING BACKWARD':3},
            embedding_contract=dict(weight_shape=[151669,1024],padding_idx=None))
        validate_model(n,r,[c,c])
        with self.assertRaises(AssertionError):validate_model(n,r,[c])
        bad=copy.deepcopy(c);bad['comparisons'][0]['mismatches']=1
        with self.assertRaises(AssertionError):validate_model(n,r,[c,bad])
        bad=copy.deepcopy(c);bad['runtime_sources']={'changed':True}
        with self.assertRaises(AssertionError):validate_model(n,r,[c,bad])

if __name__=='__main__':unittest.main()
