"""Synthetic result validation tests, never counted as hardware evidence."""
import json
import tempfile
from pathlib import Path
import unittest
from check_training import check_run


class ResultTests(unittest.TestCase):
    def fixture(self,root,mode):
        folder=root/'train'; folder.mkdir()
        for rank in range(2):
            train=dict(rank=rank,world_size=2,verdict='TRAIN_LEG_PASS',passed=6,total=6,
                       perf=dict(steps=3,batch=1,seq=32),loss_curve=[3,2,1])
            counts=[dict(operator='silu',backend='flaggems',stage=s,reason='completed',count=84)
                    for s in ('forward','backward')] if mode=='on' else [dict(operator='silu',backend='native',stage='forward',
                    reason='experimental_disabled' if mode=='off' else 'capability_unverified',count=84)]
            adapter=dict(status='PASS',scope_failed=False,counts=counts)
            hits={'GEMS SILU FORWARD':84,'GEMS SILU BACKWARD':84} if mode=='on' else {}
            for name,data in [('train_leg_result',train),('adapter',adapter),('kernel_hits',hits)]:
                (folder/f'{name}_rank{rank}.json').write_text(json.dumps(data))

    def test_modes(self):
        for mode in ('off','on','unqualified'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); self.fixture(root,mode)
                self.assertEqual(check_run(root,mode)['status'],'pass')

    def test_missing_real_kernel_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,'on')
            (root/'train/kernel_hits_rank1.json').write_text('{}')
            with self.assertRaises(AssertionError): check_run(root,'on')

    def test_upstream_failure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root,'off')
            path=root/'train/train_leg_result_rank0.json'
            data=json.loads(path.read_text()); data['verdict']='TRAIN_LEG_FAIL'; path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError): check_run(root,'off')


if __name__=='__main__': unittest.main()
