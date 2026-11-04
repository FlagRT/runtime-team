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

    def measured_fixture(self,root,mode):
        self.fixture(root,mode)
        for rank in (0,1):
            folder=root/'train'
            path=folder/f'adapter_rank{rank}.json'; data=json.loads(path.read_text())
            data['startup']={'workload_required':True,'workload':{'identity':'synthetic'}}
            if mode=='unqualified': data['counts'][0]['reason']='workload_unverified'
            path.write_text(json.dumps(data))
            (folder/f'workload_rank{rank}.json').write_text(json.dumps({'workload':{'identity':'synthetic'}}))
            guard=folder/f'step-control/rank{rank}'; guard.mkdir(parents=True)
            (guard/'report.json').write_text(json.dumps(dict(status='PASS',steps=3,hooks_restored=True)))

    def test_measured_three_paths(self):
        for mode in ('off','on','unqualified'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); self.measured_fixture(root,mode)
                self.assertEqual(check_run(root,mode,measured=True)['status'],'pass')

    def test_rank_identity_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.measured_fixture(root,'off')
            (root/'train/workload_rank1.json').write_text('{"workload":{"identity":"wrong"}}')
            with self.assertRaises(AssertionError): check_run(root,'off',measured=True)

    def test_guard_not_completed_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.measured_fixture(root,'off')
            path=root/'train/step-control/rank1/report.json'
            data=json.loads(path.read_text()); data['steps']=2; path.write_text(json.dumps(data))
            with self.assertRaises(AssertionError): check_run(root,'off',measured=True)


if __name__=='__main__': unittest.main()
