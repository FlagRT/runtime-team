"""CPU controls for the expected-rejection evidence checker."""
import json
from pathlib import Path
import tempfile
import unittest
from check_training import check_blocked


class RejectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name); self.root=self.output/'train'; self.root.mkdir()
        for r in (0,1):
            self.write(f'adapter_rank{r}.json',dict(status='BLOCKED',counts=[],events_tail=[],startup=dict(
                require_target=True,workload_required=True,workload={'id':'same'},
                checks=[dict(reason='workload_unverified',decision='native')])))
            self.write(f'workload_rank{r}.json',dict(workload={'id':'same'}))
            self.write(f'kernel_hits_rank{r}.json',{})
            self.write(f'step-control/rank{r}/report.json',dict(steps=0,hooks_restored=True))

    def write(self,name,data):
        p=self.root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data))

    def test_complete_two_rank_rejection(self):
        self.assertEqual(check_blocked(self.output,1)['status'],'pass')

    def test_no_success_or_timeout(self):
        for rc in (0,124,137):
            with self.assertRaises(AssertionError): check_blocked(self.output,rc)

    def test_missing_rank_is_not_pass(self):
        (self.root/'adapter_rank1.json').unlink()
        with self.assertRaises(FileNotFoundError): check_blocked(self.output,1)

    def test_training_or_hits_is_not_pre_entry_rejection(self):
        self.write('kernel_hits_rank0.json',{'GEMS SILU FORWARD':1})
        with self.assertRaises(AssertionError): check_blocked(self.output,1)
        self.write('kernel_hits_rank0.json',{})
        self.write('train_leg_result_rank0.json',{})
        with self.assertRaises(AssertionError): check_blocked(self.output,1)

    def test_wrong_reason_is_not_pass(self):
        p=self.root/'adapter_rank0.json'; a=json.loads(p.read_text())
        a['startup']['checks'][0]['reason']='provider_missing'; self.write(p.name,a)
        with self.assertRaises(AssertionError): check_blocked(self.output,1)

    def test_step_or_rank_mismatch_is_not_pass(self):
        self.write('step-control/rank0/report.json',dict(steps=1,hooks_restored=True))
        with self.assertRaises(AssertionError): check_blocked(self.output,1)
        self.write('step-control/rank0/report.json',dict(steps=0,hooks_restored=True))
        self.write('workload_rank1.json',dict(workload={'id':'different'}))
        with self.assertRaises(AssertionError): check_blocked(self.output,1)


if __name__=='__main__': unittest.main()
