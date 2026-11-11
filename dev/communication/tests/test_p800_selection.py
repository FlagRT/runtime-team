"""Host validation tests only: synthetic metadata is not P800 evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from run_p800_selected import snapshot, validate_single


class P800SelectionTests(unittest.TestCase):
    def test_only_completely_idle_cards_selected(self):
        rows=[]
        for card,memory,util in ((0,0,0),(1,72,0),(2,0,1),(3,1,0)):
            fields=['00000000:03:00.0','0',str(card)]+['0']*17
            fields[17],fields[19]=str(memory),str(util)
            rows.append(' '.join(fields)+' "P800 OAM"')
        with patch('run_p800_selected.subprocess.check_output',return_value='\n'.join(rows)):
            self.assertEqual(sorted(snapshot()[1]),[0])

    def fixture(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);(root/'manual').mkdir()
        source=Path(__file__).resolve().parents[1]/'results/20261028-deepfm/01-manual/rank0.json'
        data=json.loads(source.read_text())
        data.update(world_size=1,local_batch=64,backend='flagcx',device='cuda:0',
                    evidence_scope='P800 integration',runtime_info={'name':'kunlun'})
        return root,data

    def check(self,data,root):
        (root/'manual/rank0.json').write_text(json.dumps(data))
        validate_single(root,'manual')

    def test_complete_single_card_record(self):
        root,data=self.fixture();self.check(data,root)

    def test_pending_teardown_rejected(self):
        root,data=self.fixture();data['teardown']='pending'
        with self.assertRaises(ValueError):self.check(data,root)

    def test_nonfinite_loss_rejected(self):
        root,data=self.fixture();data['training']['losses'][0]=float('nan')
        with self.assertRaises(ValueError):self.check(data,root)

    def test_wrong_gradient_count_rejected(self):
        root,data=self.fixture();data['training']['gradient_communication']['gradient_allreduce_calls']=0
        with self.assertRaises(ValueError):self.check(data,root)


if __name__=='__main__':unittest.main()
