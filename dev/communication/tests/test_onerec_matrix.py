import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from summarize_onerec_matrix import summarize_matrix


class MatrixTests(unittest.TestCase):
    def fixture(self,root):
        manifest=dict(repeats=3,steps=2,seq_len=64,runs=[])
        for repeat in range(1,4):
            for cap in (64,128):
                name=f'{repeat:02d}-bucket{cap}'
                manifest['runs'].append(dict(name=name,repeat=repeat,bucket_mib=cap,exit_code=0))
                path=root/name;path.mkdir()
                (path/'execution.json').write_text('{"exit_code":0}')
                for rank in range(2):
                    d=dict(rank=rank,model='OpenOneRec/OneRec-8B',parameter_count=8391013376,
                        trainable_parameter_count=8391013376,backend='flagcx',revision='fixed',
                        dtype='bfloat16',optimizer='SGD',lr=.01,seq_len=64,global_batch=2,local_batch=1,
                        comm_mode='bucketed',bucket_bytes=cap*1024**2,teardown='CLOSED',
                        changed_parameter_tensors=387,implementation_sha256={'source':'same'},
                        memory={'used_mb':37000},
                        initial_sha256={str(i):'a' for i in range(399)},
                        final_sha256={str(i):'b' for i in range(399)},
                        communication={'gradient_allreduce_calls':(251 if cap==64 else 126)*2,
                                       'gradient_payload_bytes':16782026752*2},
                        steps=[dict(step=i,gradients_finite=True,loss=15.,seconds_with_finite_checks=1.)
                               for i in range(2)])
                    (path/f'rank{rank}.json').write_text(json.dumps(d))
        (root/'matrix.json').write_text(json.dumps(manifest))

    def test_complete_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            result=summarize_matrix(root)
            self.assertEqual(result['verdict'],'PASS')
            self.assertEqual(result['metrics']['128']['runs'],3)

    def test_missing_trial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            path=root/'matrix.json';obj=json.loads(path.read_text());obj['runs'].pop()
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):summarize_matrix(root)

    def test_duplicate_trial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            path=root/'matrix.json';obj=json.loads(path.read_text());obj['runs'][-1]=obj['runs'][0]
            path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):summarize_matrix(root)

    def test_nonzero_launcher(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            (root/'01-bucket64/execution.json').write_text('{"exit_code":1}')
            with self.assertRaises(ValueError):summarize_matrix(root)

    def test_cross_run_parameter_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            for rank in range(2):
                path=root/f'03-bucket128/rank{rank}.json';obj=json.loads(path.read_text())
                obj['final_sha256']['0']='different';path.write_text(json.dumps(obj))
            self.assertEqual(summarize_matrix(root)['verdict'],'FAIL')

    def test_wrong_sequence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            for rank in range(2):
                path=root/f'02-bucket64/rank{rank}.json';obj=json.loads(path.read_text())
                obj['seq_len']=16;path.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):summarize_matrix(root)


if __name__=='__main__':unittest.main()
