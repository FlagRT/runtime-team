import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('summarize_onerec',Path(__file__).resolve().parents[1]/'training/summarize_onerec.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SummaryTests(unittest.TestCase):
    def fixture(self,p):
        (p/'execution.json').write_text(json.dumps({'exit_code':0}))
        for r in range(2):
            d=dict(rank=r,model='OpenOneRec/OneRec-8B',parameter_count=8391013376,
                   trainable_parameter_count=8391013376,steps=[dict(step=0,gradients_finite=True,loss=15.)],
                   teardown='CLOSED',changed_parameter_tensors=1,
                   communication=dict(gradient_allreduce_calls=399,gradient_payload_bytes=16782026752),
                   backend='hccl',revision='test',dtype='bfloat16',optimizer='SGD',lr=.01,
                   seq_len=16,global_batch=2,local_batch=1,
                   initial_sha256={str(k):'a' for k in range(399)},final_sha256={str(k):'b' for k in range(399)})
            (p/f'rank{r}.json').write_text(json.dumps(d))

    def test_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            self.assertEqual(module.summarize(p,1)['verdict'],'PASS')

    def test_launcher_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            (p/'execution.json').write_text('{"exit_code":1}')
            self.assertEqual(module.summarize(p,1)['verdict'],'FAIL')

    def test_missing_steps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            self.assertEqual(module.summarize(p,3)['verdict'],'FAIL')

    def test_different_final_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            f=p/'rank1.json';obj=json.loads(f.read_text());obj['final_sha256']['0']='c';f.write_text(json.dumps(obj))
            self.assertEqual(module.summarize(p,1)['verdict'],'FAIL')

    def test_bucketed_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.fixture(p)
            for rank in range(2):
                f=p/f'rank{rank}.json'; obj=json.loads(f.read_text())
                obj.update(comm_mode='bucketed', bucket_bytes=64*1024**2)
                obj['communication']['gradient_allreduce_calls']=251
                f.write_text(json.dumps(obj))
            self.assertEqual(module.summarize(p,1)['verdict'],'PASS')
            obj['communication']['gradient_allreduce_calls']=250
            f.write_text(json.dumps(obj))
            self.assertEqual(module.summarize(p,1)['verdict'],'FAIL')

    def test_unknown_mode_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); self.fixture(p)
            for rank in range(2):
                f=p/f'rank{rank}.json'; obj=json.loads(f.read_text())
                obj['comm_mode']='unsupported'; f.write_text(json.dumps(obj))
            self.assertEqual(module.summarize(p,1)['verdict'],'FAIL')


if __name__ == '__main__':unittest.main()
