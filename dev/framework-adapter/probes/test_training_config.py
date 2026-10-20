"""CPU control tests only; no hardware qualification is produced."""
import json
import tempfile
from pathlib import Path
import unittest
import subprocess
import sys
import os
from types import SimpleNamespace
import _adapter_package
from flagrt_adapter.config import load_config, policies_from_config
from flagrt_adapter.policy import Environment, InputSpec, Capability


class TrainingConfigTests(unittest.TestCase):
    def test_cli_import_is_lightweight(self):
        import flagrt_adapter
        env=dict(os.environ,PYTHONPATH=str(Path(flagrt_adapter.__file__).resolve().parents[1]))
        result=subprocess.run([sys.executable,'-c',
            'import flagrt_adapter.train_cli; import sys; assert "torch" not in sys.modules'],env=env,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def load(self, **changes):
        data=dict(schema=1,enabled=False,operators={'silu':dict(shapes=[[1,32,3072]],dtypes=['torch.float32'])},records=None)
        data.update(changes)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'config.json'
            path.write_text(json.dumps(data))
            return load_config(path)

    def test_three_policies(self):
        env=Environment(*(['synthetic']*7))
        spec=InputSpec('npu','torch.float32',(1,32,3072),True)
        config,records=self.load()
        self.assertEqual(policies_from_config(config,records,env)['silu'].reason(env,spec),'experimental_disabled')
        config,records=self.load(enabled=True)
        self.assertEqual(policies_from_config(config,records,env)['silu'].reason(env,spec),'capability_unverified')
        record=Capability(env,'silu','torch.float32',(1,32,3072),('forward','backward'),'synthetic-only',status='passed')
        self.assertIsNone(policies_from_config(config,(record,),env)['silu'].reason(env,spec))

    def test_disabled_does_not_require_records(self):
        _,records=self.load(records='/nonexistent/records.json')
        self.assertEqual(records,())

    def test_enabled_missing_records_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.load(enabled=True,records='/nonexistent/records.json')

    def test_bad_switch_and_unknown_fields(self):
        for changes in ({'enabled':'false'},{'schema':True},{'extra':1},
                        {'operators':{'unknown':{}}},{'operators':{'silu':{'shapes':[[0]],'dtypes':['torch.float32']}}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                self.load(**changes)

    def test_runpy_disabled_and_exception_cleanup(self):
        from flagrt_adapter.training import execute_training
        env=Environment(*(['synthetic']*7))
        config,records=self.load()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); entry=root/'entry.py'
            entry.write_text('import torch\nimport torch.nn.functional as F\nx=torch.ones(2,requires_grad=True)\nF.silu(x).sum().backward()\n')
            report=execute_training(entry,config,records,env,SimpleNamespace(),None,root/'ok.json')
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['counts'][0]['reason'],'experimental_disabled')
            with self.assertRaises(ValueError):
                execute_training(entry,config,records,env,SimpleNamespace(),None,root/'ok.json')
            entry.write_text('raise RuntimeError("training failed")\n')
            with self.assertRaisesRegex(RuntimeError,'training failed'):
                execute_training(entry,config,records,env,SimpleNamespace(),None,root/'fail.json')
            self.assertTrue(json.loads((root/'fail.json').read_text())['scope_failed'])
            entry.write_text('pass\n')
            self.assertEqual(execute_training(entry,config,records,env,SimpleNamespace(),None,root/'after.json')['status'],'PASS')


if __name__=='__main__':
    unittest.main()
