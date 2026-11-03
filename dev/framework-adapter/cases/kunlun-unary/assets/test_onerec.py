"""CPU-only OneRec admission controls; not 8B training or accelerator evidence."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flagrt_adapter import Environment
from onerec_probe import verify_architecture, prepare_onerec, OPERATORS, TOLERANCE
from model_admission import prepare_model
from probe import selected_shapes


class OneRecControls(unittest.TestCase):
    def setUp(self):
        self.config=dict(model_type='qwen3',num_hidden_layers=36,hidden_size=4096,
                         intermediate_size=12288,hidden_act='silu')
        self.report=dict(model_files={'shard':'sha'},workload={'model':'OpenOneRec/OneRec-8B'},
                         runtime_sources={'backend':'sha'},environment=asdict(Environment(*(['x']*7))))

    def test_full_architecture_required(self):
        verify_architecture(self.config)
        for key in self.config:
            wrong=dict(self.config); wrong[key]='wrong'
            with self.assertRaisesRegex(ValueError,'architecture'): verify_architecture(wrong)

    def test_shape_profile_is_narrow(self):
        self.assertEqual(selected_shapes('silu','onerec16'),((1,16,12288),))
        self.assertEqual(OPERATORS['silu']['dtypes'],['torch.bfloat16'])
        for operator,profile in (('sigmoid','onerec16'),('silu','unknown')):
            with self.assertRaises(ValueError): selected_shapes(operator,profile)

    def test_existing_onerec_tolerance_unchanged(self):
        self.assertEqual(TOLERANCE,dict(atol=1e-3,rtol=1e-2))

    def test_candidate_requires_oracle_and_no_model_config(self):
        for values in (dict(adapter_config=Path('x'),records=Path('y'),reference=Path('z')),
                       dict(adapter_config=None,records=None,reference=Path('z')),
                       dict(adapter_config=None,records=Path('y'),reference=None)):
            with self.assertRaisesRegex(ValueError,'candidate requires'):
                prepare_onerec(SimpleNamespace(mode='candidate',**values),self.report)

    def test_candidate_is_explicit_not_auto_approval(self):
        with patch('flagrt_adapter.load_records',return_value=()), \
             patch('flagrt_adapter.startup.prepare_policies',return_value=({},{})) as prepare:
            r=copy.deepcopy(self.report)
            prepare_onerec(SimpleNamespace(mode='candidate',records=Path('x'),reference=Path('y'),adapter_config=None),r)
            self.assertTrue(r['qualification_experiment']); self.assertFalse(r['model_approved'])
            self.assertEqual(prepare.call_args.args[0]['schema'],2)

    def test_normal_on_still_requires_model_review(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json'
            p.write_text(json.dumps(dict(schema=2,enabled=True,require_target=True,records=None,operators=OPERATORS)))
            with self.assertRaisesRegex(ValueError,'schema 3'):
                prepare_onerec(SimpleNamespace(mode='on',adapter_config=p),copy.deepcopy(self.report))

    def test_bfloat16_is_bound_to_measured_identity(self):
        r=copy.deepcopy(self.report)
        with patch('flagrt_adapter.startup.prepare_policies',return_value=({},{})) as prepare:
            prepare_model(SimpleNamespace(mode='native'),r,'silu',[1,16,12288],'torch.bfloat16')
            self.assertEqual(prepare.call_args.args[0]['operators'],OPERATORS)
        changed=copy.deepcopy(self.report)
        with patch('flagrt_adapter.startup.prepare_policies',return_value=({},{})):
            prepare_model(SimpleNamespace(mode='native'),changed,'silu',[1,16,12288],'torch.float32')
        self.assertNotEqual(r['workload_identity'],changed['workload_identity'])


if __name__=='__main__': unittest.main()
