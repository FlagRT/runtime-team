"""CPU controls for collecting actual model identity and interpreting blocks."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from flagrt_adapter.startup import TargetBlocked
from model_admission import workload_from_report, prepare_model, record_expected_block


class ModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.report=dict(model_files={'model.py':'sha'}, workload={'model':'DeepFM','batch':32,'lr':.001},
                         runtime_sources={'backend.py':'sha'}, public_config={'dropout':.2})
        self.ops={'sigmoid':dict(shapes=[[32]],dtypes=['torch.float32'])}

    def test_measured_settings_all_change_identity(self):
        base=workload_from_report(self.report,self.ops,'bridge1')
        for section, field, value in (('model_files','model.py','new'),('workload','batch',16),
                ('workload','lr',.01),('public_config','dropout',0),('runtime_sources','backend.py','new')):
            report=copy.deepcopy(self.report); report[section][field]=value
            self.assertNotEqual(base,workload_from_report(report,self.ops,'bridge1'))
        self.assertNotEqual(base,workload_from_report(self.report,self.ops,'bridge2'))

    def test_mode_label_does_not_manufacture_approval(self):
        base=workload_from_report(self.report,self.ops,'bridge')
        for mode in ('on','denied','native'):
            self.assertEqual(base,workload_from_report(dict(self.report,mode=mode),self.ops,'bridge'))

    def test_expected_block_is_not_training_pass(self):
        r=dict(model_loaded=False,optimizer_created=False,steps=[],optimizer_steps=0)
        exc=TargetBlocked('synthetic',{'checks':[{'reason':'workload_denied'}]})
        record_expected_block(SimpleNamespace(mode='strict'),r,exc)
        self.assertEqual(r['status'],'blocked'); self.assertFalse(r['provider_imported'])

    def test_incorrect_or_late_block_is_not_accepted(self):
        exc=TargetBlocked('synthetic',{'checks':[{'reason':'workload_denied'}]})
        for r in ({'model_loaded':True},{'optimizer_created':True},{'steps':[1]}, {'optimizer_steps':1}):
            with self.assertRaises(AssertionError): record_expected_block(SimpleNamespace(mode='strict'),r,exc)
        for reason in ('workload_unverified','shape'):
            with self.assertRaises(AssertionError):
                record_expected_block(SimpleNamespace(mode='strict'),{},TargetBlocked('x',{'checks':[{'reason':reason}]}))

    def test_legacy_config_cannot_enable_model_path(self):
        from dataclasses import asdict
        from flagrt_adapter.policy import Environment
        r=dict(self.report,environment=asdict(Environment(*(['synthetic']*7))))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'config.json'
            path.write_text(json.dumps(dict(schema=2,enabled=True,require_target=True,records=None,operators=self.ops)))
            with self.assertRaisesRegex(ValueError,'schema 3'):
                prepare_model(SimpleNamespace(mode='on',adapter_config=path),r,'sigmoid',[32])


if __name__=='__main__': unittest.main()
