"""CPU-only guardrails for P800 resource selection; no accelerator commands."""
import unittest
from run import idle_p800, match_xpu_node, normalize_bus, workload_labels
from probe import baked_inventory_acceptable, BAKED_ARTIFACTS, GEMS


SAMPLE = '''|   5  P800 OAM           N/A   | 00000000:9C:00.0 N/A | 0 |
| N/A   36C N/A 92W / 400W |      0MiB / 98304MiB |      0% Default |
|                           |      0MiB / 96MiB | Disabled |
| XPU XI CI PID Type Process name XPU Memory |
'''


class ResourceTests(unittest.TestCase):
    def test_model_labels_use_actual_fixed_workload(self):
        for stage in ('model','onerec'):
            self.assertEqual(workload_labels(stage,'sigmoid',32),dict(operator='silu',batch_size=1))
        self.assertEqual(workload_labels('deepfm','silu',16),dict(operator='sigmoid',batch_size=16))

    def test_isolated_operator_is_not_a_model_batch(self):
        self.assertEqual(workload_labels('qualification','tanh',32),dict(operator='tanh',batch_size=None))

    def test_idle(self):
        self.assertEqual(idle_p800(SAMPLE,5)['chip'],'P800')

    def test_nonzero_memory_or_compute(self):
        for sample in (SAMPLE.replace('0MiB / 98304','4MiB / 98304'),SAMPLE.replace('0% Default','1% Default')):
            with self.assertRaisesRegex(ValueError,'not idle'):
                idle_p800(sample,5)

    def test_process_with_zero_rounded_memory(self):
        with self.assertRaisesRegex(ValueError,'process'):
            idle_p800(SAMPLE+'| 5 N/A N/A 123 C python 0MiB |',5)

    def test_wrong_or_unknown_chip(self):
        for sample in (SAMPLE.replace('P800','GPU'),SAMPLE.replace('5  P800','4  P800'),''):
            with self.assertRaisesRegex(ValueError,'identify'):
                idle_p800(sample,5)

    def test_exact_baked_artifacts_only(self):
        item=dict(head=GEMS,status='\n'.join(sorted(BAKED_ARTIFACTS)),diff_stat='',tracked_diff='')
        self.assertTrue(baked_inventory_acceptable(item))
        for change in ({'head':'other'},{'diff_stat':'modified'},{'tracked_diff':'modified'},
                       {'status':item['status']+'\n?? src/flag_gems/new.py'},{'status':''}):
            self.assertFalse(baked_inventory_acceptable(dict(item,**change)))

    def test_pci_index_differs_from_device_minor(self):
        actual=match_xpu_node(SAMPLE,5,{'/dev/xpu1':'0000:9c:00.0','/dev/xpu5':'0000:16:00.0'})
        self.assertEqual(actual['device_node'],'/dev/xpu1')
        self.assertEqual(actual['smi_index'],5)
        self.assertEqual(normalize_bus('00000000:9C:00.0'),'0000:9c:00.0')

    def test_ambiguous_or_missing_mapping_rejected(self):
        for nodes in ({},{'/dev/xpu0':'0000:16:00.0'},
                      {'/dev/xpu1':'0000:9c:00.0','/dev/xpu2':'0000:9c:00.0'}):
            with self.assertRaisesRegex(ValueError,'mapping'): match_xpu_node(SAMPLE,5,nodes)

    def test_mapping_still_rejects_busy_device(self):
        with self.assertRaisesRegex(ValueError,'not idle'):
            match_xpu_node(SAMPLE.replace('0MiB / 98304','4MiB / 98304'),5,{'/dev/xpu1':'0000:9c:00.0'})


if __name__ == '__main__':
    unittest.main()
