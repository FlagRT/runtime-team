"""Evidence mutations must never turn missing/failed execution into PASS."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from summarize_repeated import summarize


class RepeatedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'evidence'
        shutil.copytree(Path(__file__).resolve().parents[1]/'results/20261014-host', self.root)

    def change(self, file, function):
        path=self.root/file; data=json.loads(path.read_text()); function(data)
        path.write_text(json.dumps(data))

    def test_valid_records(self): self.assertEqual(summarize(self.root)['runs'],6)

    def test_nonzero_launcher_rejected(self):
        self.change('execution.json', lambda d: d['runs'][0].update(exit_code=134))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_missing_rank_rejected(self):
        (self.root/'01-manual/rank1.json').unlink()
        with self.assertRaises(FileNotFoundError): summarize(self.root)

    def test_wrong_scope_rejected(self):
        self.change('execution.json', lambda d:d.update(scope='Ascend integration'))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_wrong_call_count_rejected(self):
        self.change('01-fused/rank0.json', lambda d:d['training']['gradient_communication'].update(gradient_allreduce_calls=0))
        with self.assertRaises(ValueError): summarize(self.root)

    def test_duplicate_trial_rejected(self):
        self.change('execution.json', lambda d:d['runs'].append(d['runs'][0]))
        with self.assertRaises(ValueError): summarize(self.root)

    def kunlun_fixture(self):
        # Mutation of host fixtures tests validation only; not hardware evidence.
        self.change('execution.json',lambda d:d.update(scope='Kunlun integration'))
        for path in self.root.glob('*/rank*.json'):
            self.change(path.relative_to(self.root),lambda d:d.update(
                backend='flagcx',evidence_scope='P800 integration',
                device='cuda:'+str(d['rank']),runtime_info={'name':'kunlun'}))

    def test_kunlun_metadata_accepted(self):
        self.kunlun_fixture()
        self.assertEqual(summarize(self.root)['scope'],'Kunlun integration')

    def test_kunlun_wrong_vendor_rejected(self):
        self.kunlun_fixture()
        self.change('01-manual/rank0.json',lambda d:d.update(runtime_info={'name':'nvidia'}))
        with self.assertRaises(ValueError):summarize(self.root)

    def test_kunlun_cpu_fallback_rejected(self):
        self.kunlun_fixture()
        self.change('01-manual/rank0.json',lambda d:d.update(device='cpu'))
        with self.assertRaises(ValueError):summarize(self.root)

    def native_fixture(self):
        # Validator-only mutation; these are not accelerator results.
        for repeat in range(1,4):
            old=f'{repeat:02d}-fused';name=f'{repeat:02d}-native'
            shutil.copytree(self.root/old,self.root/name)
            for rank in range(2):
                self.change(f'{name}/rank{rank}.json',lambda d:d.update(
                    comm_mode='native',training={**d['training'],
                    'gradient_communication':{'observed':False}}))
            self.change('execution.json',lambda d:d['runs'].append(
                dict(name=name,mode='native',repeat=repeat,exit_code=0,timed_out=False)))

    def test_native_matrix_requires_opt_in(self):
        self.native_fixture()
        with self.assertRaises(ValueError):summarize(self.root)

    def test_native_matrix_complete(self):
        self.native_fixture()
        self.assertEqual(summarize(self.root,True)['runs'],9)

    def test_native_matrix_missing_trial_rejected(self):
        with self.assertRaises(ValueError):summarize(self.root,True)

    def test_native_counters_not_invented(self):
        self.native_fixture()
        self.change('01-native/rank0.json',lambda d:d['training'].update(gradient_communication={'observed':True}))
        with self.assertRaises(ValueError):summarize(self.root,True)

    def observed_fixture(self):
        self.native_fixture()
        manifest=json.loads((self.root/'execution.json').read_text())
        for run in manifest['runs']:
            if run['mode']!='native':continue
            old=run['name'];new=old.replace('native','observed')
            (self.root/old).rename(self.root/new)
            run.update(name=new,mode='observed')
            for rank in range(2):
                def mutate(d):
                    d['comm_mode']='observed'
                    t=d['training'];size=t['parameter_count']*4
                    t['ddp_bucket_sizes_bytes']=[size-4,4]
                    t['gradient_communication']={'gradient_allreduce_calls':100,'gradient_payload_bytes':size*50}
                self.change(f'{new}/rank{rank}.json',mutate)
        (self.root/'execution.json').write_text(json.dumps(manifest))

    def test_observed_two_buckets_accepted(self):
        self.observed_fixture()
        self.assertEqual(summarize(self.root,include_observed=True)['runs'],9)

    def test_observed_bad_bucket_coverage_rejected(self):
        self.observed_fixture()
        self.change('01-observed/rank0.json',lambda d:d['training'].update(ddp_bucket_sizes_bytes=[1]))
        with self.assertRaises(ValueError):summarize(self.root,include_observed=True)

    def callback_fixture(self):
        self.observed_fixture()
        for path in self.root.glob('*observed/rank*.json'):
            def mutate(d):
                t=d['training'];sizes=t['ddp_bucket_sizes_bytes']
                t['ddp_logging_bucket_sizes_bytes']=[sum(sizes)]  # Vendor diagnostic is stale.
                t['ddp_bucket_metadata_source']='GradBucket callbacks per measured backward'
                t['ddp_bucket_trace']=[{'step':step,'buckets':[
                    {'index':index,'bytes':size} for index,size in enumerate(sizes)]} for step in range(50)]
            self.change(path.relative_to(self.root),mutate)

    def test_actual_callbacks_override_stale_diagnostic_without_guessing(self):
        self.callback_fixture()
        self.assertEqual(summarize(self.root,include_observed=True)['verdict'],'PASS')

    def test_missing_backward_trace_rejected(self):
        self.callback_fixture()
        self.change('01-observed/rank0.json',lambda d:d['training']['ddp_bucket_trace'].pop())
        with self.assertRaises(ValueError):summarize(self.root,include_observed=True)

    def test_wrong_bucket_index_rejected(self):
        self.callback_fixture()
        self.change('01-observed/rank0.json',lambda d:d['training']['ddp_bucket_trace'][0]['buckets'][1].update(index=0))
        with self.assertRaises(ValueError):summarize(self.root,include_observed=True)

    def test_incomplete_per_step_bytes_rejected(self):
        self.callback_fixture()
        self.change('01-observed/rank0.json',lambda d:d['training']['ddp_bucket_trace'][0]['buckets'][1].update(bytes=8))
        with self.assertRaises(ValueError):summarize(self.root,include_observed=True)

    def test_old_p800_stale_metadata_does_not_get_relabelled_pass(self):
        self.observed_fixture()
        self.change('01-observed/rank0.json',lambda d:d['training'].update(
            ddp_bucket_sizes_bytes=[d['training']['parameter_count']*4]))
        with self.assertRaises(ValueError):summarize(self.root,include_observed=True)


if __name__ == '__main__': unittest.main()
