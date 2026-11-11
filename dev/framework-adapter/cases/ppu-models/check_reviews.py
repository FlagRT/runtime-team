"""CPU-only check of persisted failed review; no model execution."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from flagrt_adapter.config import load_config
from flagrt_adapter.records import environment_from_record
from flagrt_adapter.workload import Workload
from flagrt_adapter.startup import prepare_policies, TargetBlocked

p=argparse.ArgumentParser();p.add_argument('review',type=Path);a=p.parse_args()
d=json.loads((a.review/'model-failed-evidence.json').read_text())
actual=environment_from_record(d['environment']);workload=Workload(**d['workload_identity'])
c,r=load_config(a.review/'denied.json')
_,s=prepare_policies(c,r,actual,workload=workload)
assert s['checks'] and all(x['reason']=='workload_denied' for x in s['checks'])
c,r=load_config(a.review/'strict-denied.json')
try:prepare_policies(c,r,actual,workload=workload)
except TargetBlocked as e:
    assert e.report['checks'] and all(x['reason']=='workload_denied' for x in e.report['checks'])
else:raise AssertionError('Strict model denial did not block')
print(json.dumps(dict(scope='CPU control, no model or accelerator execution',status='PASS',
                     adapter=importlib.metadata.version('flagrt-framework-adapter'),
                     model=workload.model_id,checks=['failed-review-native','failed-review-strict-block']),indent=2))
