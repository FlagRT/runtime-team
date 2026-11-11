"""Replay a real failed model review through unified startup (CPU control)."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from admission import configuration,OPERATORS
from flagrt_adapter.config import load_config
from flagrt_adapter.records import environment_from_record
from flagrt_adapter.startup import prepare_policies,TargetBlocked
from flagrt_adapter.workload import Workload

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():p.error('Do not overwrite evidence')
candidate=json.loads((a.review/'candidate.json').read_text())
assert candidate['status']=='fail' and candidate['qualification_only']
actual=environment_from_record(candidate['environment']);workload=Workload(**candidate['workload_identity'])
config,records=load_config(a.review/'on.json')
config['require_target']=False
_,report=prepare_policies(config,records,actual,workload=workload)
assert len(report['checks'])==2 and all(r['reason']=='workload_denied' for r in report['checks'])
config['require_target']=True
try:prepare_policies(config,records,actual,workload=workload)
except TargetBlocked as exc:
    assert all(c['reason']=='workload_denied' for c in exc.report['checks'])
else:raise AssertionError('Failed workload enabled')
a.output.write_text(json.dumps(dict(status='PASS',scope='CPU startup replay, no hardware execution',
    checks=2,native_reason='workload_denied',strict='BLOCKED',target_calls=0),indent=2)+'\n')
print('PASS: failed workload native/strict controls, no target call')
