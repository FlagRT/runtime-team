"""Manual model decision after checking native/repeat/candidate evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
for name in ('native','repeat','candidate','operator-review','output'):
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--decision',choices=['approve','reject'],required=True)
a=p.parse_args()
n,r,c=(json.loads(f.read_text()) for f in (a.native,a.repeat,a.candidate))
assert n['status']==r['status']=='pass' and n['mode']=='native' and r['mode']=='native_repeat'
for key in ('environment','workload_identity','initial_state_sha256','model_files','runtime_sources'):
    assert n[key]==r[key]==c[key],key
assert len(n['comparisons'])==len(r['comparisons'])==294
assert all(x['mismatches']==0 and x['max_abs']==0 for x in r['comparisons'])
assert c['mode']=='candidate' and c['qualification_only']
if a.decision=='approve':
    assert c['status']=='pass' and c['optimizer_steps']==3 and len(c['comparisons'])==294
    assert all(x['status']=='pass' for x in c['steps']) and len(c['steps'])==3
    assert all(x['mismatches']==0 for x in c['comparisons'])
    assert c['provider_calls']=={'GEMS EMBEDDING FORWARD':6,'GEMS EMBEDDING BACKWARD':6}
else: assert c['status']=='fail'
shutil.copytree(a.operator_review,a.output)
for label,path in [('native',a.native),('repeat',a.repeat),('candidate',a.candidate)]:
    shutil.copy2(path,a.output/(label+'.json'))
record=dict(environment=c['environment'],workload=c['workload_identity'],
    status='passed' if a.decision=='approve' else 'failed',revision=1,
    evidence='candidate.json',evidence_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest())
(a.output/'workload-reviews.json').write_text(json.dumps(dict(schema=1,reviews=[record]),indent=2)+'\n')
print(a.decision,a.output)
