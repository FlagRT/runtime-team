"""Persist inspected failed model evidence and test CPU fail-closed decisions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def create(evidence, operator_review, output):
    data=json.loads(evidence.read_text())
    assert data['status']=='fail' and data['mode']=='candidate'
    assert data['optimizer_steps']==0 and data['dispatcher_restored']
    assert any(row.get('mismatches') for row in data['comparisons'])
    shutil.copytree(operator_review,output)
    shutil.copy2(evidence,output/'model-failed-evidence.json')
    review=dict(environment=data['environment'],workload=data['workload_identity'],status='failed',revision=1,
                evidence='model-failed-evidence.json',evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    (output/'model-reviews.failed.json').write_text(json.dumps(dict(schema=1,reviews=[review]),indent=2)+'\n')
    for mode in ('denied','strict-denied'):
        config=json.loads((output/'unknown.json').read_text())
        config.update(require_target=mode=='strict-denied',workload_reviews='model-reviews.failed.json')
        (output/(mode+'.json')).write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('evidence','operator-review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();create(a.evidence,a.operator_review,a.output)
