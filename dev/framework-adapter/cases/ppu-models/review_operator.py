"""Materialize operator-only review AFTER an explicit result inspection.

Does not grant any model approval. Normal on is deliberately not emitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def create(evidence, out):
    data=json.loads(evidence.read_text())
    rows=data['cases']
    assert data['status']=='PASS' and data['stage']=='qualification'
    assert len(rows)==3 and {r['repeat'] for r in rows}=={0,1,2}
    first=rows[0]
    for row in rows:
        assert row['status']=='pass' and row['mode']=='direct' and row['restored']
        assert row['operator']=='silu' and row['shape']==first['shape'] and row['dtype']==first['dtype']
        assert row['hits']=={'GEMS SILU FORWARD':1,'GEMS SILU BACKWARD':1}
    assert data['environment']['backend']=='ppu'
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(evidence,out/'operator-evidence.json')
    record=dict(environment=data['environment'],operator='silu',dtype=first['dtype'],shape=first['shape'],
                stages=['forward','backward'],evidence='operator-evidence.json',
                evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),status='passed',revision=1)
    (out/'operator-records.json').write_text(json.dumps(dict(schema=1,records=[record]),indent=2)+'\n')
    for mode in ('off','unknown'):
        config=dict(schema=3,enabled=mode!='off',require_target=False,
                    operators={'silu':dict(shapes=[first['shape']],dtypes=[first['dtype']])},
                    records='operator-records.json',workload_reviews=None)
        (out/(mode+'.json')).write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args(); create(a.evidence,a.output)
