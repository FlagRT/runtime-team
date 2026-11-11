"""Package an explicitly reviewed ReLU result; never called by a test runner."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decision',choices=['approve-operator','approve-model','reject-model'],required=True)
    p.add_argument('--operator-review',type=Path)
    p.add_argument('--reviewer',required=True)
    a=p.parse_args();r=json.loads(a.evidence.read_text())
    if a.decision=='approve-operator':
        assert r['status']=='PASS' and r['stage']=='qualification' and len(r['cases'])==3
        for row in r['cases']:
            assert row['status']=='pass' and row['operator']=='relu' and row['shape']==[32,400]
            assert row['dtype']=='torch.float32' and row['restored'] and row['atol']==row['rtol']==0
            assert row['hits']=={'GEMS RELU FORWARD':1,'GEMS THRESHOLD BACKWARD':1}
            assert all(row[k]==0 for k in ('forward_max_abs','backward_max_abs','cpu_forward_max_abs','cpu_backward_max_abs'))
        a.output.mkdir(exist_ok=False)
        shutil.copy2(a.evidence,a.output/'operator-evidence.json')
        write(a.output/'records.json',dict(schema=1,records=[dict(environment=r['environment'],
            operator='relu',dtype='torch.float32',shape=[32,400],stages=['forward','backward'],
            evidence='operator-evidence.json',evidence_sha256=sha(a.evidence),status='passed',revision=1)]))
    else:
        assert r['mode']=='candidate' and r['qualification_only'] and a.operator_review
        if a.decision=='approve-model':
            assert r['status']=='pass' and r['optimizer_steps']==3 and len(r['comparisons'])==294
            assert len(r['steps'])==3 and all(x['status']=='pass' for x in r['steps'])
            assert all(not x.get('mismatches',0) for x in r['comparisons'])
            assert r['provider_calls']=={'GEMS RELU FORWARD':9,'GEMS THRESHOLD BACKWARD':9}
        else:assert r['status']=='fail'
        shutil.copytree(a.operator_review,a.output)
        shutil.copy2(a.evidence,a.output/'model-evidence.json')
        write(a.output/'model-reviews.json',dict(schema=1,reviews=[dict(environment=r['environment'],
            workload=r['workload_identity'],status='passed' if a.decision=='approve-model' else 'failed',
            revision=1,evidence='model-evidence.json',evidence_sha256=sha(a.evidence))]))
    for mode in ('off','on','unknown'):
        write(a.output/(mode+'.json'),dict(schema=3,enabled=mode!='off',require_target=mode=='on',
            operators={'relu':dict(shapes=[[32,400]],dtypes=['torch.float32'])},records='records.json',
            workload_reviews='model-reviews.json' if mode=='on' and a.decision=='approve-model' else None))
    write(a.output/'review-note.json',dict(decision=a.decision,reviewer=a.reviewer,
        source_sha256=sha(a.evidence),date='2026-11-10',scope='PPU FP32 ReLU 32x400; finite input and first-order gradients'))
