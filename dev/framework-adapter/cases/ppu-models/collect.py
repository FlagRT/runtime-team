"""Collect stopped case evidence; keep large model/tensor bytes on server."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    report=dict(runs={},inputs={},tensor_backup='remote only; SHA manifest included')
    # Include startup failures that never created a run directory (e.g. sudo).
    members=sorted(a.root.glob('*.host.log'))
    for d in sorted(a.root.iterdir()):
        if not d.is_dir(): continue
        if (d/'inputs.json').is_file():
            expected=json.loads((d/'inputs.json').read_text())
            for name,value in expected.items():
                assert sha(d/name)==value,'Input changed: '+str(d/name)
            report['inputs'][d.name]=expected
            continue
        if not (d/'summary.json').is_file(): continue
        summary=json.loads((d/'summary.json').read_text())
        if 'stop' in summary: assert summary['stop']['Pid']==0 and not summary['stop']['Running']
        r=json.loads((d/'result.json').read_text()) if (d/'result.json').is_file() else {}
        row=dict(host_status=summary['status'],model_status=r.get('status'),mode=r.get('mode'),
                 steps=r.get('steps'),optimizer_steps=r.get('optimizer_steps'),
                 comparison_count=len(r.get('comparisons',[])),provider_calls=r.get('provider_calls'),
                 failure=[x for x in r.get('comparisons',[]) if x.get('mismatches',0)],
                 stop=summary.get('stop'),error=r.get('error',summary.get('error')))
        if (d/'tensors').is_dir():
            tensors={str(f.relative_to(d)):dict(sha256=sha(f),bytes=f.stat().st_size)
                     for f in sorted((d/'tensors').rglob('*')) if f.is_file()}
            (a.output/(d.name+'-tensor-sha.json')).write_text(json.dumps(tensors,indent=2)+'\n')
            row['tensor_count']=len(tensors)
            row['tensor_bytes']=sum(x['bytes'] for x in tensors.values())
        report['runs'][d.name]=row
        for f in sorted(d.rglob('*')):
            if not f.is_file():continue
            relative=f.relative_to(d)
            if set(relative.parts)&{'tensors','packages','triton-cache','gems-cache','__pycache__'}:continue
            members.append(f)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    file_shas={str(f.relative_to(a.root)):sha(f) for f in members}
    (a.output/'evidence-sha.json').write_text(json.dumps(file_shas,indent=2)+'\n')
    with tarfile.open(a.output/'evidence.tar.gz','w:gz') as tar:
        for f in members:tar.add(f,arcname=str(f.relative_to(a.root)))
    (a.output/'archive.sha256').write_text(sha(a.output/'evidence.tar.gz')+'  evidence.tar.gz\n')
    print(json.dumps(dict(runs=len(report['runs']),files=len(members),output=str(a.output))))
