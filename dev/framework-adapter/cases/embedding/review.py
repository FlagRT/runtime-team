"""Explicit reviewer command, separate from test runner; immutable evidence bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--qualification',type=Path,required=True)
p.add_argument('--decision',choices=['approve'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();d=json.loads(a.qualification.read_text())
assert d['status']=='PASS' and len(d['cases'])==24
assert {(r['width'],r['pattern'],r['repeat']) for r in d['cases']}=={
    (w,s,n) for w in (1,16) for s in ('unique','repeated','all_same','boundary') for n in range(3)}
assert all(r['status']=='pass' and r['restored'] and r['hits']=={
    'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1} for r in d['cases'])
assert d['contract']=={'operator':'embedding','weight_shapes':[[3850,16],[3850,1]],
    'weight_dtype':'torch.float32','indices_shape':[32,5],'indices_dtype':'torch.int64',
    'padding_idx':-1,'scale_grad_by_freq':False,'sparse':False,'layout':'contiguous',
    'max_norm':None,'atol':1e-5,'rtol':1e-4}
a.output.mkdir(parents=True,exist_ok=False)
shutil.copy2(a.qualification,a.output/'qualification.json')
sha=hashlib.sha256(a.qualification.read_bytes()).hexdigest()
records=[dict(environment=d['environment'],signature=dict(weight_shape=[3850,w],indices_shape=[32,5],
    weight_dtype='torch.float32',indices_dtype='torch.int64'),status='passed',stages=['forward','backward'],
    evidence='qualification.json',evidence_sha256=sha,revision=1) for w in (16,1)]
(a.output/'records.json').write_text(json.dumps(dict(schema=1,kind='embedding-dense-no-padding-v1',records=records),indent=2)+'\n')
print('Approved exact operator contract only:',a.output)
