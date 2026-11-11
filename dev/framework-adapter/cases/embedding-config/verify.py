"""Offline original-evidence verification; no model approval is issued."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser()
for name in ('root','collection','audit','build','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
assert not a.output.exists()
manifest=json.loads((a.collection/'evidence-sha.json').read_text())
for path,digest in manifest.items():assert sha(a.root/path)==digest,path
states=json.loads((a.audit/'containers.json').read_text());assert len(states)==5
assert all(not s['Running'] and s['Pid']==0 and not s['OOMKilled'] for s in states.values())
runs={}
for label in ('native','repeat','candidate','off','unknown'):
    root=a.root/('unified-'+label+'-01')
    r=json.loads((root/'result.json').read_text());s=json.loads((root/'summary.json').read_text())
    assert s['stop']['Pid']==0 and not s['stop']['Running'] and not s['stop']['OOMKilled']
    failures=[x for x in r['comparisons'] if x.get('mismatches')]
    if label=='candidate':
        assert r['status']=='fail' and r['optimizer_steps']==2
        assert len(failures)==1 and failures[0]['name']=='step2/state/mlp.mlp.0.bias'
        assert failures[0]['mismatches']==331
        assert r['provider_calls']=={'GEMS EMBEDDING FORWARD':4,'GEMS EMBEDDING BACKWARD':4}
    else:
        assert r['status']=='pass' and r['optimizer_steps']==3 and len(r['comparisons'])==294
        assert not r['provider_calls'] and not failures
        assert all(x.get('max_abs',0)==0 for x in r['comparisons'])
    assert r['dispatcher_restored'] and all(x['dispatcher_restored'] for x in r['steps'])
    runs[label]=dict(status=r['status'],steps=r['optimizer_steps'],comparisons=len(r['comparisons']),
        target_calls=r['provider_calls'],failures=failures,
        max_abs=max(x.get('max_abs',0) for x in r['comparisons']))
tensor_manifest=json.loads((a.collection/'unified-native-01-tensor-sha.json').read_text())
assert len(tensor_manifest)==294
for name,item in tensor_manifest.items():
    f=a.audit/name;assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'],name
baseline=json.loads((a.build/'local.baseline.json').read_text());assert baseline['overall']=='PASS'
tests=0
for command in baseline['commands']:
    assert command['exit_code']==0
    if command['name'].startswith('test_'):
        match=re.search(r'Ran (\d+) tests?',(a.build/(command['name']+'.log')).read_text())
        assert match;tests+=int(match[1])
assert tests==178
summary=dict(scope='PPU unified embedding DeepFM regression',evidence_files=len(manifest),
    cpu_package_tests=tests,wheel=baseline['wheel'],runs=runs,stopped_containers=5,
    reference_tensors_verified=294,decision='Candidate denied; keep native; investigate numerical difference')
a.output.write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'}))
