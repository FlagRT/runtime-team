"""Offline evidence verification, distinguishing model admission from routing."""
import argparse
import hashlib
import json
from pathlib import Path
from admission import check_model_routes

p=argparse.ArgumentParser()
for name in ('root','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
files=json.loads(a.manifest.read_text())
for name,sha in files.items():assert hashlib.sha256((a.root/name).read_bytes()).hexdigest()==sha,name
runs={}
for directory in sorted(a.root.iterdir()):
    if not (directory/'summary.json').is_file():continue
    s=json.loads((directory/'summary.json').read_text());r=json.loads((directory/'result.json').read_text())
    assert s['stop']['Pid']==0 and not s['stop']['Running'] and not s['stop']['OOMKilled']
    assert len(s['idle_observations'])==3
    row=dict(host_status=s['status'],result_status=r['status'],stop=s['stop'],environment=r['environment'])
    if r['stage'] in ('qualification','routes'):
        assert s['status']=='PASS' and r['status']=='PASS'
        assert len(r['cases'])==(24 if r['stage']=='qualification' else 6)
        assert all(c['status']=='pass' and c['restored'] for c in r['cases'])
        for c in r['cases']:
            target=r['stage']=='qualification' or c['mode']=='on'
            assert c['hits']==({'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1} if target else {})
            if r['stage']=='routes':assert c['post_scope_native']
        row.update(cases=len(r['cases']),forward_max_abs=max(c['forward_max_abs'] for c in r['cases']),
                   backward_max_abs=max(c['backward_max_abs'] for c in r['cases']))
    else:
        assert r['dispatcher_restored']
        for step in r['steps']:check_model_routes(step['routes'],r['mode'])
        if r['status']=='fail':
            assert r['mode']=='candidate' and s['status']=='ERROR'
            assert any(c.get('mismatches',0) for c in r['comparisons'])
            row['first_failure']=next(c for c in r['comparisons'] if c.get('mismatches',0))
            row['model_admission']='rejected'
        else:
            assert r['status']=='pass' and s['status']=='PASS'
            assert r['optimizer_steps']==3 and len(r['steps'])==3 and len(r['comparisons'])==294
            assert all(c.get('mismatches',0)==0 for c in r['comparisons'])
            target=r['mode'] in ('candidate','on')
            assert r['provider_calls']==({'GEMS EMBEDDING FORWARD':6,'GEMS EMBEDDING BACKWARD':6} if target else {})
            if not target:assert all(c.get('max_abs',0)==0 for c in r['comparisons'])
        row.update(mode=r['mode'],optimizer_steps=r['optimizer_steps'],comparisons=len(r['comparisons']),
            max_abs=max(c.get('max_abs',0) for c in r['comparisons']),provider_calls=r['provider_calls'],
            gradient_copies=sum(c.get('gradient_copies',0) for c in r['steps']))
    runs[directory.name]=row
expected={'embedding-qual-01','embedding-routes-01','embedding-routes-02',
          'embedding-native-01','embedding-repeat-01','embedding-candidate-01','embedding-off-01','embedding-unknown-01'}
if runs['embedding-candidate-01']['result_status']=='pass':expected.add('embedding-on-01')
assert set(runs)==expected,(set(runs),expected)
result=dict(status='PASS',scope='evidence consistency; model pass/fail remains explicit',runs=runs,verified_files=len(files))
a.output.write_text(json.dumps(result,indent=2)+'\n')
print('Verified',len(runs),'runs;',len(files),'files')
