"""Offline checks over collected immutable evidence, no accelerator access."""
import argparse
import hashlib
import json
from pathlib import Path
from relu_routes import check_model_routes


def read(path):return json.loads(path.read_text())


def verify(root, manifest):
    for name,digest in read(manifest).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
    runs={}
    for d in sorted(root.iterdir()):
        if not (d/'summary.json').is_file():continue
        s=read(d/'summary.json');r=read(d/'result.json')
        assert s['stop']['Pid']==0 and not s['stop']['Running'] and not s['stop']['OOMKilled'],d.name
        assert len(s['idle_observations'])==3
        row=dict(status='PASS',input_sha256=s['input_sha256'],stop=s['stop'])
        if d.name=='relu-candidate-01':
            assert s['status']=='ERROR' and r['status']=='fail'
            assert 'Strict target unexpectedly used native' in r['error']
            assert r['optimizer_steps']==0 and len(r['comparisons'])==0 and r['dispatcher_restored']
            assert r['provider_calls']=={'GEMS RELU FORWARD':3,'GEMS THRESHOLD BACKWARD':3}
            check_model_routes(r['steps'][0]['routes'],'candidate')
            row.update(status='PRESERVED_HARNESS_FAILURE',error=r['error'])
            runs[d.name]=row
            continue
        assert s['status']=='PASS' and r['status'].lower()=='pass',d.name
        if d.name.startswith('inventory-'):
            assert r['optimizer_steps']==1 and r['stage']=='complete'
            assert not r['inventory']['signatures_truncated']
            row.update(model=r['model'],parameter_count=r['parameter_count'],call_totals=r['inventory']['call_totals'])
        elif r.get('stage') in ('qualification','routes'):
            assert len(r['cases'])==3 and all(x['status']=='pass' and x['restored'] for x in r['cases'])
            for x in r['cases']:
                assert x['forward_max_abs']==x['backward_max_abs']==0
                target=x['mode'] in ('direct','on')
                assert x['hits']==({'GEMS RELU FORWARD':1,'GEMS THRESHOLD BACKWARD':1} if target else {})
                if r['stage']=='routes':assert x['post_scope_native']
            row['cases']=r['cases']
        else:
            assert r['optimizer_steps']==3 and len(r['steps'])==3 and len(r['comparisons'])==294
            assert all(x['status']=='pass' and x['dispatcher_restored'] for x in r['steps'])
            assert not any(x.get('mismatches',0) for x in r['comparisons'])
            target=r['mode'] in ('candidate','on')
            assert r['provider_calls']==({'GEMS RELU FORWARD':9,'GEMS THRESHOLD BACKWARD':9} if target else {})
            for x in r['steps']:check_model_routes(x['routes'],r['mode'])
            maximum=max((x.get('max_abs',0) for x in r['comparisons']),default=0)
            row.update(mode=r['mode'],steps=3,comparisons=294,max_abs=maximum,provider_calls=r['provider_calls'],
                       environment=r['environment'],workload=r['workload_identity'],tolerance=r['tolerance'])
        runs[d.name]=row
    assert len(runs)==14, sorted(runs)
    return dict(status='PASS',runs=runs,verified_files=len(read(manifest)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    report=verify(a.root,a.manifest);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',len(report['runs']),'runs,',report['verified_files'],'files')
