"""Offline verify transport, stop receipts, and numerical claims from tensors."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import torch

def load_archive(root,name):
    collection=root/(name+'-collection')
    receipt=json.loads((collection/'receipt.json').read_text())
    manifest=json.loads((collection/'files.json').read_text())
    archive=collection/'evidence.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
    blobs={}
    with tarfile.open(archive) as t:
        assert set(t.getnames())==set(manifest)
        for m in t:
            assert m.isfile()
            b=t.extractfile(m).read()
            assert len(b)==manifest[m.name]['size']
            assert hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']
            blobs[m.name]=b
    state=json.loads((collection/'container.json').read_text())
    assert state['Status']=='exited' and not state['Running'] and state['Pid']==0 and not state['OOMKilled']
    summary=json.loads(blobs['summary.json'])
    assert summary['stop']==state
    assert all(x['memory_mib']<=2 and x['processes']==0 for x in summary['idle_observations'])
    return blobs,dict(name=name,**receipt,state=state['Status'],pid=state['Pid'])

def tensor(b): return torch.load(io.BytesIO(b),map_location='cpu',weights_only=True)
def diff(a,b):return float((a.double()-b.double()).abs().max())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    diag,receipt=load_archive(a.root,'embedding-diagnosis-01')
    direct,direct_receipt=load_archive(a.root,'embedding-direct-02')
    failed,failed_receipt=load_archive(a.root,'embedding-direct-01')
    r=json.loads(diag['result.json']);d=json.loads(direct['result.json'])
    assert r['status']==d['status']=='pass' and len(r['results'])==60
    assert json.loads(failed['result.json'])['status']=='error'
    assert 'PackageNotFoundError' in failed['probe.log'].decode()
    assert len(r['native_comparisons'])==294 and all(x['max_abs']==0 for x in r['native_comparisons'])
    native=json.loads(diag['native/result.json'])
    for item in native['comparisons']:
        old=torch.load(a.reference/'final-audit/tensors'/item['file'],map_location='cpu',weights_only=True)
        assert torch.equal(old,tensor(diag['native/tensors/'+item['file']]))
    assert hashlib.sha256(diag['captured.pt']).hexdigest()==r['capture_sha256']==d['samples_sha256']
    for row in r['results']:
        assert hashlib.sha256(diag[row['file']]).hexdigest()==row['sha256']
        assert row['forward_vs_native']['max_abs']==0
    pairwise=[]
    for sample in r['pairwise']:
        for pair in sample['comparisons']:
            key=lambda t:f"tensors/{sample['sample']}-{t[0]}-{t[1]}.pt"
            x=tensor(diag[key(pair['a'])])['backward'];y=tensor(diag[key(pair['b'])])['backward']
            assert diff(x,y)==pair['max_abs']
            assert int((x!=y).sum())==pair['unequal']
        pairwise.append(dict(sample=sample['sample'],max_abs=max(p['max_abs'] for p in sample['comparisons']),
            by_path={path:max(p['max_abs'] for p in sample['comparisons'] if p['a'][1]==p['b'][1]==path)
                     for path in ('direct','standalone','unified')}))
    assert len(d['results'])==16
    for row in d['results']:
        values=tensor(direct[f"direct-{row['sample']}-{row['control']}.pt"])[row['path']]
        differences=[diff(x,y) for i,x in enumerate(values) for y in values[i+1:]]
        assert max(differences)==row['repeat_max_abs']
        assert sum(x>0 for x in differences)==row['nonidentical_pairs']
    out=dict(status='verified',scope='PPU DeepFM embedding numerical localization; not model approval',
        environment=r['environment'],adapter_package_identity=r['adapter_package_identity'],
        native_exact_comparisons=294,three_path_trials=60,forward_exact_trials=60,
        provider_arguments_exact_trials=60,repeatability=pairwise,direct_control_results=d['results'],
        provider=d['provider'],archives=[receipt,failed_receipt,direct_receipt],
        raw_result_sha256={'diagnosis':hashlib.sha256(diag['result.json']).hexdigest(),
                           'direct':hashlib.sha256(direct['result.json']).hexdigest()})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print('Verified',sum(x['files'] for x in out['archives']),'files; 294 exact native comparisons; 60 trials')
    print(json.dumps(d['results'],indent=2))

if __name__=='__main__':main()
