"""Explicit Qwen-only decisions after operator/model evidence inspection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def validate_operator(d):
    assert d['status']=='PASS' and len(d['cases'])==12
    assert {(r['width'],r['pattern'],r['repeat']) for r in d['cases']}=={
        (1024,s,n) for s in ('unique','repeated','all_same','boundary') for n in range(3)}
    assert all(r['status']=='pass' and r['restored'] and r['hits']=={
        'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1} for r in d['cases'])
    assert d['contract']==dict(operator='embedding',weight_shapes=[[151669,1024]],
        weight_dtype='torch.float32',indices_shape=[1,32],indices_dtype='torch.int64',
        padding_idx=-1,scale_grad_by_freq=False,sparse=False,layout='contiguous',
        max_norm=None,atol=1e-5,rtol=1e-4)

def validate_model(native,repeat,candidates):
    assert native['status']==repeat['status']=='pass'
    assert native['mode']=='native' and repeat['mode']=='native_repeat'
    expected=3*(1+2*native['parameter_tensors'])
    assert native['parameter_tensors']==310 and expected==1863
    assert len(native['comparisons'])==len(repeat['comparisons'])==expected
    assert len(candidates)==2,'Two completed candidates required before approval'
    for run in [repeat,*candidates]:
        for key in ('environment','workload_identity','initial_parameter_sha256','model_files','runtime_sources'):
            assert native[key]==run[key],key
        assert run['status']=='pass' and run['optimizer_steps']==3
        assert len(run['comparisons'])==expected and len(run['steps'])==3
        assert [x['name'] for x in native['comparisons']]==[x['name'] for x in run['comparisons']]
        assert all(x['status']=='pass' and x['dispatcher_restored'] for x in run['steps'])
        assert all(x['mismatches']==0 for x in run['comparisons'])
    assert all(x['max_abs']==0 for x in repeat['comparisons'])
    for run in candidates:
        assert run['mode']=='candidate' and run['qualification_only']
        assert run['provider_calls']=={'GEMS EMBEDDING FORWARD':3,'GEMS EMBEDDING BACKWARD':3}
        assert run['embedding_contract']==dict(weight_shape=[151669,1024],padding_idx=None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=('operator','model'),required=True)
    p.add_argument('--qualification',type=Path)
    p.add_argument('--native',type=Path);p.add_argument('--repeat',type=Path)
    p.add_argument('--candidate',type=Path,action='append')
    p.add_argument('--operator-review',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='operator':
        d=json.loads(a.qualification.read_text());validate_operator(d)
        a.output.mkdir(parents=True,exist_ok=False)
        shutil.copy2(a.qualification,a.output/'qualification.json')
        record=dict(environment=d['environment'],signature=dict(weight_shape=[151669,1024],
            indices_shape=[1,32],weight_dtype='torch.float32',indices_dtype='torch.int64'),
            status='passed',stages=['forward','backward'],evidence='qualification.json',
            evidence_sha256=hashlib.sha256(a.qualification.read_bytes()).hexdigest(),revision=1)
        data=dict(schema=1,kind='embedding-dense-no-padding-v1',records=[record])
        (a.output/'records.json').write_text(json.dumps(data,indent=2)+'\n')
    else:
        assert a.candidate and len(a.candidate)==2 and len({p.resolve() for p in a.candidate})==2, 'Need two distinct candidate report paths'
        native,repeat=(json.loads(x.read_text()) for x in (a.native,a.repeat))
        candidates=[json.loads(x.read_text()) for x in a.candidate]
        validate_model(native,repeat,candidates)
        shutil.copytree(a.operator_review,a.output)
        for name,path in [('native.json',a.native),('repeat.json',a.repeat),
                          *[(f'candidate-{i+1}.json',x) for i,x in enumerate(a.candidate)]]:
            shutil.copy2(path,a.output/name)
        # Approval evidence includes both candidate reports and their identities.
        evidence=dict(native_sha256=hashlib.sha256(a.native.read_bytes()).hexdigest(),
            repeat_sha256=hashlib.sha256(a.repeat.read_bytes()).hexdigest(),
            candidates=[dict(file=f'candidate-{i+1}.json',sha256=hashlib.sha256(x.read_bytes()).hexdigest())
                        for i,x in enumerate(a.candidate)],decision='approve',checks='two full candidates and exact native repeat')
        data=json.dumps(evidence,indent=2)+'\n';(a.output/'model-decision.json').write_text(data)
        record=dict(environment=candidates[0]['environment'],workload=candidates[0]['workload_identity'],
            status='passed',revision=1,evidence='model-decision.json',evidence_sha256=hashlib.sha256(data.encode()).hexdigest())
        (a.output/'workload-reviews.json').write_text(json.dumps(dict(schema=1,reviews=[record]),indent=2)+'\n')
    print('Approved exact',a.stage,'scope',a.output)

if __name__=='__main__':main()
