"""Offline verification of Qwen qualification, review and three-way evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from qwen_review import validate_operator,validate_model
from flagrt_adapter.config import load_config
from flagrt_adapter.records import environment_from_record
from flagrt_adapter.startup import prepare_policies
from flagrt_adapter.workload import Workload

p=argparse.ArgumentParser()
for n in ('collection','input','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args()
manifest=json.loads((a.collection/'evidence-sha.json').read_text())
archive=a.collection/'evidence.tar.gz'
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert sha==(a.collection/'archive.sha256').read_text().split()[0]
blobs={}
with tarfile.open(archive) as t:
    assert set(t.getnames())==set(manifest)
    for m in t:
        assert m.isfile()
        b=t.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==manifest[m.name]
        blobs[m.name]=b
def result(name):return json.loads(blobs['qemb-'+name+'/result.json'])
qualification=result('qual-01');validate_operator(qualification)
native,repeat,candidates=result('native-01'),result('repeat-01'),[result('candidate-01'),result('candidate-02')]
validate_model(native,repeat,candidates)
startup_checks={}
for mode,expected in [('off','experimental_disabled'),('on',None),('unknown','workload_unverified')]:
    config,records=load_config(a.input/'review'/f'{mode}.json')
    _,report=prepare_policies(config,records,environment_from_record(native['environment']),
                             workload=Workload(**native['workload_identity']))
    assert len(report['checks'])==1 and report['checks'][0]['reason']==expected
    startup_checks[mode]=expected
namespace={};exec(compile((a.input/'embedding_admission.py').read_text(),'frozen admission','exec'),namespace)
rows={}
states=json.loads(blobs['cleanup.host.log'])
for label in ('native-01','repeat-01','candidate-01','candidate-02','off-01','unknown-01','on-01'):
    r=result(label);state=states['qemb-'+label]
    assert not state['Running'] and state['Pid']==0 and not state['OOMKilled']
    assert r['status']=='pass' and r['optimizer_steps']==3 and len(r['comparisons'])==1863
    for k in ('environment','workload_identity','initial_parameter_sha256','model_files','runtime_sources'):
        assert r[k]==native[k],(label,k)
    if label!='native-01':assert all(x['mismatches']==0 for x in r['comparisons'])
    for step in r['steps']:
        assert step['status']=='pass' and step['dispatcher_restored']
        namespace['check_model_routes'](step['routes'],r['mode'])
    expected={'GEMS EMBEDDING FORWARD':3,'GEMS EMBEDDING BACKWARD':3} if r['mode'] in ('candidate','on') else {}
    assert r['provider_calls']==expected
    maximum=max(x.get('max_abs',0.) for x in r['comparisons'])
    if r['mode'] in ('native_repeat','off','unknown'):assert maximum==0
    rows[label]=dict(mode=r['mode'],steps=3,comparisons=1863,max_abs=maximum,
        provider_calls=r['provider_calls'],loss=[s['loss'] for s in r['steps']],
        sampled_parameter_update=r['sampled_parameter_update'],
        result_sha256=manifest['qemb-'+label+'/result.json'])
assert len(states)==8
assert len({s['StartedAt'] for s in states.values()})==8, 'Distinct container executions required'
tensors=json.loads((a.collection/'qemb-native-01-tensor-sha.json').read_text())
assert len(tensors)==1863
out=dict(status='verified',scope='PPU FP32 Qwen3-Embedding-0.6B embedding-only synthetic causal-LM 3 steps',
    environment=native['environment'],adapter_package_identity=native['adapter_package_identity'],
    qualification=dict(cases=12,forward_max_abs=max(c['forward_max_abs'] for c in qualification['cases']),
                       backward_max_abs=max(c['backward_max_abs'] for c in qualification['cases'])),
    runs=rows,verified_light_files=len(manifest),archive_sha256=sha,
    cpu_startup_replay=startup_checks,
    native_tensors=dict(count=len(tensors),bytes=sum(x['bytes'] for x in tensors.values()),
        storage='remote originals; local SHA manifest, not downloaded'),containers_stopped=len(states))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
