"""Checked port of frozen Qwen training to embedding-only schema-4 admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace(text,old,new,count=1):
    if text.count(old)!=count: raise ValueError('Source anchor changed: '+old[:100])
    return text.replace(old,new)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','wheel','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();here=Path(__file__).parent
    if not a.wheel.is_file(): raise ValueError('Missing wheel')
    expected=json.loads((a.source/'inputs.json').read_text())
    actual={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.source.rglob('*') if f.is_file() and f.name!='inputs.json'}
    if actual!=expected:raise ValueError('Source manifest mismatch')
    shutil.copytree(a.source,a.output)
    wheels=list((a.output/'wheels').glob('flagrt_framework_adapter-*.whl'))
    if len(wheels)!=1:raise ValueError('Expected one adapter wheel')
    wheels[0].unlink();shutil.copy2(a.wheel,a.output/'wheels'/a.wheel.name)
    # Old SiLU approvals have no role in this experiment. Only delete copies in
    # this newly created derivative, never the original review/evidence bundle.
    for f in (a.output/'review').glob('*'):
        if not f.is_file():raise ValueError('Unexpected nested review')
        f.unlink()
    admission=(here/'admission.py').read_text()
    admission=replace(admission,"weight_shape=[3850,width],indices_shape=[32,5]","weight_shape=[151669,width],indices_shape=[1,32]")
    admission=replace(admission,'for width in (16,1)','for width in (1024,)')
    admission=admission.replace("'completed'):2","'completed'):1").replace("reason):2","reason):1")
    admission=replace(admission,"expected.update({('relu','native','forward','operator_disabled'):3,\n                         ('sigmoid','native','forward','operator_disabled'):1})",
        "expected.update({('silu','native','forward','operator_disabled'):28})")
    (a.output/'embedding_admission.py').write_text(admission)
    namespace={};exec(compile(admission,'admission','exec'),namespace)
    for mode in ('off','on','unknown'):
        (a.output/'review'/f'{mode}.json').write_text(json.dumps(namespace['configuration'](mode),indent=2)+'\n')
    text=(a.output/'model_probe.py').read_text()
    text=replace(text,'from flagrt_adapter.bridge import ElementwiseAdapter','from flagrt_adapter.adapter import FrameworkAdapter')
    text=replace(text,"from candidate import prepare_candidate\n    actual,policies=prepare_candidate(a,r,'silu',[1,32,3072])",
        "from embedding_admission import prepare_model,check_model_routes\n    r['runtime_sources'].update({name:file_hash(Path(__file__).parent/name) for name in\n        ('model_probe.py','model_admission.py','environment.py','embedding_admission.py')})\n    actual,policies=prepare_model(a,r)")
    text=replace(text,'ElementwiseAdapter(policies,actual,runtime,gems)','FrameworkAdapter(policies,actual,runtime,gems)')
    text=replace(text,"('silu','silu_backward')","('embedding','embedding_backward','embedding_dense_backward')")
    text=replace(text,"{'GEMS SILU FORWARD':28,'GEMS SILU BACKWARD':28}","{'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1}")
    text=text.replace('Expected exactly 28 forward/backward calls','Expected one embedding forward/backward call')
    text=replace(text,"                if any(k[1]!='flaggems' for k in adapter.operator_counts):\n                    raise AssertionError('Strict model target unexpectedly used native')",
        "                check_model_routes(row['routes'],a.mode)")
    text=replace(text,"operator_counts[('silu','native','forward',reason)]!=28","operator_counts[('embedding','native','forward',reason)]!=1")
    text=replace(text,"            prefix=f'step{step+1}/'","            check_model_routes(row['routes'],a.mode)\n            row['gradient_copies']=adapter.gradient_copies if adapter else 0\n            prefix=f'step{step+1}/'")
    text=text.replace('schema-3 configuration','schema-4 configuration')
    text=replace(text,"    model=model.to('cuda:0').train()",
        "    embedding=model.get_input_embeddings()\n    if tuple(embedding.weight.shape)!=(151669,1024) or embedding.padding_idx is not None:\n        raise ValueError('Unexpected Qwen embedding contract')\n    r['embedding_contract']=dict(weight_shape=list(embedding.weight.shape),padding_idx=embedding.padding_idx)\n    model=model.to('cuda:0').train()")
    (a.output/'model_probe.py').write_text(text)
    env=(a.output/'environment.py').read_text()
    env=replace(env,"('GEMS SILU FORWARD','GEMS SILU BACKWARD')","('GEMS EMBEDDING FORWARD','GEMS EMBEDDING BACKWARD')")
    (a.output/'environment.py').write_text(env)
    probe=(here.parent/'embedding/probe.py').read_text()
    for old,new in [("[[3850,16],[3850,1]]","[[151669,1024]]"),("[32,5]","[1,32]"),
                    ('for width in (16,1)','for width in (1024,)'),('3850','151669'),
                    ('3849','151668'),('32,5,width','1,32,width'),('arange(160)','arange(32)'),
                    ('reshape(32,5)','reshape(1,32)'),('full((32,5)','full((1,32)')]:
        probe=probe.replace(old,new)
    probe=probe.replace('DeepFM-shape','Qwen-shape')
    probe=replace(probe,'    import torch\n','    import torch\n    torch.set_num_threads(2)\n')
    (a.output/'unary_probe.py').write_text(probe)
    runner=(a.output/'run.py').read_text().replace('/input/review/operator-records.json','/input/review/records.json')
    (a.output/'run.py').write_text(runner)
    for f in a.output.glob('*.py'):compile(f.read_text(),str(f),'exec')
    manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='inputs.json'}
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen',len(manifest),'files',a.output)


if __name__=='__main__':main()
