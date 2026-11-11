"""Freeze embedding adapter/model routes from immutable prior DeepFM inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from admission import OPERATORS

def replace(text,old,new):
    if text.count(old)!=1: raise ValueError('Expected one source anchor: '+old[:80])
    return text.replace(old,new)

p=argparse.ArgumentParser()
for name in ('source','wheel','review','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
for name in ('lib','public','model'):
    shutil.copytree(a.source/name,a.output/name,ignore=shutil.ignore_patterns('__pycache__','._*'))
(a.output/'wheels').mkdir()
provider=list((a.source/'wheels').glob('flag_gems-*.whl'));assert len(provider)==1
for f in (provider[0],a.wheel):shutil.copy2(f,a.output/'wheels'/f.name)
shutil.copytree(a.review,a.output/'review')
for mode in ('off','on','unknown'):
    (a.output/'review'/f'{mode}.json').write_text(json.dumps(dict(schema='embedding-case-v1',mode=mode,operators=OPERATORS),indent=2)+'\n')
for name in ('image.baseline.yaml','resources.py','model_probe.py','model_admission.py','run.py'):
    shutil.copy2(a.source/name,a.output/name)
here=Path(__file__).parent
shutil.copy2(here/'routes.py',a.output/'unary_probe.py')
shutil.copy2(here/'admission.py',a.output/'embedding_admission.py')
text=(a.source/'environment.py').read_text()
text=replace(text,"('GEMS RELU FORWARD','GEMS THRESHOLD BACKWARD')","('GEMS EMBEDDING FORWARD','GEMS EMBEDDING BACKWARD')")
(a.output/'environment.py').write_text(text)
text=(a.source/'deepfm_probe.py').read_text()
text=replace(text,'from relu_routes import check_model_routes',
    'from embedding_admission import prepare_model, check_model_routes')
text=text.replace('relu','embedding')
text=replace(text,'from flagrt_adapter.bridge import ElementwiseAdapter',
    'from flagrt_adapter.embedding import EmbeddingAdapter')
start=text.index("    if a.mode == 'candidate':")
end=text.index("    r['model_loaded']=True",start)
text=text[:start]+"    actual,policies=prepare_model(a,r)\n"+text[end:]
text=replace(text,"ElementwiseAdapter(policies,actual,runtime,gems)",
    "EmbeddingAdapter(policies['embedding'],actual,runtime,gems)")
text=replace(text,"'embedding_routes.py'","'embedding_admission.py'")
text=replace(text,"('embedding','threshold_backward')","('embedding','embedding_backward','embedding_dense_backward')")
text=replace(text,"{'GEMS RELU FORWARD':3,'GEMS THRESHOLD BACKWARD':3}",
    "{'GEMS EMBEDDING FORWARD':2,'GEMS EMBEDDING BACKWARD':2}")
text=text.replace('Expected three target forward/backward per step','Expected two target forward/backward per step')
text=replace(text,"operator_counts[('embedding','native','forward',reason)]!=3",
    "operator_counts[('embedding','native','forward',reason)]!=2")
text=replace(text,"            delta=dict(hits.counts-previous)",
    "            row['gradient_copies']=adapter.gradient_copies if adapter else 0\n            delta=dict(hits.counts-previous)")
(a.output/'deepfm_probe.py').write_text(text)
for f in a.output.glob('*.py'):compile(f.read_text(),str(f),'exec')
manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
          for f in sorted(a.output.rglob('*')) if f.is_file()}
(a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen',len(manifest),'files',a.output)
