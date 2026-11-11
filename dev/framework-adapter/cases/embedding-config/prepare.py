"""Freeze unified-config regression inputs without overwriting historical cases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from admission import configuration


def replace(text,old,new):
    if text.count(old)!=1: raise ValueError('Expected one source anchor: '+old[:80])
    return text.replace(old,new)


def main():
    p=argparse.ArgumentParser()
    for name in ('source','wheel','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not a.wheel.is_file() or not a.wheel.name.startswith('flagrt_framework_adapter-'):
        raise ValueError('Adapter wheel path is missing or invalid')
    # Verify the source snapshot before deriving a new one; preserve its bytes.
    expected=json.loads((a.source/'inputs.json').read_text())
    actual={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.source.rglob('*') if f.is_file() and f.name!='inputs.json'}
    if actual!=expected: raise ValueError('Frozen source manifest mismatch')
    shutil.copytree(a.source,a.output)
    wheels=list((a.output/'wheels').glob('flagrt_framework_adapter-*.whl'))
    if len(wheels)!=1: raise ValueError('Need one prior adapter wheel')
    # Only the newly created private derivative is modified.
    wheels[0].unlink(); shutil.copy2(a.wheel,a.output/'wheels'/a.wheel.name)
    for name in ('native.json','repeat.json','candidate.json','workload-reviews.json'):
        path=a.output/'review'/name
        if path.exists(): path.unlink() # stale model approvals cannot authorize changed code
    for mode in ('off','on','unknown'):
        (a.output/'review'/f'{mode}.json').write_text(json.dumps(configuration(mode),indent=2)+'\n')
    shutil.copy2(Path(__file__).with_name('admission.py'),a.output/'embedding_admission.py')
    probe=a.output/'deepfm_probe.py'
    text=replace(probe.read_text(),'from flagrt_adapter.embedding import EmbeddingAdapter',
                 'from flagrt_adapter.adapter import FrameworkAdapter')
    text=replace(text,"EmbeddingAdapter(policies['embedding'],actual,runtime,gems)",
                 'FrameworkAdapter(policies,actual,runtime,gems)')
    text=text.replace('require schema-3 configuration','require schema-4 configuration')
    probe.write_text(text)
    for f in a.output.glob('*.py'):compile(f.read_text(),str(f),'exec')
    manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='inputs.json'}
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen',len(manifest),'files',a.output)


if __name__=='__main__':main()
