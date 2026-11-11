"""Derive a new immutable diagnostic input; retain prior evidence untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def main():
    p=argparse.ArgumentParser()
    for name in ('source','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--direct-samples',type=Path)
    a=p.parse_args()
    expected=json.loads((a.source/'inputs.json').read_text())
    actual={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.source.rglob('*') if f.is_file() and f.name!='inputs.json'}
    assert actual==expected,'Source snapshot changed'
    shutil.copytree(a.source,a.output)
    ref=a.output/'reference';ref.mkdir()
    shutil.copy2(a.reference/'native.json',ref/'result.json')
    shutil.copytree(a.reference/'final-audit/tensors',ref/'tensors')
    shutil.copy2(Path(__file__).with_name('diagnose.py'),a.output/'diagnose.py')
    run=a.output/'run.py';text=run.read_text()
    start=text.index("    if a.stage != 'model':")
    end=text.index("    run(['docker','exec',name,'timeout'",start)
    command="    command=['/input/diagnose.py','--out','/results/result.json']\n"
    if a.direct_samples:
        shutil.copy2(a.direct_samples,a.output/'captured.pt')
        shutil.copy2(Path(__file__).with_name('direct_replay.py'),a.output/'direct_replay.py')
        command="    command=['/input/direct_replay.py','--samples','/input/captured.pt','--out','/results/result.json']\n"
    text=text[:start]+command+text[end:]
    run.write_text(text)
    for f in a.output.glob('*.py'):compile(f.read_text(),str(f),'exec')
    manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='inputs.json'}
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=a.output.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as t:t.add(a.output,arcname=a.output.name)
    print(len(manifest),'files',hashlib.sha256(archive.read_bytes()).hexdigest(),archive)

if __name__=='__main__':main()
