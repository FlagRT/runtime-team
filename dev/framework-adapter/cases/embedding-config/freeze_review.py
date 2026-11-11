"""New snapshot with reviewed evidence; never edit a previously executed input."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--review',type=Path)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
expected=json.loads((a.source/'inputs.json').read_text())
actual={str(f.relative_to(a.source)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in a.source.rglob('*') if f.is_file() and f.name!='inputs.json'}
assert expected==actual,'Frozen source differs'
shutil.copytree(a.source,a.output)
if a.review:
    for f in a.review.iterdir():
        assert f.is_file() and f.suffix=='.json'
        assert f.name not in ('on.json','off.json','unknown.json'),'Review cannot replace switches'
        shutil.copy2(f,a.output/'review'/f.name)
manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
          for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='inputs.json'}
(a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=a.output.with_suffix('.tar.gz')
with tarfile.open(archive,'x:gz') as t:t.add(a.output,arcname=a.output.name)
print(len(manifest),'files',hashlib.sha256(archive.read_bytes()).hexdigest(),archive)
