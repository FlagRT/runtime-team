"""Freeze qualification from the existing PPU baseline; refuse existing output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
for name in ('lib','public','model','wheels'):
    shutil.copytree(a.source/name,a.output/name,ignore=shutil.ignore_patterns('__pycache__','._*'))
for name in ('image.baseline.yaml','resources.py','environment.py','run.py'):
    shutil.copy2(a.source/name,a.output/name)
shutil.copy2(Path(__file__).with_name('probe.py'),a.output/'unary_probe.py')
for f in a.output.glob('*.py'): compile(f.read_text(),str(f),'exec')
manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
          for f in sorted(a.output.rglob('*')) if f.is_file()}
(a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Frozen',len(manifest),'files',a.output)
