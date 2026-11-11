"""Freeze PPU model case, public code and exact wheels; never overwrite."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser()
p.add_argument('--public-root',type=Path,required=True)
p.add_argument('--adapter-wheel',type=Path,required=True)
p.add_argument('--gems-wheel',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--review',type=Path)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
assets=Path(__file__).resolve().parent
for name in ('run.py','deepfm_probe.py','environment.py','unary_probe.py','resources.py','test_case.py'):
    shutil.copy2(assets/name,a.output/name)
# Share existing comparison/admission helpers; freeze exact bytes into input.
for name in ('model_probe.py','model_admission.py'):
    shutil.copy2(assets.parent.parent/'kunlun-unary/assets'/name,a.output/name)
shutil.copytree(a.public_root/'dev/lib',a.output/'lib',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(a.public_root/'dev/device-context/prototype/runtime',a.output/'public/runtime',
                ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(a.public_root/'dev/model-defs/deepfm/v1',a.output/'model',
                ignore=shutil.ignore_patterns('__pycache__'))
shutil.copy2(a.public_root/'dev/images/thead-ppu-runtime/v2/image.baseline.yaml',a.output/'image.baseline.yaml')
(a.output/'wheels').mkdir()
for wheel in (a.adapter_wheel,a.gems_wheel): shutil.copy2(wheel,a.output/'wheels'/wheel.name)
if a.review: shutil.copytree(a.review,a.output/'review')
files={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
       for f in sorted(a.output.rglob('*')) if f.is_file()}
(a.output/'inputs.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps({'files':len(files),'output':str(a.output)}))
