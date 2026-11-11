"""Freeze a metadata-only trace using the already-verified PPU runner baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def replace(text, old, new):
    if text.count(old) != 1: raise ValueError('Source changed: '+old[:80])
    return text.replace(old,new)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('model-input','deepfm-input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    case=Path(__file__).resolve().parent
    for name in ('lib','public','wheels'):
        shutil.copytree(a.model_input/name,a.output/name,ignore=shutil.ignore_patterns('__pycache__','._*'))
    for name in ('image.baseline.yaml','environment.py','model_probe.py','model_admission.py',
                 'onerec_probe.py','candidate.py','resources.py'):
        shutil.copy2(a.model_input/name,a.output/name)
    shutil.copy2(a.deepfm_input/'deepfm_probe.py',a.output/'deepfm_probe.py')
    shutil.copytree(a.deepfm_input/'model',a.output/'model',ignore=shutil.ignore_patterns('__pycache__','._*'))
    for name in ('probe.py','trace_ops.py'):shutil.copy2(case/name,a.output/name)
    source=a.model_input/'run.py';text=source.read_text()
    text=replace(text,"choices=['qualification', 'model']","choices=['inventory']")
    text=replace(text,"p.add_argument('--model', type=Path, required=True)",
                 "p.add_argument('--model', type=Path, required=True)\np.add_argument('--kind',choices=['deepfm','qwen','onerec'],required=True)")
    begin=text.index("    if a.stage == 'qualification':")
    end=text.index("    run(['docker','exec',name,'timeout'",begin)
    text=text[:begin]+"    command=['/input/probe.py','--runtime-root','/input/public','--model','/model',\n             '--out','/results/result.json','--kind',a.kind]\n"+text[end:]
    compile(text,'run.py','exec');(a.output/'run.py').write_text(text)
    (a.output/'port.json').write_text(json.dumps(dict(
        runner_source=str(source),runner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_input=str(a.model_input),deepfm_input=str(a.deepfm_input)),indent=2)+'\n')
    manifest={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen',len(manifest),'files',a.output)
