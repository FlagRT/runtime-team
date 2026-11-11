"""Freeze existing model regressions for PPU; checked textual port, no math changes.

Every source input and generated file is hashed. Port only environment, model
mount, resource bounds and Qwen's explicit candidate-qualification entry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError('Source changed; review port: '+old[:100])
    return text.replace(old,new)

def prepare(a):
    case=Path(__file__).resolve().parent
    dfm=case.parent/'ppu-deepfm/assets'
    xpu=case.parent/'kunlun-unary/assets'
    out=a.output
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    def read(path):
        sources[str(path.relative_to(case.parent))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path.read_text()
    def write(name,text):
        compile(text,name,'exec')
        (out/name).write_text(text)
    shape=(1,32,3072) if a.model_kind=='qwen' else (1,16,12288)
    dtype='torch.float32' if a.model_kind=='qwen' else 'torch.bfloat16'
    env=read(dfm/'environment.py')
    env=replace(env,"operator='sigmoid'","operator='silu'")
    env=replace(env,"('GEMS SIGMOID FORWARD','GEMS SIGMOID BACKWARD')","('GEMS SILU FORWARD','GEMS SILU BACKWARD')")
    write('environment.py',env)
    write('probe.py','from environment import inspect_environment as run, Hits\n')
    write('resources.py',read(dfm/'resources.py'))
    write('model_admission.py',read(xpu/'model_admission.py'))
    qwen=read(xpu/'model_probe.py')
    qwen=replace(qwen,'P800','PPU',count=2)
    qwen=replace(qwen,'runtime/backends/kunlun/backend.py','runtime/backends/ppu/backend.py')
    # Candidate experiment has no model approval; normal on still uses schema 3.
    qwen=replace(qwen,"actual,policies=prepare_model(a,r,'silu',[1,32,3072])","from candidate import prepare_candidate\n    actual,policies=prepare_candidate(a,r,'silu',[1,32,3072])")
    qwen=replace(qwen,"if a.mode=='on'","if a.mode in ('on','candidate')",count=3)
    qwen=replace(qwen,"p.add_argument('--adapter-config',type=Path)","p.add_argument('--adapter-config',type=Path); p.add_argument('--records',type=Path)")
    qwen=replace(qwen,"'unknown','strict'),required=True)","'unknown','strict','candidate'),required=True)")
    qwen=replace(qwen,"if (a.mode not in ('native','native_repeat'))!=bool(a.adapter_config): p.error('Model adapter modes require schema-3 configuration')", "if a.mode=='candidate':\n        if a.adapter_config or not a.records: p.error('Candidate requires operator records only')\n    elif a.records or (a.mode not in ('native','native_repeat'))!=bool(a.adapter_config):\n        p.error('Normal modes require schema-3 configuration')")
    write('model_probe.py',qwen)
    one=read(xpu/'onerec_probe.py')
    one=replace(one,'runtime/backends/kunlun/backend.py','runtime/backends/ppu/backend.py')
    write('onerec_probe.py',one)
    write('candidate.py',read(case/'candidate.py'))
    unary=read(dfm/'unary_probe.py')
    unary=replace(unary,'shape = (32,)',f'shape = {shape!r}')
    unary=replace(unary,"[('sigmoid', torch.sigmoid)]","[('silu', F.silu)]")
    tol='(torch.float32, 1e-5, 1e-4)' if a.model_kind=='qwen' else '(torch.bfloat16, 2e-2, 2e-2)'
    unary=replace(unary,'(torch.float32, 1e-5, 1e-4)',tol)
    size=shape[0]*shape[1]*shape[2]
    unary=replace(unary,'torch.linspace(-5, 5, 32)',f'torch.linspace(-5, 5, {size})')
    unary=replace(unary,'torch.linspace(-1, 1, 32)',f'torch.linspace(-1, 1, {size})')
    write('unary_probe.py',unary)
    runner=read(dfm/'run.py')
    runner=replace(runner,"p.add_argument('--reference', type=Path)","p.add_argument('--reference', type=Path)\np.add_argument('--model', type=Path, required=True)")
    runner=replace(runner,"'mem_limit': '8g'","'mem_limit': '64g'")
    runner=replace(runner,"[f'{inp}:/input:ro', f'{out}:/results']","[f'{inp}:/input:ro', f'{out}:/results', f'{a.model.resolve()}:/model:ro']")
    script='model_probe.py' if a.model_kind=='qwen' else 'onerec_probe.py'
    runner=replace(runner,"command=['/input/deepfm_probe.py','--runtime-root','/input/public','--model','/input/model',\n                 '--out','/results/result.json','--batch-size','32','--steps','3','--mode',a.mode]", f"command=['/input/{script}','--runtime-root','/input/public','--model','/model',\n                 '--out','/results/result.json','--steps','3','--mode',a.mode]\n        if a.mode == 'candidate': command += ['--records','/input/review/operator-records.json']")
    runner=replace(runner,"'480'","'1800'")
    runner=replace(runner,"'probe.log', 510","'probe.log', 1830")
    runner=replace(runner,"manifest = json.loads", "import shutil\n    assert a.model.is_dir(), 'Missing model directory'\n    assert shutil.disk_usage(root).free > 150 * 1024**3, 'Need 150 GiB free for model evidence'\n    manifest = json.loads")
    write('run.py',runner)
    shutil.copytree(a.public_root/'dev/lib',out/'lib',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(a.public_root/'dev/device-context/prototype/runtime',out/'public/runtime',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(a.public_root/'dev/images/thead-ppu-runtime/v2/image.baseline.yaml',out/'image.baseline.yaml')
    (out/'wheels').mkdir()
    for w in (a.adapter_wheel,a.gems_wheel): shutil.copy2(w,out/'wheels'/w.name)
    if a.review: shutil.copytree(a.review,out/'review')
    (out/'port.json').write_text(json.dumps(dict(model=a.model_kind,shape=shape,dtype=dtype,source_sha256=sources),indent=2)+'\n')
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),files=len(manifest))))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('public-root','adapter-wheel','gems-wheel','output'): p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--model-kind',choices=('qwen','onerec'),required=True)
    p.add_argument('--review',type=Path)
    prepare(p.parse_args())
