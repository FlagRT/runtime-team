"""Freeze a ReLU experiment from the reviewed PPU DeepFM case, never edit inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from prepare import replace


def prepare(source, wheel, output, review=None):
    output.mkdir(parents=True, exist_ok=False)
    for name in ('lib', 'public', 'model'):
        shutil.copytree(source/name, output/name,
                        ignore=shutil.ignore_patterns('__pycache__', '._*'))
    (output/'wheels').mkdir()
    provider=list((source/'wheels').glob('flag_gems-*.whl'))
    assert len(provider)==1
    for f in (provider[0],wheel): shutil.copy2(f,output/'wheels'/f.name)
    for name in ('image.baseline.yaml','resources.py','model_probe.py','model_admission.py'):
        shutil.copy2(source/name,output/name)
    shutil.copy2(Path(__file__).parent/'relu_routes.py',output/'relu_routes.py')
    if review: shutil.copytree(review,output/'review')
    text=(source/'environment.py').read_text()
    text=replace(text,"('GEMS SIGMOID FORWARD','GEMS SIGMOID BACKWARD')",
                 "('GEMS RELU FORWARD','GEMS THRESHOLD BACKWARD')")
    (output/'environment.py').write_text(text)
    text=(source/'unary_probe.py').read_text()
    text=replace(text,"(op, op + '_backward')","('relu', 'threshold_backward')")
    text=replace(text,"for op in ('silu', 'sigmoid'):","for op in ('relu', 'threshold'):")
    text=replace(text,"shape = (32,)","shape = (32, 400)")
    text=replace(text,"[('sigmoid', torch.sigmoid)]","[('relu', F.relu)]")
    text=replace(text,"[(torch.float32, 1e-5, 1e-4)]","[(torch.float32, 0, 0)]")
    text=replace(text,"torch.linspace(-5, 5, 32)","torch.linspace(-5, 5, 12800)")
    text=replace(text,"            grad = torch.linspace(-1, 1, 32)",
                 "            raw.flatten()[6400:6402] = torch.tensor([0.0, -0.0])\n            grad = torch.linspace(-1, 1, 12800)")
    text=replace(text,"include=[op, op + '_backward']","include=['relu', 'threshold_backward']")
    text=replace(text,"saved = x if op == 'silu' else y","saved = x")
    text=replace(text,"getattr(torch.ops.aten, op + '_backward').default(dy, saved)",
                 "torch.ops.aten.threshold_backward.default(dy, saved, 0)")
    text=replace(text,"f'GEMS {op.upper()} BACKWARD': 1","'GEMS THRESHOLD BACKWARD': 1")
    (output/'unary_probe.py').write_text(text)
    text=(source/'deepfm_probe.py').read_text()
    text=text.replace('sigmoid','relu')
    text=replace(text,"from model_admission import prepare_model, expected_native_reason, record_expected_block",
                 "from model_admission import prepare_model, expected_native_reason, record_expected_block\nfrom relu_routes import check_model_routes")
    text=replace(text,"'deepfm_probe.py','model_admission.py','model_probe.py','environment.py'",
                 "'deepfm_probe.py','model_admission.py','model_probe.py','environment.py','relu_routes.py'")
    text=replace(text,"shapes=[[a.batch_size]]","shapes=[[a.batch_size,400]]")
    text=replace(text,"prepare_model(a,r,'relu',[a.batch_size])","prepare_model(a,r,'relu',[a.batch_size,400])")
    text=replace(text,"('relu','relu_backward')","('relu','threshold_backward')")
    text=replace(text,"# Independent numerical qualification. No model approval is manufactured.",
        "# Independent model qualification through the adapter; only operator approval exists.")
    text=replace(text,"actual,policies=environment_from_record(r['environment']),None", """actual=environment_from_record(r['environment'])
        from flagrt_adapter.policy import OperatorPolicy
        from flagrt_adapter.records import load_records
        policies={'relu':OperatorPolicy(actual,((a.batch_size,400),),('torch.float32',),
            experimental=True,capabilities=load_records('/input/review/records.json'),operator='relu')}""")
    text=replace(text,"a.mode in ('native','native_repeat','candidate')", "a.mode in ('native','native_repeat')")
    text=replace(text,"gems.use_gems(include=['relu','relu_backward']) if a.mode=='candidate' else (adapter if adapter is not None else nullcontext())",
                 "adapter if adapter is not None else nullcontext()")
    text=replace(text,"{'GEMS SIGMOID FORWARD':1,'GEMS SIGMOID BACKWARD':1}",
                 "{'GEMS RELU FORWARD':3,'GEMS THRESHOLD BACKWARD':3}")
    text=text.replace('Expected one target forward/backward per step','Expected three target forward/backward per step')
    text=replace(text,"operator_counts[('relu','native','forward',reason)]!=1",
                 "operator_counts[('relu','native','forward',reason)]!=3")
    text=replace(text,"                if adapter is not None and any(k[1]!='flaggems' for k in adapter.operator_counts):\n                    raise AssertionError('Strict target unexpectedly used native')", "                check_model_routes(row['routes'],a.mode)")
    text=replace(text,"            prefix=f'step{step+1}/'", "            check_model_routes(row['routes'],a.mode)\n            prefix=f'step{step+1}/'")
    (output/'deepfm_probe.py').write_text(text)
    text=(source/'run.py').read_text()
    text=replace(text,"choices=['qualification', 'model']","choices=['qualification', 'routes', 'model']")
    text=replace(text,"if a.stage == 'qualification':\n        command=['/input/unary_probe.py','--stage','qualification']",
                 "if a.stage != 'model':\n        command=['/input/unary_probe.py','--stage',a.stage]")
    (output/'run.py').write_text(text)
    for f in output.glob('*.py'): compile(f.read_text(),str(f),'exec')
    manifest={str(f.relative_to(output)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(output.rglob('*')) if f.is_file()}
    (output/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Frozen',len(manifest),'files',output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source','wheel','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--review',type=Path)
    a=p.parse_args();prepare(a.source,a.wheel,a.output,a.review)
