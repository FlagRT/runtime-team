"""Bounded single-card case; frozen inputs, public strict baseline, always stop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback
import time
import yaml
from resources import require_idle

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--run-id', required=True)
p.add_argument('--card', type=int, required=True)
p.add_argument('--stage', choices=['qualification', 'model'], required=True)
p.add_argument('--mode', choices=['native','native_repeat','candidate','off','on','unknown'])
p.add_argument('--reference', type=Path)
a = p.parse_args()
if a.stage == 'model' and (not a.mode or (a.mode != 'native' and not a.reference)):
    p.error('Model mode and native reference required')
assert re.fullmatch(r'[a-z0-9-]+', a.run_id) and 0 <= a.card < 16
root = a.root.resolve()
inp = Path(__file__).resolve().parent
out = root / a.run_id
out.mkdir(exist_ok=False)
name = 'cgu135-ppu-' + a.run_id
image = 'sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8'
tag = 'flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64'
summary = dict(status='ERROR', image=image, physical_card=a.card, commands=[])
env = dict(os.environ, PATH=str(Path('/bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin')) + ':' + os.environ['PATH'],
           FLAGRT_WORKSPACE_ROOT=str(root))
attempted = False


def run(command, log, timeout=120):
    with (out/log).open('x') as f:
        result = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    summary['commands'].append(dict(command=command, log=log, exit_code=result.returncode))
    if result.returncode:
        raise RuntimeError(f'{log} exited {result.returncode}')


try:
    manifest = json.loads((inp/'inputs.json').read_text())
    for relative, expected in manifest.items():
        path = (inp/relative).resolve()
        assert inp in path.parents and path.is_file(), 'Invalid input manifest path'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Input hash mismatch: '+relative
    if a.reference:
        ref = a.reference.resolve()
        assert root in ref.parents, 'Reference must be in this run root'
        data = json.loads((ref/'result.json').read_text())
        assert data['status'] == 'pass' and data['mode'] == 'native'
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    summary['idle_observations']=[]
    for sample in range(3):
        smi = subprocess.check_output(['ppu-smi'], text=True, timeout=20)
        (out/f'resource-before-{sample}.log').write_text(smi)
        summary['idle_observations'].append(require_idle(smi,a.card))
        if sample < 2: time.sleep(1)
    driver = re.search(r'Driver Version:\s*(\S+)', smi)
    assert driver, 'Missing driver identity'
    assert subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'], text=True).strip() == image
    wheels = sorted((inp/'wheels').glob('*.whl'))
    assert len(wheels) == 2, 'Need exactly one adapter and one provider wheel'
    summary['input_sha256'] = {str(f.relative_to(inp)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in inp.rglob('*') if f.is_file()}
    demo = out/'demo'
    demo.mkdir()
    compose = {'name': name, 'x-flagrt': {'image_baseline': str(inp/'image.baseline.yaml'), 'mounted_repos': {}},
               'services': {'test': {'image': tag, 'container_name': name, 'network_mode': 'none',
                  'restart': 'no', 'shm_size': '2g', 'mem_limit': '8g', 'pids_limit': 512,
                  'devices': [f'/dev/{x}:/dev/{x}' for x in ['alixpu','alixpu_ctl',f'alixpu_ppu{a.card}']],
                  'volumes': [f'{inp}:/input:ro', f'{out}:/results'] + ([f'{a.reference.resolve()}:/reference:ro'] if a.reference else []),
                  'environment': {'CUDA_VISIBLE_DEVICES': '0', 'NCCL_SOCKET_IFNAME': 'bond0',
                     'GEMS_VENDOR': 'nvidia', 'PPU_DRIVER_ID': driver[1], 'CASE_IMAGE_ID': image,
                     'PYTHONPATH': '/results/packages:/input/public', 'DO_NOT_TRACK': '1',
                     'PYTHONDONTWRITEBYTECODE': '1', 'TRITON_CACHE_DIR': '/results/triton-cache',
                     'FLAGGEMS_CACHE_DIR': '/results/gems-cache'},
                  'extra_hosts': ['ai-server:127.0.0.1'], 'command': ['sleep','infinity']}}}
    (demo/'docker-compose.yml').write_text(yaml.safe_dump(compose, sort_keys=False))
    run(['/bmcp_lvm_fs/cgu135/framework-ppu-20261110/control-venv/bin/python',str(inp/'lib/baseline.py'),'export',str(demo)], 'export.log')
    attempted = True
    run(['bash',str(inp/'lib/up.sh'),str(demo),'--strict'], 'strict-start.log')
    assert subprocess.check_output(['docker','inspect',name,'--format','{{.Image}}'],text=True).strip() == image
    run(['docker','exec',name,'python3','-m','pip','install','--no-deps','--no-index',
         '--target','/results/packages', *['/input/wheels/'+w.name for w in wheels]], 'install.log')
    if a.stage == 'qualification':
        command=['/input/unary_probe.py','--stage','qualification']
    else:
        command=['/input/deepfm_probe.py','--runtime-root','/input/public','--model','/input/model',
                 '--out','/results/result.json','--batch-size','32','--steps','3','--mode',a.mode]
        if a.reference: command += ['--reference','/reference']
        if a.mode in ('off','on','unknown'):
            command += ['--adapter-config','/input/review/'+a.mode+'.json']
    run(['docker','exec',name,'timeout','--kill-after=10s','480','python3','-u',*command], 'probe.log', 510)
    assert json.loads((out/'result.json').read_text())['status'].lower() == 'pass'
    summary['status'] = 'PASS'
except Exception:
    summary['error'] = traceback.format_exc()
finally:
    try:
        if attempted and subprocess.run(['docker','inspect',name],capture_output=True,timeout=20).returncode == 0:
            subprocess.run(['docker','stop','--time','10',name],capture_output=True,timeout=40)
            summary['stop'] = json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .State}}'], text=True, timeout=20))
            if summary['stop']['Running'] or summary['stop']['Pid']:
                summary['status'] = 'ERROR'
    except Exception:
        summary['status'] = 'ERROR'
        summary['cleanup_error'] = traceback.format_exc()
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['status'] == 'PASS' else 2)
