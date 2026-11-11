"""One-card, owned-container cann-shmem before/after check; no shared image edits.

Stages are explicit so the pip dependency plan can be reviewed before install.
Full logs live outside the repository. Requires public baseline tooling + PyYAML.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml

IMAGE = 'sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15'
INDEX = 'https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/'
NAME = 'cgu135-cann-shmem-1106'
PYTHON = '/usr/local/python3.11.15/bin/python3'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--stage', choices=['start', 'before', 'plan', 'install', 'after', 'stop'], required=True)
    p.add_argument('--device', type=int, default=14)
    p.add_argument('--version', help='Explicit cann-shmem candidate; omitted uses index default')
    a = p.parse_args()
    root = a.root.resolve(); public = root / 'public'; out = root / 'results'
    out.mkdir(exist_ok=True)
    env = dict(os.environ, FLAGRT_WORKSPACE_ROOT=str(public))

    def run(cmd, label, timeout=420, required=True):
        log = out / (label + '.log')
        with log.open('x') as f:
            start = time.time()
            try:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                rc = 124
            record = {'command': cmd, 'exit_code': rc, 'seconds': time.time()-start}
        (out / (label + '.command.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(f'{label}: exit={rc}', flush=True)
        if required and rc:
            raise RuntimeError(f'{label} failed: see {log}')
        return rc

    def inside(args, label, timeout=420, required=True):
        return run(['docker', 'exec', NAME, 'timeout', '--kill-after=10s', str(timeout-20), PYTHON] + args,
                   label, timeout, required)

    if a.stage == 'start':
        assert a.device >= 0
        if subprocess.run(['docker','inspect',NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('Refusing existing container')
        baseline = public / 'dev/images/ascend-train-comm/v3/image.baseline.yaml'
        data = yaml.safe_load(baseline.read_text()); assert data['image']['id'] == IMAGE
        shutil.copyfile(baseline, root/'image.baseline.yaml')
        demo = root/'demo'; demo.mkdir()
        service = dict(image=data['image']['ref'], container_name=NAME, labels={'owner':'cgu135'},
            network_mode='host', shm_size='2g', mem_limit='12g', cpus=4, restart='no', command=['sleep','infinity'],
            devices=[f'/dev/davinci{a.device}:/dev/davinci{a.device}', '/dev/davinci_manager:/dev/davinci_manager',
                     '/dev/devmm_svm:/dev/devmm_svm','/dev/hisi_hdc:/dev/hisi_hdc'],
            volumes=[str(root/'input')+':/input:ro',str(out)+':/results',
                     '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'],
            environment={'ASCEND_RT_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1'})
        (demo/'docker-compose.yml').write_text(yaml.safe_dump({'name':NAME,'x-flagrt':{
            'image_baseline':'../image.baseline.yaml','mounted_repos':{}},'services':{'runtime-dev':service}},sort_keys=False))
        run([sys.executable,str(public/'dev/lib/baseline.py'),'check','--image',str(root)],'image-check')
        run([sys.executable,str(public/'dev/lib/baseline.py'),'export',str(demo)],'baseline-export')
        run(['bash',str(public/'dev/lib/up.sh'),str(demo),'--strict'],'strict-start')
        info=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
        assert info['Image'] == IMAGE and info['Config']['Labels']['owner']=='cgu135'
        (out/'container.json').write_text(json.dumps({k:info[k] for k in ('Image','Mounts','State')},indent=2))
        (out/'device.json').write_text(json.dumps(info['HostConfig']['Devices'],indent=2))
    elif a.stage == 'stop':
        info=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
        assert info['Config']['Labels']['owner']=='cgu135' and info['Image']==IMAGE
        run(['docker','stop','--time','10',NAME],'stop')
        (out/'stopped-state.json').write_text(subprocess.check_output(['docker','inspect','--format','{{json .State}}',NAME],text=True))
    else:
        info=json.loads(subprocess.check_output(['docker','inspect',NAME]))[0]
        assert info['Image']==IMAGE and info['State']['Running'] and info['Config']['Labels']['owner']=='cgu135'
        if a.stage in ('before','after'):
            inside(['-m','pip','list','--format=json'],a.stage+'-packages')
            inside(['-c','import importlib.util; s=importlib.util.find_spec("shmem"); print(s)'],a.stage+'-shmem-spec')
            rc=inside(['/input/minirepro_shmem.py'],a.stage+'-minirepro',required=False)
            if a.stage == 'after' and rc == 0:
                inside(['/input/silu_check.py'],'after-silu',required=False)
        elif a.stage == 'plan':
            label='pip-plan-'+(a.version or 'latest')
            spec='cann-shmem'+('=='+a.version if a.version else '')
            inside(['-m','pip','install','--no-cache-dir','--dry-run','--report','/results/'+label+'.json',
                    spec,'-i',INDEX],label,timeout=600)
        elif a.stage == 'install':
            plan=json.loads((out/('pip-plan-'+(a.version or 'latest')+'.json')).read_text())
            items=plan['install']
            # Only the suggested package; refuse implicit replacement of the locked stack.
            assert len(items)==1 and items[0]['metadata']['name'].lower().replace('_','-')=='cann-shmem', 'Review unexpected dependencies'
            item=items[0]; version=item['metadata']['version']; download=item['download_info']
            assert download['url'].startswith('https://ascend.devcloud.huaweicloud.com/')
            expected=download['archive_info']['hashes']['sha256']
            inside(['-m','pip','download','--no-deps','--dest','/results/wheels',f'cann-shmem=={version}','-i',INDEX],'pip-download',timeout=600)
            wheels=list((out/'wheels').glob('*.whl')); assert len(wheels)==1
            assert hashlib.sha256(wheels[0].read_bytes()).hexdigest()==expected
            inside(['-m','pip','install','--no-index','--no-deps','--report','/results/pip-install.json',
                    '/results/wheels/'+wheels[0].name],'pip-install',timeout=600)
        run(['docker','diff',NAME],a.stage+'-container-diff')


if __name__=='__main__':
    main()
