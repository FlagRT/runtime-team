"""One-card v3 native startup controls, using public baseline and installed wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import traceback
import yaml

IMAGE='sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15'


def manifest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*'))
            if p.is_file() and not any(x in ('.git','__pycache__') for x in p.parts) and p.suffix!='.pyc'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('public-root','source','runtime','wheel','output'): p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--device',type=int,required=True); a=p.parse_args()
    if a.output.exists() or a.device<0: p.error('Fresh output and coordinated nonnegative device required')
    for field in ('public_root','source','runtime','wheel','output'): setattr(a,field,getattr(a,field).resolve())
    image_file=a.public_root/'dev/images/ascend-train-comm/v3/image.baseline.yaml'
    image=yaml.safe_load(image_file.read_text()); assert image['image']['id']==IMAGE
    a.output.mkdir(parents=True); demo=a.output/'demo'; demo.mkdir()
    shutil.copyfile(image_file,a.output/'image.baseline.yaml')
    name='cgu135-startup-'+a.output.name
    service=dict(image=image['image']['ref'],container_name=name,labels={'owner':'cgu135'},
        network_mode='none',mem_limit='12g',shm_size='2g',restart='no',command=['sleep','infinity'],
        devices=[f'/dev/davinci{a.device}:/dev/davinci{a.device}','/dev/davinci_manager:/dev/davinci_manager',
                 '/dev/devmm_svm:/dev/devmm_svm','/dev/hisi_hdc:/dev/hisi_hdc'],
        volumes=[str(a.source)+':/code:ro',str(a.runtime)+':/runtime:ro',str(a.wheel)+':/package/'+a.wheel.name+':ro',
                 str(a.output)+':/results','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'],
        environment={'PYTHONDONTWRITEBYTECODE':'1','ASCEND_RT_VISIBLE_DEVICES':'0','FLAGRT_VERIFIED_IMAGE_ID':IMAGE})
    (demo/'docker-compose.yml').write_text(yaml.safe_dump({'name':name,'x-flagrt':{'image_baseline':'../image.baseline.yaml'},'services':{'runtime-dev':service}},sort_keys=False))
    hashes={'source':manifest(a.source),'runtime':manifest(a.runtime)}
    wheel_hash=hashlib.sha256(a.wheel.read_bytes()).hexdigest()
    (a.output/'snapshot-manifest.json').write_text(json.dumps(hashes,indent=2))
    report={'status':'ERROR','image':IMAGE,'device':a.device,'wheel_sha256':wheel_hash,'container':name,'commands':[]}
    env=dict(os.environ,FLAGRT_WORKSPACE_ROOT=str(a.public_root))
    def execute(cmd,label):
        report['commands'].append({'label':label,'command':cmd})
        with (a.output/(label+'.log')).open('x') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=420)
        if result.returncode: raise RuntimeError(f'{label}: exit {result.returncode}')
    started=False
    try:
        if subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise RuntimeError('Container exists')
        execute([sys.executable,str(a.public_root/'dev/lib/baseline.py'),'check','--image',str(a.output)],'image-check')
        execute([sys.executable,str(a.public_root/'dev/lib/baseline.py'),'export',str(demo)],'baseline-export')
        started=True; execute(['bash',str(a.public_root/'dev/lib/up.sh'),str(demo),'--strict'],'strict-start')
        info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        assert info['Image']==IMAGE
        (a.output/'container.json').write_text(json.dumps({'Image':info['Image'],'Devices':info['HostConfig']['Devices'],'Mounts':info['Mounts']},indent=2))
        execute(['docker','exec',name,'python','-m','pip','install','--no-index','--no-deps','--target','/results/installed','/package/'+a.wheel.name],'install')
        execute(['docker','exec','-e','PYTHONPATH=/results/installed',name,'python','-c',
                 'import flagrt_adapter; from pathlib import Path; assert Path("/results/installed") in Path(flagrt_adapter.__file__).resolve().parents; print(flagrt_adapter.__file__,flagrt_adapter.__version__)'],'installed-import')
        execute(['docker','exec','-e','PYTHONPATH=/results/installed',name,'timeout','--kill-after=10s','300',
                 'python','-B','/code/probes/test_startup_npu.py','--runtime-root','/runtime','--out','/results/result.json'],'test')
        assert json.loads((a.output/'result.json').read_text())['status']=='pass'
        assert hashes=={'source':manifest(a.source),'runtime':manifest(a.runtime)}
        assert hashlib.sha256(a.wheel.read_bytes()).hexdigest()==wheel_hash
        report['status']='PASS'
    except Exception: report['error']=traceback.format_exc(); print(report['error'])
    finally:
        if started:
            r=subprocess.run(['docker','stop','--time','10',name],capture_output=True,text=True)
            report['stop_returncode']=r.returncode
            report['container_state']=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True).stdout.strip()
            if r.returncode: report['status']='ERROR'
        (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print('overall: '+report['status']); return 0 if report['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
