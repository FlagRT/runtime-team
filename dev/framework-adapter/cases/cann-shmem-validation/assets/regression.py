"""Staged v3 overlay qualification/regression in a new owned two-card container.

Review bundles are supplied separately after inspecting qualification results.
No kernel changes, implicit approvals, retries or Git operations.
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

IMAGE='sha256:e13a15d0d02b86e9cd7b1656df3c6b52838c258f270b6b894a91c2b074da5f15'
SHMEM='308a1c5a3dfe039d77d1159e27f3b1c16f38f4990a19c1fc68c7c052dee2b3f5'
PYTHON='/usr/local/python3.11.15/bin/python3'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['init','operator','controls','native','repeat','candidate','off','on','unqualified','stop'],required=True)
    p.add_argument('--devices',default='14,15'); p.add_argument('--model',type=Path)
    a=p.parse_args(); root=a.root.resolve(); inp=root/'input'; results=root/'results'; results.mkdir(exist_ok=True)
    name='cgu135-overlay-'+root.name
    env=dict(os.environ,FLAGRT_WORKSPACE_ROOT=str(inp/'public'))
    def run(cmd,label,timeout=1200):
        with (results/(label+'.log')).open('x') as f:
            start=time.time()
            try: rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired: rc=124
        (results/(label+'.command.json')).write_text(json.dumps(dict(command=cmd,exit_code=rc,seconds=time.time()-start),indent=2)+'\n')
        print(label,rc,flush=True)
        if rc: raise RuntimeError(label+' failed; inspect its log; no automatic retry')
    def getinfo():
        info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        assert info['Image']==IMAGE and info['Config']['Labels']['owner']=='cgu135'
        return info
    if a.stage=='init':
        assert a.model and a.model.is_dir()
        devices=[int(x) for x in a.devices.split(',')]; assert len(set(devices))==len(devices)==2 and min(devices)>=0
        if subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0: raise RuntimeError('Container exists')
        for filename,digest in json.loads((inp/'manifest.json').read_text())['files'].items():
            assert hashlib.sha256((inp/filename).read_bytes()).hexdigest()==digest,filename
        wheels=list(inp.glob('*.whl')); assert len(wheels)==2
        shmem=next(f for f in wheels if f.name.startswith('cann_shmem'))
        adapter=next(f for f in wheels if f.name.startswith('flagrt_'))
        assert hashlib.sha256(shmem.read_bytes()).hexdigest()==SHMEM
        (root/'reviews').mkdir()
        baseline=inp/'public/dev/images/ascend-train-comm/v3/image.baseline.yaml'
        shutil.copyfile(baseline,root/'image.baseline.yaml'); image=yaml.safe_load(baseline.read_text()); assert image['image']['id']==IMAGE
        demo=root/'demo'; demo.mkdir()
        service=dict(image=image['image']['ref'],container_name=name,labels={'owner':'cgu135'},network_mode='host',
            ipc='host',ulimits={'memlock':-1},mem_limit='48g',cpus=8,init=True,restart='no',command=['sleep','infinity'],
            devices=[f'/dev/davinci{d}:/dev/davinci{d}' for d in devices]+['/dev/davinci_manager:/dev/davinci_manager','/dev/devmm_svm:/dev/devmm_svm','/dev/hisi_hdc:/dev/hisi_hdc'],
            volumes=[str(inp)+':/input:ro',str(results)+':/results',str(root/'reviews')+':/reviews:ro',
                     str(inp/'runtime')+':/runtime:ro',str(a.model.resolve())+':/model:ro',
                     '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro'],
            environment={'PYTHONDONTWRITEBYTECODE':'1','ASCEND_RT_VISIBLE_DEVICES':'0,1','DC_ROOT':'/runtime',
                'DC_BACKEND':'ascend','DC_DIST_BT':'hccl','DC_MODEL':'/model','MAX_STEPS':'3','BATCH':'1','SEQ':'32','LR':'1e-5',
                'HCCL_WHITELIST_DISABLE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                'FLAGRT_VERIFIED_IMAGE_ID':IMAGE,'FLAGRT_CANN_SHMEM_WHEEL':'/input/'+shmem.name})
        (demo/'docker-compose.yml').write_text(yaml.safe_dump({'name':name,'x-flagrt':{'image_baseline':'../image.baseline.yaml','mounted_repos':{}},'services':{'runtime-dev':service}},sort_keys=False))
        run([sys.executable,str(inp/'public/dev/lib/baseline.py'),'check','--image',str(root)],'image-check')
        run([sys.executable,str(inp/'public/dev/lib/baseline.py'),'export',str(demo)],'baseline-export')
        run(['bash',str(inp/'public/dev/lib/up.sh'),str(demo),'--strict'],'strict-start')
        info=getinfo(); assert info['State']['Running']
        (results/'container.json').write_text(json.dumps({k:info[k] for k in ('Image','Mounts','HostConfig')},indent=2))
        run(['docker','exec',name,PYTHON,'-m','pip','install','--no-index','--no-deps','/input/'+shmem.name],'install-overlay')
        run(['docker','exec',name,PYTHON,'-m','pip','install','--no-index','--no-deps','/input/'+adapter.name],'install-adapter')
        run(['docker','exec',name,PYTHON,'-c',
            'import os,flagrt_adapter; from flagrt_adapter.environment import measure_compiler_build; print(flagrt_adapter.__file__); print(measure_compiler_build(os.environ["FLAGRT_CANN_SHMEM_WHEEL"]))'],'measure-overlay')
    elif a.stage=='stop':
        getinfo(); run(['docker','stop','--time','10',name],'stop')
        (results/'stopped-state.json').write_text(json.dumps(getinfo()['State'],indent=2))
    else:
        assert getinfo()['State']['Running']
        cmd=['docker','exec']
        if a.stage in ('operator','controls'):
            cmd += [name,'timeout','--kill-after=15s','900',PYTHON,'/input/qualify_overlay.py']
            if a.stage=='controls': cmd+=['--controls']
        else:
            folder='/results/'+a.stage
            cmd += ['-e','DC_OUT_DIR='+folder+'/train','-e','TRITON_CACHE_DIR='+folder+'/cache',name,
                    'timeout','--kill-after=15s','1100',PYTHON,'-m','torch.distributed.run',
                    '--standalone','--nproc_per_node=2','--max-restarts=0']
            if a.stage in ('native','repeat','candidate'):
                cfg='candidate' if a.stage=='candidate' else 'off'
                cmd+=['/input/qualify_public_training.py','--config','/reviews/'+cfg+'.json','--mode',a.stage]
                if a.stage!='native': cmd+=['--reference','/results/native/train']
            else:
                cmd+=['--module','flagrt_adapter.train_cli','--config','/reviews/'+a.stage+'.json']
        run(cmd,a.stage)
        if a.stage in ('operator','controls'):
            label='operator-controls' if a.stage=='controls' else 'operator-qualification'
            assert json.loads((results/label/'result.json').read_text())['status']=='PASS'
        elif a.stage in ('native','repeat','candidate'):
            for rank in (0,1):
                row=json.loads((results/a.stage/f'train/qualification_rank{rank}.json').read_text())
                assert row['status']=='PASS' and row['optimizer_steps']==3
        else:
            sys.path.insert(0,str(inp)); from check_training import check_run
            verdict=check_run(results/a.stage,a.stage,measured=True)
            assert verdict['status']=='pass'
            (results/(a.stage+'-check.json')).write_text(json.dumps(verdict,indent=2))


if __name__=='__main__': main()
