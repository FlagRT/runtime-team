"""One-entry case; optional isolated offline wheel install, no qualification approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import yaml

IMAGE = "sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f"
GEMS = "f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4"


def manifest(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*')) if p.is_file()
        and not any(s in ('.git','__pycache__','.pytest_cache') for s in p.relative_to(root).parts)
        and p.suffix != '.pyc'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public-root','source','gems','deps','runtime','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--records',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--stage',choices=('qualification','unary','sigmoid','qwen','dc-train'),required=True)
    p.add_argument('--second-device',type=int)
    p.add_argument('--training-config',choices=('off','on','unqualified'))
    p.add_argument('--wheel',type=Path,required=True,help='Independent adapter wheel, installed offline')
    p.add_argument('--native-repeat-only',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args()
    if not a.wheel.is_file() or a.wheel.suffix!='.whl':
        p.error('--wheel requires an existing wheel')
    if a.device<0 or a.output.exists():
        p.error('Need nonnegative physical device and NEW output directory')
    for name in ('public_root','source','gems','deps','runtime'):
        value=getattr(a,name).resolve()
        if not value.is_dir(): p.error('Missing '+name)
        setattr(a,name,value)
    if a.stage!='qualification' and (not a.records or not a.records.is_file()):
        p.error('Reviewed records required; this runner never generates them')
    if a.stage in ('qwen','dc-train') and (not a.model or not a.model.is_dir()): p.error('Model required')
    if a.stage=='dc-train' and (a.second_device is None or a.second_device<0 or a.second_device==a.device or not a.training_config):
        p.error('dc-train needs a distinct second device and training config')
    git=['git','-c','safe.directory='+str(a.gems),'-C',str(a.gems)]
    if subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()!=GEMS:
        p.error('Unexpected Gems revision')
    if subprocess.check_output(git+['status','--porcelain','--untracked-files=no'],text=True).strip():
        p.error('Modified Gems tracked sources')
    a.output=a.output.resolve()
    a.output.mkdir(parents=True)
    demo=a.output/'demo'; demo.mkdir()
    case=Path(__file__).resolve().parents[1]
    shutil.copyfile(case/'image.baseline.yaml',a.output/'image.baseline.yaml')
    config=yaml.safe_load((case/'demo/docker-compose.yml').read_text())
    config['name']='cgu135-framework-'+a.output.name
    config['x-flagrt']['mounted_repos']['FlagGems']['host_path']=str(a.gems)
    if a.stage=='dc-train':
        service=config['services']['runtime-dev']
        service['devices'].insert(1,f'/dev/davinci{a.second_device}:/dev/davinci{a.second_device}')
        service['network_mode']='host'
        service['ipc']='host'
        service.pop('shm_size',None)
        service['mem_limit']='48g'
        service['ulimits']={'memlock':-1}
        service['environment'].update(ASCEND_RT_VISIBLE_DEVICES='0,1',DC_ROOT='/runtime',
            DC_BACKEND='ascend',DC_DIST_BT='hccl',DC_MODEL='/model',DC_OUT_DIR='/results/train',
            MAX_STEPS='3',BATCH='1',SEQ='32',LR='1e-5')
    if a.wheel:
        a.wheel=a.wheel.resolve()
        service=config['services']['runtime-dev']
        if a.stage=='dc-train':
            service['volumes']=[v for v in service['volumes'] if ':/code:' not in v]
        service['volumes'] += [str(a.wheel)+':/package/'+a.wheel.name+':ro',
            str(a.source/'demo/training-configs')+':/config:ro']
    (demo/'docker-compose.yml').write_text(yaml.safe_dump(config,sort_keys=False))
    empty=a.output/'empty'; empty.mkdir()
    name='cgu135-unary-'+a.output.name
    env=dict(os.environ,FLAGRT_CONTAINER=name,FLAGRT_DEVICE=str(a.device),FLAGRT_SOURCE=str(a.source),
        FLAGRT_GEMS=str(a.gems),FLAGRT_DEPS=str(a.deps),FLAGRT_RUNTIME=str(a.runtime),
        FLAGRT_OUTPUT=str(a.output),FLAGRT_RECORDS=str(a.records.resolve().parent if a.records else empty),
        FLAGRT_MODEL=str(a.model.resolve() if a.model else empty),FLAGRT_WORKSPACE_ROOT=str(a.public_root))
    identity={k:manifest(v) for k,v in [('source',a.source),('runtime',a.runtime),('deps',a.deps)]}
    (a.output/'snapshot-manifest.json').write_text(json.dumps(identity,indent=2))
    summary={'status':'ERROR','stage':a.stage,'device':a.device,'image':IMAGE,'container':name,
             'second_device':a.second_device,'training_config':a.training_config,
             'inputs':{k:str(getattr(a,k)) for k in ('source','runtime','deps','gems','records','model')}}
    if a.wheel:
        summary['wheel']={'path':str(a.wheel),'sha256':hashlib.sha256(a.wheel.read_bytes()).hexdigest()}
    def execute(cmd,log):
        with (a.output/log).open('x') as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        if result.returncode: raise RuntimeError(f'{log}: exit {result.returncode}')
    started=False
    try:
        # Refuse collision; never stop a container not created by this run.
        if subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
            raise RuntimeError('Container name already exists')
        execute([sys.executable,str(a.public_root/'dev/lib/baseline.py'),'check','--image',str(a.output)],'image-check.log')
        execute([sys.executable,str(a.public_root/'dev/lib/baseline.py'),'export',str(demo)],'baseline-export.log')
        started=True
        execute(['bash',str(a.public_root/'dev/lib/up.sh'),str(demo),'--strict'],'strict-start.log')
        actual=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        if actual['Image']!=IMAGE: raise RuntimeError('Actual container image mismatch')
        (a.output/'container.json').write_text(json.dumps({'Image':actual['Image'],
            'Devices':actual['HostConfig']['Devices'],'Mounts':actual['Mounts']},indent=2))
        command=['python','/code/probes/'+{'qualification':'qualify_unary_npu.py',
            'unary':'test_unary_adapter_npu.py','sigmoid':'test_sigmoid_model_npu.py','qwen':'test_silu_training_npu.py'}[a.stage],
            '--runtime-root','/runtime','--out','/results/result.json'] if a.stage!='dc-train' else []
        if a.stage=='qualification':
            command+=['--image-id',IMAGE,'--operator','sigmoid','--shape','32,64']
        elif a.stage=='qwen':
            command+=['--scoped-adapter','--chip-name','Ascend910_9382','--admission-records',
                      '/records/'+a.records.name,'--model','/model','--steps','3']
        elif a.stage!='dc-train': command+=['--records','/records/'+a.records.name]
        if a.stage=='dc-train':
            command=['python','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                '/code/tools/train_with_adapter.py','--config','/code/demo/training-configs/'+a.training_config+'.json']
            if a.wheel:
                command=['python','-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    '--module','flagrt_adapter.train_cli','--config','/config/'+a.training_config+'.json']
        if a.stage=='sigmoid':
            if a.native_repeat_only: command+=['--native-repeat-only']
            if a.deterministic: command+=['--deterministic']
        # Keep image PYTHONPATH (e.g. vllm dependencies); prepend read-only overrides.
        import shlex
        if a.wheel:
            execute(['docker','exec',name,'python','-m','pip','install','--no-deps','--no-index',
                     '--target','/results/installed','/package/'+a.wheel.name],'install.log')
            check='import flagrt_adapter,flagrt_adapter.train_cli; from pathlib import Path; import json; p=Path(flagrt_adapter.__file__).resolve(); assert Path("/results/installed") in p.parents; assert not Path("/code/src/flagrt_adapter/policy.py").exists(); print(json.dumps({"path":str(p),"version":flagrt_adapter.__version__}))'
            execute(['docker','exec','-e','PYTHONPATH=/results/installed',name,'python','-c',check],'installed-import.log')
        shell='export PYTHONPATH=/gems/src:/deps:$PYTHONPATH; export TRITON_CACHE_DIR=/results/cache; exec '+shlex.join(['timeout','--kill-after=10s','900']+command)
        if a.wheel:
            shell='export PYTHONPATH=/results/installed:$PYTHONPATH; '+shell
        execute(['docker','exec',name,'bash','-lc',shell],'test.log')
        if a.stage=='dc-train':
            from check_training import check_run
            result=check_run(a.output,a.training_config)
            (a.output/'result.json').write_text(json.dumps(result,indent=2))
        else:
            result=json.loads((a.output/'result.json').read_text())
        if result['status']!='pass': raise RuntimeError('Probe did not report pass')
        if a.wheel and hashlib.sha256(a.wheel.read_bytes()).hexdigest()!=summary['wheel']['sha256']:
            raise RuntimeError('Wheel changed during test')
        for key,root in [('source',a.source),('runtime',a.runtime),('deps',a.deps)]:
            if manifest(root)!=identity[key]: raise RuntimeError('Inputs changed during test: '+key)
        summary['status']='PASS'
    except Exception:
        summary['error']=traceback.format_exc()
        print(summary['error'])
        summary['status']='FAIL' if (a.output/'result.json').exists() else 'ERROR'
    finally:
        if started:
            stopped=subprocess.run(['docker','stop','--time','10',name],capture_output=True,text=True)
            summary['stop_returncode']=stopped.returncode
            state=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True)
            summary['container_state']=state.stdout.strip()
            if stopped.returncode: summary['status']='ERROR'
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('overall: '+summary['status'])
    return {'PASS':0,'FAIL':1,'ERROR':2}[summary['status']]


if __name__=='__main__':
    raise SystemExit(main())
