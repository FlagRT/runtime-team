"""P800 private-container runner. Public strict baseline, bounded test, always stop."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import traceback
import yaml

IMAGE = 'sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a'
REF = 'flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64'
PYTHON = '/root/miniconda/envs/python310_torch29_cuda/bin/python3'
PUBLIC = 'a7e90e750fcf5a8b8adcd53858815c4d97cddac4'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(root):
    return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not any(x in p.parts for x in ('.git','__pycache__'))}


def workload_labels(stage, operator, batch_size):
    if stage in ('model','onerec'): return dict(operator='silu',batch_size=1)
    if stage=='deepfm': return dict(operator='sigmoid',batch_size=batch_size)
    return dict(operator=operator,batch_size=None)


def idle_p800(text, device):
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if re.match(r'^\|\s+'+str(device)+r'\s+P800\b',line)]
    if len(headers) != 1:
        raise ValueError('Cannot identify the selected physical P800')
    block = '\n'.join(lines[headers[0]:headers[0]+4])
    memory = re.search(r'\|\s+(\d+)MiB\s*/\s*\d+MiB\s*\|\s+(\d+)%',block)
    if not memory or tuple(map(int,memory.groups())) != (0,0):
        raise ValueError('Selected P800 is not idle (0 MiB and 0%)')
    if any(re.match(r'^\|\s+'+str(device)+r'\s+N/A\s+N/A\s+\d+',line) for line in lines):
        raise ValueError('Selected P800 still has a process')
    return {'chip':'P800','physical_device':device,'memory_mib':0,'utilization_percent':0}


def normalize_bus(bus):
    match=re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',bus)
    if not match: raise ValueError('Invalid PCI bus identity')
    domain,bus_id,slot,function=(int(v,16) for v in match.groups())
    return f'{domain:04x}:{bus_id:02x}:{slot:02x}.{function}'


def match_xpu_node(text, device, node_buses):
    """xpu-smi is PCI ordered; /dev/xpuN uses driver minors, not that index."""
    idle_p800(text,device)
    header=next(line for line in text.splitlines() if re.match(r'^\|\s+'+str(device)+r'\s+P800\b',line))
    buses=re.findall(r'[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',header)
    if len(buses)!=1: raise ValueError('Cannot identify selected PCI bus')
    bus=normalize_bus(buses[0])
    nodes=[node for node,value in node_buses.items() if normalize_bus(value)==bus]
    if len(nodes)!=1: raise ValueError('Missing or ambiguous PCI-to-device-node mapping')
    return dict(smi_index=device,pci_bus=bus,device_node=nodes[0],logical_device=0)


def resolve_xpu_node(text, device):
    nodes={}
    for path in Path('/dev').glob('xpu[0-7]'):
        info=path.stat()
        if not stat.S_ISCHR(info.st_mode): continue
        sysfs=Path(f'/sys/dev/char/{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}').resolve(strict=True)
        buses=[part for part in sysfs.parts if re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',part)]
        if buses: nodes[str(path)]=buses[-1]
    return match_xpu_node(text,device,nodes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public-root','wheel','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--stage',choices=['inspect','qualification','routes','model','deepfm','onerec'],required=True)
    p.add_argument('--operator',choices=['silu','sigmoid','tanh'],default='silu')
    p.add_argument('--shape-profile',choices=['default','onerec16'],default='default')
    p.add_argument('--batch-size',type=int,choices=(16,32),default=32)
    p.add_argument('--mode',choices=['native','native_repeat','off','on','unqualified','denied','unknown','strict','candidate'],default='off')
    p.add_argument('--records',type=Path)
    p.add_argument('--adapter-config',type=Path,help='Schema-3 model configuration inside the review bundle')
    p.add_argument('--model',type=Path,help='Read-only model directory or single-model HF cache root')
    p.add_argument('--reference',type=Path,help='Completed native model run; mounted read-only')
    p.add_argument('--steps',type=int,choices=(1,2,3),default=3)
    a=p.parse_args()
    model_stage=a.stage in ('model','deepfm','onerec')
    if a.shape_profile!='default' and (a.stage not in ('qualification','routes') or a.operator!='silu'):
        p.error('OneRec shape profile is only for isolated SiLU qualification/routes')
    if a.mode=='candidate' and a.stage!='onerec':
        p.error('Candidate is an explicit OneRec qualification experiment, not a normal model mode')
    if a.output.exists() or a.device not in range(8) or not a.wheel.is_file():
        p.error('Require a NEW output, one physical device 0..7 and existing wheel')
    if a.stage=='routes' and a.mode=='on' and (not a.records or not a.records.is_file()):
        p.error('Reviewed evidence required; this runner never generates qualifications')
    if a.mode=='unqualified' and a.records:
        p.error('Unqualified run must not load records')
    if model_stage:
        if a.mode=='candidate':
            if a.adapter_config or not a.records or not a.records.is_file():
                p.error('Candidate experiment needs single-op records, not a model approval config')
        elif a.records: p.error('Model records are selected through schema-3 --adapter-config')
        if (a.mode not in ('native','native_repeat','candidate'))!=bool(a.adapter_config):
            p.error('Model adapter modes require --adapter-config')
        if a.adapter_config and not a.adapter_config.is_file(): p.error('Missing adapter config')
        if not a.model or not a.model.is_dir(): p.error('Model stage requires an existing model directory')
        if a.mode not in ('native','strict') and (not a.reference or not (a.reference/'result.json').is_file()):
            p.error('Comparison requires a completed native reference')
        if a.mode in ('native','strict') and a.reference: p.error('Native/strict stage cannot consume another reference')
    elif a.model or a.reference or a.adapter_config or a.mode in ('native','native_repeat','denied','unknown','strict'):
        p.error('Model flags/modes are only valid for the model stage')
    a.public_root=a.public_root.resolve(); a.output=a.output.resolve(); a.wheel=a.wheel.resolve()
    runtime=a.public_root/'dev/device-context/prototype'
    baseline=a.public_root/'dev/images/kunlun-operator-runtime/v2/image.baseline.yaml'
    if not baseline.is_file() or not (runtime/'runtime/__init__.py').is_file():
        p.error('Missing fixed public snapshot')
    name='cgu135-xpu-'+a.output.name
    if subprocess.run(['docker','inspect',name],capture_output=True,timeout=30).returncode==0:
        p.error('Container name exists; will not reuse or stop an existing object')
    required_gib=40*a.steps+20 if a.stage=='onerec' else (24 if a.stage=='model' else 2)
    if shutil.disk_usage(a.output.parent).free < required_gib*1024**3:
        p.error(f'Need at least {required_gib} GiB on the output filesystem')
    a.output.mkdir()
    for name_dir in ('demo','tmp','cache','home'):
        (a.output/name_dir).mkdir()
    case=Path(__file__).resolve().parent
    snapshots={'assets':manifest(case),'runtime':manifest(runtime),'wheel':digest(a.wheel)}
    if a.adapter_config: snapshots['review']=manifest(a.adapter_config.resolve().parent)
    if a.records: snapshots['review']=manifest(a.records.resolve().parent)
    (a.output/'inputs.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    summary=dict(status='ERROR',stage=a.stage,mode=a.mode,physical_device=a.device,logical_device=0,
                 **workload_labels(a.stage,a.operator,a.batch_size),
                 container=name,image=IMAGE,public_commit=PUBLIC,wheel_sha256=digest(a.wheel),
                 source='scoped case snapshot; exact hashes in inputs.json',commands=[])
    try:
        image_env=subprocess.check_output(['docker','image','inspect',REF,'--format','{{json .Config.Env}}'],text=True,timeout=30)
        resource=subprocess.check_output(['xpu-smi'],text=True,timeout=15)
        mapping=resolve_xpu_node(resource,a.device)
        summary['device_mapping']=mapping
        (a.output/'device-mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    except Exception:
        summary.update(error=traceback.format_exc(),phase='image_or_device_mapping_before_container_creation',
                       container_start_attempted=False)
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(summary['error'],flush=True)
        print('overall: ERROR evidence:',a.output,flush=True)
        return 2
    baked=dict(line.split('=',1) for line in json.loads(image_env) if '=' in line)
    config={'name':name,'x-flagrt':{'image_baseline':str(baseline),'mounted_repos':{}},
        'services':{'runtime-dev':{'image':REF,'container_name':name,'restart':'no',
          'network_mode':'none','shm_size':'2g','mem_limit':'12g','pids_limit':512,
          'devices':[mapping['device_node']+':'+mapping['device_node'],'/dev/xpuctrl:/dev/xpuctrl'],
          'volumes':[str(a.output)+':/results',str(case)+':/case:ro',str(runtime)+':/runtime:ro',
                     str(a.wheel)+':/package/'+a.wheel.name+':ro'],
          'environment':{'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':'/results/installed:'+baked.get('PYTHONPATH',''),
                         'CUDA_VISIBLE_DEVICES':'0','VLLM_FL_PREFER':'vendor','TRITON_CACHE_DIR':'/results/cache',
                         'FLAGRT_VERIFIED_CHIP':'P800',
                         'TMPDIR':'/results/tmp','XDG_CACHE_HOME':'/results/cache','PIP_NO_CACHE_DIR':'1',
                         'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
          'command':['sleep','infinity'],'logging':{'driver':'json-file','options':{'max-size':'2m','max-file':'1'}}}}}
    if a.records:
        config['services']['runtime-dev']['volumes'].append(str(a.records.resolve().parent)+':/review:ro')
    if a.adapter_config:
        config['services']['runtime-dev']['volumes'].append(str(a.adapter_config.resolve().parent)+':/review:ro')
        summary['adapter_config_sha256']=digest(a.adapter_config)
    if model_stage:
        service=config['services']['runtime-dev']
        if a.stage=='model': service.update(mem_limit='32g',shm_size='4g')
        if a.stage=='onerec': service.update(mem_limit='96g',shm_size='4g')
        service['environment'].update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                                      TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        service['volumes'].append(str(a.model.resolve())+':/model:ro')
        if a.reference: service['volumes'].append(str(a.reference.resolve())+':/reference:ro')
    (a.output/'demo/docker-compose.yml').write_text(yaml.safe_dump(config,sort_keys=False))
    env=dict(os.environ,FLAGRT_WORKSPACE_ROOT=str(a.public_root),TMPDIR=str(a.output/'tmp'))
    def execute(command,log,timeout=360,expected_code=0):
        with (a.output/log).open('x') as output:
            result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
        summary['commands'].append(dict(command=command,log=log,exit_code=result.returncode))
        if result.returncode != expected_code:
            raise RuntimeError(f'{log}: exit {result.returncode}')
    attempted=False
    try:
        resource=subprocess.check_output(['xpu-smi'],text=True,timeout=15)
        (a.output/'resource-before.log').write_text(resource)
        summary['resource_check']=idle_p800(resource,a.device)
        if resolve_xpu_node(resource,a.device)!=mapping:
            raise RuntimeError('Device mapping changed before launch')
        summary['host_storage']={str(path):shutil.disk_usage(path)._asdict() for path in (Path('/'),a.output)}
        execute([sys.executable,str(a.public_root/'dev/lib/baseline.py'),'export',str(a.output/'demo')],'export.log')
        attempted=True
        execute(['bash',str(a.public_root/'dev/lib/up.sh'),str(a.output/'demo'),'--strict'],'strict-start.log')
        inspect=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        if inspect['Image']!=IMAGE:
            raise RuntimeError('Actual image mismatch')
        (a.output/'container.json').write_text(json.dumps({'Image':inspect['Image'],
            'devices':inspect['HostConfig']['Devices'],'mounts':inspect['Mounts']},indent=2)+'\n')
        execute(['docker','exec',name,PYTHON,'-m','pip','install','--no-deps','--no-index','--no-compile',
                 '--target','/results/installed','/package/'+a.wheel.name],'install.log')
        if model_stage:
            script={'model':'model_probe.py','deepfm':'deepfm_probe.py','onerec':'onerec_probe.py'}[a.stage]
            command=['timeout','--kill-after=10s','1800' if a.stage=='onerec' else '900',PYTHON,'-B','/case/'+script,
                     '--runtime-root','/runtime','--model','/model','--steps',str(a.steps),
                     '--mode',a.mode,'--out','/results/result.json']
            if a.stage=='deepfm': command+=['--batch-size',str(a.batch_size)]
            if a.reference: command+=['--reference','/reference']
        else:
            command=['timeout','--kill-after=10s','300',PYTHON,'-B','/case/probe.py','--runtime-root','/runtime',
                     '--stage',a.stage,'--mode',a.mode,'--operator',a.operator,
                     '--shape-profile',a.shape_profile,'--out','/results/result.json']
        if a.records: command+=['--records','/review/'+a.records.name]
        if a.adapter_config: command+=['--adapter-config','/review/'+a.adapter_config.name]
        execute(['docker','exec',name]+command,'probe.log',timeout=1860 if a.stage=='onerec' else (960 if model_stage else 360),
                expected_code=3 if a.mode=='strict' else 0)
        report=json.loads((a.output/'result.json').read_text())
        expected_status='blocked' if a.mode=='strict' else 'pass'
        if report['status']!=expected_status:
            raise RuntimeError('Probe did not report pass')
        if a.mode=='strict' and (report.get('block_reason')!='workload_denied'
                or report.get('model_loaded') or report.get('optimizer_created')
                or report.get('provider_imported') is not False or report.get('steps')):
            raise RuntimeError('Strict path was not blocked before model training/provider import')
        summary['execution_status']=report['status']
        summary['acceptance']=('expected startup rejection' if a.mode=='strict' else
                               'candidate experiment; no model approval generated' if a.mode=='candidate' else 'bounded training/probe')
        if manifest(case)!=snapshots['assets'] or manifest(runtime)!=snapshots['runtime'] or digest(a.wheel)!=snapshots['wheel']:
            raise RuntimeError('Inputs changed during run')
        if a.adapter_config and manifest(a.adapter_config.resolve().parent)!=snapshots['review']:
            raise RuntimeError('Model review bundle changed during run')
        if a.records and manifest(a.records.resolve().parent)!=snapshots['review']:
            raise RuntimeError('Single-op review bundle changed during run')
        summary['status']='PASS'
    except Exception:
        summary['error']=traceback.format_exc()
        if (a.output/'result.json').is_file():
            status=json.loads((a.output/'result.json').read_text()).get('status')
            summary['status']='FAIL' if status=='fail' else 'ERROR'
        print(summary['error'],flush=True)
    finally:
        if attempted:
            present=subprocess.run(['docker','inspect',name],capture_output=True,text=True)
            if present.returncode==0:
                stopped=subprocess.run(['docker','stop','--time','10',name],capture_output=True,text=True,timeout=40)
                state=json.loads(subprocess.check_output(['docker','inspect',name,'--format','{{json .State}}'],text=True))
                summary['stop']={'exit_code':stopped.returncode,'state':state}
                if stopped.returncode or state['Running'] or state.get('Pid'):
                    summary['status']='ERROR'
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('overall:',summary['status'],'evidence:',a.output,flush=True)
    return 0 if summary['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
