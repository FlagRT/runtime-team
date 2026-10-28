"""Known-v3 startup policy: native/off and native/degrade train; strict is BLOCKED."""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import _adapter_package
from flagrt_adapter import Environment
from flagrt_adapter.startup import provider_precheck, package_identity, TargetBlocked
from flagrt_adapter.training import execute_training


def run(a,r):
    import torch
    import triton
    sys.path.insert(0,a.runtime_root)
    import runtime
    runtime.use('ascend'); runtime.set_device(0)
    assert runtime.device_count()==1
    torch.use_deterministic_algorithms(True)
    assert 'flag_gems' not in sys.modules, 'Native runtime unexpectedly imported FlagGems'
    spec=importlib.util.find_spec('flag_gems')
    revision,identity=package_identity(spec.origin)
    env=Environment(torch.npu.get_device_name(0),torch.__version__,metadata.version('torch-npu'),
        revision,os.environ['FLAGRT_VERIFIED_IMAGE_ID'],triton.__version__,'flagtree=='+metadata.version('flagtree'))
    r.update(environment=env.__dict__,identity=identity,modes=[])
    before=torch._C._dispatch_dump_table('aten::silu')
    for mode in ('off','degrade','strict'):
        config=dict(schema=2,enabled=mode!='off',require_target=mode=='strict',
            operators={'silu':dict(shapes=[[32,64]],dtypes=['torch.float32'])},records=None)
        reason=provider_precheck(config,env.image_id)
        expected='experimental_disabled' if mode=='off' else 'known_compile_failure'
        assert reason==expected,(reason,expected)
        target=a.out.parent/(mode+'.pt'); os.environ['STARTUP_TEST_RESULT']=str(target)
        report_path=a.out.parent/(mode+'.json')
        try:
            result=execute_training(Path(__file__).with_name('startup_training_entry.py'),config,(),env,runtime,None,report_path,
                                    provider_reason=reason,identity=identity)
            assert mode!='strict'
            assert result['status']=='PASS'
            assert result['counts']==[dict(operator='silu',backend='native',stage='forward',reason=expected,count=3)]
            state=torch.load(target,weights_only=True)
            if mode=='off': native=state
            else:
                assert state['losses']==native['losses']
                for key,value in state['state'].items(): torch.testing.assert_close(value,native['state'][key],atol=0,rtol=0)
        except TargetBlocked:
            assert mode=='strict' and not target.exists()
            result=json.loads(report_path.read_text()); assert result['status']=='BLOCKED' and result['counts']==[]
        r['modes'].append(dict(mode=mode,status=result['status'],reason=reason,counts=result['counts']))
        assert before==torch._C._dispatch_dump_table('aten::silu')
        assert 'flag_gems' not in sys.modules
    r.update(status='pass',target_imported=False,target_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runtime-root',required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if a.out.exists(): p.error('Output exists')
    r={'status':'fail','scope':'startup fallback controls on v3, NOT target-kernel acceptance'}
    try: run(a,r)
    except Exception: r['error']=traceback.format_exc(); print(r['error'],flush=True)
    finally:
        with a.out.open('x') as f: json.dump(r,f,indent=2)
    print('overall: '+r['status'].upper()); raise SystemExit(0 if r['status']=='pass' else 1)
