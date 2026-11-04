#!/usr/bin/env python3
"""Prepare a drained API for an independent trial; never use restart as recovery evidence.

Runs inside rag-ljy-vllm-910c. Preserves the existing model/database command.
Default only prints the proposed command; --run performs the restart.
"""
import argparse
import fcntl
import json, os, signal, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',action='store_true')
parser.add_argument('--retrieval-backend',choices=['milvus','elasticsearch'])
parser.add_argument('--cache-bridge',default='/workspace/dev/rag_ljy/experiment_outputs/runtime_extensions/82f0cce02d30e7b44704/libnpu_cache_bridge.so')
options=parser.parse_args()
lock=open('/tmp/rag-watermarks-modes.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def health():
    with urlopen('http://127.0.0.1:8088/health', timeout=5) as r:
        return json.load(r)

servers=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        args=p.read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeError):
        continue
    if any(Path(a).name in ('check_watermarks_modes.py','check_memory_admission_live.py','run_pressure_suite.py','check_online_pressure.py','check_request_memory_modes.py','check_dynamic_admission_live.py','check_measured_watermarks.py') for a in args):
        raise RuntimeError('A pressure test is running; refusing to restart')
    if any(Path(a).name=='serve_queries.py' for a in args) and Path(args[0]).name.startswith('python'):
        servers.append((int(p.parent.name),args))
if len(servers)!=1:
    raise RuntimeError(f'Expected one query API, got {len(servers)}')
pid, old=servers[0]
h=health()
if h['inflight']!=0 or h['memory_admission']['reserved_mib']!=0:
    raise RuntimeError('API must be drained before restart')
if not h['memory_admission']['enabled']:
    raise RuntimeError('Memory admission must stay enabled')
env=dict(v.split('=',1) for v in Path(f'/proc/{pid}/environ').read_bytes().decode().rstrip('\0').split('\0'))
old_env=dict(env)
backend=options.retrieval_backend or h['retrieval_backend']
env['RETRIEVAL_BACKEND']=backend
env['RAG_PROFILE_FIRST_FORWARD']='1'
env['RAG_MEMORY_PROBE_PROCESS']='1'
env['RAG_RERANK_DOCUMENT_TOKENS']='640'
env['RAG_NPU_CACHE_BRIDGE']=options.cache_bridge
env['PYTORCH_NPU_ALLOC_CONF']='garbage_collection_threshold:0.70'
cwd=Path(f'/proc/{pid}/cwd').resolve()
new=list(old)
if '--memory-request-mib' in new:
    i=new.index('--memory-request-mib'); del new[i:i+2]
if '--memory-budget-mode' in new:
    new[new.index('--memory-budget-mode')+1]='dynamic'
else:
    new += ['--memory-budget-mode','dynamic']
if '--max-inflight' in new:
    new[new.index('--max-inflight')+1]='0'
else:
    new+=['--max-inflight','0']
for flag, value in [('--memory-safety-mib','5120'), ('--max-reranker-batch-size','32'), ('--memory-probe-interval','0.001'), ('--embedding-workers','8')]:
    if flag in new: new[new.index(flag)+1]=value
    else: new += [flag,value]
# Preserve a bounded stage waiting queue; this does not reject new requests.
if '--stage-queue-capacity' not in new:
    new+=['--stage-queue-capacity','8']
database_check=None
if backend == 'elasticsearch':
    checked=subprocess.run([old[0],str(cwd/'scripts/runtime/check_elasticsearch_memory_inputs.py')],cwd=cwd,env=env,capture_output=True,text=True,timeout=120)
    if checked.returncode:
        raise RuntimeError('Elasticsearch preflight failed before stopping the API:\n'+checked.stderr[-3000:])
    database_check=json.loads(checked.stdout)
if not options.run:
    print(json.dumps({'retrieval_backend':backend,'database_check':database_check,'proposed_command':new,'cache_bridge':options.cache_bridge,'note':'Read-only. --run restarts the drained API before an independent experiment.'},indent=2))
    raise SystemExit(0)
if not Path(options.cache_bridge).is_file():raise RuntimeError('Build the native cache bridge first')
out=cwd/'experiment_outputs/checks/dynamic_memory_admission'/backend/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out.mkdir(parents=True)
(out/'config.json').write_text(json.dumps({'retrieval_backend':backend,'database_check':database_check,'old_command':old,'new_command':new,'health_before':h,'environment':{k:v for k,v in env.items() if k.startswith('RAG_') or k in ('DC_BACKEND','PYTHONPATH','PYTORCH_NPU_ALLOC_CONF','RETRIEVAL_BACKEND')}},indent=2))
print('OUTPUT='+str(out),flush=True)
os.kill(pid,signal.SIGTERM)
end=time.monotonic()+60
while Path(f'/proc/{pid}/stat').exists():
    if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
    if time.monotonic()>end:raise RuntimeError('Original API did not exit; no replacement started')
    time.sleep(.25)
def start(command, label, environment=None):
    with (out/(label+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=cwd,env=environment or env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    Path('/tmp/rag-query-server.pid').write_text(str(proc.pid))
    return proc
proc=start(new,'api')
end=time.monotonic()+120
try:
    while time.monotonic()<end:
        if proc.poll() is not None:raise RuntimeError('New API exited')
        try:
            h=health()
            if h['status']=='ready' and h['memory_admission']['status']=='ready':
                assert h['retrieval_backend']==backend, 'API started with the wrong retrieval backend'
                assert h['max_inflight']==0 and h['memory_estimation']['enabled'] and h['memory_admission']['config']['request_budget_mib'] is None
                (out/'health_after.json').write_text(json.dumps(h,indent=2))
                print(json.dumps(h),flush=True)
                break
        except (OSError, ValueError):pass
        time.sleep(.5)
    else:raise RuntimeError('New API readiness timeout')
except BaseException:
    proc.terminate()
    try:proc.wait(30)
    except subprocess.TimeoutExpired:
        raise RuntimeError('New API did not stop; refusing to overlap rollback')
    restored=start(old,'rollback',old_env)
    print('Rollback API started, PID '+str(restored.pid),flush=True)
    raise
