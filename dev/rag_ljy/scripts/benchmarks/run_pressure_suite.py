#!/usr/bin/env python3
"""Run backend-isolated pressure checks, restore Milvus, then real-memory checks.

Resume mode waits for an already running Milvus check. Server PID must be the
known RAG API, never a general pkill. Embedding service remains running.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from check_online_pressure import http
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume-milvus-dir',type=Path);a=p.parse_args()
    out=ROOT/'experiment_outputs/checks/pressure_suite'/datetime.now().strftime('%Y%m%dT%H%M%S');out.mkdir(parents=True)
    print('SUITE='+str(out),flush=True)
    results={};proc=None
    def stop_server():
        pid=int(Path('/tmp/rag-query-server.pid').read_text().strip())
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:return
        if b'serve_queries.py' not in cmd:raise RuntimeError('PID no longer belongs to RAG API')
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if http('http://127.0.0.1:8088/health',timeout=5)[2]['inflight']==0:break
            time.sleep(.5)
        else:raise RuntimeError('API failed to drain')
        os.kill(pid,15)
        deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            stat=Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2]=='Z':return
            time.sleep(.2)
        raise RuntimeError('API shutdown timeout')
    def start_server(backend):
        nonlocal proc
        env=dict(os.environ,RETRIEVAL_BACKEND=backend,MODEL_DIR='/mnt/raid/jliu171/models',MILVUS_COLLECTION='nfcorpus_chunks_v1',ELASTICSEARCH_INDEX='nfcorpus-chunks-v1',DC_BACKEND='ascend',RAG_EMBEDDING_BACKEND='vllm')
        logpath=out/(backend+'-server.log')
        with logpath.open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u','scripts/runtime/serve_queries.py','--device','npu:0','--port','8088','--memory-admission','--memory-request-mib','2048','--memory-allow-unverified-scope'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        Path('/tmp/rag-query-server.pid').write_text(str(proc.pid))
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('API failed; inspect '+str(logpath))
            try:
                h=http('http://127.0.0.1:8088/health',timeout=5)[2]
                if h['status']=='ready' and h['memory_admission']['status']=='ready':return
            except Exception:pass
            time.sleep(.5)
        raise RuntimeError('API readiness timeout')
    def run(name,command):
        with (out/(name+'.log')).open('w') as log:
            code=subprocess.call([sys.executable,'-u']+command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        results[name]=code;print(name+' exit='+str(code),flush=True)
    error=None;switched=False
    try:
        if a.resume_milvus_dir:
            end=time.monotonic()+2400
            while not (a.resume_milvus_dir/'summary.json').exists():
                if time.monotonic()>end:raise RuntimeError('Milvus run timeout')
                time.sleep(5)
            results['milvus']=json.loads((a.resume_milvus_dir/'summary.json').read_text())['status']
        else:run('milvus',['scripts/benchmarks/check_online_pressure.py','--backend','milvus'])
        stop_server();switched=True;start_server('elasticsearch')
        run('elasticsearch',['scripts/benchmarks/check_online_pressure.py','--backend','elasticsearch'])
    except Exception as e:error=repr(e)
    finally:
        if switched:
            try:stop_server();start_server('milvus')
            except Exception as e:error=str(error)+'; restore failed: '+repr(e)
    if error is None:
        run('memory',['scripts/checks/check_memory_admission_live.py','--device','npu:0'])
    result={'results':results,'error':error,'note':'Main API restored to Milvus. Inspect per-run summaries; exit code 0 alone does not establish every test passed.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
