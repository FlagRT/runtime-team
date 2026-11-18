#!/usr/bin/env python3
"""Supervise the two-die RAG workflow; reuse verified services already running.

Container startup entry: OneRec vLLM on die6, embedding/rerank on die7,
Selectable Elasticsearch/Milvus retrieval, API8088. Never changes database contents or other containers.
"""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT/'.venv/bin/python')


def get(port, path='/health'):
    try:
        with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}{path}', timeout=3) as response:
            body=response.read().decode()
            return json.loads(body) if body else {}
    except Exception:
        return None


def commands():
    return {
        'generation': [PYTHON, '-u', 'scripts/runtime/serve_generation_vllm.py', '--physical-die', '6', '--run'],
        'embedding': [PYTHON, '-u', 'scripts/runtime/serve_embeddings.py'],
        'api': [PYTHON, '-u', 'scripts/runtime/serve_queries.py', '--device', 'npu:0', '--port', '8088',
            '--generation', '--generation-model', '/mnt/onerec/OneRec-8B',
            '--generation-max-input-tokens', '4096', '--generation-max-new-tokens', '1024',
            '--max-inflight', '0', '--stage-queue-capacity', '8', '--embedding-workers', '8',
            '--retrieval-workers', '4', '--generation-workers', '4', '--max-reranker-batch-size', '32',
            '--memory-admission', '--memory-budget-mode', 'dynamic', '--memory-probe-interval', '.01',
            '--memory-safety-mib', '5120', '--memory-high-watermark', '.85', '--memory-low-watermark', '.75',
            '--memory-allow-unverified-scope'],
    }


def verify_service_die(port, die):
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().decode().rstrip('\0').split('\0')
            if 'vllm.entrypoints.openai.api_server' not in args or '--port' not in args:
                continue
            if args[args.index('--port')+1] != str(port):
                continue
            env = dict(item.split('=', 1) for item in (path.parent/'environ').read_bytes().decode().rstrip('\0').split('\0'))
            matches.append(env.get('ASCEND_RT_VISIBLE_DEVICES') == str(die)
                           and env.get('RAG_ASCEND_PINNED_DIE') == str(die))
        except (OSError, UnicodeError):
            continue
    if matches != [True]:
        raise RuntimeError(f'Service on {port} is not uniquely pinned to physical die {die}')


def main():
    os.chdir(ROOT)
    backend=os.getenv('RAG_WORKFLOW_BACKEND','elasticsearch')
    if backend not in ('elasticsearch','milvus'):
        raise ValueError('RAG_WORKFLOW_BACKEND must be elasticsearch or milvus')
    lock=open('/tmp/rag-vllm-workflow.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    out=ROOT/'experiment_outputs/runtime/vllm_workflow'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    Path('/tmp/rag-vllm-workflow-output').write_text(str(out))
    print('OUTPUT='+str(out),flush=True)
    env=dict(os.environ)
    env.update(ASCEND_RT_VISIBLE_DEVICES='7', RAG_ASCEND_PINNED_DIE='7', DC_BACKEND='ascend',
        RETRIEVAL_BACKEND=backend, RAG_RETRIEVAL_BACKENDS='elasticsearch,milvus', RAG_GENERATION_BACKEND='vllm',
        RAG_GENERATION_URL='http://127.0.0.1:8090', RAG_GENERATION_DIE='6',
        RAG_EMBEDDING_BACKEND='vllm', RAG_MEMORY_PROBE_PROCESS='1',
        RAG_RERANK_DOCUMENT_TOKENS='640', PYTORCH_NPU_ALLOC_CONF='garbage_collection_threshold:0.70',
        RAG_NPU_CACHE_BRIDGE=str(ROOT/'experiment_outputs/runtime_extensions/82f0cce02d30e7b44704/libnpu_cache_bridge.so'))
    configs=commands()
    (out/'commands.json').write_text(json.dumps(configs,indent=2))
    children={}
    stopping=False
    def stop(signum, frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def require_running():
        if stopping:
            raise InterruptedError('Workflow stopping')
        for name,child in children.items():
            if child.poll() is not None:
                raise RuntimeError(f'{name} exited with {child.returncode}; see {out}/{name}.log')
    def ensure(name,port,predicate):
        current=get(port)
        if current is not None:
            if not predicate(current):
                raise RuntimeError(f'Existing {name} on {port} has incompatible configuration')
            print(f'REUSED={name}',flush=True)
            return
        with (out/(name+'.log')).open('w') as log:
            child=subprocess.Popen(configs[name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[name]=child
        if name=='api':
            Path('/tmp/rag-query-server.pid').write_text(str(child.pid))
        (out/'processes.json').write_text(json.dumps({k:v.pid for k,v in children.items()}))
        end=time.monotonic()+240
        while time.monotonic()<end:
            require_running()
            current=get(port)
            if current is not None and predicate(current):
                print(f'READY={name}',flush=True)
                return
            time.sleep(1)
        raise RuntimeError(f'{name} readiness timeout; see {out}/{name}.log')
    try:
        ensure('generation',8090,lambda _: 'onerec-8b' in {m['id'] for m in (get(8090,'/v1/models') or {}).get('data',[])})
        verify_service_die(8090,6)
        ensure('embedding',8089,lambda _: 'rag-embedding' in {m['id'] for m in (get(8089,'/v1/models') or {}).get('data',[])})
        verify_service_die(8089,7)
        ensure('api',8088,lambda h: h.get('status')=='ready' and h.get('retrieval_backend')==backend
            and set(h.get('retrieval_backends',[]))=={'milvus','elasticsearch'}
            and h.get('generation',{}).get('execution_profile',{}).get('backend')=='vllm'
            and h.get('memory_admission',{}).get('status')=='ready'
            and h.get('memory_admission_by_stage',{}).get('generation',{}).get('status')=='ready')
        (out/'health_ready.json').write_text(json.dumps(get(8088),indent=2))
        print('WORKFLOW=ready',flush=True)
        while not stopping:
            require_running()
            time.sleep(1)
    except InterruptedError:
        pass
    finally:
        # Drain API first; then stop only model processes started by this supervisor.
        for name in ('api','embedding','generation'):
            child=children.get(name)
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(180 if name=='api' else 40)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()


if __name__=='__main__':
    main()
