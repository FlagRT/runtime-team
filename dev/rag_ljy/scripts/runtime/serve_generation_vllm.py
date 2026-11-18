#!/usr/bin/env python3
"""Start standalone OneRec vLLM on physical card 3, die 6 (default).

Default is read-only preflight. --run replaces this process with vLLM.
This does not switch the RAG API or reinterpret its single-device memory budget.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

PROBE = '''
import ctypes,json
from rag_engine.device import prepare_torch_device
prepare_torch_device('npu:0')
import runtime
lib=ctypes.CDLL('libascendcl.so')
logical=ctypes.c_int32(-1); physical=ctypes.c_int32(-1)
a=lib.aclrtGetDevice(ctypes.byref(logical))
b=lib.aclrtGetPhyDevIdByLogicDevId(logical.value,ctypes.byref(physical))
if a or b: raise RuntimeError('CANN physical device verification failed')
print('PROBE='+json.dumps({'physical_die':physical.value,'logical_device':logical.value,'stats':runtime.memory_stats(0)}))
'''


def launch_command(args):
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
        '--model', str(args.model), '--served-model-name', 'onerec-8b',
        '--dtype', 'bfloat16', '--enforce-eager', '--no-enable-prefix-caching',
        '--enable-chunked-prefill', '--block-size', '128',
        '--gpu-memory-utilization', str(args.memory_fraction),
        '--max-model-len', '5120', '--max-num-seqs', '4',
        '--max-num-batched-tokens', '4096',
        '--host', '127.0.0.1', '--port', str(args.port)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--physical-die', type=int, choices=[6,7], default=6)
    parser.add_argument('--model', type=Path, default=Path('/mnt/onerec/OneRec-8B'))
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--memory-fraction', type=float, default=.55)
    args = parser.parse_args()
    if not .3 <= args.memory_fraction <= .70 or not 1 <= args.port <= 65535:
        parser.error('memory-fraction must be 0.30..0.70; port must be valid')
    from rag_engine.generation import checkpoint_manifest
    manifest = checkpoint_manifest(args.model)
    config = json.loads((args.model/'config.json').read_text())
    if config.get('architectures') != ['Qwen3ForCausalLM']:
        raise RuntimeError('Require the verified OneRec Qwen3ForCausalLM checkpoint')
    # Detect a port collision before initializing any accelerator process.
    with socket.socket() as port:
        # A stopped server can leave completed connections in TIME_WAIT.
        # Match the HTTP server's reuse policy without allowing a live listener.
        port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port.bind(('127.0.0.1', args.port))
    env = dict(os.environ)
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(args.physical_die),
               RAG_ASCEND_PINNED_DIE=str(args.physical_die), DC_BACKEND='ascend',
               TRITON_ENABLE_TASKQUEUE='true', HF_HUB_OFFLINE='1', DO_NOT_TRACK='1')
    acl = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages'
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), acl, env.get('PYTHONPATH','')])
    result = subprocess.run([sys.executable,'-c',PROBE],env=env,cwd=ROOT,
                            capture_output=True,text=True,timeout=90,check=True)
    probe = json.loads(next(line[6:] for line in result.stdout.splitlines() if line.startswith('PROBE=')))
    if probe['physical_die'] != args.physical_die or probe['logical_device'] != 0:
        raise RuntimeError('Unexpected physical device binding')
    from rag_engine.memory_probe import validate_memory_stats
    stats = validate_memory_stats(probe['stats'])
    target = stats['total_mb'] * args.memory_fraction
    if stats['free_mb'] - target < 5120 or stats['used_mb'] + target >= .85 * stats['total_mb']:
        raise RuntimeError('Insufficient measured device memory for engine allocation plus 5GiB safety')
    command = launch_command(args)
    record = {'preflight':'passed','command':command,'device':probe,
              'memory_source':'device-context.runtime.memory_stats',
              'generation_model':manifest,'run_requested':args.run,
              'scope':'standalone vLLM deployment; RAG routing and admission unchanged',
              'memory_fraction_note':'vLLM engine allocation target, not a per-request budget or hard runtime cap'}
    print(json.dumps({k:v for k,v in record.items() if k!='generation_model'}),flush=True)
    if not args.run:
        return
    out=ROOT/'experiment_outputs/checks/onerec_vllm'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    (out/'launch.json').write_text(json.dumps(record,indent=2))
    (out/'serve_generation_vllm.py').write_text(Path(__file__).read_text())
    print('OUTPUT='+str(out),flush=True)
    os.execve(sys.executable,command,env)


if __name__ == '__main__':
    main()
