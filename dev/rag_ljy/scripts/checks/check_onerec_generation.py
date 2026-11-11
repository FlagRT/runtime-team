#!/usr/bin/env python3
"""One real guarded OneRec generation. Default validates files on CPU only.

This smoke check is not a RAG/database or 85% watermark acceptance test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _bootstrap import load_project_env

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import fcntl
import json
import logging
import os
import shutil
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


async def run(args, out):
    from rag_engine.generation import GenerationLimits, OneRecGenerator
    from rag_engine.memory_probe import DeviceMemoryMonitor
    from rag_engine.admission import MemoryAdmissionConfig, MemoryAdmissionController
    monitor = DeviceMemoryMonitor(args.device, interval_s=.1)
    rows = []
    monitor.on_sample = lambda row: rows.append({'monotonic': time.monotonic(), **row})
    monitor.start()
    owner = SimpleNamespace(fault=None, cache_snapshot={})
    executor = ThreadPoolExecutor(max_workers=1)
    generator = None
    try:
        loop = asyncio.get_running_loop()
        def load():
            return OneRecGenerator(args.model, device=args.device,
                limits=GenerationLimits(args.max_input_tokens, args.max_new_tokens))
        generator = await loop.run_in_executor(executor, load)
        generator.configure_memory_guard(owner, 5120)
        # Wait for a sample started after weights were loaded, so they are part
        # of measured usage rather than an omitted resident cost.
        loaded = time.monotonic()
        for _ in range(300):
            sample = monitor.snapshot()
            if sample.get('status') == 'ok' and sample.get('sample_started_monotonic', 0) >= loaded:
                break
            await asyncio.sleep(.1)
        controller = MemoryAdmissionController(MemoryAdmissionConfig(enabled=True,
            estimate_requests=True, allow_unverified_scope=args.allow_unverified_scope))
        estimate = generator.memory_estimator.estimate(args.max_input_tokens, args.max_new_tokens)
        decision = controller.reserve('onerec-smoke', sample, request_budget_mib=estimate['estimated_mib'])
        (out/'admission.json').write_text(json.dumps({'decision':decision, 'estimate':estimate,'sample':sample}, indent=2))
        if not decision['accepted']:
            raise RuntimeError('Single generation rejected by memory admission: '+decision['reason'])
        try:
            result = await loop.run_in_executor(executor, generator.generate,
                args.query, args.sources, args.max_new_tokens)
        finally:
            controller.release('onerec-smoke')
        if owner.fault:
            raise RuntimeError(owner.fault)
        (out/'response.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return {'result':'passed', 'scope':'standalone real generation; not RAG or watermark acceptance',
                'execution_profile':generator.execution_profile, 'usage':result['usage'],
                'memory_estimator_fault':owner.fault,
                'peak_sampled_used_percent':max((r['stats']['memory_used_percent'] for r in rows if r.get('stats')), default=None)}
    finally:
        await monitor.close()
        executor.shutdown(wait=True)
        (out/'memory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--model', type=Path, default=Path('/mnt/onerec/OneRec-8B'))
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--max-input-tokens', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--allow-unverified-scope', action='store_true')
    args = parser.parse_args()
    load_project_env()
    from rag_engine.generation import checkpoint_manifest, GenerationLimits
    GenerationLimits(args.max_input_tokens, args.max_new_tokens)
    manifest = checkpoint_manifest(args.model)
    if not args.run:
        print(json.dumps({'preflight':'passed', 'model':manifest, 'run_requested':False}, indent=2))
        return 0
    lock = open('/tmp/rag-watermarks-modes.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if argv and Path(argv[0]).name.startswith('python') and any(Path(a).name=='serve_queries.py' for a in argv):
            raise RuntimeError('A query API is running; use check_generation_live.py instead of loading a second OneRec')
    # Exact supplied evidence makes an unsupported/free-form answer detectable.
    args.query = 'Which planet is the largest in the Solar System? Give a short answer with a source citation.'
    args.sources = [{'document_id':'smoke-planet','chunk_id':'smoke-planet-1',
        'title':'Solar System facts','source_uri':'test-fixture://solar-system',
        'text':'Jupiter is the largest planet in the Solar System. Earth is the third planet from the Sun.'}]
    out = ROOT/'experiment_outputs/checks/onerec_generation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out), flush=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(out/'run.log')])
    os.environ.setdefault('RAG_MEMORY_PROBE_PROCESS','1')
    files = [Path(__file__), *(ROOT/'src/rag_engine'/name for name in (
        'generation.py','forward_guard.py','memory_estimate.py','admission.py','memory_probe.py'))]
    for path in files:
        shutil.copyfile(path, out/path.name)
    config = {'argv':sys.argv, 'input':vars(args), 'manifest':manifest,
              'visible_device_nodes':[str(p) for p in Path('/dev').glob('davinci[0-9]*')],
              'packages':{name:importlib.metadata.version(name) for name in ('torch','torch-npu','transformers')},
              'device_memory_source':'device-context.runtime.memory_stats',
              'pressure_source':'real model execution only; one supplied-source generation request'}
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str))
    try:
        result = asyncio.run(run(args, out))
        response = json.loads((out/'response.json').read_text())
        if 'jupiter' not in response['answer'].lower() or '[1]' not in response['answer']:
            result.update(result='failed', error='Output did not state the supplied fact with its citation')
    except Exception as error:
        logging.exception('OneRec smoke failed')
        result = {'result':'failed', 'error':str(error), 'error_type':type(error).__name__}
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print('RESULT='+result['result'], flush=True)
    return 0 if result['result']=='passed' else 1


if __name__ == '__main__':
    sys.exit(main())
