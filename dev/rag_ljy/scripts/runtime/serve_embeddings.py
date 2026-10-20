#!/usr/bin/env python3
"""Start the default vLLM embedding engine on this container's visible NPU.

Default endpoint: http://127.0.0.1:8089/v1/embeddings
The current container maps physical NPU 7 to logical 0. RAG_DEVICE does not
control this separate process; ASCEND_RT_VISIBLE_DEVICES selects its NPU.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import os
from pathlib import Path
import sys

from _bootstrap import load_project_env


def main():
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(os.getenv('EMBEDDING_MODEL_PATH', '/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B')))
    parser.add_argument('--port', type=int, default=8089)
    parser.add_argument('--memory-fraction', type=float, default=0.15)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--synchronous', action='store_true',
                        help='diagnostic workaround: set ASCEND_LAUNCH_BLOCKING=1; may reduce throughput')
    args = parser.parse_args()
    if not args.model.is_dir() or not 0 < args.memory_fraction < 1 or args.device_index < 0:
        parser.error('Require a local model directory, memory fraction in (0,1), and nonnegative device index')
    if args.synchronous:
        os.environ['ASCEND_LAUNCH_BLOCKING'] = '1'
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(args.device_index)
    # This image's Triton uses _npu_getCurrentRawStreamNoWait. Disabling its
    # task queue bypasses queued torch_npu producers (observed at Q/K -> RoPE),
    # yielding different embeddings for identical input. Override the shared
    # Compose base's legacy TRITON_ENABLE_TASKQUEUE=false for this engine only.
    os.environ['TRITON_ENABLE_TASKQUEUE'] = 'true'
    os.environ['DO_NOT_TRACK'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    # The container's runtime PYTHONPATH override omits CANN's ACL bindings.
    cann_python = Path('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
    if cann_python.is_dir():
        os.environ['PYTHONPATH'] = str(cann_python) + os.pathsep + os.environ.get('PYTHONPATH', '')
    cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', str(args.model), '--served-model-name', 'rag-embedding',
           '--runner', 'pooling', '--dtype', 'bfloat16', '--enforce-eager',
           '--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--block-size', '128',
           '--pooler-config', '{"pooling_type":"LAST","use_activation":true}',
           '--gpu-memory-utilization', str(args.memory_fraction), '--max-model-len', '8192',
           '--max-num-seqs', '8', '--host', '127.0.0.1', '--port', str(args.port)]
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
