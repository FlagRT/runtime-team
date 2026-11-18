"""OneRec HTTP generation with exact RAG prompt tokens and vLLM-owned KV allocation."""
from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .generation import GenerationLimits, PromptBuilder, checkpoint_manifest


class GenerationServiceError(RuntimeError):
    pass


def call_json(url, body=None, timeout=300):
    request = Request(url, data=json.dumps(body).encode() if body is not None else None,
                      headers={'Content-Type': 'application/json'})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except Exception as error:
        raise GenerationServiceError('OneRec vLLM request failed') from error


def verify_engine(model_path, url, physical_die, *, proc_root=Path('/proc')):
    """Only attach to the verified local service whose device we monitor."""
    parsed = urlparse(url)
    if parsed.hostname not in ('localhost', '127.0.0.1') or parsed.scheme != 'http':
        raise ValueError('Generation requires a local HTTP vLLM service')
    matches = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().rstrip('\0').split('\0')
            if 'vllm.entrypoints.openai.api_server' not in argv:
                continue
            def flag(name):
                return argv[argv.index(name)+1] if name in argv else None
            if flag('--port') != str(parsed.port or 80):
                continue
            env = dict(v.split('=', 1) for v in (path.parent/'environ').read_bytes().decode().rstrip('\0').split('\0'))
            if (Path(flag('--model')).resolve() != Path(model_path).resolve()
                    or flag('--dtype') != 'bfloat16' or flag('--served-model-name') != 'onerec-8b'
                    or env.get('ASCEND_RT_VISIBLE_DEVICES') != str(physical_die)
                    or env.get('RAG_ASCEND_PINNED_DIE') != str(physical_die)):
                raise ValueError('OneRec service model/dtype/device differs from monitored configuration')
            matches.append({'pid': int(path.parent.name), 'max_model_len': int(flag('--max-model-len')),
                            'max_num_seqs': int(flag('--max-num-seqs')), 'physical_die': physical_die})
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise ValueError('Expected one verified OneRec vLLM process')
    return matches[0]


class VLLMOneRecGenerator:
    backend = 'vllm'

    def __init__(self, model_path, *, url='http://127.0.0.1:8090', physical_die=6, limits=None):
        if physical_die not in (6, 7):
            raise ValueError('This deployment uses physical card 3, die 6 or 7')
        if str(physical_die) == os.getenv('RAG_ASCEND_PINNED_DIE'):
            raise ValueError('Remote generation must use its dedicated die, separate from reranking')
        self.model_path = Path(model_path)
        self.url, self.physical_die = url.rstrip('/'), physical_die
        self.device = 'remote-vllm'
        self.limits = limits or GenerationLimits()
        engine = verify_engine(self.model_path, self.url, physical_die)
        if engine['max_model_len'] < self.limits.max_input_tokens + self.limits.max_new_tokens:
            raise ValueError('Generation token limits exceed vLLM context length')
        self.manifest = checkpoint_manifest(self.model_path)
        models = call_json(self.url+'/v1/models', timeout=5)
        if 'onerec-8b' not in {model['id'] for model in models['data']}:
            raise ValueError('OneRec vLLM model unavailable')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
        self.builder = PromptBuilder(self.tokenizer, self.limits)
        self._tokenizer_lock = threading.Lock()
        # Compatibility metadata only: there is deliberately no estimate() here.
        self.memory_estimator = SimpleNamespace(fingerprint=self.manifest['config_sha256'])
        self.execution_profile = {'backend': 'vllm', 'model_path': str(self.model_path),
            'url': self.url, 'model_name': 'onerec-8b', 'dtype': 'bfloat16',
            'physical_die': physical_die, 'engine': engine,
            'max_input_tokens': self.limits.max_input_tokens, 'max_new_tokens': self.limits.max_new_tokens,
            'generation_reserved_mib': 0, 'allocation_policy': 'vllm_managed_kv_pool'}

    def configure_memory_guard(self, owner, safety_mib):
        # Per-device admission runs in the API event loop, not the HTTP worker.
        self.safety_mib = safety_mib

    def validate_request(self, query, max_new_tokens):
        if not 1 <= max_new_tokens <= self.limits.max_new_tokens:
            raise ValueError('max_new_tokens exceeds configured generation limit')
        with self._tokenizer_lock:
            self.builder.validate_query(query)

    def generate(self, query, hits, max_new_tokens=256):
        self.validate_request(query, max_new_tokens)
        with self._tokenizer_lock:
            prompt = self.builder.build(query, hits)
        count = len(prompt['input_ids'])
        # Exact locally assembled chat tokens: no second template or truncation.
        result = call_json(self.url+'/v1/completions', {'model': 'onerec-8b',
            'prompt': prompt['input_ids'], 'max_tokens': max_new_tokens, 'temperature': 0,
            'stream': False})
        choice, usage = result['choices'][0], result['usage']
        answer = choice['text'].strip()
        if not answer or usage['prompt_tokens'] != count or not 1 <= usage['completion_tokens'] <= max_new_tokens:
            raise GenerationServiceError('Invalid OneRec generation response or token accounting')
        if choice['finish_reason'] not in ('stop', 'length'):
            raise GenerationServiceError('Unexpected OneRec finish reason')
        return {'answer': answer, 'sources': prompt['sources'], 'context_truncated': prompt['context_truncated'],
            'usage': {'input_tokens': count, 'output_tokens': usage['completion_tokens'], 'max_new_tokens': max_new_tokens},
            'finish_reason': 'eos' if choice['finish_reason'] == 'stop' else 'length',
            'generation_model': str(self.model_path), 'generation_backend': 'vllm',
            'execution_memory_estimate': {'reserved_mib': 0, 'input_tokens': count,
                'max_new_tokens': max_new_tokens, 'policy': 'vllm_managed_kv_pool'}}
