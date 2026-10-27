"""Qwen3 embeddings from a separately managed local vLLM pooling server."""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .embedding import Qwen3Embedder


class VllmQwen3Embedder(Qwen3Embedder):
    """Reuse query/document formatting, without loading a Transformers model.

    Engine placement, dtype and pooling are configured by serve_embeddings.py.
    Tokenize locally to preserve the existing truncation/special-token contract.
    """
    def __init__(self, model_path: Path, device='auto', max_length=8192,
                 output_dims=1024,
                 instruction='Given a web search query, retrieve relevant passages that answer the query',
                 *, url='http://127.0.0.1:8089/v1', model_name='rag-embedding', timeout=120):
        if not model_path.exists():
            raise FileNotFoundError(f'Embedding model is missing: {model_path}')
        if not 32 <= output_dims <= 1024 or max_length < 1 or timeout <= 0:
            raise ValueError('Invalid embedding dimensions, maximum length or timeout')
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self._tokenizer_lock = threading.Lock()
        self.max_length, self.output_dims, self.instruction = max_length, output_dims, instruction
        self.url, self.model_name, self.timeout = url.rstrip('/'), model_name, timeout
        self.device = 'remote-vllm'  # Never claim the HTTP engine uses the caller's device.
        self.backend = 'vllm'
        self.dtype = 'server-configured'

    def require_connection(self):
        with build_opener(ProxyHandler({})).open(self.url + '/models', timeout=self.timeout) as response:
            models = json.load(response)
        if self.model_name not in {item.get('id') for item in models.get('data', [])}:
            raise ValueError('Configured vLLM embedding model is not being served')

    def _request(self, payload):
        request = Request(self.url + '/embeddings', data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with build_opener(ProxyHandler({})).open(request, timeout=self.timeout) as response:
            return json.load(response)

    def _encode(self, texts, batch_size=8):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        vectors = []
        for start in range(0, len(texts), batch_size):
            # Fast tokenizers mutate padding/truncation settings per call. Protect
            # tokenization only; HTTP calls may overlap for vLLM batching.
            with self._tokenizer_lock:
                batch = self.tokenizer(texts[start:start + batch_size], truncation=True,
                                       max_length=self.max_length, padding=False)['input_ids']
            response = self._request({'model': self.model_name, 'input': batch,
                                      'encoding_format': 'float'})
            records = response.get('data', [])
            if len(records) != len(batch) or sorted(r.get('index', -1) for r in records) != list(range(len(batch))):
                raise ValueError('Invalid embedding response indices/count')
            for record in sorted(records, key=lambda r: r['index']):
                vector = record['embedding']
                if not isinstance(vector, list) or len(vector) != 1024:
                    raise ValueError('Expected Qwen3-Embedding-0.6B 1024-dimensional output')
                vector = [float(x) for x in vector[:self.output_dims]]
                norm = math.sqrt(sum(x*x for x in vector))
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError('Nonfinite or zero embedding')
                vectors.append([x / norm for x in vector])
        return vectors


def create_embedder(model_path, device='auto', **kwargs):
    """Use vLLM by default for document/query embeddings; no automatic fallback."""
    import os
    backend = os.getenv('RAG_EMBEDDING_BACKEND', 'vllm').strip().lower()
    if backend == 'transformers':
        return Qwen3Embedder(model_path, device=device, **kwargs)
    if backend == 'vllm':
        embedder = VllmQwen3Embedder(
            model_path, device=device, **kwargs,
            url=os.getenv('RAG_VLLM_EMBEDDING_URL', 'http://127.0.0.1:8089/v1'),
            model_name=os.getenv('RAG_VLLM_EMBEDDING_MODEL', 'rag-embedding'))
        embedder.require_connection()
        return embedder
    raise ValueError('RAG_EMBEDDING_BACKEND must be transformers or vllm')
