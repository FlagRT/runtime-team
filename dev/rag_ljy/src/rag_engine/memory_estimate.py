"""CPU-only, model/shape-aware incremental-memory estimates for RAG admission.

This is a conservative analytical estimate, not an empirically certified bound.
Resident weights/engine pools are already in measured device usage. Unknown
model/execution configurations fail closed instead of falling back to 2 GiB.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import threading
from time import monotonic
from urllib.parse import urlparse
from importlib.metadata import version, PackageNotFoundError

MIB = 1024 ** 2


def embedding_execution_profile(model_path, url, proc_root=Path('/proc')):
    """Verify this prototype's local vLLM deployment instead of assuming BF16."""
    endpoint = urlparse(url)
    if endpoint.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Remote embedding execution configuration is not verified')
    port = str(endpoint.port or 80)
    matches = []
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().decode().rstrip('\0').split('\0')
        except (OSError, UnicodeError):
            continue
        if 'vllm.entrypoints.openai.api_server' not in argv:
            continue
        def argument(name, default=None):
            if name in argv:
                index = argv.index(name)
                if index+1 >= len(argv):
                    raise ValueError('Incomplete embedding launch configuration')
                return argv[index+1]
            return next((v.split('=', 1)[1] for v in argv if v.startswith(name+'=')), default)
        if argument('--port', '8000') != port:
            continue
        if Path(argument('--model', '')).resolve() != Path(model_path).resolve():
            raise ValueError('Embedding engine model differs from the tokenizer/configuration')
        dtype = argument('--dtype')
        if dtype not in ('bfloat16', 'float16', 'half', 'float', 'float32'):
            raise ValueError('Embedding dtype must be explicit and supported')
        if argument('--max-model-len') != '8192' or '--enforce-eager' not in argv:
            raise ValueError('Embedding execution length/graph configuration requires revalidation')
        if argument('--quantization') is not None or argument('--kv-cache-dtype', 'auto') != 'auto':
            raise ValueError('Quantized embedding/cache memory estimator is not implemented')
        if argument('--runner') != 'pooling' or '--no-enable-prefix-caching' not in argv:
            raise ValueError('Embedding cache accounting requires verified pooling without prefix caching')
        matches.append({'runner': 'pooling', 'kv_cache': 'preallocated_in_device_usage', 'dtype': dtype, 'dtype_bytes': 4 if dtype in ('float', 'float32') else 2,
                        'max_model_len': 8192, 'enforce_eager': True,
                        'max_num_seqs': argument('--max-num-seqs'),
                        'memory_fraction': argument('--gpu-memory-utilization')})
    if len(matches) != 1:
        raise ValueError('Cannot uniquely verify local embedding engine configuration')
    return matches[0]


@dataclass(frozen=True)
class ModelShape:
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    dtype_bytes: int
    use_cache: bool

    @classmethod
    def from_config(cls, config, dtype_bytes=2):
        if config.get('model_type') != 'qwen3':
            raise ValueError('Memory estimation currently supports verified Qwen3 model shapes only')
        keys = ('hidden_size', 'intermediate_size', 'num_attention_heads',
                'num_key_value_heads', 'head_dim', 'num_hidden_layers', 'vocab_size')
        values = [config[k] for k in keys]
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
            raise ValueError('Invalid model shape')
        if dtype_bytes not in (2, 4):
            raise ValueError('Unsupported model dtype')
        return cls(*values, dtype_bytes, bool(config.get('use_cache', True)))


def forward_components(shape, tokens, batch, *, causal_logits):
    if not isinstance(tokens, int) or not isinstance(batch, int) or min(tokens, batch) < 1:
        raise ValueError('Positive integer sequence length and batch required')
    b, n, d = batch, tokens, shape.dtype_bytes
    # Dense/eager upper envelope, even for implementations using fused attention.
    # Matmul, mask addition, FP32 softmax and cast may coexist transiently.
    attention = b * shape.heads * n * n * (3*d + 4)
    mask = b * n * n * 4
    linear = b * n * (16*shape.hidden + 8*shape.intermediate + 4*shape.heads*shape.head_dim) * d
    kv = b * n * 2 * shape.kv_heads * shape.head_dim * shape.layers * d if shape.use_cache else 0
    logits = b * n * shape.vocab * d if causal_logits else b * n * shape.hidden * d
    return {'attention_mib': attention/MIB, 'mask_mib': mask/MIB,
            'activations_mib': linear/MIB, 'kv_cache_mib': kv/MIB, 'output_mib': logits/MIB}


class RequestMemoryEstimator:
    version = 'qwen3-execution-v3'

    def __init__(self, embedding_shape, reranker_shape, embedding_tokenizer,
                 reranker_tokenizer, instruction, *, max_length=8192,
                 document_token_limit=1024, factor=2.0, workspace_mib=512):
        if not math.isfinite(factor) or factor < 1 or not math.isfinite(workspace_mib) or workspace_mib < 0:
            raise ValueError('Invalid estimator allowances')
        if not 1 <= document_token_limit <= max_length or max_length != 8192:
            raise ValueError('Unsupported sequence/document limits')
        self.embedding_shape, self.reranker_shape = embedding_shape, reranker_shape
        self.embedding_tokenizer, self.reranker_tokenizer = embedding_tokenizer, reranker_tokenizer
        self.instruction = instruction
        self.max_length, self.document_token_limit = max_length, document_token_limit
        self.factor, self.workspace_mib = factor, workspace_mib
        self.lock = threading.Lock()
        self.fingerprint = None
        self.execution_profile = None
        self.fault = None
        self.flash_verified = False
        self.flash_candidate = False
        self.cache_snapshot = {"reclaimable_mib":0, "sampled_monotonic":float("inf")}
        self.generation_estimator = None

    @classmethod
    def from_pipeline(cls, pipeline):
        from transformers import AutoTokenizer
        embedder, reranker = pipeline.embedder, pipeline.reranker
        if getattr(embedder, 'backend', '') != 'vllm':
            raise ValueError('Dynamic estimator requires the supported vLLM embedding + Transformers reranker deployment')
        if embedder.max_length != 8192 or reranker.max_length != 8192:
            raise ValueError('Sequence limits changed; recalibrate memory estimator')
        configs, tokenizers, hashes = [], [], {}
        for name, obj in (('embedding', embedder), ('reranker', reranker)):
            directory = Path(obj.model_path)
            raw = (directory/'config.json').read_bytes()
            configs.append(json.loads(raw))
            tokenizers.append(AutoTokenizer.from_pretrained(str(directory), local_files_only=True))
            hashes[name] = hashlib.sha256(raw + (directory/'tokenizer.json').read_bytes()).hexdigest()
        dtype = str(reranker.model.dtype)
        if dtype not in ('torch.bfloat16', 'torch.float16', 'torch.float32'):
            raise ValueError('Unsupported reranker dtype')
        engine = embedding_execution_profile(embedder.model_path, embedder.url)
        estimator = cls(ModelShape.from_config(configs[0], engine['dtype_bytes']),
                        ModelShape.from_config(configs[1], 4 if dtype == 'torch.float32' else 2),
                        *tokenizers, embedder.instruction,
                        document_token_limit=int(__import__("os").getenv("RAG_RERANK_DOCUMENT_TOKENS", "1024")))
        packages = {}
        for package in ('torch', 'torch-npu', 'transformers', 'vllm', 'vllm-ascend'):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = 'not_installed_under_this_distribution_name'
        estimator.execution_profile = {'embedding': engine, 'reranker_dtype': dtype,
            'reranker_attention': getattr(reranker.model.config, '_attn_implementation', None),
            'document_token_limit': estimator.document_token_limit,
            'reranker_use_cache': configs[1].get('use_cache', True), 'reranker_logits_to_keep': 0,
            'packages': packages,
            'allocator_config': __import__('os').getenv('PYTORCH_NPU_ALLOC_CONF', ''),
            'allocator_fraction': getattr(reranker, 'allocator_fraction', None),
            'cache_reclaim_backend': getattr(reranker, 'cache_reclaim_backend', None)}
        generator = getattr(pipeline, 'generator', None)
        if generator is not None:
            if str(generator.device) != str(reranker.device):
                raise ValueError('Generation and reranking must share the monitored device')
            estimator.generation_estimator = generator.memory_estimator
            estimator.version = 'qwen3-execution-v4-generation'
            estimator.execution_profile['generation'] = generator.execution_profile
            hashes['generation'] = generator.memory_estimator.fingerprint
        hashes['execution'] = hashlib.sha256(json.dumps(estimator.execution_profile, sort_keys=True).encode()).hexdigest()
        estimator.fingerprint = hashes
        estimator.flash_candidate = (str(reranker.device).startswith('npu') and dtype in ('torch.bfloat16', 'torch.float16')
            and estimator.execution_profile['reranker_attention'] == 'sdpa' and not reranker.model.training)
        return estimator

    def verify_flash_operators(self, names):
        if not self.flash_candidate or 'aclnnFlashAttentionScore' not in names:
            return False
        self.flash_verified = True
        self.execution_profile['verified_reranker_operators'] = sorted(names)
        self.execution_profile['memory_model'] = 'fused_attention_lifetime_max_v3'
        self.execution_profile['padding_policy'] = 'request_max_length'
        self.execution_profile['cache_credit_policy'] = 'exclusive_idle_snapshot_with_reservation_promotion'
        self.fingerprint['execution'] = hashlib.sha256(json.dumps(self.execution_profile, sort_keys=True).encode()).hexdigest()
        return True

    def forward_estimate(self, tokens, batch, *, embedding=False):
        shape = self.embedding_shape if embedding else self.reranker_shape
        if self.flash_verified and embedding:
            # vLLM allocated its KV arena at startup; runtime.device usage already includes it.
            shape = replace(shape, use_cache=False)
        components = forward_components(shape, tokens, batch, causal_logits=not embedding)
        factor = self.factor
        if self.flash_verified:
            components['attention_mib'] = batch * tokens * shape.heads * shape.head_dim * 4 * shape.dtype_bytes / MIB
            components['mask_mib'] = batch * tokens * tokens * (shape.dtype_bytes + 1) / MIB if not embedding else 0
            # Layer temporaries and the final full-vocabulary projection do not coexist.
            # KV and causal masks survive across both phases. Keep workspace + 15% allowance (validated against recorded cold/warm peaks).
            peak = components['kv_cache_mib'] + components['mask_mib'] + max(
                components['output_mib'], components['activations_mib'] + components['attention_mib'])
            factor = 1.15
        else:
            peak = sum(components.values())
        estimate = math.ceil(peak * factor + self.workspace_mib)
        return {'estimated_mib': estimate, 'tokens': tokens, 'batch': batch,
                'components': components, 'peak_live_mib': peak, 'factor': factor, 'workspace_mib': self.workspace_mib,
                'attention_accounting': 'verified_fused' if self.flash_verified else 'dense_fallback'}

    def plan(self, request, available_mib, cache_credit_mib=0):
        """Batch is an execution upper limit; all candidates still get scored.

        Select a real executable batch before reserving its shape-specific budget.
        This does not discount the estimate or enqueue a refused request.
        """
        result = self.estimate(request)
        result['requested_reranker_batch_size'] = request.reranker_batch_size
        for batch in range(result['rerank']['batch'], 0, -1):
            rerank = self.forward_estimate(result['rerank']['tokens'], batch)
            growth = max(1024, math.ceil(.1 * rerank['estimated_mib'])) if self.flash_verified else 0
            credit = min(max(0, cache_credit_mib), rerank['estimated_mib'])
            result.update(rerank=rerank, allocator_growth_allowance_mib=growth, cache_credit_mib=credit,
                          planned_reranker_batch_size=batch,
                          estimated_incremental_mib=result['embedding']['estimated_mib'] + rerank['estimated_mib'] - credit + growth
                          + result.get('generation', {}).get('estimated_mib', 0))
            if result['estimated_incremental_mib'] <= available_mib:
                break
        return result

    def estimate(self, request):
        text = request.query
        # Dedicated tokenizers: never mutate the inference worker's tokenizer.
        with self.lock:
            embedding_tokens = len(self.embedding_tokenizer(
                f'Instruct: {self.instruction}\nQuery:{text}', truncation=True,
                max_length=self.max_length, padding=False)['input_ids'])
            query_tokens = len(self.reranker_tokenizer.encode(text, add_special_tokens=False))
            framing = len(self.reranker_tokenizer.encode(
                f'<Instruct>: {self.instruction}\n<Query>: \n<Document>:', add_special_tokens=False))
        # Actual reranker enforces the document-token ceiling before building pairs.
        # 128 also covers system prefix/suffix and BPE boundary effects; execution
        # checks use exact padded shapes before any input tensor moves to the NPU.
        rerank_tokens = min(self.max_length, query_tokens + framing + self.document_token_limit + 128)
        # /query embeds ONE query; embedding_batch_size=8 does not mean 8 queries.
        embedding = self.forward_estimate(max(1, embedding_tokens), 1, embedding=True)
        rerank_batch = min(request.reranker_batch_size, request.coarse_top_k)
        rerank = self.forward_estimate(max(1, rerank_tokens), rerank_batch)
        allocator_growth = max(1024, math.ceil(.1 * rerank['estimated_mib'])) if self.flash_verified else 0
        result = {'version': self.version, 'status': 'analytical_uncalibrated',
                'estimated_incremental_mib': embedding['estimated_mib'] + rerank['estimated_mib'] + allocator_growth,
                'allocator_growth_allowance_mib': allocator_growth,
                'embedding': embedding, 'rerank': rerank,
                'document_token_limit': self.document_token_limit, 'model_fingerprint': self.fingerprint,
                'execution_profile': self.execution_profile,
                'resident_weights': 'already included in measured device usage',
                'policy': 'stage peak envelopes plus allocator growth; exclusive verified idle-cache credit is promoted when released; not a hard memory cap'}
        if getattr(request, 'generate', False):
            if self.generation_estimator is None:
                raise ValueError('Generation is unavailable for this estimator')
            generation = self.generation_estimator.estimate(
                self.generation_estimator.limits.max_input_tokens, request.max_new_tokens)
            result['generation'] = generation
            result['estimated_incremental_mib'] += generation['estimated_mib']
        return result
