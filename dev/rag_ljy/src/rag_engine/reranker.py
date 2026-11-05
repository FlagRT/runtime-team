"""Qwen3 generative reranker implemented using its official yes/no scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from contextlib import nullcontext

from .device import inference_dtype, prepare_torch_device


class Qwen3Reranker:
    PREFIX = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        max_length: int = 8192,
        instruction: str = (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    ) -> None:
        self.model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reranker model is missing: {model_path}. "
                "Run scripts/data/download_models.py first."
            )

        torch, torch_device = prepare_torch_device(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        import os
        self.allocator_fraction = None
        if str(torch_device).split(':')[0] == 'npu' and 'garbage_collection_threshold:' in os.getenv('PYTORCH_NPU_ALLOC_CONF', ''):
            # Enables native unused-block GC; this sets a ceiling and allocates nothing.
            self.allocator_fraction = .99
            torch.npu.set_per_process_memory_fraction(self.allocator_fraction, torch_device)
        from .npu_cache import cache_reclaimer
        if str(torch_device).split(':')[0] == 'npu':
            self.empty_cache, self.cache_reclaim_backend = cache_reclaimer(torch.npu)
        else:
            self.empty_cache, self.cache_reclaim_backend = None, 'not_npu'
        self.torch = torch
        self.device = torch_device
        self.max_length = max_length
        import os
        self.profile_next_forward = os.getenv("RAG_PROFILE_FIRST_FORWARD") == "1"
        self.memory_guard = None
        self.document_token_limit = None
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), padding_side="left", local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=inference_dtype(torch, torch_device),
            local_files_only=True,
        ).to(torch_device)
        self.model.eval()

        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(
            self.PREFIX, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.SUFFIX, add_special_tokens=False
        )
        if self.max_length <= len(self.prefix_tokens) + len(self.suffix_tokens):
            raise ValueError("max_length is too small for the reranker prompt")

    def configure_memory_guard(self, estimator, safety_mib):
        from .forward_guard import ForwardMemoryGuard
        import runtime
        module = getattr(self.torch, self.device.type)
        for method in ('synchronize', 'memory_allocated', 'memory_reserved', 'reset_peak_memory_stats',
                       'max_memory_allocated', 'max_memory_reserved', 'empty_cache'):
            if not callable(getattr(module, method, None)):
                raise ValueError('Dynamic guard requires allocator instrumentation: '+method)
        self.profile_next_forward = True
        self.document_token_limit = estimator.document_token_limit
        self.memory_guard = ForwardMemoryGuard(
            estimator, lambda: runtime.memory_stats(self.device.index or 0),
            self.empty_cache or module.empty_cache, safety_mib=safety_mib, allocator=module)

    def reclaim_unused_memory(self):
        import runtime, logging, json
        module = getattr(self.torch, self.device.type)
        module.synchronize()
        before = runtime.memory_stats(self.device.index or 0)
        import time
        started = time.monotonic()
        if self.memory_guard is not None:
            self.memory_guard.estimator.cache_snapshot = {'reclaimable_mib':0, 'sampled_monotonic':float('inf')}
        (self.empty_cache or module.empty_cache)()
        duration_ms = (time.monotonic()-started)*1000
        if self.memory_guard is not None:
            self.memory_guard.estimator.cache_snapshot = self.memory_guard.cache_capacity()
        after = runtime.memory_stats(self.device.index or 0)
        logging.getLogger(__name__).info('event=idle_pressure_cache_reclaim readings=%s', json.dumps({'before':before,'after':after,'duration_ms':duration_ms,'backend':self.cache_reclaim_backend}))

    def _bounded_document(self, document):
        if self.document_token_limit is None:
            return document
        tokens = self.tokenizer.encode(document, add_special_tokens=False)
        if len(tokens) <= self.document_token_limit:
            return document
        import logging
        logging.getLogger(__name__).warning(
            'event=rerank_document_truncated original_tokens=%d limit=%d',
            len(tokens), self.document_token_limit)
        return self.tokenizer.decode(tokens[:self.document_token_limit], skip_special_tokens=False)

    def score(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 4,
    ) -> list[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        pairs = [self._format_pair(query, self._bounded_document(document)) for document in documents]
        scores: list[float] = []
        content_length = self.max_length - len(self.prefix_tokens) - len(
            self.suffix_tokens
        )

        # Tokenize on CPU once. A stable padding shape across this request avoids
        # accumulating separate large cached output blocks for every candidate batch.
        encoded_all = self.tokenizer(pairs, padding=False, truncation="longest_first",
                                     max_length=content_length, return_attention_mask=False)
        all_ids = [self.prefix_tokens + ids + self.suffix_tokens for ids in encoded_all['input_ids']]
        padded_length = max((len(ids) for ids in all_ids), default=0)
        start = 0
        while start < len(pairs):
            encoded = {'input_ids': all_ids[start:start + batch_size]}
            actual_batch = len(encoded['input_ids'])
            plan = None
            if self.memory_guard is not None:
                actual_batch, plan = self.memory_guard.choose_batch(
                    [padded_length] * actual_batch, actual_batch)
                encoded['input_ids'] = encoded['input_ids'][:actual_batch]
            inputs = self.tokenizer.pad(
                encoded,
                padding="max_length",
                max_length=padded_length,
                return_tensors="pt",
            )
            observation = self.memory_guard.observe_forward(plan) if self.memory_guard is not None else nullcontext()
            profiler = self.torch.profiler.profile(activities=[self.torch.profiler.ProfilerActivity.CPU]) if self.profile_next_forward else nullcontext()
            with observation, profiler as trace:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.inference_mode():
                    final_logits = self.model(**inputs).logits[:, -1, :].float()
                    binary_logits = self.torch.stack(
                        [final_logits[:, self.false_token_id], final_logits[:, self.true_token_id]], dim=1)
                    probabilities = self.torch.softmax(binary_logits, dim=1)[:, 1]
                scores.extend(probabilities.cpu().tolist())
                start += actual_batch
                # Release references before estimating the next model forward.
                del inputs, final_logits, binary_logits, probabilities
            if self.profile_next_forward:
                import logging, json
                names = sorted({e.key for e in trace.key_averages() if any(k in e.key.lower() for k in ('attention', 'flash', 'fusion'))})
                logging.getLogger(__name__).info('event=attention_operator_profile operators=%s', json.dumps(names))
                if self.memory_guard is not None:
                    self.memory_guard.estimator.verify_flash_operators(names)
                self.profile_next_forward = False

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        documents = [candidate["_source"]["text"] for candidate in candidates]
        scores = self.score(query, documents, batch_size=batch_size)
        ranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["_rerank_score"] = float(score)
            ranked.append(item)
        return sorted(
            ranked,
            key=lambda hit: (-hit["_rerank_score"], -hit["_rrf_score"]),
        )[:top_k]

    def _format_pair(self, query: str, document: str) -> str:
        return (
            f"<Instruct>: {self.instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
