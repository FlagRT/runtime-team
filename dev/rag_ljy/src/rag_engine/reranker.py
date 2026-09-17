"""Qwen3 generative reranker implemented using its official yes/no scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .npu import inference_dtype, prepare_torch_device


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
        device: str = "npu:0",
        max_length: int = 8192,
        instruction: str = (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Reranker model is missing: {model_path}. "
                "Run scripts/download_models.py first."
            )

        torch, torch_device = prepare_torch_device(device)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch_device
        self.max_length = max_length
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), padding_side="left", local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=inference_dtype(torch, device),
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

    def score(
        self,
        query: str,
        documents: list[str],
        batch_size: int = 4,
    ) -> list[float]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        pairs = [self._format_pair(query, document) for document in documents]
        scores: list[float] = []
        content_length = self.max_length - len(self.prefix_tokens) - len(
            self.suffix_tokens
        )

        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                batch_pairs,
                padding=False,
                truncation="longest_first",
                max_length=content_length,
                return_attention_mask=False,
            )
            encoded["input_ids"] = [
                self.prefix_tokens + token_ids + self.suffix_tokens
                for token_ids in encoded["input_ids"]
            ]
            inputs = self.tokenizer.pad(
                encoded,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with self.torch.inference_mode():
                final_logits = self.model(**inputs).logits[:, -1, :].float()
                binary_logits = self.torch.stack(
                    [
                        final_logits[:, self.false_token_id],
                        final_logits[:, self.true_token_id],
                    ],
                    dim=1,
                )
                probabilities = self.torch.softmax(binary_logits, dim=1)[:, 1]
            scores.extend(probabilities.cpu().tolist())
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
