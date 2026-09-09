"""Qwen3 embedding inference with explicit Ascend NPU placement."""

from __future__ import annotations

from pathlib import Path

from .npu import inference_dtype, prepare_torch_device


class Qwen3Embedder:
    def __init__(
        self,
        model_path: Path,
        device: str = "npu:0",
        max_length: int = 8192,
        output_dims: int = 1024,
        instruction: str = (
            "Given a web search query, retrieve relevant passages that answer the query"
        ),
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding model is missing: {model_path}. "
                "Run scripts/download_models.py first."
            )
        if output_dims < 32 or output_dims > 1024:
            raise ValueError("Qwen3-Embedding-0.6B output_dims must be 32..1024")

        torch, torch_device = prepare_torch_device(device)
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.functional = torch.nn.functional
        self.device = torch_device
        self.max_length = max_length
        self.output_dims = output_dims
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), padding_side="left", local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path),
            torch_dtype=inference_dtype(torch, device),
            local_files_only=True,
        ).to(torch_device)
        self.model.eval()

    def encode_queries(
        self, queries: list[str], batch_size: int = 8
    ) -> list[list[float]]:
        instructed = [
            f"Instruct: {self.instruction}\nQuery:{query}" for query in queries
        ]
        return self._encode(instructed, batch_size)

    def encode_documents(
        self, documents: list[str], batch_size: int = 8
    ) -> list[list[float]]:
        return self._encode(documents, batch_size)

    def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with self.torch.inference_mode():
                output = self.model(**batch)
                embeddings = self._last_token_pool(
                    output.last_hidden_state, batch["attention_mask"]
                )
                embeddings = embeddings[:, : self.output_dims]
                embeddings = self.functional.normalize(embeddings, p=2, dim=1)
            vectors.extend(embeddings.float().cpu().tolist())
        return vectors

    def _last_token_pool(self, hidden_states, attention_mask):
        left_padded = bool(
            (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
        )
        if left_padded:
            return hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        indices = self.torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[indices, sequence_lengths]
