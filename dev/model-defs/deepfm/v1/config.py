"""Construction config for the DeepFM reference model (see model.py).

Plain dataclass, no framework/environment coupling — the values here are the
only thing a caller needs to change to point this reference model at their
own feature schema.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DeepFMConfig:
    # Vocab size of each sparse categorical field, in field order. Callers
    # must supply their own real schema (this has no meaningful default);
    # build.py's smoke test uses an illustrative placeholder.
    field_dims: List[int]
    embed_dim: int = 16
    mlp_dims: Tuple[int, ...] = (400, 400, 400)
    dropout: float = 0.2
