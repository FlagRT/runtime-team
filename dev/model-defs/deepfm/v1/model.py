"""Standard DeepFM (Guo et al., 2017) reference implementation.

Vendored and lightly modernized from the mainstream, widely-used open-source
reference https://github.com/rixwew/pytorch-fm (MIT License, Copyright (c)
2019 rixwew; see ./LICENSE in this directory) — `DeepFactorizationMachineModel`
and its `FeaturesLinear` / `FeaturesEmbedding` / `FactorizationMachine` /
`MultiLayerPerceptron` building blocks in `torchfm/model/dfm.py` and
`torchfm/layer.py`. Only the pieces needed for DeepFM are kept (the upstream
repo also has xDeepFM/AFM/DCN/PNN/etc., out of scope here).

Modernization vs. upstream (behavior-preserving):
- Field offsets are a registered buffer (`register_buffer`) instead of a
  NumPy array recomputed with `x.new_tensor(...)` on every forward call —
  same math, moves with the module across devices/dtypes for free.
- Dropped the `numpy` dependency entirely (upstream used `np.long`, which
  NumPy >= 1.24 removed) — one less version-sensitive dependency.

Design constraint (deliberate): this file only uses stable `torch.nn`
primitives (Embedding / Linear / BatchNorm1d / Dropout / Sequential). It does
not hardcode a device, dtype, or import any accelerator-specific package
(e.g. `torch_npu`). That is a caller concern — see `build.py:pick_device()`.
This lets the same file be copied as-is into any chip/runtime container and
just work, which is the point of keeping it here as a cross-team reference.
"""

from typing import Sequence

import torch
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """Order-1 (wide/linear) term: per-field 1-dim embedding lookup + bias.

    :param field_dims: vocab size of each sparse field, e.g. ``[1000, 2000, 50]``.
    :param output_dim: output width (1 for a binary CTR-style logit).
    """

    def __init__(self, field_dims: Sequence[int], output_dim: int = 1):
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        offsets = [0]
        for dim in field_dims[:-1]:
            offsets.append(offsets[-1] + dim)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Long tensor of shape ``(batch_size, num_fields)``."""
        x = x + self.offsets.unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module):
    """Shared k-dim embedding table for both the FM and the deep component.

    :param field_dims: vocab size of each sparse field.
    :param embed_dim: latent factor dimension ``k``.
    """

    def __init__(self, field_dims: Sequence[int], embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        offsets = [0]
        for dim in field_dims[:-1]:
            offsets.append(offsets[-1] + dim)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Long tensor of shape ``(batch_size, num_fields)``.
        :returns: Float tensor of shape ``(batch_size, num_fields, embed_dim)``.
        """
        x = x + self.offsets.unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(nn.Module):
    """Order-2 term: ``0.5 * sum((sum_i v_i)^2 - sum_i v_i^2)``."""

    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Float tensor of shape ``(batch_size, num_fields, embed_dim)``."""
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(nn.Module):
    """Deep component: stacked Linear -> BatchNorm1d -> ReLU -> Dropout blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        output_layer: bool = True,
    ):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Float tensor of shape ``(batch_size, input_dim)``."""
        return self.mlp(x)


class DeepFM(nn.Module):
    """DeepFM: wide (linear) + FM (order-2) + deep (MLP), sharing one embedding table.

    Reference: H. Guo et al., "DeepFM: A Factorization-Machine based Neural
    Network for CTR Prediction", IJCAI 2017.

    :param field_dims: vocab size of each sparse categorical field. Each input
        column ``x[:, i]`` must already be 0-indexed within field ``i``
        (i.e. in ``[0, field_dims[i])``) — the module handles the
        cross-field offset internally.
    :param embed_dim: latent factor dimension shared by FM and deep component.
    :param mlp_dims: hidden layer widths of the deep component.
    :param dropout: dropout probability inside the deep component.
    """

    def __init__(
        self,
        field_dims: Sequence[int],
        embed_dim: int = 16,
        mlp_dims: Sequence[int] = (400, 400, 400),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: Long tensor of shape ``(batch_size, num_fields)``.
        :returns: Float tensor of shape ``(batch_size,)``, sigmoid CTR probability.
        """
        embed_x = self.embedding(x)
        logit = (
            self.linear(x)
            + self.fm(embed_x)
            + self.mlp(embed_x.view(-1, self.embed_output_dim))
        )
        return torch.sigmoid(logit.squeeze(1))
