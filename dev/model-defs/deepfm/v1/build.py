"""Build the DeepFM reference model and run a device-agnostic smoke test.

Usage:
    python build.py

No network, no dataset, no accelerator-specific import required — this only
needs a plain `pip install torch` (see requirements.txt) and runs unmodified
on CPU or whatever accelerator the local torch build already exposes.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from config import DeepFMConfig
from model import DeepFM

# Illustrative only: 5 sparse fields with these vocab sizes. Callers should
# replace this with their own real feature schema.
DEFAULT_CONFIG = DeepFMConfig(
    field_dims=[1000, 2000, 500, 300, 50],
    embed_dim=16,
    mlp_dims=(400, 400, 400),
    dropout=0.2,
)


def build_model(config: DeepFMConfig = DEFAULT_CONFIG) -> DeepFM:
    return DeepFM(
        field_dims=config.field_dims,
        embed_dim=config.embed_dim,
        mlp_dims=config.mlp_dims,
        dropout=config.dropout,
    )


def pick_device() -> torch.device:
    """Best-effort device selection with zero vendor-specific imports.

    Only probes attributes that an already-imported torch build exposes
    (e.g. `torch_npu` monkeypatches `torch.npu` in when the caller imports
    it first). This file never imports a vendor package itself, so it stays
    usable on any machine regardless of which accelerator SDK is installed.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_test(
    config: DeepFMConfig = DEFAULT_CONFIG, batch_size: int = 32, steps: int = 3
) -> None:
    """Construct the model, then run a few random-tensor forward+backward
    steps to confirm it builds, trains, and never produces NaN/Inf.

    This is a usability check, not a numerical-accuracy validation — it uses
    random labels, so the loss trajectory is not meaningful. Validating
    accuracy on a real dataset is a separate, follow-up exercise for whoever
    adopts this reference model against their own feature schema.
    """
    device = pick_device()
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"[smoke_test] device={device} field_dims={config.field_dims} "
        f"embed_dim={config.embed_dim} mlp_dims={config.mlp_dims} params={num_params}"
    )

    model.train()
    for step in range(steps):
        x = torch.stack(
            [torch.randint(0, dim, (batch_size,)) for dim in config.field_dims],
            dim=1,
        ).to(device)
        y = torch.randint(0, 2, (batch_size,)).float().to(device)

        optimizer.zero_grad()
        pred = model(x)
        assert pred.shape == (batch_size,), f"unexpected output shape {tuple(pred.shape)}"
        assert torch.isfinite(pred).all(), "non-finite values in prediction"
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        print(f"[smoke_test] step={step} loss={loss.item():.4f}")

    print("[smoke_test] PASS")


if __name__ == "__main__":
    smoke_test()
