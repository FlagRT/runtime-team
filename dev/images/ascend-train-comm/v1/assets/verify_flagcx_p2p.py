#!/usr/bin/env python3
"""Two-rank P2P canary proving ProcessGroupFlagOS selected FlagCX."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
import sys

import torch_fl  # noqa: F401 - Torch-FL must own PrivateUse1 before consumers
import flagcx
import torch
import torch.distributed as dist


PAYLOAD_ELEMENTS = (1, 4, 257, 65536)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def pattern(source_rank: int, iteration: int, elements: int) -> torch.Tensor:
    """Create exactly representable FP32 values on CPU for full receive checks."""
    base = source_rank * 512 + iteration * 256
    return (torch.arange(elements, dtype=torch.float32) % 251) + base


def resolve_flagcx_backend(device: torch.device) -> tuple[str, str]:
    """Resolve the registered c10d Backend and read its runtime-owned name."""
    default_group = dist.distributed_c10d._get_default_group()
    backend = default_group._get_backend(device)
    backend_type = f"{type(backend).__module__}.{type(backend).__name__}"
    names = []
    for attribute in ("name", "_get_backend_name"):
        method = getattr(backend, attribute, None)
        if not callable(method):
            continue
        try:
            value = method()
        except Exception:
            continue
        if value is not None:
            names.append(f"{attribute}={value}")
    backend_name = ",".join(names) or "unavailable"
    if "flagcx" not in backend_name.lower():
        raise RuntimeError(
            "flagos runtime backend did not identify as FlagCX: "
            f"backend_type={backend_type}, backend_name={backend_name}"
        )
    return backend_type, backend_name


def stream_handle(stream: object) -> int:
    value = getattr(stream, "cuda_stream", 0)
    return int(value or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream-mode", choices=("default", "nondefault"), default="default",
    )
    args = parser.parse_args()

    if os.environ.get("FLAGCX_TORCH_BACKEND") != "flagos":
        raise RuntimeError("FLAGCX_TORCH_BACKEND must be exactly 'flagos'")
    if os.environ.get("HCCL_WHITELIST_DISABLE") != "1":
        raise RuntimeError("HCCL_WHITELIST_DISABLE must be exactly '1'")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.flagos.set_device(local_rank)
    device = torch.device(f"flagos:{local_rank}")
    dist.init_process_group("flagos", timeout=timedelta(seconds=60))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError(
                f"FlagCX P2P canary requires exactly 2 ranks, got {world_size}"
            )
        public_name, inner_name = resolve_flagcx_backend(device)

        default_stream = torch.flagos.current_stream(local_rank)
        default_handle = stream_handle(default_stream)
        test_stream = None
        test_handle = default_handle
        if args.stream_mode == "nondefault":
            test_stream = torch.flagos.Stream(device=local_rank)
            test_handle = stream_handle(test_stream)
            if test_handle == default_handle:
                raise RuntimeError(
                    "nondefault stream unexpectedly reused the default handle"
                )

        payload_results = []
        for iteration, elements in enumerate(PAYLOAD_ELEMENTS):
            context = torch.flagos.stream(test_stream) if test_stream else nullcontext()
            with context:
                if test_stream is not None:
                    current_handle = stream_handle(
                        torch.flagos.current_stream(local_rank)
                    )
                    if current_handle != test_handle:
                        raise RuntimeError(
                            "nondefault stream was not current during communication: "
                            f"current={current_handle}, expected={test_handle}"
                        )
                if rank == 0:
                    outbound = pattern(0, iteration, elements).to(device)
                    inbound = torch.empty(elements, dtype=torch.float32, device=device)
                    dist.send(outbound, dst=1)
                    dist.recv(inbound, src=1)
                    expected = pattern(1, iteration, elements)
                else:
                    inbound = torch.empty(elements, dtype=torch.float32, device=device)
                    dist.recv(inbound, src=0)
                    expected = pattern(0, iteration, elements)
                    outbound = pattern(1, iteration, elements).to(device)
                    dist.send(outbound, dst=0)
            if test_stream is not None:
                test_stream.synchronize()
            torch.flagos.synchronize()
            observed = inbound.cpu()
            if not torch.equal(observed, expected):
                mismatch = int(torch.ne(observed, expected).sum().item())
                raise RuntimeError(
                    f"sentinel mismatch on rank {rank}, payload {elements}: "
                    f"mismatch_elements={mismatch}"
                )
            payload_results.append({
                "elements": elements,
                "bytes": elements * 4,
                "received_sha256": tensor_sha256(observed),
            })

        if test_stream is not None:
            restored = stream_handle(torch.flagos.current_stream(local_rank))
            if restored != default_handle:
                raise RuntimeError(
                    "stream context did not restore the default stream: "
                    f"restored={restored}, expected={default_handle}"
                )
        dist.barrier()
        print(json.dumps({
            "schema_version": 1,
            "status": "passed",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "public_backend": "flagos",
            "public_process_group": public_name,
            "inner_backend": inner_name,
            "stream_mode": args.stream_mode,
            "default_stream_handle": default_handle,
            "test_stream_handle": test_handle,
            "payloads": payload_results,
        }, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
