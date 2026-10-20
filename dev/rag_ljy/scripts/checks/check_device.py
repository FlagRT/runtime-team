#!/usr/bin/env python3
"""Verify device binding and a real tensor result through the RAG adapter."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from _bootstrap import load_project_env

from rag_engine.device import device_selection, inference_dtype, prepare_torch_device


def main() -> None:
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto or a logical device such as npu:3")
    args = parser.parse_args()
    torch, device = prepare_torch_device(args.device)
    dtype = inference_dtype(torch, device)
    tensor = torch.ones(4, device=device, dtype=dtype)
    result = (tensor * 2).float().cpu().tolist()
    if result != [2.0] * 4:
        raise RuntimeError(f"Unexpected tensor result: {result}")
    backend_name, _, ordinal = device_selection(args.device)
    print(f"torch={torch.__version__} backend={backend_name or 'native'}")
    print(f"selected_device={device} dtype={dtype} result={result}")
    if backend_name:
        import runtime

        print(f"runtime={runtime.__file__}")
        print(f"device_count={runtime.device_count()}")
        print(f"memory={runtime.memory_stats(ordinal)}")
        if not runtime.probe_device(ordinal):
            raise RuntimeError("Runtime device probe failed")
    print("RAG_DEVICE_PASS")


if __name__ == "__main__":
    main()
