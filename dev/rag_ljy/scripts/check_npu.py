#!/usr/bin/env python3
"""Verify that the active Python can execute a tensor operation on Ascend NPU."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if not args.device.startswith("npu:"):
        parser.error("--device must be an NPU device such as npu:0")
    import torch
    import torch_npu

    if not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but no Ascend NPU is available")
    device = torch.device(args.device)
    torch.npu.set_device(device)
    tensor = torch.ones(4, device=device)
    print(f"torch={torch.__version__}")
    print(f"torch_npu={torch_npu.__version__}")
    print(f"device_count={torch.npu.device_count()}")
    print(f"selected_device={device}")
    print(f"tensor={tensor}")


if __name__ == "__main__":
    main()
