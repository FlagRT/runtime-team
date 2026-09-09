#!/usr/bin/env python3
"""Verify that the active Python can execute a tensor operation on Ascend NPU."""

import torch
import torch_npu


def main() -> None:
    tensor = torch.ones(4, device="npu:0")
    print(f"torch={torch.__version__}")
    print(f"torch_npu={torch_npu.__version__}")
    print(f"device_count={torch.npu.device_count()}")
    print(f"tensor={tensor}")


if __name__ == "__main__":
    main()
