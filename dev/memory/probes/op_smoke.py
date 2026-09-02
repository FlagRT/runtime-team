#!/usr/bin/env python3
"""单算子冒烟:flag_gems rms_norm / rotary_embedding 在 flagos 上是否可执行(不卡)。"""
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402
import torch_fl  # noqa: E402
import flagos_boot  # noqa: E402
import flag_gems  # noqa: E402


def main():
    dev = "flagos:0"
    print("flag_gems.device =", flag_gems.device, flush=True)
    x = torch.randn(2, 32, 128, dtype=torch.bfloat16, device=dev)
    print("x dev:", x.device, flush=True)

    # rms_norm
    from flag_gems import rms_norm as rms_norm_fn
    w = torch.randn(128, dtype=torch.bfloat16, device=dev)
    t0 = time.time()
    y = rms_norm_fn(x, (128,), w, 1e-6)
    torch.npu.synchronize()
    print(f"rms_norm OK {time.time()-t0:.2f}s shape={tuple(y.shape)}", flush=True)

    # rotary_embedding(如果可导入)
    try:
        from flag_gems import rotary_embedding as rot_fn  # noqa: F401
        print("rotary_embedding importable", flush=True)
    except Exception as e:
        print(f"rotary_embedding import skip: {type(e).__name__}: {str(e)[:100]}", flush=True)


if __name__ == "__main__":
    main()
