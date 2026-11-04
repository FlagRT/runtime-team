import torch, torch_npu
print("torch", torch.__version__, "torch_npu", torch_npu.__version__)
print("device_count:", torch.npu.device_count())
print("is_available:", torch.npu.is_available())
