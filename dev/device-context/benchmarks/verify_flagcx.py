import torch, torch_fl, flagcx
import torch.distributed as dist
print("flagcx file:", flagcx.__file__)
print("flagcx backend registered:", "flagcx" in dist.Backend.backend_capability)
