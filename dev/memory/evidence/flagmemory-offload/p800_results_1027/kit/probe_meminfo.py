import torch

f, t = torch.cuda.mem_get_info(0)
print("MEMGI free_GiB=%.2f total_GiB=%.2f" % (f / 2**30, t / 2**30), flush=True)
