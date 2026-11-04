import torch, flag_gems
for trial in range(3):
    a = torch.randn((64,64), dtype=torch.float32, device=flag_gems.device)
    b = torch.randn((64,64), dtype=torch.float32, device=flag_gems.device)
    ref = torch.mm(a.float().cpu(), b.float().cpu())
    with flag_gems.use_gems():
        out = torch.mm(a,b)
    d = (out.float().cpu()-ref).abs().max().item()
    print(f"trial{trial}: max_abs_diff={d:.4e} rel={(d/ref.abs().max().item()):.4e}")
    if trial==0:
        print("  ref[0,:4] =", [round(x,4) for x in ref[0,:4].tolist()])
        print("  out[0,:4] =", [round(x,4) for x in out.float().cpu()[0,:4].tolist()])
        # 对照：vendor 原生 torch.mm（不进 use_gems）
        a2=a.clone(); b2=b.clone()
        native = torch.mm(a2,b2).float().cpu()
        print("  native[0,:4] =", [round(x,4) for x in native[0,:4].tolist()], " maxdiff_vs_ref=", (native-ref).abs().max().item())
