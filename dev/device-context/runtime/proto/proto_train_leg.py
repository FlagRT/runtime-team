"""训练腿自验证（V1）：锁定训练镜像 + 2 卡分布式微调。

经我们的统一运行时原型接入设备（runtime.use("flagos")），通信走 torch.distributed，
其底层由镜像注册为 FlagCX（public=flagos / inner=flagcx）。

验证点（对应验收标准 1 训练腿）：
  1. 两进程各自绑定到不同卡（rank → device），设备上下文可用
  2. 模型在 flagos 上加载，前向/反向可执行
  3. 梯度经集合通信同步（all_reduce 求平均），loss 正常下降
  4. 集合通信正确性对照：all_reduce / all_gather / P2P 各一组证据
  5. 全程无死锁、无数据错乱；记录吞吐
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, "/mnt/raid/hliu553/runtime-team/dev/device-context")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch_fl  # noqa: F401  —— 必须先于 torch 导入（镜像约束）
import runtime  # noqa: E402  —— 我们的统一运行时原型

MODEL = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"
STEPS = int(os.environ.get("MAX_STEPS", "20"))
BATCH = int(os.environ.get("BATCH", "4"))
SEQ = int(os.environ.get("SEQ", "128"))


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("flagos", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()

    res: dict = {"rank": rank, "world_size": world, "model": MODEL, "checks": {}}

    # ── 1. 经统一运行时 API 绑定设备 ──
    runtime.use("flagos")
    runtime.set_device(local_rank)
    dev_id = f"flagos:{local_rank}"
    n_dev = runtime.device_count()
    st = runtime.create_stream()
    res["checks"]["device_context"] = {
        "ok": n_dev >= 2 and st is not None,
        "detail": f"rank{rank} 绑定 {dev_id}；可见设备 {n_dev}；统一流 {st!r}"}

    # ── 2. 模型加载 ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, dtype=torch.float32).to(dev_id)
    model.train()
    LR = float(os.environ.get("LR", "1e-5"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8)
    res["checks"]["model_load"] = {"ok": True, "detail": f"加载到 {dev_id}"}

    # ── 3. 集合通信正确性对照（训练前）──
    # all_reduce：各 rank 贡献 rank+1，求和后应为 world*(world+1)/2
    t = torch.tensor([float(rank + 1)], device=dev_id)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expect_ar = float(world * (world + 1) / 2)
    ar_ok = abs(float(t.item()) - expect_ar) < 1e-6
    # all_gather：收集各 rank 的 rank 值
    g_in = torch.tensor([float(rank)], device=dev_id)
    g_out = [torch.zeros(1, device=dev_id) for _ in range(world)]
    dist.all_gather(g_out, g_in)
    g_vals = sorted(float(x.item()) for x in g_out)
    ag_ok = g_vals == [float(i) for i in range(world)]
    # P2P：rank0 <-> rank1 收发校验
    if world == 2:
        if rank == 0:
            send = torch.arange(8, dtype=torch.float32, device=dev_id)
            recv = torch.empty(8, dtype=torch.float32, device=dev_id)
            dist.send(send, dst=1)
            dist.recv(recv, src=1)
            expect_p2p = torch.arange(8, dtype=torch.float32, device=dev_id) * 2
        else:
            recv = torch.empty(8, dtype=torch.float32, device=dev_id)
            dist.recv(recv, src=0)
            send = torch.arange(8, dtype=torch.float32, device=dev_id) * 2
            dist.send(send, dst=0)
            expect_p2p = torch.arange(8, dtype=torch.float32, device=dev_id)
        p2p_ok = bool(torch.allclose(recv, expect_p2p))
        p2p_detail = f"rank{rank} 接收一致={p2p_ok}"
    else:
        p2p_ok, p2p_detail = True, "world!=2，跳过"
    res["checks"]["comm_all_reduce"] = {
        "ok": ar_ok, "detail": f"实测 {float(t.item()):.1f} / 期望 {expect_ar:.1f}"}
    res["checks"]["comm_all_gather"] = {"ok": ag_ok, "detail": f"收集到 {g_vals}"}
    res["checks"]["comm_p2p"] = {"ok": p2p_ok, "detail": p2p_detail}

    # ── 4. 微调循环（梯度经 all_reduce 同步）──
    torch.manual_seed(1234 + rank)
    losses = []
    t0 = time.time()
    tokens_done = 0
    for step in range(STEPS):
        # 合成数据：确定性地构造输入（两卡用同一 seed 保证一致性）
        g = torch.Generator().manual_seed(999 + step)
        ids = torch.randint(0, 30000, (BATCH, SEQ), generator=g).to(dev_id)
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度同步：all_reduce 求和后取平均（等价于 DDP 的梯度平均）
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = float(loss.item())
        losses.append(round(lv, 4))
        tokens_done += BATCH * SEQ
        if rank == 0 and (step % 5 == 0 or step == STEPS - 1):
            print(f"[step {step:3d}] loss={lv:.4f}", flush=True)
    torch.flagos.synchronize()
    dt = time.time() - t0

    first, last = losses[0], losses[-1]
    # 判据：整体下降（末值低于首值）且无 NaN
    decreased = last < first
    no_nan = all(not (l != l) for l in losses)
    res["checks"]["loss_decrease"] = {
        "ok": decreased and no_nan,
        "detail": f"首 {first:.4f} → 末 {last:.4f}（{STEPS} 步）；无 NaN={no_nan}"}
    res["perf"] = {
        "steps": STEPS,
        "batch": BATCH, "seq": SEQ,
        "tokens_per_s_per_rank": round(tokens_done / dt, 1),
        "tokens_per_s_total": round(tokens_done * world / dt, 1),
        "elapsed_s": round(dt, 2),
    }
    res["loss_curve"] = losses

    ok = all(v["ok"] for v in res["checks"].values())
    res["verdict"] = "TRAIN_LEG_PASS" if ok else "TRAIN_LEG_FAIL"
    res["passed"] = sum(1 for v in res["checks"].values() if v["ok"])
    res["total"] = len(res["checks"])

    out = f"/mnt/raid/hliu553/runtime-team/scratch/train_leg_result_rank{rank}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[rank{rank}] {res['verdict']} {res['passed']}/{res['total']} "
          f"| loss {first:.4f}→{last:.4f} | "
          f"{res['perf']['tokens_per_s_total']} tok/s(两卡合计) -> {out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
