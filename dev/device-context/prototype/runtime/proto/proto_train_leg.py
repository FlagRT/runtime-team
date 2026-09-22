"""训练腿自验证：2 卡分布式微调（**后端无关**）。

设备经统一运行时原型接入（`runtime.use(BACKEND)`），通信走 `torch.distributed`。

**后端无关化的由来（2026-09-14，昆仑芯接入）**
  原实现硬编码 `flagos` 后端、无条件 `import torch_fl` 与 910C 路径，只能在昇腾锁定镜像里跑。
  现改为**由环境变量驱动，默认值保持 910C 原行为** —— 于是同一份脚本可在两处运行，
  这正是「统一 API：换芯片只改一行」的体现（也正是《新芯片接入手册》要收录的用法）。

验证点（对应验收标准 1 训练腿）：
  1. 两进程各自绑定到不同卡（rank → device），设备上下文可用
  2. 模型加载，前向/反向可执行
  3. 梯度经集合通信同步（all_reduce 求平均），loss 正常下降
  4. 集合通信正确性对照：all_reduce / all_gather / P2P 各一组证据
  5. 全程无死锁、无数据错乱；记录吞吐

环境变量（括号内为默认值 = 910C 训练腿原口径）
  DC_BACKEND   运行时后端名（`flagos`；昆仑芯用 `kunlun`）
  DC_DIST_BT   进程组后端（`flagos`；昆仑芯见 --help 探测结果，如 `xccl`）
  DC_ROOT      device-context 根路径（`/mnt/raid/hliu553/runtime-team/dev/device-context`）
  DC_MODEL     模型路径（`/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`）
  DC_OUT_DIR   结果输出目录（`/mnt/raid/hliu553/runtime-team/scratch`）
  MAX_STEPS / BATCH / SEQ / LR   训练超参（20 / 4 / 128 / 1e-5）

用法（昆仑芯 P800）：
  # 用卡前先 `xpu-smi` 挑**空闲且连续同组**的卡；共享机上被他人占用的卡会触发设备侧报错
  CUDA_VISIBLE_DEVICES=6,7 DC_BACKEND=kunlun FLAGCX_ADAPTOR=klx \
  DC_ROOT=/workspace/prototype DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B \
  DC_OUT_DIR=/workspace/out \
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 proto_train_leg.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import timedelta

# ── 后端无关化：全部由环境变量驱动，默认值 = 910C 训练腿原口径 ──
BACKEND = os.environ.get("DC_BACKEND", "flagos")
#: 进程组后端默认映射 —— **注意同一集合通信库在不同芯片上注册的后端名不同**：
#:   910C：flagcx 被镜像注册为 `flagos`（单后端名即可）
#:   P800：flagcx 注册为 `flagcx`，且需显式 `import flagcx`；设备走 flagcx、CPU 走 gloo
#:   （P800 侧路径与 xliu969 已验证的 Route A 一致）
_DIST_BT_DEFAULT = {
    "flagos": "flagos",
    "kunlun": "cpu:gloo,cuda:flagcx",
}
DIST_BT = os.environ.get("DC_DIST_BT") or _DIST_BT_DEFAULT.get(BACKEND, "gloo")
ROOT = os.environ.get("DC_ROOT", "/mnt/raid/hliu553/runtime-team/dev/device-context")
MODEL = os.environ.get("DC_MODEL", "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
OUT_DIR = os.environ.get("DC_OUT_DIR", "/mnt/raid/hliu553/runtime-team/scratch")

sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

if "flagcx" in DIST_BT:
    # 显式导入才会把 `cuda:flagcx` 注册为 c10d 后端（P800 路线 A 的集合通信）
    import flagcx  # noqa: F401,E402

if BACKEND == "flagos":
    # 910C 训练镜像约束：torch_fl 与 torch_npu 运行时不可共存，需显式导入 torch_fl。
    # 昆仑芯（kunlun）后端不需要，故按后端条件导入。
    import torch_fl  # noqa: F401,E402

import runtime  # noqa: E402  —— 我们的统一运行时原型

STEPS = int(os.environ.get("MAX_STEPS", "20"))
BATCH = int(os.environ.get("BATCH", "4"))
SEQ = int(os.environ.get("SEQ", "128"))


def _preflight_env_check() -> None:
    """环境前提检查：把**已实测的上游坑**在开跑前明确告警，避免踩坑者误判。

    背景（实测，2026-09-14）：昆仑芯 P800 上 `XPU_EVENT_KL3_ENABLE=1` **且**存在设备侧
    集合通信时，设备事件同步会概率性**永久挂死**（18 次运行 16 次，≈89%）。
    自旋点在厂商 `libcuda.so`（实为 `libxpucuda.so`）内的
    `cudaEventRecordWithFlags` / `cudaDeviceSynchronize`；属厂商层缺陷，已上报。

    本方向训练腿（transformers 纯 torch）与推理腿（vLLM）**均不依赖 FlagGems**，
    故可在不设置该变量下进行 —— 但必须显式告知，不能悄悄改条件。
    """
    kl3 = os.environ.get("XPU_EVENT_KL3_ENABLE")
    if BACKEND == "kunlun" and kl3 == "1":
        print(
            "[preflight][WARN] BACKEND=kunlun 且 XPU_EVENT_KL3_ENABLE=1。\n"
            "  已知上游缺陷 KUNLUN-KL3-EVENT-SYNC-HANG：该组合下多进程设备集合通信\n"
            "  概率性永久挂死（实测 ≈89%；自旋于厂商 libcuda.so / libxpucuda.so）。\n"
            "  本方向训练腿/推理腿均不依赖 FlagGems，建议**不设置该变量**后重跑；\n"
            "  证据与最小复现：prototype/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md\n"
            "  若确需在开启该变量下取证据，请把本告警与结果一并记录，**不得作为通过依据**。",
            flush=True)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    _preflight_env_check()
    dist.init_process_group(DIST_BT, timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()

    res: dict = {
        "rank": rank, "world_size": world, "model": MODEL,
        "backend": BACKEND, "dist_backend": DIST_BT,
        "checks": {},
    }

    # ── 1. 经统一运行时 API 绑定设备 ──
    runtime.use(BACKEND)
    runtime.set_device(local_rank)
    res["device_type"] = runtime.current().device_type
    dev_id = f"{runtime.current().device_type}:{local_rank}"
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
    # 经统一 API 同步（后端无关；原为 torch.flagos.synchronize()）
    runtime.synchronize(local_rank)
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

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"train_leg_result_rank{rank}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[rank{rank}] {res['verdict']} {res['passed']}/{res['total']} "
          f"| backend={BACKEND} dist={DIST_BT} | loss {first:.4f}→{last:.4f} | "
          f"{res['perf']['tokens_per_s_total']} tok/s(两卡合计) -> {out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
