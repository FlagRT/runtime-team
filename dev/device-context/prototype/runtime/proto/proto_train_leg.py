"""训练腿自验证：2 卡分布式微调（**后端无关**）。

设备经统一运行时原型接入（`runtime.use(BACKEND)`），通信走 `torch.distributed`。

**后端无关化的由来（2026-09-14，昆仑芯接入）**
  原实现硬编码 910C 训练腿的设备后端与容器路径（路线 B 时期），只能在昇腾锁定镜像里跑。
  现改为**由环境变量驱动，默认值保持 910C 当前口径** —— 于是同一份脚本可在三处运行，
  这正是「统一 API：换芯片只改一行」的体现（也正是《新芯片接入手册》要收录的用法）。

验证点（对应验收标准 1 训练腿）：
  1. 两进程各自绑定到不同卡（rank → device），设备上下文可用
  2. 模型加载，前向/反向可执行
  3. 梯度经集合通信同步（all_reduce 求平均），loss 正常下降
  4. 集合通信正确性对照：all_reduce / all_gather / P2P 各一组证据
  5. 全程无死锁、无数据错乱；记录吞吐

环境变量（括号内为默认值 = **910C 训练腿当前口径**）
  DC_BACKEND   运行时后端名（`ascend` = torch_npu；昆仑芯用 `kunlun`；寒武纪用 `cambricon`）
  DC_DIST_BT   进程组后端（910C=`hccl`；昆仑芯见 --help 探测结果，如 `xccl`；寒武纪必须显式给）
  DC_ROOT      device-context 根路径（`/mnt/raid/hliu553/runtime-team/dev/device-context`）
  DC_MODEL     模型路径（`/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`）
  DC_OUT_DIR   结果输出目录（`/mnt/raid/hliu553/runtime-team/scratch`）
  MAX_STEPS / BATCH / SEQ / LR   训练超参（20 / 4 / 128 / 1e-5）

⚠️ **910C 口径变更（2026-09-22）**：全组统一基座定的是 Qwen3-0.6B **训推**、
且原型接入走**厂商 torch 插件路线** ⇒ 910C 两条腿**统一 `torch_npu`**
（`DC_BACKEND=ascend`、`DC_DIST_BT=hccl`）。
原训练腿走路线 B（torch_fl）只是被锁定训练镜像"禁止两个插件同进程共存"逼出来的权宜例外；
**该例外已取消，且路线 B 后端已从原型删除** —— 本脚本不再有该路径的任何分支。

用法（910C，训推统一 torch_npu）：
  # 解释器必须是有 torch_npu 的那个（容器内：/mnt/raid/hliu553/venvs/venv-infer-a/bin/python）
  # 该解释器内**没有其他占用 PrivateUse1 的插件**（物理隔离，不触发镜像的共存校验）
  ASCEND_RT_VISIBLE_DEVICES=0,1 DC_BACKEND=ascend DC_DIST_BT=hccl \
  DC_ROOT=/mnt/raid/hliu553/runtime-team/dev/device-context/prototype \
  DC_MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
  python -m torch.distributed.run --standalone --nproc_per_node=2 proto_train_leg.py

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

# ── 后端无关化：全部由环境变量驱动，默认值 = 910C 训练腿当前口径 ──
BACKEND = os.environ.get("DC_BACKEND", "ascend")   # 2026-09-22：910C 训推统一 torch_npu
#: 进程组后端默认映射 —— **注意同一集合通信库在不同芯片上注册的后端名不同**：
#:   910C：统一 torch_npu 后走**原生 HCCL**（`hccl`）
#:   P800：flagcx 注册为 `flagcx`，且需显式 `import flagcx`；设备走 flagcx、CPU 走 gloo
#:   （P800 侧路径与 xliu969 已验证的 Route A 一致）
#:   MLU590：**刻意不给默认值**（见下方 cambricon 分支）—— 后端名必须实测后显式指定
_DIST_BT_DEFAULT = {
    # 910C 统一 torch_npu（2026-09-22）⇒ 集合通信走 torch_npu 原生 **HCCL**。
    # 实测可用后端（torch_npu 2.11.0）：gloo / nccl / xccl / ucc / mpi / **hccl** / lccl
    "ascend": "hccl",
    "kunlun": "cpu:gloo,cuda:flagcx",
}
DIST_BT = os.environ.get("DC_DIST_BT") or _DIST_BT_DEFAULT.get(BACKEND, "gloo")

# ⚠️ 寒武纪（cambricon）：**刻意不给默认值，且不给就报错退出**。
#    理由（手册 §9 坑 3「同一集合通信库在不同芯片注册的后端名不同」）：
#      三家芯片各自注册的后端名互不相同（910C=`hccl`、P800=`flagcx`、MLU590 待实测）
#      ⇒ **不可类推**；
#      而 `_DIST_BT_DEFAULT.get(BACKEND, "gloo")` 的兜底是 `gloo`，那会**静默退化为纯 CPU
#      集合通信**：训练脚本照样跑完、loss 照样下降，但**设备侧集合通信根本没被验证**
#      —— 这类"看起来通过"的结果比失败更糟，故此处显式拦住。
if BACKEND == "cambricon" and not os.environ.get("DC_DIST_BT"):
    print(
        "[cambricon] 未设置 DC_DIST_BT，拒绝以兜底值 `gloo` 继续（那会静默退化为纯 CPU\n"
        "  集合通信，训练脚本仍会跑完，但设备侧通信未被验证）。\n"
        "  寒武纪的集合通信后端名**必须实测后显式指定**，不能从 910C(`hccl`) / P800(`flagcx`) 类推。\n"
        "  探测方法（容器内，只读）：\n"
        "    python3 -c \"import torch_mlu, torch; print(torch.mlu.is_available(), torch.mlu.device_count())\"\n"
        "    # 逐个尝试进程组后端，记录可用者（设备侧通常写作 <dev>:<backend>，如 mlu:cncl）\n"
        "    #   候选：cncl（寒武纪 CCL） / flagcx（若镜像内已装 flagcx-ascend 之外的 mlu 适配）\n"
        "  探测结果请回填《寒武纪接入方案》与\n"
        "    runtime/backends/cambricon/backend.py 顶部「未实测清单」第 10 条。\n"
        "  确认后按 DC_DIST_BT=\"cpu:gloo,mlu:<backend>\" 重跑。",
        flush=True,
    )
    raise SystemExit(2)
ROOT = os.environ.get("DC_ROOT", "/mnt/raid/hliu553/runtime-team/dev/device-context")
MODEL = os.environ.get("DC_MODEL", "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
OUT_DIR = os.environ.get("DC_OUT_DIR", "/mnt/raid/hliu553/runtime-team/scratch")

sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

if "flagcx" in DIST_BT:
    # 显式导入才会把 `cuda:flagcx` 注册为 c10d 后端（P800 路线 A 的集合通信）
    import flagcx  # noqa: F401,E402

import runtime  # noqa: E402  —— 我们的统一运行时原型

# ⚠️ **必须在 init_process_group 之前经后端触发厂商扩展加载**（2026-09-22 实测踩到，审计台账第 15 条）：
#   `hccl` / `flagcx` 这类**集合通信后端名**只有厂商扩展被 import 后才在 c10d 里注册，
#   而统一 API 的后端是**懒加载**的（`use()` 本身不触发厂商扩展导入）。
#   实测 910C（torch_npu 2.11.0，容器关 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`）：
#     直接 `init_process_group("hccl")` → `AssertionError: Unknown backend type hccl`；
#     先 `use(BACKEND)` + 触碰一次设备后即通过。
#   ⇒ 纪律：**凡要用厂商设备串或厂商集合通信后端名之前，先经后端触碰一次设备**。
#     （与 conformance runner 那处同源：第 11 条 = 拼设备串、本条 = 拼集合通信后端名。）
runtime.use(BACKEND)
runtime.current().device_count()

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
    # 经统一 API 同步（后端无关）
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
