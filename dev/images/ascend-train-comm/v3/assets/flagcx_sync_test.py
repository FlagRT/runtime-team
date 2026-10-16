#!/usr/bin/env python3
"""FlagCX c10d collective correctness check (2 ranks, self-judging).

    torchrun --nproc_per_node=2 flagcx_sync_test.py [--iters N]   (default N=5)

Cases, each run with and without torch.npu.synchronize() before reading:
  broadcast              -> dist.broadcast
  all_reduce             -> dist.all_reduce (SUM)
  all_gather_list        -> dist.all_gather(list_of_tensors, ...)
  all_gather_into_tensor -> dist.all_gather_into_tensor (the primitive DDP init uses)

Every call carries a payload unique to (case, iteration), so a result left
over from an earlier call is detected as FAIL instead of passing by accident
(the round 3/4 version reused one payload and could not tell stale from fresh).

Results are aggregated across ranks (all_reduce MIN + synchronize, verified
correct) and iterations. Exit code is identical on all ranks:
  0  every case passes, synced and unsynced -> FlagCX c10d semantics correct
  1  any case fails (the current state; see README for the expected pattern)
"""
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

CASES = ["broadcast", "all_reduce", "all_gather_list", "all_gather_into_tensor"]


def run_case(case, rank, ws, dev, tag, use_sync):
    """Run one collective with payload base `tag`; return (ok, got, expected)."""
    if case == "broadcast":
        t = torch.full((4,), tag if rank == 0 else -1, dtype=torch.int64, device=dev)
        dist.broadcast(t, src=0)
        out, expected = (lambda: t.cpu().tolist()[0]), tag
    elif case == "all_reduce":
        t = torch.full((4,), tag + rank, dtype=torch.int64, device=dev)
        dist.all_reduce(t)
        out, expected = (lambda: t.cpu().tolist()[0]), sum(tag + k for k in range(ws))
    elif case == "all_gather_list":
        loc = torch.full((4,), tag + rank, dtype=torch.int64, device=dev)
        outs = [torch.full((4,), -1, dtype=torch.int64, device=dev) for _ in range(ws)]
        dist.all_gather(outs, loc)
        out, expected = (lambda: [o.cpu().tolist()[0] for o in outs]), [tag + k for k in range(ws)]
    else:
        loc = torch.full((4,), tag + rank, dtype=torch.int64, device=dev)
        flat = torch.full((4 * ws,), -1, dtype=torch.int64, device=dev)
        dist.all_gather_into_tensor(flat, loc)
        out, expected = (lambda: flat.cpu().tolist()[::4]), [tag + k for k in range(ws)]
    if use_sync:
        torch.npu.synchronize()
    got = out()
    # 读取后一律同步再进入下一组:未同步的上一组若仍在执行,其张量被释放后由缓存
    # 分配器复用,迟到的写入会落到下一组的张量上,使各组结果互相污染。
    torch.npu.synchronize()
    return got == expected, got, expected


def main():
    iters = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 5
    dist.init_process_group(backend="flagcx")
    rank, ws = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dev = f"npu:{local_rank}"

    names = [f"{c}+{m}" for c in CASES for m in ("nosync", "sync")]
    passed = dict.fromkeys(names, True)
    for i in range(iters):
        for ci, case in enumerate(CASES):
            for mi, mode in enumerate(("nosync", "sync")):
                tag = 1000 * (i + 1) + 100 * ci + 10 * mi
                ok, got, exp = run_case(case, rank, ws, dev, tag, mode == "sync")
                passed[f"{case}+{mode}"] &= ok
                if not ok:
                    print(f"[rank{rank}] iter{i} {case}+{mode}: got={got} expected={exp} -> FAIL", flush=True)

    torch.npu.synchronize()
    dist.barrier()
    flags = torch.tensor([int(passed[n]) for n in names], dtype=torch.int64, device=dev)
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    torch.npu.synchronize()
    result = dict(zip(names, (bool(v) for v in flags.cpu().tolist())))
    all_ok = all(result.values())
    if rank == 0:
        print(f"\n=== VERDICT (all ranks, iters={iters}) ===", flush=True)
        for n in names:
            print(f"  {n:32s} {'PASS' if result[n] else 'FAIL'}", flush=True)
        print(f"overall: {'PASS' if all_ok else 'FAIL'}", flush=True)

    # 跳过 destroy_process_group 与解释器正常退出:FlagCX 已知的退出期
    # free(): invalid pointer SIGABRT(lock.yaml known_issues teardown-sigabrt)会把
    # 判定覆盖成 torchrun 失败。判定已汇总并打印,直接按判定退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
