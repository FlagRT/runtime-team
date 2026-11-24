"""connector-ascend 容器侧执行体：直接驱动 flagmemory.tiering.offload.npu.NpuDmaCopyBackend
做字节级验证——不经 vLLM，不受调度器/显存压力策略影响，判据对应 npu/RUNBOOK.md「六判据」
里的 ③D2H 字节级逐字一致 / ④H2D 新旧块号分叉在场 / ⑤审计（本探针覆盖 GPU 读回=host 池两方，
第三方「独立重算参考」用复制前先存一份的 expected 张量代替——纯内存搬运场景下，搬运前的原始
内容本身即是可信参考）/ ⑥destroy 双形态（带在途任务 / 空闲幂等）。

①工厂懒加载 / ②锁页实测 两条判据不在本探针内——那两条已由
`tests/tiering/test_offload_realcard.py -k npu`（导入与别名）与 FlagMemory 既有锁页探测
覆盖，本探针只补「真实搬运行为对不对」这一段此前没有可重复脚本的空白。

输出：末行打印一行 JSON（机器可判），退出码 0=全部通过、1=任一判据失败。
"""

import json
import sys
import time

sys.path.insert(0, "/workspace/FlagMemory")

import torch
import torch_npu  # noqa: F401

from flagmemory.tiering.offload.npu.npu_copy_backend import NpuDmaCopyBackend

RESULT = {"criteria": {}}


def wait_for(events, timeout_s=10):
    deadline = time.time() + timeout_s
    while not events and time.time() < deadline:
        time.sleep(0.05)
    return bool(events)


def main() -> int:
    device = torch.device("npu:0")
    torch.npu.set_device(device)

    num_blocks, bpb = 8, 1024
    gpu = torch.zeros((num_blocks, bpb), dtype=torch.float32, device=device)
    for i in range(num_blocks):
        gpu[i] = torch.arange(bpb, dtype=torch.float32, device=device) + i * 100000.0
    cpu = torch.zeros((num_blocks, bpb), dtype=torch.float32, device="cpu")

    backend = NpuDmaCopyBackend()
    backend.init({"layer0": gpu}, {"layer0": cpu}, device, torch.npu.Stream(), torch.npu.Stream())

    expected_block3 = gpu[3].clone()

    ev1: list = []
    backend.launch_copy(src_blocks=[3], dst_blocks=[5], is_store=True, event_idx=0, events_list=ev1)
    got1 = wait_for(ev1)
    torch.npu.synchronize()
    store_ok = got1 and torch.equal(cpu[5], expected_block3.cpu())
    RESULT["criteria"]["D2H_byte_exact"] = bool(store_ok)

    new_content = torch.full((bpb,), -999.0, dtype=torch.float32, device=device)
    gpu[3].copy_(new_content)
    torch.npu.synchronize()

    ev2: list = []
    backend.launch_copy(src_blocks=[5], dst_blocks=[7], is_store=False, event_idx=1, events_list=ev2)
    got2 = wait_for(ev2)
    torch.npu.synchronize()
    load_ok = got2 and torch.equal(gpu[7], expected_block3)
    coexist_ok = torch.equal(gpu[3], new_content)
    RESULT["criteria"]["H2D_byte_exact"] = bool(load_ok)
    RESULT["criteria"]["old_new_block_coexist"] = bool(coexist_ok)

    ev3: list = []
    backend2 = NpuDmaCopyBackend()
    backend2.init({"layer0": gpu}, {"layer0": cpu}, device, torch.npu.Stream(), torch.npu.Stream())
    backend2.launch_copy(src_blocks=[0], dst_blocks=[1], is_store=True, event_idx=0, events_list=ev3)
    inflight_shutdown_ok = True
    try:
        backend2.shutdown()
    except Exception:
        inflight_shutdown_ok = False
    idempotent_shutdown_ok = True
    try:
        backend2.shutdown()
    except Exception:
        idempotent_shutdown_ok = False
    RESULT["criteria"]["destroy_inflight_drain"] = bool(inflight_shutdown_ok)
    RESULT["criteria"]["destroy_idempotent"] = bool(idempotent_shutdown_ok)

    backend.shutdown()

    overall = all(RESULT["criteria"].values())
    RESULT["result"] = "PASS" if overall else "FAIL"
    RESULT["executed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    for k, v in RESULT["criteria"].items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    print("SUMMARY_JSON::" + json.dumps(RESULT))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
