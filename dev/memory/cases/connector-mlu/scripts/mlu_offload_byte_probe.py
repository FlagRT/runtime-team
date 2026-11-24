"""connector-mlu 容器侧执行体：直接驱动 flagmemory.tiering.offload.mlu.MluCopyBackend
做字节级验证——不经 vLLM，不受调度器/显存压力策略影响。方法与 connector-ascend/connector-ppu
同构（见 dev/memory/cases/connector-ascend/scripts/npu_offload_byte_probe.py 的说明）。

输出：末行打印一行 JSON（机器可判），退出码 0=全部通过、1=任一判据失败。
"""

import json
import sys
import time

sys.path.insert(0, "/workspace/FlagMemory")

import torch
import torch_mlu  # noqa: F401,E402

from flagmemory.tiering.offload.mlu.mlu_copy_backend import MluCopyBackend  # noqa: E402

RESULT = {"criteria": {}}


def wait_for(events, timeout_s=10):
    deadline = time.time() + timeout_s
    while not events and time.time() < deadline:
        time.sleep(0.05)
    return bool(events)


def main() -> int:
    device = torch.device("mlu:0")
    torch.mlu.set_device(device)

    num_blocks, bpb = 8, 1024
    gpu = torch.zeros((num_blocks, bpb), dtype=torch.float32, device=device)
    for i in range(num_blocks):
        gpu[i] = torch.arange(bpb, dtype=torch.float32, device=device) + i * 100000.0
    cpu = torch.zeros((num_blocks, bpb), dtype=torch.float32, device="cpu")

    backend = MluCopyBackend()
    backend.init({"layer0": gpu}, {"layer0": cpu}, device, torch.mlu.Stream(), torch.mlu.Stream())

    expected_block3 = gpu[3].clone()

    ev1: list = []
    backend.launch_copy(src_blocks=[3], dst_blocks=[5], is_store=True, event_idx=0, events_list=ev1)
    got1 = wait_for(ev1)
    torch.mlu.synchronize()
    store_ok = got1 and torch.equal(cpu[5], expected_block3.cpu())
    RESULT["criteria"]["D2H_byte_exact"] = bool(store_ok)

    new_content = torch.full((bpb,), -999.0, dtype=torch.float32, device=device)
    gpu[3].copy_(new_content)
    torch.mlu.synchronize()

    ev2: list = []
    backend.launch_copy(src_blocks=[5], dst_blocks=[7], is_store=False, event_idx=1, events_list=ev2)
    got2 = wait_for(ev2)
    torch.mlu.synchronize()
    load_ok = got2 and torch.equal(gpu[7], expected_block3)
    coexist_ok = torch.equal(gpu[3], new_content)
    RESULT["criteria"]["H2D_byte_exact"] = bool(load_ok)
    RESULT["criteria"]["old_new_block_coexist"] = bool(coexist_ok)

    ev3: list = []
    backend2 = MluCopyBackend()
    backend2.init({"layer0": gpu}, {"layer0": cpu}, device, torch.mlu.Stream(), torch.mlu.Stream())
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
