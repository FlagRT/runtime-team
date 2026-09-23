# v3 待执行验证脚本（阶段性临时文件）

> **状态**：本目录是从上一台机器的 session scratchpad 里拉进 git 仓库的临时诊断/验证脚本，
> 目的是让新机器/新 session 能接着跑，不用重新写一遍。**不是最终归档资产**——
> 一旦对应的验证真正跑完、结论写回了各 `v3/REBUILD.md` + `lock.yaml`，本目录应该整体删除。
> 详见 `../HANDOFF-v3-910C.md`。

## 文件说明

| 文件 | 用途 | 依赖的镜像/资源 |
|---|---|---|
| `count_device_containers.sh` | 数当前机器上挂了 `--device` 的容器数量（带卡容器并发自查用） | 无（纯 `docker inspect`） |
| `wait_capacity.sh` | 轮询等到带卡容器数 ≤2（留 1 个名额）就退出 | 上面那个脚本 |
| `race_and_validate.sh` | 等到并发降到 ≤2，立刻起 1 个卡容器跑 `v3_step5_validate.py`（vLLM smoke test + 训练/推理共存检查），跑完自动清理 | `ascend-operator-runtime-comm:2.0.0-...-arm64`；`Qwen3-Embedding-0.6B` 模型（见下方"新机器前置条件"） |
| `v3_step5_validate.py` | 实际的验证逻辑：torch_npu 默认后端检查、torch_fl guard 检查、vllm_fl 平台注册检查、真实 `LLM(...).encode()` 推理调用 | 同上 |
| `race_task1.sh` | 等到并发降到 ≤2，起 1 个卡容器跑 `flagcx_sync_test.py`（见下） | `ascend-operator-runtime-comm:2.0.0-...-arm64`；`Qwen2.5-1.5B` 模型 |
| `flagcx_sync_test.py` | **优先级最高、还没跑过**：测试"FlagCX broadcast/all_gather 返回错误结果"是不是漏了 `torch.npu.synchronize()`（对照私有 fork commit `5d545c9` 的修复思路） | 同上，2 卡 |
| `train_qwen_1_5b_npu_syncpatch.py` | 上面测试如果证实是漏 sync，用这个跑一遍真实训练脚本确认端到端也修复（有已知局限，脚本内有注释） | 同上 |
| `diag_collectives.py` | 更早期的 collective 诊断脚本，`broadcast`/`all_gather` 返回错误结果的原始复现脚本（已经跑出结论，仅留作对照） | 同上 |

## 新机器前置条件（跑之前必须确认）

1. **真机 Ascend 910C 硬件 + driver**（`/usr/local/Ascend/driver`）。
2. **带卡容器并发规则**：机器级别硬上限 3 个同时挂 `--device` 的容器，起容器前必须 `docker ps` 自查
   （见 `dev/stack.lock.910c.v2.yaml` 的 `rules` 段）。上一台机器验证卡了很久就是因为
   这个上限一直被别的团队用满，新机器需要重新确认卡资源情况，不能假设空闲。
3. **模型文件路径**：`race_and_validate.sh`/`race_task1.sh` 里写死了
   `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B` 和 `/mnt/raid/hliu553/models/Qwen2.5-1.5B`——
   这是上一台机器的 RAID 挂载路径，**新机器大概率没有同名挂载点**，跑之前要改成新机器上的
   实际模型路径（或换成新机器上可访问的模型源）。
4. **镜像需要重建**：docker 镜像本身不会跟着 git 走，新机器要先跑
   `dev/images/ascend-operator-runtime/v3/build.sh` + `dev/images/ascend-train-comm/v3/build.sh`
   （已验证可重建，见各自 `REBUILD.md`），拿到本地 image id 后再回来跑这几个验证脚本。
