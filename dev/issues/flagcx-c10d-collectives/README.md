# FlagCX c10d collective：异步返回与列表形式 `all_gather` 结果错位

## 问题与现状

- **一句话**：FlagCX 的 c10d ProcessGroup 实现有两类缺陷——① `broadcast`、`all_reduce`、
  `all_gather_into_tensor` 返回时数据未写完（调用方按同步语义立即读取得到旧值或全零，显式
  `torch.npu.synchronize()` 后正确）；② 列表形式 `dist.all_gather` 读到**上一次调用**的数据，
  显式同步也无效。前者使 PyTorch 原生 `DistributedDataParallel` 不可用（DDP 启动检查
  `_verify_params_across_processes` 在 C++ 层命中类别 ①，报 "rank0 有 338 个参数、其他 rank 0 个"），
  后者没有调用方规避手段，只能避开列表形式。
- **归属方向**：FlagCX 通信方向。需求文本：
  [`决策备忘录 附录 B-2 及其 2026-10-10 复验修正段`](../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)。
- **状态（2026-10-11）**：未修复，2026-10-11 第 4 次运行判定与前 3 次逐行一致。本组的规避是在每个
  collective 之后显式 `torch.npu.synchronize()` 并不使用列表形式 `all_gather`——2 卡手动 DDP 可训
  （Qwen2.5-1.5B 24 步 loss 2.8911 → 2.5528），原生 DDP 封装不可用。Python 层补丁（vllm-plugin-FL
  覆盖层）覆盖不到 c10d 路径，修复必须落在 FlagCX 自己的 c10d 实现。

| collective | 未同步即读取 | `torch.npu.synchronize()` 后读取 | 缺陷类别 |
|---|---|---|---|
| `broadcast` | FAIL | PASS | ① 异步返回 |
| `all_reduce` | FAIL | PASS | ① 异步返回 |
| `all_gather_into_tensor`（DDP 启动检查所用） | FAIL | PASS | ① 异步返回 |
| `all_gather`（输出为张量列表） | FAIL | **FAIL**：得到上一次调用的结果或全零 | ② 结果错位，同步无效 |

- **类别 ② 的代码位置与推断**（**推断，待 FlagCX 方向确认**）：
  `FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp` `flagcxBackend::allgather` 先把结果汇集到临时扁平
  缓冲 `outputFlattened`，再以 `copy_(..., non_blocking=true)` 拷回输出列表；该拷贝疑未与
  `flagcxAllGather` 所在流建立先后顺序，且 `outputFlattened` 未 `recordStream`，于是拷贝读到尚未写入、
  已被缓存分配器复用的旧缓冲。
- **连带现象**：未同步的调用若在执行中途其张量被释放，迟到的写入会落到复用该内存的后续张量上
  （测试早期版本中观察到 `broadcast` 源 rank 的张量被上一次调用的数据覆盖）。
- 退出期另有已知的 `free(): invalid pointer` SIGABRT（低优先级，结果已在退出前输出，不影响判定；
  见 [`ascend-train-comm/v3/lock.yaml`](../../images/ascend-train-comm/v3/lock.yaml) known_issues `teardown-sigabrt`）。

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | **现行训练腿发布物**：`flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`（npu1-11 image id `e13a15d0d02b`，FlagCX 公开主干 `4e0e0cbcbf721169ca82348080f8353aebfe2c31`）—— [`dev/stack.lock.910c.v4.yaml`](../../stack.lock.910c.v4.yaml) `lock.train`，血统目录 [`dev/images/ascend-train-comm/v3`](../../images/ascend-train-comm/v3/README.md) | 按 [`ascend-train-comm/v3/REBUILD.md`](../../images/ascend-train-comm/v3/REBUILD.md) 构建，或从其 `ARCHIVE.md` 所列离线包 `docker load`；image id 由 `up.sh --strict` 按镜像基线核对 |
| 运行环境 | **训练腿** demo 运行环境 [`dev/images/ascend-train-comm/v3/demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml) | 随仓。该镜像全部 repo 为固化资产，宿主检出不影响运行 |
| 卡 | 2 × 910C | 启动前 `docker ps` 确认本机无其他带卡容器（昇腾驱动对带卡容器独占） |
| 模型与数据 | **不需要**（脚本只跑 collective，不加载模型、不联网） | — |
| 模型目录变量 | `FLAGRT_MODELS_DIR` 指向一个实体目录 | demo 把它挂为 `/workspace/models:ro`；缺省值 `<仓根>/models` 在本机是软链，容器内无法解析，故启动时须显式给值（npu1-11：`/home/xliu969/models`） |

### 单入口命令（容器内）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagrt-v3-demo-910c \
  torchrun --nproc_per_node=2 dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py --iters 20
```

脚本（本目录不复制，按相对路径引用）：
[`dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py`](../../images/ascend-train-comm/v3/assets/flagcx_sync_test.py)。
对 4 个 collective 各做「未同步即读取 / 同步后读取」两组，每次调用使用唯一数据（据此区分"本次结果"
与"上一次调用的遗留结果"），跨 rank（`all_reduce` MIN）、跨迭代汇总后判定。未同步组的失败为偶发，
单次可能通过，故取 `--iters 20`。容器启动与收尾见 [`NOTES-step-by-step.md`](NOTES-step-by-step.md)。

### 预期结果（问题存在时，退出码 1）

```
=== VERDICT (all ranks, iters=20) ===
  broadcast+nosync                 FAIL
  broadcast+sync                   PASS
  all_reduce+nosync                FAIL
  all_reduce+sync                  PASS
  all_gather_list+nosync           FAIL
  all_gather_list+sync             FAIL
  all_gather_into_tensor+nosync    FAIL
  all_gather_into_tensor+sync      PASS
overall: FAIL
```

判定段之后 torchrun 会打印 `ChildFailedError`（退出码 1 的正常表现），不是另一个错误。

## 修复验收

**判据**：上述单入口命令退出码 0、判定行 `overall: PASS`（八组全部 PASS，即调用方在当前流上直接读取结果
无需额外同步，且列表形式 `all_gather` 结果正确）。**另需**原生 `DistributedDataParallel` 双卡验证通过
（`DistributedDataParallel` 封装、不含手动同步的自判定脚本，待补，见
[`v3-round4-真机验证结果与移交指导`](../../../docs/v3-round4-真机验证结果与移交指导-20261009.md) 第七节 B
与 [`dev/images/TODO.md`](../../images/TODO.md) 第 2 项）。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`flagcx_sync_test.py`](../../images/ascend-train-comm/v3/assets/flagcx_sync_test.py)（引用，不复制） | 复现 / 验证 | 见「单入口命令」 | 退出码 0 且 `overall: PASS` | 2026-10-11 npu1-11 demo 环境，`overall: FAIL`（退出码 1，缺陷复现），判定与 2026-10-10 的 3 次逐行一致。证据：[`assets/evidence/recheck-20261011.txt`](assets/evidence/recheck-20261011.txt)（判定段原文与失败样例见 [`flagcx-collectives-20261010.txt`](../../images/ascend-train-comm/v3/assets/evidence/flagcx-collectives-20261010.txt)） |
| 原生 DDP 双卡脚本 | 验证 | 待补 | 退出码 0 | 未实现 |

修复到货后，v3 完整镜像 `repro_status` 由 🟡 升 🟢
（[`ascend-train-comm/v3/README.md`](../../images/ascend-train-comm/v3/README.md)），训练侧不再需要手动同步写法。
