# 锁定训练镜像双卡通信正确性执行记录

## 结论

在 `dev/stack.lock.910c.v1.yaml` 指定的训练镜像内，使用 2 个 Rank 对 AllReduce、AllGather、P2P 和异步 AllReduce 进行 20 轮确定性验证。FP32、BF16 两种 dtype 共执行 320 次通信调用，结果为 **320/320 PASS，0 失败**。

本记录对应本周计划中“all_reduce / all_gather / P2P 三类通信正确性对照方案与证据”。它是小张量功能验证，不是通信性能基准，也不替代训练腿完整微调结果。

## 环境

| 项 | 值 |
| --- | --- |
| 服务器 | `npu1-27` |
| 训练镜像 | `flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64` |
| 本机镜像 ID | `sha256:3b9e08f231d0e80d37341e375e6bb13066e03a67794b78a3d2874c1770c6b6bc` |
| 容器 | `flagos-proto-train-910c` |
| Python / PyTorch | 3.11.15 / 2.10.0+cpu |
| 设备后端 | `flagos`（锁定镜像内由 FlagCX 实现通信） |
| Rank / 设备 | rank0→`flagos:0`、rank1→`flagos:1` |

## 用例与结果

| 操作 | dtype | 每 Rank 调用数 | 两 Rank 合计 | 结果对照 |
| --- | --- | ---: | ---: | --- |
| AllReduce SUM | FP32、BF16 | 40 | 80 | rank 贡献值求和后逐元素等于 3 |
| AllGather | FP32、BF16 | 40 | 80 | 两端均收集到 rank0、rank1 的确定值 |
| P2P Send/Recv | FP32、BF16 | 40 | 80 | 双向 8 元素张量逐元素相等 |
| 异步 AllReduce | FP32、BF16 | 40 | 80 | `wait` + 设备同步后逐元素等于 3 |
| **合计** | 2 种 | **160** | **320** | **320/320 PASS** |

- rank0：160/160，通过；总执行时间 8.855 秒。
- rank1：160/160，通过；总执行时间 8.828 秒。
- 两个 Rank 均为 0 失败，进程正常退出。

上述耗时包含进程组初始化、Python 循环和同步，仅用于记录本次执行成本，不作为带宽或时延数据。

## 异步完成语义观察

异步 AllReduce 提交后立即调用 `Work.is_completed()`，两 Rank、两种 dtype 共 80 次观察均返回 `true`。最终结果在执行 `Work.wait()` 和设备同步后全部正确。

因此，本轮只能确认“等待并同步后的结果正确”，不能用 `is_completed() == true` 单独证明设备侧结果已经可被下游流安全消费。后续需结合统一 Stream/Event 接口补充跨流消费验证，再冻结异步完成语义。

## 复现

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nproc_per_node=2 \
  --master_port=29610 \
  /tmp/communication_correctness.py \
  --iterations 20 \
  --out-dir /tmp/comm-results-lianzhongyou
```

仓库脚本：`dev/communication/probes/communication_correctness.py`

原始结果：

- `results/20260909/communication_correctness_rank0.json`，SHA-256 `b918b674aff35f112a87a47d7b3ac07cfec64c9732c13f509eae645f5aaaf440`
- `results/20260909/communication_correctness_rank1.json`，SHA-256 `342ebd41c34df9a73cc3f9c90b835693d12072ebace1765c030c521c9abcd6df`

## 未覆盖项

- 未覆盖 ReduceScatter、Broadcast、不同消息规模和 8/16 卡性能。
- 未注入单进程异常退出或通信超时。
- 未证明 `is_completed()` 与设备侧完成之间的严格时序关系。
- 历史 FlagCX 缺陷仍需在 FlagCX 仓库整理为可编译、可测试的干净提交。
