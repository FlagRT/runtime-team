# 通信接口约定 v1（本地待确认）

负责人：尤联忠。依据战略目标 §5，设备接口参考 `kistich/device-context@f81adaf`。
通信章节已补齐；下游确认与新代码真机回归尚未完成，因此未达到公共分支合入门槛。

## 环境与后端

当前权威配置为 `dev/stack.lock.910c.v2.yaml` 的 `lock.train`，仍为原训练镜像
`0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`。
`candidates.train` 仅登记候选，不能据此切换验收镜像。通信接口使用 PyTorch
`torch.distributed`，该镜像公开后端名是 `flagos`，由 FlagCX 实现。
设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，先导入 `torch_fl`，再导入 `torch`，不加载 `torch_npu`。

P800 在设备方向分支中使用 `cpu:gloo,cuda:flagcx`，且有 KL3 条件性挂死记录；
这些是另一个芯片实例的限制，不是本接口在 910C 上的新测试结果。

## 通信组生命周期

1. 每个进程先按 `LOCAL_RANK` 绑定可见设备，再调用 `init_process_group("flagos", timeout=...)`。
   `RANK`、`WORLD_SIZE`、rendezvous 配置由启动器统一提供；当前验收为同机 2 Rank。
2. 初始化完成后通过 `get_rank/get_world_size` 查询身份。所有成员保持一致的 collective
   顺序、形状、dtype 和归约操作；当前探针不支持多线程并发提交或多组交错。
3. 状态为 NEW → READY → DRAINING → CLOSED；任何初始化/通信/同步失败进入 FAILED。
   这些是调用方管理规则，未新增自研通信组实现。
4. 正常关闭：停止提交 → 等待未完成 Work → 设备同步 → 每个成员调用
   `destroy_process_group()` → 启动器等待所有进程正常退出。
   不承诺 PyTorch 原始 destroy 接口可以重复调用；调用方用状态防止二次销毁。
5. FAILED 组不再提交新任务、不做可能永久阻塞的退出 barrier。保留原始异常，由启动器
   对本次进程组施加总超时，先 TERM、宽限后 KILL；重建由调度重新拉起全组。
   不承诺同进程损坏后端可恢复，也不自动重放可能已部分完成的 collective。

## 操作与能力声明

| 操作 | 输入/输出约定 | 当前证据 |
| --- | --- | --- |
| AllReduce SUM | 各 Rank 同形状/dtype；原地更新；参考值为各 Rank 贡献之和 | 2 Rank，FP32/BF16，8 元素 |
| AllGather | 输出列表长为 world size；各输出与输入同形状/dtype；按 Rank 排列 | 2 Rank，FP32/BF16，4 元素 |
| Send/Recv | 对等 src/dst、形状、dtype；显式双向次序避免双方阻塞 send | 2 Rank，FP32/BF16，8 元素 |
| async AllReduce | 保留输入和 Work，完成依赖建立后再消费/复用缓冲 | wait + 全设备同步后的数值正确 |
| ReduceScatter/Broadcast | API 已纳入计划；不宣称当前探针已验收 | 未覆盖 |
| 跨流消费/并行通信计算 | 须验证通信完成到消费流的真实依赖 | 待真机验证 |
| 多节点、8/16 卡、性能 | 独立的后续规模化任务 | 本轮不声明支持结论 |

不支持与未验证必须区分：表中未覆盖项表示本方向没有验收证据，不能推断后端不支持。
当前探针在进程组初始化后检查 world size 并拒绝非 2 Rank 配置；业务调用的其他组合需经能力验证后再开放，禁止静默降级。

## 异步与缓冲所有权

`async_op=True` 的返回、`is_completed()` 为 true、`wait()` 返回、设备完成、跨流可见
是不同观察点。历史 80 次立即查询均为 true，并不足以证明设备提前完成或接口有缺陷。
当前已验证消费路径为 `work.wait()` + `torch.flagos.synchronize()` 后做 Host 对照。
全设备同步是当前正确性基线，有性能代价；尚不承诺通信计算重叠。

跨流接入须同时满足两件事：通信完成后在正确的流记录 Event，并让消费流 wait；张量
保留到消费完成，必要时 `record_stream(consumer)` 防止缓存分配器提前复用。
`record_stream` 管的是生命周期，不能代替执行依赖。具体流/事件由设备方向组件提供，
通信方向不另造 Stream/Event 实现。后端内部流是否遵循调用方当前流需实测，不能仅凭
在调用方流记录 Event 就认为覆盖了通信内部流。

## 错误与验收结果

设备方向的 `runtime.translate_error` 接收完整原始异常及 location，返回 category / disposition。
调用方按 disposition 处理；通信组超时或成员退出先视为整个组失败，协调调度重建，
不因为某个 Rank 可继续运行就恢复提交。设备致命错误交设备方向 `recover_device`，
何时恢复、检查点重放由监控/调度编排，通信库不自行重置共享设备。

`run_acceptance.py` 用独立 run_id/输出目录归档 launcher.log、各 Rank JSON、acceptance.json。
完整 PASS 必须同时满足：

- 两个当前运行结果齐全，身份、dtype、轮数、计数一致且所有数值校验通过；
- 每个 Rank 记录 `phase=group_destroyed`；
- 启动器退出码为 0，未超时，也未发现需要强杀的残留进程组。

数值 PASS 后 SIGABRT 仍为整次 FAIL，保留 `numeric_ok=true` 以供定位。
外层总超时不等于后端本身支持有界 wait；kill 也不证明驱动资源已释放，真机操作方仍须
复查卡资源。当前进程组监督针对遵循 torchrun 会话的子进程，主动 setsid 脱离会话的程序不在覆盖内。

## 对接与合入门槛

设备方向确认完成依赖与生命周期；调度方向接入退出码和 acceptance.json；performance
引用原始结果及测试环境，不把主机模拟测试并入真机性能报告。
本次完成本地章节、监督器及主机回归；下游人员确认、910C 新代码回归和完整训练腿联测
仍待执行。没有下游确认记录，不能将“章节写完”标为“全组契约已冻结”。
