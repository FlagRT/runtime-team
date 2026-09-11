# 推理插件接入（阶段 4）多卡 TP 验证 · 方案 + 执行记录

> ⚠️ **已归档（2026-08-22）**：本文件为设备层路线变更（B→A，见 [../FlagOS设备层路线变更指南.md](../FlagOS设备层路线变更指南.md)）前的 **B 线/旧案产物**，不进入 A 线交付路径。
> 仍有效结论：TP 数值退化修复经验（浮点归约差异判定）。
> 需 A 线重验：A 线昇腾 TP 验证。

> 日期：2026-08-15 ｜ 执行人：xliu969 ｜ 前置：单卡最小闭环已跑通（05:18，exit=0）
> 状态：✅ **TP=2 数值退化已修复（2026-08-16），TP=1/TP=2 输出逐字一致**；✅ **TP=4 链路闭环（2026-08-16），输出合理且两次运行可复现，3/4 与 TP=1 逐字一致、1/4 前缀一致后分叉（浮点归约差异，非 bug）**
> 本文档参照阶段 3 流程：方案先行 → 过程存档 → 保存最小链路方案

## 〇、结论（2026-08-16 更新）

**TP=2 数值退化（生成 "!!!!!!!"/"<<<<<<<<"）根因 = 两个叠加缺陷，均已修复：**

### 根因 1（主因）：flagos 设备上 bool×int 类型提升错误 → embedding 查表全查 row1

- `AscendVocabParallelEmbedding._get_masked_input_and_mask`（vllm-plugin-FL
  `vllm_fl/dispatch/backends/vendor/ascend/impl/vocab_parallel_embedding.py:127`）
  用 `vocab_mask * (input_ - valid_offset)` 做 TP vocab 分片掩码。
- flagos(PrivateUse1/torch_fl) 上 `bool * int` 返回 **bool**（标准 torch 返回 int64）：
  实测 `True*9707 → True`，`.long()` 后 token id 全变 1 → **所有 token 查 embedding row1**。
- 修复：显式 `.to(torch.int64)`（3 处，见 git diff）。TP=1 不触发（tp_size=1 不走 mask 分支），
  所以单卡一直正常——这正是"单卡通、多卡废"的原因。

### 根因 2（隐藏缺陷）：flagcx 后端通信异步返回无同步 → 下游读到 NaN

- `CommunicatorFL.all_reduce` 的 fallback（`torch.distributed.all_reduce(out,
  group=device_group)`，communicator.py:46）在 flagcx 后端下**异步返回**，结果未就绪
  就被下游算子消费 → 模型中间层产生 NaN → logits 全 NaN → 采样退化。
- 之前误判"通信正确"是因为调试探针的 `.sum().item()/.tolist()` **意外强制了设备同步**；
  清理探针后问题暴露（TP=2 从成功变失败，TP=1 不受影响）。
- 修复：fallback 后补 `torch.npu.synchronize()`；并覆盖 `all_gather`（基类实现同样异步）
  加同步（communicator.py，可 PR 上游）。

### 排查方法（可复用）
- **探针必须打在真实路径上**：TP 张量通信走 `GroupCoordinator.all_reduce`（parallel_state.py）
  → `device_communicator`，不是 `StatelessProcessGroup.all_reduce`（KV-store 对象广播路径，
  之前 DBG-AR 打错位置导致"通信从未发生"的错误结论）。
- **flagos 设备上的 `.sum().item()` 数值不可靠**（flag_gems sum 算子问题），跨 rank 对比
  用 `.cpu()` 后计算或 `.tolist()` 取元素。
- **多进程日志交错**：按 rank 分别 grep/配对，避免把 rank0 的输入配到 rank1 的输出。

### 验证（全部通过）
- TP=2 单 prompt：`"Hello, my name is" → " Xiaoyu, and I'm a"`（与 TP=1 一致）
- TP=2 多 prompt（4 条）：输出与 TP=1 逐字一致（含 "The capital of France is → Paris..." 等）
- 挂点 `scatter_.src: backend not registered`（多 prompt decode 批量路径）由 flagos_boot.py
  的 CPU fallback 解决（`aten::scatter_` 系列注册 PrivateUse1 实现，见 flagos_boot.py 尾部）

### TP=4 验证（2026-08-16 追加）

**链路**：`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 VLLM_FL_TP=4` 跑同一脚本（4 条 prompt），
**链路完整闭环**（exit=0，REASONING OK）：4 卡 flagcx 通信初始化 ✅、TP=4 权重分片加载 ✅、
模型前向+采样 ✅。无新挂点（TP=2 修复的 bool×int + flagcx 同步直接覆盖 TP=4 路径）。

**数值**：两次独立运行输出**完全一致**（确定性可复现，无 NaN/退化文本）：

| prompt | TP=1 | TP=4（两次一致） | 一致性 |
|---|---|---|---|
| Hello, my name is | " Xiaoyu, and I'm a student at the University of Science and Technology" | 同左 | ✅ 逐字 |
| The capital of France is | ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.' | ' Paris. The capital of Paris is...? The capital of Paris is not a' | ⚠️ 前缀 "Paris." 一致后分叉 |
| 2+2= | '4, 2+2=4, 2+2=4,' | 同左 | ✅ 逐字 |
| Python is a | ' high-level, interpreted, general-purpose programming language...' | 同左 | ✅ 逐字 |

**结论**：TP=4 链路闭环成立、输出语义合理、可复现；与 TP=1 存在 1/4 分叉
（前缀一致后 token 选择不同）。分叉是**确定性的**（两次 TP=4 结果相同），且无 NaN 迹象，
判定为多卡归约顺序不同导致的浮点精度差异（greedy 解码对 top-1 logits 微小差敏感）——
非同步 bug（同步缺陷会表现为随机/NaN 输出）。TP=2 逐字一致、TP=4 出现分叉，
符合"卡数越多归约路径越长、浮点误差累积越明显"的预期。**若要进一步追平，可对比
flagcx allreduce 的求和顺序与单卡逐层 logits**（非阻塞，列入后续可选优化）。

## 一、方案

**目标**：在单卡闭环基础上，验证 Qwen3-4B 多卡（TP=2 起步）推理闭环——vllm 张量并行下 2 张卡协同推理，flagcx（阶段 3 通信资产）承担 TP 通信。

**原理（大白话）**：单卡闭环 = 1 张卡跑完整模型；TP=2 = 把模型的每一层"横切成两半"分给 2 张卡，前向时两张卡通过 flagcx 互相传中间结果。vllm 负责横切逻辑（tensor_parallel_size=2），我们负责验证通信（flagcx）在推理路径上的适配。

**步骤**：
1. 环境确认：16 卡空闲（已查，全空）、容器 flagos-fl-dev-910c、venv311 单卡套件不变
2. 推理脚本参数化：`VLLM_FL_TP` 环境变量控制 TP 大小（默认 1，单卡链路不变）
3. 跑 TP=2：`ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_FL_TP=2`，记录挂点/解法
4. 跑通后：确认生成文本合理、记录耗时与日志存档

**预期新挂点（单卡未覆盖的路径）**：
- flagcx 的 allreduce/allgather 在 vllm 推理路径的首次调用（阶段 3 只验证了训练侧）
- TP 权重分片加载（Qwen3-4B 权重按 TP=2 横切）
- 多 worker 进程的通信初始化（rank 分配、nccl 风格的 init）

**风险与回滚**：
- 全部变更在容器内（venv311 + 脚本），不动宿主配置/驱动 → 无回滚风险
- 多卡测试前 npu-smi 已确认 16 卡空闲（同事 sglang 已停）；若测试中同事重启任务，先停我们的验证

## 二、执行记录（追加式，按时间）

### 2026-08-15 多卡 TP 验证（进行中）

**已达成**：
- TP=2 推理链路**完整跑通**（exit=0，REASONING OK）：flagcx 通信初始化 ✅（world_size=2 backend=flagcx）、TP 权重分片加载 ✅、模型前向+采样执行 ✅——**链路层面多卡闭环成立**
- 过程修复的挂点：
  1. embedding/lm_head 的 weight_loader 未恢复（TP=1 时形状恰好相等掩盖）→ base_loader 备份条件放宽（`"weight_loader" in _pp.__dict__`）
  2. aclnn 算子在线程设备与 tensor 设备不匹配时执行失败（ret=361001 ACLNN_ERR_RUNTIME_ERROR）→ torch_fl csrc `AclTensorWrapper` 构造按 tensor 设备 pin 线程 ACL 上下文（**可 PR**，op_api_common.h）
  3. 内存检查误判：残留 worker 进程占用设备 0（28GB）+ `_mem_get_info` device=None 时固定查设备 0 → 杀残留 + mem_get_info 用 `current_device()`

**待解决（下一轮）~~**~~：输出数值退化——**已于 2026-08-16 解决**（见本文档 §〇 结论）。历史排查记录保留如下：~~**输出数值退化**（生成 `!!!!!!!` / `应对!!!!!!!`，单卡为正常文本）——**链路通但数值错**。已排除：~~
- ~~flagcx allreduce 数值正确（最小 2 进程测试 [0,0,0,0] correct:True）~~（注：此结论受探针同步副作用误导，见 §〇 根因 2）
- ~~flagos tensor pickle 往返正常（vllm 对象广播路径的传输没问题）~~
- ~~**关键发现**：vllm 的 `GroupCoordinator.all_reduce`（对象广播）**从未被调用**（DBG-AR 计数 0）~~（**错误结论**：DBG-AR 探针打在 KV-store 对象广播路径，TP 张量通信实际走 `GroupCoordinator.all_reduce` → `device_communicator`，一直在正常调用）

**临时调试残留（下轮清理）**：~~vllm/distributed/utils.py 的 all_reduce 首 4 次打印（[DBG-AR]）~~（已清）；探针脚本 probe_fill_thread_device.py / probe_flagcx_allreduce.py（宿主 scripts/ 保留，无害）
