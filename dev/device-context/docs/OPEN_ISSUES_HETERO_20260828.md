# 异构训练开放问题与待办工作项（Open Issues & Backlog）

> 基线日期：2026-08-28 ｜ 来源：`docs/hetero_progress_and_roce_proposal.md` §7 遗留与开放问题
> 用途：把阶段性总结中的遗留项**正式化**为可跟踪、可分配、可验证的工作项；状态随工作推进更新。
> 编号规则：**O**pen Issue，按优先级排序（P0 阻塞验收 / P1 近期 / P2 中期 / P3 长期）。

## 工作项总览

| ID | 工作项 | 类别 | 优先级 | 状态 |
|----|--------|------|--------|------|
| O1 | P2+P6+P7+P8+P9 干净 diff 提交上游 `FlagRT/FlagCX` `kistich/ascend-dev1.0` | 上游提交 | **P0** | ✅ 完成（`a1e7e0f`，2026-08-28） |
| O2 | 昇腾 DAG 引擎解锁（CANN UVA 落地：补 `hostGetDevicePointer` + 设备侧 reduce 节点） | 架构优化 | P2 | ⬜ 未开始 |
| O3 | `flagcxGetLastError` 存根完善（缺陷 4，错误诊断） | 错误诊断 | P2 | ⬜ 未开始 |
| O4 | socket 协议无 tag 匹配加固（opId/序列号校验） | 协议加固 | P3 | ⬜ 未开始 |
| O5 | 诊断打点剥离（net.cc/proxy.cc 的 P1-P4 残留） | 代码清理 | P1（O1 前置） | ✅ 完成（2026-08-28，日志 ~60MB→234B/轮） |
| O6 | RoCE 组网推进（4090 RoCE ↔ 910C RoCE 接入 10.92.x 骨干） | 基础设施 | P1 | ⬜ 待 IT |

---

## 逐项说明

### O1（P0）—— 五修复合并提交上游 `FlagRT/FlagCX` `kistich/ascend-dev1.0`
- **背景**：P2（死锁）/ P6（数据错乱）/ P7（OOM）/ P8（设备侧 reduce）/ P9（net.cc 完成判定）全部在本地工作树实测闭环（30/30 集合级 + 50 步训练），但**均未 commit 到 FlagCX 源码仓库**。
- **前置**：O5（剥离诊断打点）——工作树残留 P1-P4 打印，直接提交会污染上游。
- **范围**：`launch_kernel.cc` / `group.cc` / `cann_adaptor.cc` / `uni_runner.cc` / `net.cc` / `cuda_adaptor.cc` / `flagcx_device_reduce.cu` / Makefile（COMPILE_KERNEL 拆分）/ nvidia.mk（C++17、sm_89）/ nvidia_gencode.mk。
- **验证**：合并后跑 `test_ag_hetero.py` 10 轮 + 50 步训练回归；提交说明引用本仓库看板批次。

### O2（P2）—— 昇腾 DAG 引擎解锁
- **背景**：CANN UVA **已实测可用**（`aclrtMallocHost` + `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer` 全通，`aclnnInplaceAdd` 以 host 映射地址为输入算出正确结果）。真缺口是 **cann adaptor 结构体 `hostGetDevicePointer` 字段留 NULL**（上游适配未写完）。
- **需要的两部分**：
  1. cann adaptor 补 `hostGetDevicePointer`（对接 `aclrtHostGetDevicePointer`）；
  2. 昇腾设备侧 reduce kernel（host FIFO 轮询语义）——或先用 **aclnn 折中**实现 DAG 的 reduce 节点（复用 P8 已验证的 `aclnnInplaceAdd` 路径）。
- **收益**：解锁 DAG 引擎的片级流水线重叠（Ring/Sliced/Tree AllReduce），在 RoCE 数据面之上进一步逼近同构吞吐。
- **风险**：需新 kernel 开发（昇腾侧无现成实现，参考 `adaptor/kernel/nvidia/` 的 `.cu` 与 `du/` 燧原实现）；当前朴素路径 + 设备侧 reduce 已可跑，故定为可选优化、不阻塞。

### O3（P2）—— `flagcxGetLastError` 存根完善（缺陷 4）
- **背景**：`flagcxGetLastError` 目前是 TODO 存根，**吞掉错误信息**（本轮 P7 排障时"flagcxComm is not fully initialized"就是 `flagcxGetLastError(NULL)` 的固定兜底串，误导定位）。
- **需要**：实现错误码表 + 线程局部错误记录；`DEVCHECK` 失败时记录到线程局部错误（P7 的 cudaMallocAsync OOM 就是 DEVCHECK 静默返回，无任何日志）。
- **验证**：注入已知错误（如显存不足），确认错误串可精确定位到调用点。

### O4（P3）—— socket 协议无 tag 匹配加固
- **背景**：socket 收发匹配完全依赖 ctrlSock 上"先交换 4 字节 size"的 FIFO 握手，**无 tag/序列号**；多 op 排队 + send 延迟场景存在理论错位风险。
- **现状**：P9（eventSynchronize）已消除当前实际路径的竞态（30/30 全过），此为协议级防御性加固，非紧急。
- **需要**：握手阶段附加 opId/序列号（或 size+校验），收发两侧校验；注意与旧版本兼容。
- **验证**：构造多 op 并发压测场景，确认零错位。

### O5（P1，O1 前置）—— 诊断打点剥离
- **背景**：net.cc/proxy.cc 等仍残留 P1-STATE / P1-COPY / P1-PROGRESS / P2-COPY / P4-SEND-DATA / P4-RECV-DATA 等诊断打印（每步 ~60MB stderr）。
- **需要**：逐文件清理；保留可开关的 `TRACE`/`INFO` 宏路径（默认关闭）。
- **收益**：① O1 提交干净；② 单步 sync 预计再快 ~10s（P9 后已从 47.5s 降到 26.6s，剥离后仍有空间）。

### O6（P1，待 IT）—— RoCE 组网推进
- **背景**：4090 RoCE（192.168.10.0/24，无网关）与 910C RoCE（10.120.73.0/24，VLAN2173）均未接入 10.92.x 骨干；此前 traceroute 死在 hop8=10.92.100.66。
- **需要（IT）**：4090 RoCE 网段接入骨干 + 配置网关 + 放行 **UDP 4791** + **无损 QoS（PFC/ECN）**。
- **入口**：打通后启动 `docs/hetero_progress_and_roce_proposal.md` §6.3 三件事（修 IB 适配器 → 同构 RDMA → 异构 RDMA → GDR 实测）。

---

## 状态图例

⬜ 未开始 ｜ 🔄 进行中 ｜ ✅ 完成 ｜ ❌ 取消/不再需要

## 更新约定

- 每项完成时更新状态 + 补充验证数据（与看板批次一致）。
- 新发现的遗留项随时追加（保持编号递增）。
