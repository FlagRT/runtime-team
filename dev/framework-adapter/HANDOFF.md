# framework-adapter · 开发续接

更新：2026-09-17。这是可重读的工程记录，不代替实时资源检查。先读 [STATUS](STATUS.md)，再看日期证据；不要从聊天摘要猜环境、测试通过情况或 Git 状态。

## 固定约定

- 负责人顾宬 / cgu135；个人分支 `cgu135/framework-adapter`。9 月 17 日按用户确认及根 README 改为：个人分支 commit，快进同步本地 dev-1.0，dev 合入个人分支解决冲突，个人分支 merge 回 dev 后检查无分叉并 push。不走 PR，不直接在共享分支开发、不 rebase/强推共享分支；main 仍须正式 PR。
- runtime-team 收拢方向入口、原型、探针和证据。已有 FL 子库改动在独立 `vllm-plugin-FL` 分支 `cgu135/safe-op-fallback`，不能假定主仓提交会同步子库，也不把它自动装入官方镜像。
- 周报使用本周新增数字，区分 CPU/NPU、单算子/模型、原型/验收；重复轮次不累加。不写凭证，不擅自同步 OneDrive。
- 不停他人容器、不抢卡、不修改宿主驱动或规避设备隔离。是否有空闲计算和是否获得设备访问是两回事。

## 当前事实与入口

- 公共基线：`origin/dev-1.0@f4d0ddd`，已合入个人本地分支。锁文件是 [v2](../stack.lock.910c.v2.yaml)，训练候选镜像未切换；本轮只用锁定推理镜像。
- PR [#16](https://github.com/FlagRT/runtime-team/pull/16)：2026-09-17 查询 CLOSED、未合并；用户明确要求依根 README 直接 merge/push 到 dev-1.0。本次整理包含 9 月 16 日新结果，提交状态以 Git 实查为准。
- 27：`cgu135@10.120.72.27`，连接凭证不在仓库。用户确认可用；运行前仍要重查资源和权限。
- 本人容器 `flagos-proto-infer-910c`，ID 前缀 `7a13465c20ee`、label `owner=cgu135`；本轮创建，仅映射 davinci0。结果完成后停止，重用前核对归属和配置。
- 宿主持久目录：`/home/cgu135/framework-adapter-910c/acceptance-20260916` → 容器 `/work`。模型 `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B` → `/model:ro`。
- 镜像 ID `2e56022ae5b3…`，digest、模型 SHA256、包版本、脚本 SHA256 均在 [证据目录](docs/evidence-model-20260916/)。不能只对 tag。
- 新入口：`probes/qwen_embedding_baseline.py`；服务器最终执行快照 `/work/qwen_embedding_baseline-v3.py`。本地最终脚本需与 `probe-v3.sha256` 相符。前两版保留在宿主目录，不覆盖旧日志。
- 9 月 16 日收尾已下载最终日志，确认本地/远端 v3 脚本哈希一致、12 项控制测试与语法检查通过，见 [本地核验](docs/evidence-model-20260916/local-verification.md)。当日未提交/推送；9 月 17 日按新流程同步，仍无 OneDrive 写入。

## 本轮结果及边界

- CPU FP32：真实模型 2 组输入通过，8 组代表算子用例通过（6 RMSNorm、2 SiLU）；12 项无硬件控制测试通过。
- NPU FP16：最终 v3 在设备初始化报 `aclInit 507899 / Resource_Busy`，device_count=0；模型/算子 NPU 通过数为 0。内核日志 `Conflict open udevid` 指向设备 0 的命名空间占用冲突，未确认拥有者或唯一根因。
- 关键接入发现：Transformers `Qwen3RMSNorm` 自己展开归一化，不经过 `F.rms_norm`，历史拦截原型不能直接覆盖它。vLLM 融合入口只做了源码核对，未运行引擎，不可混称已观测调用。
- 所有 CPU 结果为诊断 eager、小输入、前向，不是模型检索质量/服务/性能/训练验收，也不是优化算子或回退验证。

## 下一步顺序

2026-09-16 19:08 只读复查：宿主 npu-smi 未列计算进程；运行中的 flagos-proto-train-910c 和 flaggems-cann9.0.0 均映射全部 16 个 davinci 设备节点（后者 privileged）。这只是映射/运行快照，不证明具体占用者；本次没有重跑 aclInit。本人推理容器仍 exited，未修改远端状态，SSH 已退出。

19:22 再查：rag-ljy-vllm-910c 已运行，带卡容器总数达到 3（另外两个同上）；设备 2/3 有 Python 计算，设备 0 未列进程。按锁文件并发规则，没有启动本人的第 4 个带卡容器，也没有执行初始化测试。因此没有新增 aclInit 错误/恢复结论。本人容器保持 exited，SSH 已退出，待协调串行窗口或允许共用的环境。

1. 优先整理接入与回退需求，由组内向算子组确认算子清单、对应关系、调用契约和支持条件；先确认已有机制再写必要适配，不直接扩展为自动识别所有自定义算子。
2. 协调可分配的设备，或明确被允许共用的公共推理容器。不能自行进入他人的运行环境执行探针；先确认配置/设备分配，再恢复测试。
3. 在锁定镜像运行同一探针的 NPU FP16/BF16 模式，保留新输出文件名；不修改容差以掩盖错误。
4. 补充真实 vLLM Embedding 路径观测，分别核对融合 RMSNorm（含 residual）与 SiLU-and-Mul；CPU 的普通模块结果不能替代。
5. 用实际输入与调用链修订接入方案，确认模块入口/注册入口、支持条件与执行后上抛边界；请求至少 1 个下游 review。
6. 更新 STATUS、日期报告和本文件，再按用户授权及标准 merge 流程提交/推送。源码和证据存在宿主及本地，不只留在容器可写层。
