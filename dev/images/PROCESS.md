# dev/images 过程记录 —— 候选血统的构建与验证过程

> 只记录"怎么走到今天这个状态"的过程性叙述（尝试了什么、为什么换路线、round 之间的
> 因果关系、STOP CONDITION 的发现与处理）。**当前状态的事实性结论**见 `image_list.md`
> （索引，含层级视图）与各 `<name>/vN/lock.yaml`（权威，逐层 pin + changelog）。
> 本文件按芯片 → 版本线 → round 组织，是跨系列（`ascend-operator-runtime` +
> `ascend-train-comm` 等）的综合叙述，只做跨系列的因果串联——单个系列自己的完整技术细节
> 仍以其自己的 `REBUILD.md` 为准。
>
> 设计动机与黄金准则见
> `docs/运行时基座建设-黄金准则与方案存档.v1.md`；本文件是该准则下 910C 候选血统
> 迭代过程的存档，是否切换生效版本由总组在 `dev/stack.lock.910c.*.yaml` 裁定。

---

## 昇腾 910C

### v1 → v2：从"自建底座 + pip wheel 组合"切到"按 BAAI·FlagTree 官方手册路径构建"（2026-09-12）

v1（现网生效版本）不是按 BAAI·FlagTree 官方 ascend3.5 手册的构建路径产出的：推理腿
直接用**华为昇腾官方** `vllm-ascend`；训练腿用 **BAAI 内部**手搭的 CANN 9.0.0 底座 +
pip 装上游 `triton==3.5.0` 官方 wheel + **华为昇腾官方** `triton-ascend==3.2.1`
（从 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 镜像里拷出的 wheel，非公开 PyPI 分发）
+ FlagOS 组件（Torch-FL / FlagGems / FlagCX，FlagCX 依赖 2 个 owner 私有 patch）。
版本纪元对齐 ascend3.5，但构建方式自成一路，不是手册规定的路径。

`dev/images/*/v2/` 第一次真正按手册路径构建：**BAAI·FlagTree 官方**预构建基座镜像
（取代 BAAI 内部手搭底座）+ checkout FlagTree 上游 `triton_v3.5.x` 分支 pinned commit
`15ec1a6cbc8d51f597f46459a500e96f3812c58f` + `FLAGTREE_BACKEND=ascend` 现场编译
（取代 pip 装 wheel 组合）。实测 `triton.__version__ == 3.5.1`（手册版本线表格暗示
3.5.0，按实测记录，未采信假设）。FlagCX 层改用 `FlagRT/FlagCX` 组织仓公开主干 tip
commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`，**不再依赖任何 owner 私有
commit/patch**，`lock.yaml` 不再有 `gaps` 段。

**关键实测结论（后续 v3 的设计依据）**：torch_fl 与 torch_npu 的冲突是**进程内
import 顺序**约束，不是"能否共装"的约束——先 `torch_fl` 后 `torch_npu`：安全；反过来
先 `torch_npu` 后 `torch_fl`：`torch_fl` 立即抛出清晰 `RuntimeError`，fail-loud。
v2 因此不卸载基座自带的 `torch_npu`，两者共装但从未真正被用于计算（默认路径仍是
`torch_fl`/flagos）。

真机 2 卡验证：Torch-FL/FlagGems 算子自检通过；FlagCX all_reduce/all_gather/p2p/
async_all_reduce 40/40 通过（fp32+bf16）。已知遗留问题：集合通信正确完成后，进程
退出阶段有 `free(): invalid pointer` SIGABRT（已隔离验证与 torch_npu 无关，疑似
FlagCX/Torch-FL 退出期清理顺序冲突），不影响通信正确性本身。

**候选状态**：未进入 `dev/stack.lock.910c.v1.yaml` 的 `lock:`（现网仍锁 v1），
是否切换由总组另行裁定。

### v3：Route A 默认 + 训练/推理统一血统（起草 2026-09-22，round 1-3 实机验证）

**目标**：在 v2 基础上迭代，把设备后端默认路径从"torch_npu 与 torch_fl 共装但从未
真正被用于计算"（v2 结论）切换成"torch_npu 是唯一默认路径，torch_fl 保留但降级为
需要显式反向操作才能激活的 opt-in 组件"（**Route A**，对齐
`docs/运行时基座建设-黄金准则与方案存档.v1.md` GR4）；训练腿通信从"flagos 适配"
切到"HCCL 适配"（torch_npu 原生 + FlagCX 自己注册的 `flagcx` c10d 后端）；新增
vLLM 推理插件层，使同一血统的镜像既能跑训练也能跑 vLLM 推理服务。

#### round 1（2026-09-22）：vllm-ascend——STOP CONDITION，已被 round 2 取代

vLLM 推理插件选用华为昇腾官方 `vllm-ascend`。精确 pin commit 核实无误
（`367b8e62da799870a7476ce34f5f7658589a8aad`，与
`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 内 `pip show vllm-ascend` 输出一致）。
**`docker build --network=host` 在安装层失败**：`vllm-ascend` 硬性依赖
`triton-ascend==3.2.1`，但公开 PyPI 索引从未发布过这个精确版本（只有
`3.2.0rc2`/`rc3`/`rc4`/正式版 `3.2.0`）——这个精确版本号历来只能从
`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` 官方镜像里提取 wheel 获得，不是一条能被
`pip install .` 自动解析出来的公开依赖。按任务书要求未自行绕过（提取 wheel /
联系维护者 / vendor triton-ascend 3.2.0 均未执行，留待日后评估），如实记录为
STOP CONDITION，未产出设计 tag 镜像实体；另建一个跳过该层的诊断变体镜像
（`...-arm64-DIAGNOSTIC-novllm`，image id `4bab61434602`）验证其余部分：Route A
默认生效、torch_fl guard fail-loud、FlagCX `all_reduce`/P2P `send`/`recv` 通信正确
（均真机确认）。

#### round 2（2026-09-23）：改用 vllm-plugin-FL——构建成功

重新核查 BAAI·FlagTree 官方 wiki（真实 clone 读取）发现：该 wiki 的 ascend3.5 线
自己的"Run Qwen vLLM benchmark"一节走的根本不是 vllm-ascend，而是 FlagGems +
`vllm-plugin-FL`（`https://github.com/flagos-ai/vllm-plugin-FL`）。核实其
`pyproject.toml` 核心依赖只有 `pyyaml`，不含 `triton-ascend`，不会撞上 round 1
那堵公开 PyPI 墙；`release/0.2` 分支与本血统 vLLM 0.20.2 精确匹配（`main` 分支对应
vLLM 0.24.0，不用）；entry points 确认走 `vllm.platform_plugins`/`vllm.general_plugins`
机制；`setup.py` 的 `SUPPORTED_VENDORS = ("cuda",)` 确认不设 `VLLM_VENDOR` 时装的是
跳过原生 C++ 扩展的纯 Python 插件，ascend 走 `device_type == "npu"` 识别，与本血统
Route A 默认的 torch_npu 直接对应。

精确 pin：`git ls-remote ... release/0.2` 实测取得
`8b059122e32b9ac47b9820a9c7b1bb95077481a4`（2026-09-22 现场取得）。**构建结果**：
`docker build --network=host` 设计 tag 全部 27 步成功，image id `9ad551058f2f`，
`verify_runtime.py --static` 通过（`vllm_fl_installed: true`）。同步重建
`ascend-train-comm:v3`（父镜像换成新 operator-runtime，FlagCX 层本身未变，image id
`43f3e2f70b4c`）。

**FlagRT 私有 fork 的发现与决策**：本机另有一个 `FlagRT/vllm-plugin-FL` 组织 fork，
已合并一个 PR（commit `5d545c9`，作者 `seanl`）修复了两个真实 bug：① PrivateUse1
后端 `bool*int64` 类型提升错误（导致 token id 坍缩成 1）；② `flagcx` backend
`all_reduce`/`all_gather` 异步返回缺 `torch.npu.synchronize()`。该 fork 与
`flagos-ai/vllm-plugin-FL` `release/0.2` 已分叉（非祖先关系），且验证环境是
torch_fl（flagos）路径，不是本血统默认的 torch_npu（Route A）路径。按"纯公开血统"
要求，本层**刻意不用**这个私有 fork/patch——这两个 bug 是否会在 torch_npu 路径上
实际触发，只能靠真机推理的真实输出判断，不能靠读代码预先断言。

FlagGems pin 未改动（官方 README 建议不同 commit，按"先用已 pin 的 commit 试"的
要求，静态构建/装配层面完全通过）。真机 Step 5 本轮会话未能执行——机器带卡容器
并发数持续超过并发上限 3（其他工程师的活跃 sweep/benchmark 负载，非本任务所起），
按"respect shared-machine rules"未强行起容器，记为 PENDING。

#### round 3（2026-09-23，卡资源释放后继续）：两条腿分别推进，各自发现新结论

本机带卡容器并发数从交接时的 5 回落到 2，未换机器即在原机器继续。

**训练腿（`ascend-train-comm:v3`）**：round 2 私有 fork 提出的"flagcx backend 异步
返回，缺 `torch.npu.synchronize()`"假说，用真机 2 卡直接测试**证实成立**——
`broadcast`/`all_gather` 无 sync 时复现原 STOP CONDITION（broadcast 目标 rank 收不到
数据，all_gather 恒返回全零），加 sync 后两个原语两个 rank 全部转为 PASS。**结论
更新**：2026-09-22 的原 STOP CONDITION 是诊断脚本遗漏同步导致的误报，不是 FlagCX
本体数据损坏。追加跑猴子补丁版训练脚本确认了预先记录的限制：DDP 内部走 C++ 层
`torch._C._distributed_c10d._verify_params_across_processes`，不经过 Python
猴子补丁，仍报参数不一致——**真正的修复点必须落在 FlagCX/vllm_fl 自己的 c10d 实现
里**（私有 fork commit `5d545c9` 实际打的位置），不能靠训练脚本调用方绕过。真实
loss/吞吐数据因此仍未产出，卡在"是否吸收私有 fork 的 sync 修复"这个需要总组拍板的
决定上。

**推理腿（`ascend-operator-runtime:v3`）**：coexistence 检查（torch_npu Route A +
flagcx + triton + vllm/vllm_fl 同进程）真机 **PASSED**——之前担心的两个私有 fork
PrivateUse1 类型提升 bug 均**未在本血统触发**。真实推理烟雾测试两轮尝试：第一轮
撞上验证脚本自身的 API 版本问题（`task="embed"` 在本血统实装的 vllm 0.20.2 里已被
拆成 `runner`/`convert` 两个参数，现场核实新签名并修复脚本）；第二轮（脚本修复后
重跑）**发现新 STOP CONDITION**：`torch.cos` 在 NPU 上被 FlagGems 的 Triton-Ascend
算子透明接管，针对 RoPE cos/sin cache 这个张量形状选择了"grid-stride-loop"风格
kernel，该分支生成代码间接 `import shmem`——这个模块既未装在镜像里，也不是公开
PyPI 包（404），`Dockerfile.repro` 从未提及。直接后果：`CompilationError` →
vLLM engine core 初始化失败 → 未产出任何真实 `.encode()`/`generate()` 输出。
round 1/2 的静态自检与训练侧 collective 验证从未触发这条代码路径，是本轮真实跑一次
模型前向计算才第一次暴露。**未尝试绕过**（候选绕过方式——装未经核实的 `shmem`
wheel / 关闭 FlagGems 让 `cos` 落回 torch_npu 原生实现 / 手写空 `shmem` stub——均属
"自行决定的绕过"，可能掩盖 FlagGems 在 Ascend 上更广泛的缺口，留待项目负责人
裁定）。

**退出期 SIGABRT**：v3 默认路径（不导入 torch_fl）下**确认复现**，推翻了"可能因
不装 torch_fl 而不复现"的推论，根因需重新定位（不是 STOP CONDITION，不影响已打印
的正确结果）。

**结论（round 3 收尾，2026-09-23）**：两条血统的 `repro_status` 维持
🟡 partial-repro。技术层面已查清楚阻塞点，剩两项需要总组拍板的决策：
① 是否吸收 FlagRT 私有 fork 的 sync 修复进 v3 默认血统；② `shmem` 依赖缺口三个
候选方向（找官方来源补进 Dockerfile / 关闭 grid-stride-loop kernel 路径规避 /
评估对 embedding 类模型完全不激活 FlagGems）。详见
`ascend-operator-runtime/v3/REBUILD.md`「ROUND 2/3 真机验证结果」与
`ascend-train-comm/v3/REBUILD.md`「ROUND 3」。

**候选状态**：未进入 `dev/stack.lock.910c.v2.yaml` 的 `lock:`/`candidates:`，
是否登记候选、是否切换由总组另行裁定——鉴于两项决策仍待拍板，**不建议在完成前
登记为候选**。
