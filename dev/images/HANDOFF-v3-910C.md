# 交接文档：910C 训练/推理统一基座 v3（Route A 化）

> 写于 2026-09-23，从当前机器切换到新机器时用。新 session 打开后建议先读完本文档全文，
> 再决定下一步，不用重新翻聊天记录。原始任务书见
> `docs/运行时层原型验证-战略目标-910C.v1.md`（v3 相关的具体诉求以下面「任务目标」段为准）。

## 任务目标（浓缩版）

把 910C 训练/推理统一基座从 v2 迭代到 v3，满足 6 条约束：

1. **Route A 为默认生效后端**：torch_npu 是实际生效的设备后端（官方 autoload），不是"共存但不用"。
2. **训练通信切到 HCCL adaptor**：FlagCX 消费路径对齐 `train_qwen_1_5b_npu.py` 的实测路径，不再走 flagos adaptor。
3. **训练 + 推理统一同一血统**：同一镜像既能跑 DDP 训练又能跑 vLLM 推理。
4. **FlagGems/FlagTree 保留可插拔，不强制激活**。
5. **torch_fl 降为可选、非默认**。
6. **纯净公共基座**：只用公开 commit，不引入 owner 私有依赖缺口。

验收要求：不能只是"不报错"，必须真机跑出真实数据（loss 曲线/吞吐、真实推理输出）；
遇到需要私有 patch / 真机数值错误 / 官方组件驱动冲突 / 结果明显劣于历史水平这几种情况，
要停下来汇报，不能自行绕过决定。

## 现状总结（按时间顺序）

### Phase 1 — 设计稿（已完成）
`dev/images/ascend-operator-runtime/v3/` + `dev/images/ascend-train-comm/v3/` 的
`Dockerfile.repro`/`lock.yaml`/`REBUILD.md`/`build.sh`/`ARCHIVE.md` 全部写好，6 条约束
逐条落实，开放问题都标注为"待真机核实"而非静默猜测。

### Phase 2 Round 1 — 真机验证第一轮（vllm-ascend 路线）
真机确认（不用重测）：
- Route A 默认生效：`torch._C._get_privateuse1_backend_name() == "npu"`，零特殊 import 顺序 ✅
- torch_fl opt-in、抢注后 import 立刻 fail-loud RuntimeError ✅
- FlagGems/FlagTree 可插拔、不默认激活 ✅
- FlagCX `all_reduce`、P2P send/recv 真机 2 卡结果正确 ✅

两个 STOP CONDITION：
- **vllm-ascend 构建失败**：硬依赖 `triton-ascend==3.2.1`，公开 PyPI 从未发布过这个版本
  （只有 3.2.0rc2/rc3/rc4/3.2.0）。→ **已在 Round 2 解决**（见下）。
- **FlagCX `broadcast`/`all_gather` 真机返回错误结果**：`dist.broadcast` 静默 no-op，
  `dist.all_gather`（tensor-list 形式）恒返回全零，导致 `train_qwen_1_5b_npu.py` 在
  `DDP()` 构造阶段直接失败（各 rank 参数形状对不上）。**未解决，见下方「待办 P0」**。
  详见 `dev/images/ascend-train-comm/v3/REBUILD.md`「PHASE 2 STOP CONDITION」一节。
- 附带发现：v2 已知的退出期 SIGABRT 在 v3（不装 torch_fl）下依然复现，推翻了"可能因
  torch_fl 消失而不复现"的猜测，根因待重新定位。

### 方向修正 — vllm-ascend → vllm-plugin-FL
**用一手来源核实**（不是转述仓库文档）：`git clone` FlagTree 官方 wiki 仓库
（`flagos-ai/FlagTree.wiki`）通读，发现官方手册"跑 vLLM 推理"那节根本没用 `vllm-ascend`，
用的是 `vllm-plugin-FL`（`github.com/flagos-ai/vllm-plugin-FL`，FlagOS 自己的 vLLM 插件）。
进一步核实 `vllm-plugin-FL` 仓库本身：`release/0.2` 分支对应 vLLM v0.20.2（跟我们基座
完全一致，`main` 分支对应 v0.24.0 不匹配）；`pyproject.toml` 核心依赖只有 `pyyaml`；
`requirements/ascend.txt` 是空的——**完全不依赖 triton-ascend**，不会踩 vllm-ascend 那个坑。

### Phase 2 Round 2 — 真机验证第二轮（切到 vllm-plugin-FL）
- 精确 pin：`git ls-remote https://github.com/flagos-ai/vllm-plugin-FL.git release/0.2`
  → commit `8b059122e32b9ac47b9820a9c7b1bb95077481a4`。
- `ascend-operator-runtime:v3` 用 vllm-plugin-FL 重建**成功**（27/27 步），
  image `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`
  （id `9ad551058f2f`）——**不再是 DIAGNOSTIC-novllm 变体，这是真的**。
- `ascend-train-comm:v3` 在新 parent 上重建**成功**，
  image `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`
  （id `43f3e2f70b4c`）。
- **真机验证（vLLM 推理 smoke test + 训练/推理共存检查）：卡资源被占满，从 Round 2 开始
  到现在都没跑成，状态是 PENDING（不是"跳过"，不是"假设通过"）。** 验证脚本已经写好，
  见 `dev/images/v3-pending-validation/`。
- **重要发现**：本机存在一个本地私有 fork
  `/home/xliu969/runtime-team/vllm-plugin-FL/`（`.gitignore` 排除，不进主仓库）：
  ```
  origin          git@github.com:FlagRT/vllm-plugin-FL.git
  flagos-upstream https://github.com/flagos-ai/vllm-plugin-FL.git
  ```
  其中 commit `5d545c9`（PR #1，分支 `xliu969/ascend-sync-fix`，作者 `seanl <vlev02@qq.com>`）
  修了两个真实 bug：
  1. `vocab_parallel_embedding`：PrivateUse1 上 `bool*int` 类型提升错误导致 token id 全变 1，
     补丁显式转 `int64`。
  2. `communicator.py`：flagcx 后端的 `all_reduce`/`all_gather` **返回是异步的**，不加
     `torch.npu.synchronize()` 直接读结果会拿到脏数据——补丁在两处都加了 sync。
  
  第 2 点跟 Round 1 发现的 FlagCX broadcast/all_gather "返回错误结果" 高度疑似是同一类问题
  （诊断脚本当时没加 sync）。**这是接下来最该优先验证的假设**。

### Phase 2 Round 3 — 已布置，Task 1 卡资源阻塞，Task 2 已完成
- **Task 1（sync 假设测试，P0，未跑）**：验证脚本 `flagcx_sync_test.py` 已写好
  （对照测试：不加 sync 复现原 STOP CONDITION / 加 `torch.npu.synchronize()` 看是否修复），
  卡资源一直不够，没跑成。
- **Task 2（editable-install + 挂载可行性，已完成）**：
  - vllm-plugin-FL、FlagGems：纯 Python 无编译步骤，`pip install -e` + `docker run -v`
    挂载可以**零重建热切换版本**（真机验证过：装一次之后换挂载目录内容，不重装不重启，
    新进程直接读到新代码）。
  - Torch-FL、FlagTree：都有真实编译步骤，挂载切换后仍需重新编译，达不到零重建效果。
  - 真实代价：挂载切换后 `pip freeze`/`pip show` 版本号不会跟着变（装的时候就冻结了），
    溯源得靠 `git rev-parse HEAD` 挂载目录，不能依赖 pip freeze。
  - Dockerfile 具体改法已设计（默认仍烘焙纯公开 `flagos-ai` 主干保证自包含可复现，
    `docker run -v` 覆盖用于测试），**草案，未应用**，等待拍板。

## 卡资源现状（截至交接时）

机器上带卡容器长期超过"≤3"的硬上限，已经持续 10+ 小时：
`sgl-c256-tp4-20260922-prefill-{0,1,2,3}`（4 个，共占满全部 16 张卡）+
`flaggems-cann9.0.0`（占满全部 16 张卡，已跑 15+ 小时）+ 后来又多了一个 `x-benchmark`
（也挂了全部 16 张卡）。这不像临时性任务，更像长期占用。**新机器需要重新确认卡资源情况，
不能假设跟这台机器一样紧张，也不能假设已经空闲。**

## 待办清单（按优先级）

| # | 事项 | 阻塞条件 | 产出应写回哪里 |
|---|---|---|---|
| P0 | 跑 `v3-pending-validation/race_task1.sh`（驱动 `flagcx_sync_test.py`），验证"漏 sync"假设 | 需要 2 卡窗口 | `dev/images/ascend-train-comm/v3/REBUILD.md` + `lock.yaml` |
| P0 | 根据上一条结论分叉：成立→回去补跑 `train_qwen_1_5b_npu_syncpatch.py` 或改造正式训练脚本拿真实 loss/吞吐；不成立→FlagCX 公开 commit 本身有 bug，需上报/找 FlagCX 维护者 | 依赖上一条 | 同上 |
| P1 | 跑 `v3-pending-validation/race_and_validate.sh`（驱动 `v3_step5_validate.py`）：真实 vLLM 推理 smoke test + 训练/推理共存检查 | 需要 1 卡窗口；模型路径需按新机器调整（见 `v3-pending-validation/README.md`） | `dev/images/ascend-operator-runtime/v3/REBUILD.md` + `lock.yaml`，`image_list.md` 的 repro_status |
| P1 | **需要总组/接手人拍板**：FlagRT 私有 fork 的 sync 修复要不要正式采用进 v3 默认血统（打破"纯公共基座"换真实 bug 修复，还是继续用纯公开 commit 追查/等上游修） | 无 | 拍板后回填 Dockerfile.repro + lock.yaml 的 provenance |
| P2 | **需要拍板**：editable-install + 挂载切换的 Dockerfile 改法要不要正式采用（草案已就绪，未应用） | 无 | `ascend-operator-runtime/v3/Dockerfile.repro` |
| P2 | 全部真机验证通过后，`repro_status` 从 🟡 partial-repro 升级到 🟢，再考虑要不要/何时把 v3 写进 `dev/stack.lock.910c.v2.yaml`（**明确不要在验证完成前自动做**，需总组确认） | 依赖以上全部 | `image_list.md`，`stack.lock` 更新单独走审批 |
| P3 | 协调卡资源：`sgl-c256-tp4-*`/`flaggems-cann9.0.0`/`x-benchmark` 长期占用，建议直接找 owner 沟通释放窗口，而不是继续被动重试 | — | — |
| 收尾 | `dev/images/v3-pending-validation/` 整个目录是临时的，等上面的验证都跑完、结论写回各 `REBUILD.md`/`lock.yaml` 后应该整体删除 | 依赖以上全部 | — |

## 新机器前置条件

1. 真机 Ascend 910C 硬件 + driver（`/usr/local/Ascend/driver`）。
2. docker + 出网（github.com、pypi.org、harbor.baai.ac.cn；`github.com` 网页/WebFetch 类工具
   在这台机器的沙箱里会因域名安全校验失败超时，改用 `git clone`/`curl` 到
   `api.github.com`/`raw.githubusercontent.com` 是可行的，别在 WebFetch 上浪费时间）。
3. SSH 访问 `git@github.com:FlagRT/...`（私有 fork 用，`vllm-plugin-FL` 已验证过；
   `FlagCX`/`FlagGems`/`Torch-FL` 等其他 FlagRT 组织仓同理）。
4. docker 镜像不会跟着 git 走，需要重新构建：先跑
   `dev/images/ascend-operator-runtime/v3/build.sh` 再跑
   `dev/images/ascend-train-comm/v3/build.sh`（构建步骤已验证可重复），
   拿到本机 image id 后再去跑 `v3-pending-validation/` 里的验证脚本。
5. 带卡容器并发规则：机器级别硬上限 3 个同时挂 `--device` 的容器，起容器前
   `docker ps` 自查（见 `dev/stack.lock.910c.v2.yaml` 的 `rules` 段）。
6. 模型文件路径：验证脚本里写死的是这台机器的 RAID 路径
   （`/mnt/raid/hliu553/models/...`），新机器大概率不一样，跑之前先改，
   具体见 `v3-pending-validation/README.md`。

## 新机器上手前两件事（准备好的 prompt，直接复制粘贴用）

- `dev/images/HANDOFF-v3-910C-prompt-docker-storage.md` —— home 目录空间有限，
  docker 存储位置需要先搬到大容量盘（参考这台机器的模式：`/mnt/raid/docker`）。
- `dev/images/HANDOFF-v3-910C-prompt-lan-sync.md` —— 局域网从这台机器
  （`npu1-27`，IP `10.120.73.79`/`10.120.73.80`/`10.120.72.27`）同步已构建好的
  两个 v3 镜像（合计约 31GB，走 `docker save | ssh | docker load` 流式传输）+
  两个模型文件（约 4GB），比重新构建/重新下载快很多。**建议先做存储迁移，
  再做同步**，不然同步过来的东西没地方放。

## 文件索引

- `dev/images/ascend-operator-runtime/v3/`、`dev/images/ascend-train-comm/v3/` ——
  正式交付物（Dockerfile.repro/lock.yaml/REBUILD.md/build.sh/ARCHIVE.md/assets），
  这次改动已经用真实数据回填。
- `dev/images/image_list.md` —— 索引表 + v3 候选血统说明段，`repro_status` 目前是
  🟡 partial-repro（两个镜像都是；不是 🟢，因为真机推理/训练闭环验证还没做完）。
- `dev/images/v3-pending-validation/` —— 本次新增，待执行验证脚本 + 详细 README，
  验证完应删除（见上方「收尾」）。
- `dev/stack.lock.910c.v2.yaml` —— 全程未改动，v3 还不是生效/候选版本，
  提升与否是总组裁定的事。
- `dev/device-context/910C/distributed_training/scripts/train_qwen_1_5b_npu.py` ——
  历史对照脚本（2481 步、loss 1.95、4245-5428 tok/s，HCCL adaptor 路径的原始证明）。
- `/home/xliu969/runtime-team/vllm-plugin-FL/`（本机路径，`.gitignore` 排除，不在主仓库里）——
  本地私有 fork，含关键 sync 修复 commit `5d545c9`。**这个目录本身不会跟着 git 走**，
  新机器如果要用它（比如去跑 Task 1 的对照验证），需要重新 `git clone
  git@github.com:FlagRT/vllm-plugin-FL.git` 并 `git remote add flagos-upstream
  https://github.com/flagos-ai/vllm-plugin-FL.git`，然后 `git log` 确认能看到
  `5d545c9`/`f34b4e7` 这两个 commit。

## 已构建的 docker 镜像（本机，不会跟着 git 走）

| tag | image id | 说明 |
|---|---|---|
| `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64` | `9ad551058f2f` | v3 正式（vllm-plugin-FL 版），Round 2 产出 |
| `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64` | `43f3e2f70b4c` | v3 正式（同上，叠加 FlagCX），Round 2 产出 |
| `...-arm64-DIAGNOSTIC-novllm` 两个 | `4bab61434602` / `76ad7e08b4e7` | Round 1 的诊断变体（跳过 vllm-ascend 层），已被上面两个取代，可以不管 |

新机器上这些 image id 都对不上，需要重新 `build.sh` 构建。
