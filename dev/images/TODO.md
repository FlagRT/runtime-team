# dev/images TODO —— 镜像归档 / 重建的待办

> 只跟踪本目录职责内的事（归档、重建、索引）。项目层面的诉求（谁用、路线选型、验收）见 `dev/stack.lock.910c.*.yaml`（当前生效版本见该文件自身状态）/ `docs/`。

## 已完成（2026-09-09 ~ 09-10）

- **离线归档**：`ascend-operator-runtime-0.2.0` / `ascend-train-comm-0.1.3` / `cann-base-manual-20260807` 三个 `docker save` 压缩包 + `recovered/` 重建资产，迁移到 raid（清单见各 `v1/ARCHIVE.md`，本机路径见 `v1/BACKUP.local`）。
- **配方还原**：从 npu1-27 本地镜像 `docker history` 还原 `Dockerfile`（逐字节目标）；写出 `Dockerfile.repro`（功能等价）+ `build.sh`。
- **重建验证**：`Dockerfile.repro` 实机重建两镜像，`pip freeze` 逐行一致、FlagCX `.so` 逐字节一致（详见各 `v1/REBUILD.md`）。两镜像 `repro_status` → 🟢。

## 待办

| # | 事项 | 说明 |
|---|---|---|
| 1 | 逐字节复现（可选金标准） | 需 owner 交出 FlagCX 私有 commit `55eb2ff` + 2 个 patch + 干净 python3.11.15 构建上下文。见各 `v1/lock.yaml:gaps`。总组以 issue 形式向 owner 索取。 |
| 2 | `ascend-infer-vllm` 离线副本（可选加固） | 按 digest `sha256:5cf8a2b6…` 存一份，防 quay rc 标签被 GC；连带留存拷出的 `triton_ascend 3.2.1` wheel。见 `ascend-infer-vllm/v1/ARCHIVE.md`。 |
| 3 | `image_list.md` 维护 | 新镜像/新版本入档时更新索引表与"当前生效版本"。 |
| 4 | v3 真机验证收尾 | **技术验证已完成，见下方「N2 完成」**；剩余是两项需总组拍板的决策（是否吸收私有 fork 的 sync 修复；`shmem` 依赖缺口三个候选方向），详见 `ascend-train-comm/v3/REBUILD.md`「ROUND 3」与 `ascend-operator-runtime/v3/REBUILD.md`「ROUND 2/3 真机验证结果」。 |

## N1 完成（2026-09-12）——按 BAAI·FlagTree 官方手册 ascend3.5 线重建训练腿祖先镜像

**结论：候选血统已产出并实测通过，`dev/images/ascend-operator-runtime/v2/` +
`dev/images/ascend-train-comm/v2/`。未改 `dev/stack.lock.910c.v1.yaml`——是否
切换现网生效版本由总组另行裁定。**

- **基座**：`harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2`（BAAI·FlagTree 官方公开预构建镜像，本机已存在，公开可拉，非 BAAI 内部手搭），取代 v1 的 `pytorch-plugin-fl:manual-*`。
- **triton**：FlagTree 上游仓库 `triton_v3.5.x` 分支 pinned commit `15ec1a6cbc8d51f597f46459a500e96f3812c58f`（只读上游，未动 FlagRT fork），`FLAGTREE_BACKEND=ascend` 现场编译。**实测版本 `triton.__version__ == 3.5.1`**（手册版本线表格暗示 3.5.0，按实测记录，未采信假设）。
- **triton-ascend 后端补丁**：FlagTree 的 triton 3.5.1 已把 ascend 后端重构为插件式 `backend_strategy_registry`，v1 的补丁脚本对新代码 13 条规则 12 条不命中。写了新脚本 `assets/patch_triton_ascend_flagtree.py`（新增 `"torch_fl"` category，纯增量，不影响华为官方 torch_npu 主线）。真机测试中额外发现并修复 triton 顶层驱动"2 active drivers"误判（Torch-FL 的 `torch.cuda.is_available()` 生态兼容 shim 与 FlagTree nvidia 驱动自检的冲突）。
- **FlagCX（训练腿通信层核心目标）**：改用 `FlagRT/FlagCX` 组织仓公开主干 tip commit `4e0e0cbcbf721169ca82348080f8353aebfe2c31`（同步自 flagos-ai/FlagCX 公开上游），`USE_ASCEND=1 pip install . --no-build-isolation` 端到端源码构建，**无需任何 owner 私有 commit / patch / vendor 回收件**——`dev/images/ascend-train-comm/v2/lock.yaml` 不再有 `gaps` 段，逐字节可复现（因为源头本身就是可复现的公开构建，不是"逐字节匹配一个黑盒终点"的意义上）。
- **torch_npu 共存问题实测结论**（任务书要求的关键判断）：**是进程内 import 顺序约束，不是"两个包不能共装"的约束**。先 `torch_fl` 后 `torch_npu`：安全，无异常，PrivateUse1 仍是 `"flagos"`；反过来先 `torch_npu` 后 `torch_fl`：`torch_fl` 立即抛出清晰 `RuntimeError`，fail-loud 不损坏状态。v2 因此不卸载基座自带的 `torch_npu`，也不再复用 v1 的"构建期断言不存在"，改为验证"import 顺序安全"这一真正不变量。详见 `ascend-operator-runtime/v2/REBUILD.md`「torch_npu 共存问题」。
- **真机验证**（2 卡，容器 `flagos-cand-train-910c-v2`，验证完已释放）：Torch-FL/FlagGems 算子自检通过；FlagCX `verify_flagcx_runtime.py --device` 两个 rank 均 `device_all_reduce_ok: true`；复用 `dev/communication/probes/communication_correctness.py`（未重复造轮子）40/40 all_reduce/all_gather/p2p/async_all_reduce 通过（fp32+bf16）。
- **已知遗留问题（未修复，已记录）**：集合通信成功完成、结果数值正确之后，进程退出阶段有 `free(): invalid pointer` SIGABRT（已隔离验证与 torch_npu 无关，疑似 FlagCX/Torch-FL 退出期清理顺序冲突，需 FlagCX/Torch-FL 侧协作定位）；不影响训练期间通信正确性，只影响"用退出码判断成功"的批处理式脚本。见 `ascend-train-comm/v2/lock.yaml:known_issues`。
- **归档**：两镜像 `docker save` + gzip 至 `/mnt/raid/xliu969/flagrt-image-backup/20260912/`，清单见各 `v2/ARCHIVE.md`。

## N2 完成（2026-09-23）——v3 待验证项真机跑完，两个新 STOP CONDITION 澄清/发现

**结论：本机（npu1-27）带卡容器并发数从交接时的 5 回落到 2，未换机器即在原机器
继续完成。训练腿"缺 sync"假说证实成立；推理腿 coexistence 检查通过，但发现一个
全新的、跟原先担心的两个 bug 无关的 STOP CONDITION。两项都不是"验证通过"，
是"技术层面查清楚了，剩两个需要总组拍板的决策"。详见
`ascend-train-comm/v3/REBUILD.md`「ROUND 3」与
`ascend-operator-runtime/v3/REBUILD.md`「ROUND 2/3 真机验证结果」。**

- **训练腿 sync 假说验证**（2 卡，容器 `v3-validate-flagcx-sync-910c` /
  `-followup-910c`，验证完已释放）：`flagcx_sync_test.py` 真机直接测试
  ——`broadcast`/`all_gather` 无 `torch.npu.synchronize()` 时复现原 STOP
  CONDITION（broadcast 目标 rank 收不到数据，all_gather 恒返回全零），加 sync
  后两个原语两个 rank 全部转为 PASS。**结论：2026-09-22 的原诊断是测试遗漏
  同步导致的误报，不是 FlagCX 本体数据损坏。** 追加跑
  `train_qwen_1_5b_npu_syncpatch.py`（猴子补丁 `dist.broadcast`/`all_gather`
  后跑真实训练脚本）确认了预先记录的限制：DDP 内部走 C++ 层
  `torch._C._distributed_c10d._verify_params_across_processes`，不经过
  Python 猴子补丁，仍报参数不一致——**真正的修复点必须落在 FlagCX/vllm_fl 自己
  的 c10d 实现里（私有 fork commit `5d545c9` 实际打的位置），不能靠训练脚本
  调用方绕过**，真实 loss/吞吐数据仍待"是否吸收私有修复"这个决策落地。
- **推理腿 coexistence + 推理烟雾测试**（1 卡，容器
  `v3-validate-train-910c-r2`，验证完已释放）：torch_npu Route A、FlagCX
  import、triton driver、vllm+vllm_fl platform 注册**全部 PASS**——之前担心的
  两个私有 fork PrivateUse1 类型提升 bug **均未在本血统触发**。真实
  `LLM(...).encode()` 调用：第一次撞上验证脚本自身的 API 版本问题（`task=`
  参数在本血统实装的 vllm 0.20.2 里已拆成 `runner`/`convert`，已修复脚本）；
  修复后重跑，**发现新 STOP CONDITION**：FlagGems 的 Triton-Ascend
  grid-stride-loop kernel 代码路径（RoPE cos/sin cache 计算触发）间接
  `import shmem`，这个模块既未装在镜像里也不是公开 PyPI 包（
  `pypi.tuna.tsinghua.edu.cn/simple/shmem/` 404），`Dockerfile.repro` 从未
  提及，vLLM engine core 初始化失败。round 1/2 的静态自检/训练侧验证从未触发
  这条代码路径，是本轮真实跑一次模型前向计算才第一次暴露。**按任务书要求未
  尝试绕过**（候选绕过方式均属"自行决定"，已记录三个候选方向待拍板）。
- **两个脚本迁入永久归档**：`flagcx_sync_test.py`、
  `train_qwen_1_5b_npu_syncpatch.py` → `ascend-train-comm/v3/assets/`；
  `v3_step5_validate.py`（已修复 API 用法）→
  `ascend-operator-runtime/v3/assets/`。临时目录
  `dev/images/v3-pending-validation/` 已按其自身 README 的"验证完应删除"要求
  整体删除。
- **仍未解决（需拍板，非技术阻塞）**：① 是否吸收 FlagRT 私有 fork 的 sync 修复
  进 v3 默认血统；② `shmem` 依赖缺口三个候选方向（找官方来源补进 Dockerfile /
  关闭 grid-stride-loop kernel 路径 / 评估对 embedding 模型完全不激活
  FlagGems）。两项都已把技术判断依据写清楚，等总组/接手人拍板后回填
  `Dockerfile.repro` + `lock.yaml` provenance。`repro_status` 两条血统均维持
  🟡 partial-repro，不升级为 🟢。


## N3 完成（2026-10-08）——两项决策拍板落地：声明式覆盖层 + 官方开关解堵

**结论：决策①（吸收 sync 修复）以"公开基座 + 声明过的补丁"方式落地；决策②
（shmem 缺口）用插件官方黑名单开关临时解堵并同步提跨组需求。镜像重建+真机
验证 PENDING（本机带卡容器 5 个超上限 3，排队）。** 详见
`ascend-operator-runtime/v3/lock.yaml` changelog 2026-10-08 条目与
`docs/v3-决策备忘录与shmem跨组需求-20261008.md`。

- **决策①落地物**：`assets/patches/vllm-plugin-FL/0001-fix-ascend-int64-mask-promote-flagcx-sync.patch`
  （git format-patch 原样导出自 FlagRT fork `5d545c9`，sha256
  `b97d0d8b…e23b5f928`，实测在上游 pin `8b059122e` 上干净套用零冲突）；
  `Dockerfile.repro` vllm-plugin-FL 层新增 `git apply --check`（fail-closed）；
  `build.sh` 同步拷贝补丁；lock.yaml 该层 origin → `official+custom-overlay`，
  记录 overlay_patches（来源/sha256/套用基线/保质期）。不切换私有 fork。
- **决策②落地物**：`docker-compose.flaggems-cos-off.yml`（opt-in 层，
  `VLLM_FL_FLAGOS_BLACKLIST=cos`）——vllm-plugin-FL 官方一级开关
  （`vllm_fl/worker/worker.py` 读取），把 torch.cos 拉回 torch_npu 原生，
  零代码改动；`docker compose config` 实测两层合并正确。选黑名单而非总开关：
  影响面最小、可逆、不遮掩问题。
- **upstream-first 同步启动**：修复已具备提交上游条件（干净 patch + 真机证据），
  PR 文本草稿见决策备忘录附录。
- **跨组需求已建档**：shmem 官方出处（FlagTree triton tle/dsa/ascend 扩展的
  依赖声明缺口）已写成独立需求文档，待转算子编译组。
- **下一步（按依赖排序，2026-10-08 深夜更新）**：
  ① ~~重建镜像~~ **已完成**：operator-runtime `be30a952c2eb` / train-comm
  `7028028bb62c`（含覆盖层补丁，三重证据验证，见各 REBUILD.md「ROUND 4」）。
  ② 真机双腿验证 **PENDING→迁移 npu1-11**：npu1-27 带卡容器并发超限（守候 4h
  无窗口）且 docker 存储层校验和损坏（save 不可用）；npu1-11（10.120.72.11，
  16 卡零占用）已就位：仓 dev-1.0@31c741d 同步完毕、harbor 基座拉取中、模型
  rsync 中，到货后按本目录配方本地重建+验证。
  ③ 文档回填：本轮已完成（lock.yaml×2 / REBUILD.md×2 / image_list / TODO）。
  ④ **新增待办**：v3 round 4 镜像 docker save 离线归档 raid——被 npu1-27 存储
  损坏阻塞，待 npu1-11 重建成功后在那边 save 归档（先例各 v2/ARCHIVE.md）。
  ⑤ 上游 PR 提交与跨组需求发出并行推进（PR 草稿见决策备忘录附录 A/B）。
