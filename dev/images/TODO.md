# dev/images TODO —— 镜像归档 / 重建的待办

> 只跟踪本目录职责内的事（归档、重建、索引）。项目层面的诉求（谁用、路线选型、验收）见 `dev/stack.lock.910c.v1.yaml` / `docs/`。

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
