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

## 下一期（v2 候选，本原型期不做）

| # | 事项 | 说明 |
|---|---|---|
| N1 | **按 BAAI·FlagTree 官方手册 ascend3.5 线重建训练腿祖先镜像** | 目标：用手册的 ascend3.5 基座镜像（`flagtree-ascend3.5-910c-…:202608-torch2.10.0-vllm0.20.2`）+ `triton_v3.5.x` 自编译路径，产出一个**真正可从零重建、公开可锚定**的祖先镜像，替换现 `pytorch-plugin-fl:manual-20260807-ascend-dev`（BAAI 私有 harbor、需账号、不透明手搭、无 Dockerfile）。<br>**前置核查**：手册那个基座镜像的 registry 是否公开可拉。<br>**做法**：按版本化规范开 `dev/images/ascend-operator-runtime/v2/`（新血统，非原地换）；FlagOS 组件（Torch-FL / FlagGems / FlagCX）在新基座上重编；重跑 functional-repro + 两段 demo 验收。<br>**风险**：组件此前只在现基座上验证过，换基座可能暴露集成差异。<br>**归属**：总组排期，对接战略文档 §6 步骤 5 遗留能力缺口。 |
