# dev/images TODO —— 基座镜像可复现化

> 负责：**总组**（依据各子方向 `dev/<方向>/STATUS.md` 收拢裁定；构建/登记可指派执行方，进度由总组跟踪）
> 建档：2026-09-09 ｜ 状态：进行中

---

## 现状（一句话）

`dev/stack.lock.910c.v1.yaml` 锁定的训练腿镜像**仍是 npu1-27 上那份预构建、无 digest 的原镜像**。
`dev/images/` 目前只有**从 `docker history` 还原的构造配方**（`repro_status: 🟡`），**尚未重建、未验证、未发布**。
在下列任务全部完成前，训练腿基座不具备"可复现 / 可从 registry 拉取"的保证。

---

## T1 · ascend-train-comm（训练腿，🟡 → 🟢）

- [ ] 从 npu1-27 `docker save` 原镜像到第二位置 / harbor（**先止血**：当前全球仅一份，无备份）
- [ ] 向镜像 owner（FlagPerf/FlagRT 侧，经 drinkle 对接）索取：
  - [ ] FlagCX 2 个 patch 文件本体（现只有 sha256 `bc84ea26…` / `524654689…`）
  - [ ] `flagcx-0.13.0-cp311-cp311-linux_aarch64.whl`（sha256 `eb7a2676…`），或据 commit `55eb2ff` + patch 自建
  - [ ] `assets/FlagGems-DSA-__init__.py`（一处 COPY 覆盖，未从镜像取出）
- [ ] 确认 CANN 基座来源（`sha256:a36a3022…`，疑似 FlagPerf `flagperf_ascend:v2025`）
- [ ] 按 `requirements-runtime.txt` 重建 `wheelhouse/`（arm64 / py311）
- [ ] `ascend-operator-runtime/v1/build.sh` → `ascend-train-comm/v1/build.sh` 跑通
- [ ] 重建产物与原镜像比对（`verify_flagcx_runtime.py --static` + 关键包 `pip freeze` 一致）
- [ ] 推 `harbor.baai.ac.cn`，登记可拉取 digest
- [ ] `lock.yaml` `repro_status` 改 🟢；`stack.lock` train.image 改指 digest，changelog 记一条

## T2 · ascend-operator-runtime（训练腿父层，🟡 → 🟢）

- [ ] 同 T1 的父层部分：CANN 基座来源、`wheelhouse/`、`mpich-4.1.3.tar.gz`（登记 sha256）、
      Torch-FL(`af50297…`)/FlagGems(`f7ae8e6b…`) 源码按 commit 检出
- [ ] `build.sh` 跑通并与原镜像比对
- [ ] 推 harbor + 登记 digest + `repro_status` 改 🟢

## T3 · ascend-infer-vllm（推理腿，🟢，仅加固）

- [ ] 按 digest `sha256:5cf8a2b6…` mirror 到 `harbor.baai.ac.cn`（防 quay 上 rc 标签被 GC/覆盖）
- [ ] 连带留存从该镜像拷出的 `triton_ascend 3.2.1` wheel（见 VERSIONS.md）
- [ ] device-context 定 `transformers` pin 回写 `stack.lock`（仓库内 5.15.1 / 5.5.3 两处不一致）

## T4 · 规范落地

- [ ] 各子方向在 `dev/<方向>/STATUS.md` 按 §8.2 建/维护状态文件，基座相关诉求写入其中
- [ ] 总组每阶段交界复核一次 `custom` 组件 commit 是否跟进上游、`official` 是否有新版本

---

## 验收（本 TODO 关闭条件）

`stack.lock` 两条锁定镜像均满足：`repro_status 🟢` + 带可拉取 digest + `dev/images/<name>/v1/` 配方可 `build.sh` 重建并与原镜像一致。

## 关联位置（改动同步回本文件）

- `dev/stack.lock.910c.v1.yaml` — `lock.train.registry_status*`、入锁门槛规则
- `dev/images/*/v1/lock.yaml` — `repro_status`、`gaps`
- `docs/运行时层原型验证-战略目标-910C.v1.md` §6 步骤 1、§7「训练腿镜像可分发性缺口」
