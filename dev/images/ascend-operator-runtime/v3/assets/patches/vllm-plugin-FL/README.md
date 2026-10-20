# vllm-plugin-FL 声明式覆盖层补丁（decision ①，2026-10-08 总组拍板方向落地）

本目录存放 v3 血统对上游 `flagos-ai/vllm-plugin-FL` 的**显式声明的覆盖层补丁**——
不切换到 FlagRT 私有 fork，保持"公开基座 + 声明过的补丁"性质。每个补丁一个文件，
带 sha256 记入 `lock.yaml`，上游合入后整层删除（补丁有保质期，见各补丁头部）。

## 0001-fix-ascend-int64-mask-promote-flagcx-sync.patch

- **sha256**: `b97d0d8b2a22539a2d2887bc0345d9e229ea77cc4414c24e8e5fe6ce23b5f928`
- **来源**: FlagRT fork commit `5d545c9eadd7e20a5f3a63111500bbf20a8a1571`（作者 seanl），
  `git format-patch` 原样导出，未修改内容。
- **基线**: 上游 `release/0.2` tip `8b059122e32b9ac47b9820a9c7b1bb95077481a4`。
  2026-10-08 实测 `git apply --check` 干净套用、零冲突（上游 tip 领先 fork 基点
  `885aaef` 19 个提交，但均未触碰 patch 涉及的两个文件）。
- **内容**（两个文件，+36/-3）:
  1. `vllm_fl/dispatch/backends/vendor/ascend/impl/vocab_parallel_embedding.py`
     — bool*int 显式转 int64，修复 PrivateUse1 类型提升错误（token id 坍缩成 1）。
  2. `vllm_fl/distributed/communicator.py` — `all_reduce` 后追加
     `torch.npu.synchronize()`；新增 `all_gather` 覆盖实现（逻辑与基类一致 +
     sync）。round 3 真机已证实缺 sync 假说成立（broadcast/all_gather 无 sync 复现
     错误、加 sync 转 PASS）。
- **为何只吸收这一个 commit**: 它是 round 3 假说验证的技术依据；fork 其余内容与
  上游 release/0.2 已分叉（非祖先关系），不整体采用。
- **保质期**: 修复已按 upstream-first 整理待提交上游（见
  `../../../../docs/` 决策备忘录）。上游合入并出现在 release/0.2 后，本层删除，
  Dockerfile.repro 回退到纯上游 pin。

## 应用方式

构建期应用（Dockerfile.repro 的 vllm-plugin-FL 层内，`git apply` 后再
`pip install`），见 Dockerfile.repro 对应 RUN 块注释。
