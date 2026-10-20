# ascend-train-comm v3

## 用途与现状

- 本目录构建 v3 完整镜像,**完整镜像 = 推理 + 训练超集**:
  `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`
  (npu1-11 image id `e13a15d0d02b`)。内容为基座镜像
  [`../../ascend-operator-runtime/v3/`](../../ascend-operator-runtime/v3/README.md) 全部组件 +
  FlagCX(公开主干 `4e0e0cbcbf721169ca82348080f8353aebfe2c31`)。本镜像为 现行训练腿发布物
  (见 [`dev/stack.lock.910c.v4.yaml`](../../../stack.lock.910c.v4.yaml) 的 `lock.train`);
  推理腿为华为昇腾官方 `vllm-ascend`,见同文件 `lock.infer`;
  tag 中的 `comm` 字样为历史命名,保留不改。
- 版本事实见 [`image.baseline.yaml`](image.baseline.yaml);demo 运行环境(验收配置与推荐用法)
  见 [`demo/`](demo/docker-compose.yml)(训练腿运行环境)。
- `repro_status`:🟡 partial-repro。
  - 推理:可用,前提为两项运行时配置(已写入 demo,见基座目录 README「复现」节)。
  - 训练:2 卡手动 DDP 写法(梯度 all_reduce 后显式 `torch.npu.synchronize()`)可用,
    Qwen2.5-1.5B 24 步 loss 2.8911 → 2.5528;PyTorch 原生 `DistributedDataParallel`
    封装不可用。
  - 升 🟢 条件:FlagCX c10d 层修复异步返回,原生 DDP 双卡验证通过。
- 已知限制:

| 问题 | 归属 | 详情 |
|---|---|---|
| FlagCX c10d collective 两类缺陷:① `broadcast` / `all_reduce` / `all_gather_into_tensor` 返回时数据未写完,同步后正确;DDP 初始化的 C++ 校验(`_verify_params_across_processes` → `all_gather_into_tensor`)命中此类,原生 DDP 不可用;② 列表形式 `all_gather` 得到上一次调用的结果,同步无效 | FlagCX 通信方向 | `lock.yaml` known_issues `flagcx-broadcast-allgather-wrong-result`;`REBUILD.md`「ROUND 4 真机验证——训练腿」②;需求文本:[决策备忘录附录 B-2](../../../../docs/v3-决策备忘录与shmem跨组需求-20261008.md) |
| 进程退出时 `free(): invalid pointer` SIGABRT(结果已在退出前输出,不影响判定) | FlagCX 通信方向(低优先级) | `lock.yaml` known_issues `teardown-sigabrt` |
| 默认学习率 5e-5 在 bf16 + AdamW 下第 16 步出现 NaN,需使用 1e-5 | 本组(训练脚本参数) | `REBUILD.md`「ROUND 4 真机验证——训练腿」③ |

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | 本目录完整镜像 tag(见上) | 先按基座目录构建 `ascend-operator-runtime` v3,再执行本目录 `build.sh`(见 `REBUILD.md`);或从 `ARCHIVE.md` 所列离线包 `docker load` |
| 运行环境 | `demo/` | 随仓;image id 由 `up.sh --strict` 按镜像基线核对;v3 的全部 repo 均为固化资产,宿主检出不影响运行 |
| 卡 | 2 × 910C | 启动前 `docker ps` 确认本机无其他带卡容器(昇腾驱动对带卡容器独占);多项验证共用一个容器,以 `ASCEND_RT_VISIBLE_DEVICES` 分区 |
| 模型与数据(仅训练) | Qwen2.5-1.5B;wikitext-2-raw-v1 | 模型目录经 `FLAGRT_MODELS_DIR` 挂到容器内 `/workspace/models`(npu1-11 现有副本 `/home/xliu969/models/Qwen2.5-1.5B`);数据集由脚本经 `datasets.load_dataset("Salesforce/wikitext", ...)` 拉取,需可访问 HuggingFace 或其镜像 |

### 启动容器(仓根执行)

```bash
docker ps
FLAGRT_MODELS_DIR=/home/xliu969/models dev/lib/up.sh dev/images/ascend-train-comm/v3/demo --strict
```

验证结束后 `docker compose -f dev/images/ascend-train-comm/v3/demo/docker-compose.yml down`
(绕开 `up.sh` 时先 `export WORKSPACE_ROOT=<仓根绝对路径>`)。

### 复现 FlagCX collective 缺陷

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagrt-v3-demo-910c \
  torchrun --nproc_per_node=2 dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py --iters 20
```

脚本对 `broadcast`、`all_reduce`、列表形式 `all_gather`、`all_gather_into_tensor` 各做"未同步即读取 / 同步后读取"
两组,每次调用使用唯一数据,跨 rank、跨迭代汇总判定。当前预期结果(问题存在时,退出码 1):

```
  broadcast+nosync                 FAIL
  broadcast+sync                   PASS
  all_reduce+nosync                FAIL
  all_reduce+sync                  PASS
  all_gather_list+nosync           FAIL
  all_gather_list+sync             FAIL
  all_gather_into_tensor+nosync    FAIL
  all_gather_into_tensor+sync      PASS
overall: FAIL
```

未同步组的失败为偶发,单次可能通过,故取 `--iters 20`。脚本在判定后直接退出,
不经过 FlagCX 退出期 SIGABRT,退出码即判定结果。证据:
[`assets/evidence/flagcx-collectives-20261010.txt`](assets/evidence/flagcx-collectives-20261010.txt)。

## 验证

| 脚本 | 类别 | 命令(容器内,`/workspace` 下) | 通过判据 | 最近一次结果 |
|---|---|---|---|---|
| `assets/verify_flagcx_runtime.py` | 验证 | `python3 /opt/flagrt/verify_flagcx_runtime.py --static` | 退出码 0 | round 2 通过(`REBUILD.md`「静态自检」) |
| `assets/verify_flagcx_runtime.py` | 验证 | `torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_runtime.py --device` | 两 rank `device_all_reduce_ok: true`,`device_backend_str: "flagcx"` | round 1 通过(`REBUILD.md`「真机动态验证」) |
| `assets/verify_flagcx_p2p.py` | 验证 | `torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_p2p.py` | 两 rank 4 组 payload sha256 一致,`status: "passed"` | round 1 通过(同上) |
| `assets/flagcx_sync_test.py` | 复现 / 验证 | 见「复现」 | **修复验收**:退出码 0(八组全部 PASS) | 2026-10-10 demo 环境,连续 3 次结论一致:同步后 `broadcast`、`all_reduce`、`all_gather_into_tensor` PASS,列表 `all_gather` FAIL(见上方证据)。round 3/4 结论"同步后 all_gather 正确"系测试复用相同数据所致,已修正 |
| `assets/train_qwen_1_5b_npu_syncpatch.py` | 复现 | 在 Python 层为 broadcast / all_gather 补同步后运行原生 DDP 训练脚本;脚本内写死路径 `/workspace/train_qwen_1_5b_npu.py`,需先将 `dev/device-context/910C/distributed_training/scripts/train_qwen_1_5b_npu.py` 置于该路径 | 预期失败于 DDP 初始化:`RuntimeError: DDP expects same model across all ranks ... inconsistent 0 params`(补丁位于 Python 层,不覆盖 C++ 校验路径) | round 4 以本机变体脚本执行,如预期失败(npu1-11:`~/tmp-reproV3-r4/train-leg/outputs/train_syncpatch.log`) |
| `assets/provenance/pipfreeze-*.txt` | 收数 | `build.sh` 构建后生成 | — | round 4 npu1-11:`pipfreeze-comm-v3-round4-npu11.txt` |

修复验收另需原生 DDP 验证脚本(`DistributedDataParallel` 封装、不含手动同步,双卡自判定通过),
见 [`docs/v3-round4-真机验证结果与移交指导-20261009.md`](../../../../docs/v3-round4-真机验证结果与移交指导-20261009.md) 第七节 B。

round 4 训练与 DDP 阻塞证据(npu1-11,用户 xliu969,`~/tmp-reproV3-r4/train-leg/outputs/`):
`train_lr1e5.log`(手动 DDP loss 曲线)、`probe_agit.log`(`all_gather_into_tensor`
未同步 FAIL / 同步后 PASS)。技术细节:`REBUILD.md`、`lock.yaml`。
