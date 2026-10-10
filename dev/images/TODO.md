# dev/images TODO —— 镜像归档 / 重建的待办

> 只跟踪本目录职责内的事（归档、重建、索引、case 包）。项目层面的诉求（谁用、路线选型、验收）见 `dev/stack.lock.910c.*.yaml`（当前生效版本见该文件自身状态）/ `docs/`。

## 待办

| # | 事项 | 说明 |
|---|---|---|
| 1 | v3 demo 带卡复验（PENDING） | demo 运行环境与修正后的 `v3_step5_validate.py` 尚未在带卡环境复跑；npu1-11 的 NPU 当前被他人容器全部占用。命令与判据见下方「v3 demo 带卡复验」。通过后回填两个 v3 README 的「最近一次结果」 |
| 2 | v3 case 包缺口 | ① round 4 的部分结果由仅存于 npu1-11 `~/tmp-reproV3-r4/` 的脚本得出：`train_manual_ddp.py`（手动 DDP loss）、`probe_agit.py`（`all_gather_into_tensor` 探针）、推理 `minirepro.py`，未入仓；② 两个 v3 目录尚无 `assets/evidence/`（报错截选，见 `CASE-TEMPLATE.md` 规则 4）；③ `flagcx_sync_test.py` 的退出码只反映 sync 路径，修复验收须看 rank0 判定行 |
| 3 | 跨组需求 | shmem 根治 → 算子编译方向（[决策备忘录附录 B](../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)）；FlagCX c10d 异步返回 → FlagCX 通信方向（附录 B-2）。需求文本以仓内文档形式保留，下发方式由用户决定 |
| 4 | 上游 PR | vllm-plugin-FL sync 修复（决策备忘录附录 A 草稿就绪），由用户提交；合入后删除 `ascend-operator-runtime/v3/assets/patches/vllm-plugin-FL/` 覆盖层补丁 |
| 5 | 训练脚本默认学习率 | 默认 5e-5 在 bf16 + AdamW 下第 16 步出现 NaN，改为 1e-5（`ascend-train-comm/v3/REBUILD.md`「ROUND 4 真机验证——训练腿」③） |
| 6 | framework-adapter 三级回退需求对接 | round 4 实证编译期失败（`CompilationError`）不触发 dispatch 自动回退，算子级回退目前只能手工黑名单 |
| 7 | npu1-27 docker 存储报修 | 层校验和损坏（numpy libopenblas 层），`docker save` 不可用，运行不受影响 |
| 8 | v3 镜像基线 `vllm` 的 `info: unavailable` | 经代理访问 GitHub 失败；在可联网机器上重新 `baseline.py export --image` 补全（见 `../ENV-SPEC.md` §6） |
| 9 | v1 逐字节复现（可选金标准） | 需 owner 交出 FlagCX 私有 commit `55eb2ff` + 2 个 patch + 干净 python3.11.15 构建上下文。见各 `v1/lock.yaml:gaps` |
| 10 | `ascend-infer-vllm` 离线副本（可选加固） | 按 digest `sha256:5cf8a2b6…` 存一份，防 quay rc 标签被 GC；连带留存拷出的 `triton_ascend 3.2.1` wheel。见 `ascend-infer-vllm/v1/ARCHIVE.md` |
| 11 | `image_list.md` 维护（常设） | 新镜像/新版本入档时更新索引表与"当前生效版本" |

### v3 demo 带卡复验

仓根执行；昇腾驱动对带卡容器独占，先 `docker ps` 确认本机无其他带卡容器。

```bash
FLAGRT_MODELS_DIR=/home/xliu969/models dev/lib/up.sh dev/images/ascend-train-comm/v3/demo --strict

# 推理（1 卡）：退出码 0，9 个 [STEP] 全部 OK
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0 flagrt-v3-demo-910c \
  python3 dev/images/ascend-operator-runtime/v3/assets/v3_step5_validate.py

# 训练（2 卡）：结果应与 round 4 一致（all_gather nosync FAIL / sync PASS）
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagrt-v3-demo-910c \
  torchrun --nproc_per_node=2 dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py

docker compose -f dev/images/ascend-train-comm/v3/demo/docker-compose.yml down
```

`down` 绕开了 `up.sh`，须先 `export WORKSPACE_ROOT=<仓根绝对路径>`。

## 已完成

过程叙述见 `PROCESS.md` 与各 `REBUILD.md`，此处只列结论。

- 2026-09-09 ~ 09-10：v1 两镜像离线归档、配方还原、实机重建（`pip freeze` 逐行一致、FlagCX `.so` 逐字节一致），`repro_status` 🟢。
- 2026-09-12（N1）：按 BAAI·FlagTree 官方手册 ascend3.5 线产出 v2 两镜像并真机验证，见 `PROCESS.md`「v1 → v2」。
- 2026-09-23（N2）：v3 真机验证查清训练腿缺 sync 与推理腿 shmem 缺口两项 STOP CONDITION，见 `PROCESS.md`「v3」。
- 2026-10-08（N3）：两项决策落地（vllm-plugin-FL 声明式覆盖层补丁；cos/sin 黑名单解堵），见 `PROCESS.md`「round 4」。
- 2026-10-09：npu1-11 重建 v3 两镜像并完成 round 4 真机双腿验证；镜像离线归档至 `/mnt/raid/user_cache/xliu969/v3-round4-archive/`；结论见 [`docs/v3-round4-真机验证结果与移交指导-20261009.md`](../../docs/v3-round4-真机验证结果与移交指导-20261009.md)。
- 2026-10-10：`v3_step5_validate.py` 按 vLLM 0.20.2 接口修正；v3 接入环境基线机制（两份 `image.baseline.yaml` + demo 运行环境目录，见 `../ENV-SPEC.md`）；挂载资产试点（vllm-plugin-FL）跑通。
