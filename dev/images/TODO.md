# dev/images TODO —— 镜像归档 / 重建的待办

> 只跟踪本目录职责内的事（归档、重建、索引、case 包）。项目层面的诉求（谁用、路线选型、验收）见 `dev/stack.lock.910c.*.yaml`（当前生效版本见该文件自身状态）/ `docs/`。

## 待办

| # | 事项 | 说明 |
|---|---|---|
| 1 | v3 case 包缺口 | round 4 的部分结果由仅存于 npu1-11 `~/tmp-reproV3-r4/` 的脚本得出：`train_manual_ddp.py`（手动 DDP loss）、`probe_agit.py`（`all_gather_into_tensor` 探针），未入仓。推理最小复现已以修正版入仓为 `../issues/shmem-missing/assets/minirepro_shmem.py` |
| 2 | 原生 DDP 验证脚本 | FlagCX 修复到货后需补 `DistributedDataParallel` 封装、不含手动同步的双卡自判定脚本（[移交指导第七节 B](../../docs/v3-round4-真机验证结果与移交指导-20261009.md)） |
| 3 | 跨组需求 | shmem 根治 → 算子编译方向（[决策备忘录附录 B](../../docs/v3-决策备忘录与shmem跨组需求-20261008.md)）；FlagCX c10d 两类缺陷（异步返回；列表形式 `all_gather` 结果错位，2026-10-10 复验修正）→ FlagCX 通信方向（附录 B-2 及其修正段）。需求文本以仓内文档形式保留，下发方式由用户决定。两项已按 `CASE-TEMPLATE` 发布为 case 包：[`../issues/`](../issues/README.md)（`shmem-missing`、`flagcx-c10d-collectives`） |
| 4 | 上游 PR | vllm-plugin-FL sync 修复（决策备忘录附录 A 草稿就绪），由用户提交；合入后删除 `ascend-operator-runtime/v3/assets/patches/vllm-plugin-FL/` 覆盖层补丁 |
| 5 | 训练脚本默认学习率 | 默认 5e-5 在 bf16 + AdamW 下第 16 步出现 NaN，改为 1e-5（`ascend-train-comm/v3/REBUILD.md`「ROUND 4 真机验证——训练腿」③） |
| 6 | framework-adapter 三级回退需求对接 | round 4 实证编译期失败（`CompilationError`）不触发 dispatch 自动回退，算子级回退目前只能手工黑名单 |
| 7 | npu1-27 docker 存储报修 | 层校验和损坏（numpy libopenblas 层），`docker save` 不可用，运行不受影响 |
| 8 | ~~v3 镜像基线 `vllm` 的 `info: unavailable`~~（已关闭 2026-10-12，不追） | **主动取舍，不是遗留缺口**：`info` 仅供阅读、不参与任何校验（`../ENV-SPEC.md` §3.1），缺它不影响 `check` 与任何版本锚。原归因「经代理访问 GitHub 失败」已更正：GitHub 可达，是代理的 HTTP/2 中途断流（npu1-11 上两次独立重现），改走 HTTP/1.1 即可取到——但那不是 `baseline.py` 走的路径，基线只记文档化命令自己能产出的内容，故保持 `unavailable`。归因已同步进 §6 |
| 9 | v1 逐字节复现（可选金标准） | 需 owner 交出 FlagCX 私有 commit `55eb2ff` + 2 个 patch + 干净 python3.11.15 构建上下文。见各 `v1/lock.yaml:gaps` |
| 10 | `ascend-infer-vllm` 离线副本（可选加固） | 按 digest `sha256:5cf8a2b6…` 存一份，防 quay rc 标签被 GC；连带留存拷出的 `triton_ascend 3.2.1` wheel。见 `ascend-infer-vllm/v1/ARCHIVE.md` |
| 11 | `image_list.md` 维护（常设） | 新镜像/新版本入档时更新索引表与"当前生效版本" |
| 12 | Torch-FL pin commit 可达性 | v3 固化的 Torch-FL `162582d6` 不在本机记录的任何远端分支上，可能无法从零重建；若后续镜像不再安装 Torch-FL 则随之关闭 |
| 13 | 昆仑芯 v1 离线归档去留（待定） | `kunlun-operator-runtime/v1/ARCHIVE.md` 记的是 `docker export` + `docker import --change` 重放（归档镜像是 `docker commit` 产物，`docker save` 取不到 layer blob，见该目录 `lock.yaml` 的 `known_issues: v1-image-is-docker-commit-not-build`）。v2 可用后该离线副本是否还需保留，待定 |
| 14 | ~~昆仑芯 v2 带卡验收收尾~~（已完成 2026-10-11） | 带卡自判定 `overall: PASS`，`image_list.md` 已改 🟢，证据 `kunlun-operator-runtime/v2/assets/evidence/verify-v2-card7-PASS.txt` |
| 15 | 昆仑芯 v2 离线归档（待定） | 当前无离线实体：本版本可从公开源头 ~15 分钟重建（且不再依赖宿主任何 checkout），而本机 `/var/lib/docker` 99% 满（共享机）。**已实测：`docker save` 对本镜像不可用**（14336 字节 / 零 layer blob，未改动的官方基座镜像同样如此），要归档须沿用 v1 的 `docker export` + `docker import --change` 变通。前置与做法见 `kunlun-operator-runtime/v2/ARCHIVE.md` |
| 16 | flagtree wheel 的 sha256 接进核对链（需拍板） | **原判断已更正**：官方索引页按 PEP 503 在 href 上直接公布 `#sha256=`，一次小 HTTP 请求即可取，不需下载 3.3GB；值已记在 `kunlun-operator-runtime/v2/lock.yaml` 的 `layers` 段。接进 `baked_packages` 核对链不需猜测：wheel 文件名由镜像内 dist 版本 + dist-info 的 `WHEEL: Tag` 完整推出，再查索引页同名条目。待拍板是否实现（约 40 行 + 测试）。见该 `lock.yaml` `known_issues: flagtree-wheel-sha256-not-in-check-chain` |
| 17 | ~~`/env/xvllm-plugin-FL` 的仓库身份~~（建议关闭） | 远端是厂商内网仓（`ssh://…icode.baidu.com/baidu/xpu/xvllm-plugin-FL`），本仓没有也无法取得其 checkout，**不应**登记到 `../ENV-SPEC.md` §3.2 的「仓库名 → 宿主目录」表——那张表是挂载资产的 `host_path` 缺省来源，登记一个指向不存在目录的映射只会误导。现状（`commit: unavailable` + `source_path` 记镜像内路径 + `kunlun-operator-runtime/v2/lock.yaml` `parent_image.note` 已写明与仓内 `vllm-plugin-FL` 不是同一个仓）即正确表达。待你确认后可关闭 |
| 18 | 昆仑芯 GEMM 失败形态复核 | v1 记 SIGABRT（`make_llir`），v2 以 64×64 `torch.mm` 实测为 `OutOfResources: uni_sram`，是否同一缺陷未定论。需用 v1 的 pytest 用例形状在 v2 上复跑确认。见 `kunlun-operator-runtime/v2/lock.yaml` `known_issues: gemm-failure-signature-differs-from-v1` |
| 19 | ~~昇腾侧 `check --image` 在 P800 机器上无法复验~~（已完成 2026-10-12） | 已在 npu1-11 真镜像上复跑，无回归：85 条单测全过；`check --image` 两镜像与 demo `check` 均 rc=0。两处新逻辑在昇腾上的表现符合预期——`vllm`（`from: base_image`、有 commit、镜像无对应标签）报 **INFO**「取自基座 pip 元数据…由 image id 兜底」而非 FAIL；`_assert_inherited_commits` 对其跳过（`COMMIT_LABEL` 无 `vllm` 条目）。`export --image` 重新生成两份基线转入新格式，差异只有注释/分组/`generated_at`，外加 `base_image.provides` 增 `torchvision: 0.25.0+cpu`/`torchaudio: 2.10.0+cpu`（`BASE_PROVIDES_PKGS` 扩容，与仓内 `pipfreeze-*-round4-npu11.txt` 一致，L1 事实更完整）；`commit`/`sha256`/`image.id`/`digest`/`version`/`install`/`from`/`layer` 的值与 `baked_repos` 顺序均未变，`packages` 仍指向 `pipfreeze-*-round4-npu11.txt`。带卡复核亦通过：`up.sh --strict` rc=0 起容器，`verify_flagcx_runtime.py --static` rc=0（`flagcx_commit` 与镜像标签一致），`--device` 双卡两 rank 均 `device_all_reduce_ok: true` / `device_backend_str: "flagcx"` / `torch_npu_default_check: passed`；退出期复现已知的 `teardown-sigabrt`（`free(): invalid pointer`，故 torchrun rc=1），形态与 round 1 记录一致，不影响判定 |
| 20 | ~~昆仑芯 v2 旧镜像 `eda3e0ae14eb` 回收~~（已完成 2026-10-11） | 配方简化前那次构建的产物，已 `docker rmi`（只删该 id，未 prune）；实际回收约 40MB，7.38GB 的 flagtree 层与新镜像共享故未删，删后 `check --image` 仍 rc=0 |
| 21 | `docker save` 对巨型厂商派生镜像失效的机制未定位 | 本机实测：官方基座（94GB，拉取）、v1（105GB，commit）、v2（105GB，build）`docker save` 均只出 14336 字节 / 1 blob，而本机构建的 50MB 合成镜像正常。与产出方式、拉取/构建均无关。推测是这些镜像的 layer blob 不在本机 containerd content store 内（惰性/远端快照），**未深究**；「换一台机器就正常」同样未验证。影响离线归档方式的选择，见 `kunlun-operator-runtime/v2/ARCHIVE.md` |

## 已完成

过程叙述见 `PROCESS.md` 与各 `REBUILD.md`，此处只列结论。

- 2026-09-09 ~ 09-10：v1 两镜像离线归档、配方还原、实机重建（`pip freeze` 逐行一致、FlagCX `.so` 逐字节一致），`repro_status` 🟢。
- 2026-09-12（N1）：按 BAAI·FlagTree 官方手册 ascend3.5 线产出 v2 两镜像并真机验证，见 `PROCESS.md`「v1 → v2」。
- 2026-09-23（N2）：v3 真机验证查清训练腿缺 sync 与推理腿 shmem 缺口两项 STOP CONDITION，见 `PROCESS.md`「v3」。
- 2026-10-08（N3）：两项决策落地（vllm-plugin-FL 声明式覆盖层补丁；cos/sin 黑名单解堵），见 `PROCESS.md`「round 4」。
- 2026-10-09：npu1-11 重建 v3 两镜像并完成 round 4 真机双腿验证；镜像离线归档至 `/mnt/raid/user_cache/xliu969/v3-round4-archive/`；结论见 [`docs/v3-round4-真机验证结果与移交指导-20261009.md`](../../docs/v3-round4-真机验证结果与移交指导-20261009.md)。
- 2026-10-10：`v3_step5_validate.py` 按 vLLM 0.20.2 接口修正；v3 接入环境基线机制（两份 `image.baseline.yaml` + demo 运行环境目录，见 `../ENV-SPEC.md`）；挂载资产试点（vllm-plugin-FL）跑通。
- 2026-10-10：v3 demo 运行环境带卡复验——推理腿 9/9 PASS（脚本原样执行）；`flagcx_sync_test.py` 重写为唯一数据 + 跨 rank 汇总 + 迭代，修正 FlagCX 结论（列表形式 `all_gather` 同步无效）；两个 v3 目录建立 `assets/evidence/`。见 `ascend-train-comm/v3/REBUILD.md`「2026-10-10 demo 环境带卡复验」。
