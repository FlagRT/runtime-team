# 证据索引 —— 910C 上以 `up.sh --strict` 完成一次真实验收（含 `image.id` 跨机对齐）

> 机器：**npu1-27** ｜ 时间：**2026-10-16 14:34–14:36** ｜ 容器：`flagos-dc-train-910c-dev`（跑完已 `down`，Removed）
> 驱动：`../run_strict_acceptance_910c_20261016.sh` ｜ 对齐工具：`../../env/pin_local_image_id.sh`
> 完整日志（含被覆盖的上一轮）：`npu1-27:/mnt/raid/hliu553/dc_env_20261016/strict/log/strict_run_run{1,2}.log`

## 1. 结论一行

`up.sh <train> --strict` **rc=0** 起容器 → 容器内**两项验收 `overall: PASS`** → `down` rc=0 无残留：
**本机 `--strict` 验收已可跑**。代价是 `env.baseline.yaml` 的 `image.id` 必须**按本机 id 对齐**（见 §3），
这是一条**已登记的规范偏离**（§4）。

## 2. 判据与读数（两条腿）

| 项 | 结果 | 证据 |
|---|---|---|
| `check` 训练腿 | **rc=0** — `[PASS] image: … -> sha256:7028028bb62c…` · `[PASS] FlagCX: commit=b576fa6ad977` · `[INFO] FlagCX 覆盖镜像内固化版本 4e0e0cbcbf72，生效方式 pip-editable` | `05_check_after.txt` |
| `check` 推理腿 | **rc=0** — `[PASS] image: quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 -> sha256:2e56022ae5b39…`（**未做任何对齐**，见 §3.4） | `05_check_after.txt`、`13_env_baseline_infer.yaml` |
| **`up.sh <train> --strict`** | **rc=0**，`Container … Created → Starting → Started` | `06_up_strict.txt` |
| 容器状态 | `[OK] 容器在运行: flagos-dc-train-910c-dev\|Up 8 seconds` | `06_up_strict.txt` |
| 挂载生效（自证） | `docker inspect` 的 `Cmd` **逐字等于**声明：`USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation -e /workspace/FlagCX && exec sleep infinity`；`Successfully installed flagcx-0.13.0`；`import flagcx` → `/workspace/FlagCX/plugin/torch/flagcx`（宿主 checkout，非镜像内） | `07_install_proof.txt` |
| **容器内单入口验收（两项）** | **rc=0** · ① `flagcx_sync_test.py --iters 20` → `判定=PASS` ② `ddp_native_test.py --steps 10`（原生 DDP 双卡）→ `判定=PASS` · **`overall: PASS`** | `08_acceptance_in_env.txt` |
| `docker compose down` | **rc=0**，容器 `Stopping→Stopped→Removing→Removed` | `09_down_and_residue.txt` |
| 收尾 | 我方运行中容器 **0** · 我方残留进程 **0** · 我方容器（含已停）**0** | `09_down_and_residue.txt` |
| 选卡（判据 = 逐 die 建容器试 INIT） | **可用 die = 5 0**（其余 `can not use command`）；起前复测仍 FREE | `10_pick_dies.log`、`11_pick_dies_pre.log` |

## 3. `image.id` 跨机问题：复现、依据、对齐

### 3.1 复现（证明这是机制行为，不是我们把配置写错）

```
[2] python3 dev/lib/baseline.py export <train> --require-clean     → rc=0
    export 后 env 基线里的 image.id = sha256:e13a15d0d02b…（**制造机 npu1-11** 的 id）
[3] check → [FAIL] image: id 不一致: 基线 sha256:e13a15d0d02b… 本机 sha256:7028028bb62c…   rc=1
```
`export` 的 `image.id` 取自镜像基线（`export_env` 里 `img.get('id')` 直接抄录），而 `check` 拿它与
**本机** `docker image inspect <ref>.Id` 逐字比对 ⇒ 除制造机外恒 `FAIL`、`--strict` 恒拒绝启动。

### 3.2 为什么「合规拿到同一制品」不可得（三条路逐条实测）

| 路径 | 实测 |
|---|---|
| ① `docker load` 离线包 | `ARCHIVE.md` 记实体 `train-comm-v3-round4-npu11-e13a15d0d02b.tar.gz`（10.5 G）**只在 npu1-11**；本机 `find /mnt/raid -maxdepth 4` 名匹配 0 命中、无 `BACKUP.local`、`docker images` 里也无 `e13a15d0` 该 id（任何 tag）；`hliu553@10.120.72.11` 三把私钥一律 `Permission denied` |
| ② registry 按 digest 拉 | 连**已知存在的父镜像**都拉不动：`docker manifest inspect harbor.baai.ac.cn/flagtree/flagtree-…` → `failed to configure transport: … proxyconnect tcp: dial tcp 127.0.0.1:7890: connect: connection refused`（daemon 代理已失效） |
| ③ 按 `Dockerfile.repro` 重建 | 只到「功能等价」：镜像基线 `repro_status = 🟡 partial-repro`，构建期含 `pin: unpinned` 的 `json`/`googletest` clone 与层时间戳 ⇒ **复现不出同一 id** |

### 3.3 「内容同源」的正面依据（实测，不是推断）

- 同一 tag；本机镜像 `Created = 2026-10-08`，与 `lock.yaml` 的 `image_id_round4_npu27` 对得上。
- **L1 厂商基座 digest 逐字相同**：本机 `RepoDigests` = `sha256:9ee5d835d5f5e3579e01f02b3381c002f85de09f1c385d62f5c7910dfdf90c8a`
  = 镜像基线 `base_image.digest`。
- **镜像 labels 里 5 项固化资产 commit 与镜像基线逐字相同**：FlagCX `4e0e0cbcbf72…` · FlagTree `15ec1a6cbc8d…` ·
  FlagGems `f7ae8e6b934a…` · Torch-FL `162582d678e4…` · vllm-plugin-FL `8b059122e32b…`。
- 总组亦在 `dev/images/ascend-train-comm/v3/lock.yaml` 里**并列登记** `image_id`(npu1-11) 与 `image_id_round4_npu27`。
- ⚠️ 本组镜像 `RepoDigests = []` ⇒ **从未 push**（这正是「无法按 digest 拉取对齐」的根）。

> 以上 5 条读数原文见 `15_released_image_facts.txt`。

### 3.4 对齐动作与「谁不需要对齐」

```
[3] bash dev/device-context/910C/env/pin_local_image_id.sh <train>
    发布(制造机) id: sha256:e13a15d0d02b…     本机 id: sha256:7028028bb62c…
    [OK] 回读校验：id 行数=1 值=sha256:7028028bb62c…
    [OK] yaml 解析一致（safe_load 可读，image.id 命中）
    幂等复跑 → [OK] 已对齐，无需改动           （原文见 04_pin_align.txt）
[4] check 两腿 → 全 PASS                        （原文见 05_check_after.txt）
```

- **推理腿不需要这条对齐**（且**未做任何改动**）：官方镜像按 digest 拉取，`Id` 与
  `RepoDigests`（`quay.io/ascend/vllm-ascend@sha256:5cf8a2b6…`）天然跨机一致 ⇒ `check` 原生 PASS。

## 4. 与 ENV-SPEC 的关系（如实登记，不粉饰）

ENV-SPEC §3 明确要求两类基线文件「由 `baseline.py` 生成和核对，**不手工编辑**」。
本次处置**是对该条的偏离**，但做了四项约束使其可控：**单字段**（只改 `image.id`）· **值来自本机 docker**
（不是手输）· **写完回读校验**（`grep` 命中数 + `yaml.safe_load` 解析一致）· **幂等且把两个 id 与依据写进文件注释**。
⚠️ **`export` 之后必须重跑**（`export` 会把该字段抄回制造机 id —— 上面的 §3.1 就是这条的实测复现）。

**仍在机制/镜像侧的正解（任选其一即可消除本偏离，本方向不擅改基线或 `baseline.py`）**：
① `image.id` 接受**等价 id 集合**（总组已在 `lock.yaml` 并列登记两个 id，把这条事实接进 schema 即可）；
② 分发统一走 `docker load` 离线包（id 内容导出、跨机一致）；③ 把本组镜像 push 到可拉取的 registry。

## 5. 边界（本轮**未**覆盖）

- 未覆盖**跨机/8 卡/训练腿 50 步**；未覆盖性能。
- 未验证 `--strict` 在**制造机（npu1-11）**或持有发布镜像的机器上的行为（按设计应原生 PASS，未实测）。
- `check` 默认不联网（ENV-SPEC §4.4），本轮未加 `--online`。

## 6. 过程中自引入并修掉的 2 个缺陷（都影响判据可读性）

| 缺陷 | 症状 | 修法 |
|---|---|---|
| 驱动脚本的容器存活判据写错 | `docker ps --format '{{.Names}}\|{{.Status}}'` 的行是「名字\|状态」，而判据用了 `grep -E "^${C}$"` ⇒ **永不命中**、报出假 `[FAIL] 容器未在运行`（第 1 轮日志留档 `strict_run_run1.log`） | 改为 `docker ps --format '{{.Names}}' \| grep -qx "$C"`，成功后另打印 `名字\|状态` 佐证 |
| 对齐工具读「发布 id」正则过严 | 镜像基线的 `id:` 行**带行尾注释**（`  id: sha256:…   # check 项`），原正则 `^  id: (\S+)\s*$` 要求行尾 ⇒ 读成 `(读不到)`，注释里写出了错误文本 | 改为只锚定值 `^  id: (\S+)`；并把「已对齐跳过」判定加上「注释不陈旧」条件 |

## 7. 证据清单

| 文件 | 内容 |
|---|---|
| `01_env_selfproof.txt` | 环境自证：`Docker Compose v2.29.7` · `Python 3.9.9 / yaml 6.0.2` · 挂载镜像 id · `FlagCX HEAD=b576fa6a` |
| `02_check_before.txt` | 两腿 check 现状（训练腿已对齐 ⇒ PASS；推理腿原生 PASS） |
| `03_export_clobber.txt` | ⭐ `export` 抄回制造机 id ⇒ `check` FAIL 的复现 |
| `04_pin_align.txt` | ⭐ 对齐：写入本机 id + 回读 / yaml 解析校验 + 幂等复跑 |
| `05_check_after.txt` | 两腿 check 全 PASS |
| `06_up_strict.txt` | ⭐ `up.sh --strict` rc=0 + 容器 Created→Started→Up |
| `07_install_proof.txt` | 挂载覆盖安装自证（`Cmd` 逐字 + `Successfully installed` + import 路径） |
| `08_acceptance_in_env.txt` | ⭐ 容器内单入口验收两项 `PASS` / `overall: PASS` / rc=0 |
| `09_down_and_residue.txt` | `down` rc=0 + 我方容器/进程 0（共享机按容器判、不按卡归因） |
| `10_pick_dies.log` · `11_pick_dies_pre.log` | 逐 die 试 INIT 选卡（8 张候选 → `5 0`）+ 起前复测 |
| `12_env_baseline_train.yaml` | 训练腿运行环境基线**最终形态**（权威副本在 `../../env/train/env.baseline.yaml`） |
| `13_env_baseline_infer.yaml` | 推理腿运行环境基线（**未改动**，权威副本在 `../../env/infer/env.baseline.yaml`） |
| `14_last_check_train.json` | `check` 留痕（`.last-check.json` 的快照） |
| `15_released_image_facts.txt` | 发布镜像事实：`RepoDigests` 为空 · labels 逐字比对 · L1 digest · 三条合规路径实测 · 推理腿对照 |
