# C2 / C3 收尾 · 宿主副本对齐 + 环境层补记（2026-09-30）

> 定位：审计清单 **C2**（G7 宿主工作副本陈旧）与 **C3**（G6 环境层 4 小项）的收尾记录。
> 两项都**不需要设备窗口**（C2 是 git 操作，C3 是补记）。

---

## 1 C2 · 910C 宿主副本：从「落后 208 提交 + 109 脏文件」到「对齐且干净」

### 1.1 收尾前实测（与审计登记一致，并更新了一个数字）

| 项 | 值 |
|---|---|
| 路径 | `/mnt/raid/hliu553/runtime-team` |
| 分支 / HEAD | `kistich/device-context` / **`856b24e`**（v0.1.0 时代） |
| 落后 origin | **208 个提交**（`git rev-list --count HEAD..FETCH_HEAD`；审计登记时为 201，期间又推了 7 个） |
| 脏文件 | **109 个**：84 未跟踪（`??`）· 23 已修改（`M`）· 2 已删除（`D`） |

⚠️ 另有一个**此前未登记**的现象：工作时区含 **root 属主**的 `kernel_meta/` ⇒ `git status` 会打
`warning: could not open directory 'kernel_meta/': Permission denied`（**容器内 root 建的目录，宿主侧删不掉**）。

### 1.2 先查「脏文件是否含他人改动」（这是 C2 的收尾前置条件）

实测结论：**不含**。依据三条：

1. **属主**：抽查的修改文件全部 `hliu553`；
2. **时间**：`2026-09-17 … 09-23`（都是我自己那些日子的会话产物）；
3. **范围**：47 个条目全部落在 **`dev/device-context/`（本方向子树）**；其余未跟踪项是仓库根下的
   旧实验脚本（`uva_test/`、`test_*.py`、`train_qwen_*` 等），也全部是我的。
   两个 `D`（删除）正是 **`runtime/backends/flagos/`** —— 路线 B 退出时的删除，本地从未提交。

⇒ **不含任何其他方向（`dev/communication` / `dev/memory` …）的改动**，可安全对齐。

### 1.3 处置（**不删除任何东西**）

| 步骤 | 动作 |
|---|---|
| ① 留档 | `git status --porcelain` 全量落盘 → `/mnt/raid/hliu553/host_copy_dirty_list_20260930.txt`（109 行） |
| ② 保 WIP | `git stash push -m host-copy-wip-tracked-20260930` ⇒ 23 个已修改 + 2 个删除**进 stash，可 `git stash pop` 还原** |
| ③ 试快进 | `git merge --ff-only` 被"未跟踪文件将被覆盖"挡住（47 个路径冲突） |
| ④ 移交 | 逐个 `mv` 冲突文件到备份目录时**撞上 root 属主文件**（`Permission denied`）⇒ **改换干净做法** |
| ⑤ **最终做法** | **整目录改名留档** `mv runtime-team → runtime-team.stale-856b24e-20260930-hostcopy`（**48 GB，一个文件没动**，stash 仍在其中；备份目录一并移入该目录下 `_untracked_backup_20260930/`），然后在**原路径新克隆**一份对齐副本 |

> 为什么改换做法：逐文件搬移在"部分文件属 root"时会半途而废（且越弄越乱）；整目录 `mv` 只依赖父目录写权限、
> **不需要目录内文件的权限**，且**完全可回滚**（旧副本原样在位）。

### 1.4 收尾后核验（三项）

```bash
本地 HEAD : 57cb9ddb1056767752b569744b0394104558624c
远端 tip  : 57cb9ddb1056767752b569744b0394104558624c   ← 一致
分支      : kistich/device-context
脏文件    : 0
当前文档抽查: dev/device-context/prototype/docs/INTERFACE_CONTRACT_REVISION_STATUS_REVIEW_20260929.md 可读（09-29 新文档在位）
```

⇒ **照这份副本操作不会再跑到 08–09 月的旧代码**；旧的 109 个脏文件与 stash 全部保留在留档目录里。

### 1.5 P800：明确「无宿主 git 副本」这一事实

实测：`/workspace/runtime-team`、`/data2/hliu553/runtime-team` **均不存在**；
P800 侧只有**同步过去的 `prototype/` 目录**（`/data2/hliu553/dc_regress_20260929/prototype`，宿主 `= /workspace/...`）。
⇒ 已写入 P800 README（见 §3 的同步清单），避免后来者按"应该有一份"去翻。

---

## 2 C3 · 环境层 4 小项

### 2.1 ③ 两实例 LR 实际取值 —— **补记：`1e-5`（两实例一致，均取默认值）**

| 依据 | 内容 |
|---|---|
| 取值来源 | `prototype/runtime/proto/proto_train_leg.py:184` ⇒ `LR = float(os.environ.get("LR", "1e-5"))`，优化器 `AdamW(lr=LR, eps=1e-8)` |
| 910C 实际运行 | 09-22 统一复核脚本 与 09-29 第 6 轮复跑命令**均未传 `LR`** ⇒ 实际生效值 **1e-5** |
| P800 实际运行 | 09-14 记录的命令 与 09-30 本轮复跑命令**均未传 `LR`** ⇒ 实际生效值 **1e-5** |

⇒ 两实例训练腿的 LR **都是默认 1e-5**（此前"未记录"是因为没人写下这条，不是值不明确）。

### 2.2 ④ 910C 训练腿依赖版本 —— **补记（实测，2026-09-30）**

| 组件 | 版本 | 取数方式 |
|---|---|---|
| Python | **3.12.13** | `venv-infer-a/bin/python -c "import sys"` |
| torch | **2.11.0+cu130** | 同上 |
| torch_npu | **2.11.0** | 同上 |
| transformers | **4.57.6** | 同上 |
| numpy | **2.3.5** | 同上 |
| CANN | **9.0.0** | `/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info` ⇒ `Version=9.0.0`（`required_package_runtime_version="9.0"`） |
| 解释器路径 | `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python` | **容器内路径**（该 venv 的 python 是容器内解释器，宿主直接执行会 `No such file or directory` ⇒ 必须经 `docker exec`） |
| 容器镜像 | 训练腿：`flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64`<br>本轮探针：`flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet` | `docker inspect -f '{{.Config.Image}}'` |

**P800 侧对照（此前已记录，本次复核一致）**：conda `python310_torch29_cuda` ⇒ py3.10 ·
**torch 2.9.0+cu129** · transformers 4.57.1 · **vllm 0.13.0** · **XPU-RT 5.0.21.47 / Driver 5.0.21**。

> ⚠️ 注意：**镜像与 venv 是两件事** —— CANN/驱动由镜像提供，而 python/torch 来自挂载的 venv；
> 记录训练腿依赖时必须两者都写（只写镜像 tag 会漏掉真实生效的 torch 版本）。

### 2.3 ①② 两个镜像项 —— **不是我们能关的**（如实标注）

| 项 | 状态 | 说明 |
|---|---|---|
| ① 910C 训练镜像**未发布 registry** | ⏳ **已登记诉求，交总组裁定** | 发布镜像涉及流水线与权限，属总组职责 |
| ② P800 镜像**未归档未入锁** | ⏳ 同上 | 同上 |

⇒ C3 的 4 小项中，**③④ 本轮已闭环**；**①② 保持"已登记、待总组"**，不计入我方未收尾。

---

## 3 本轮同步的文档

| 文档 | 改动 |
|---|---|
| `prototype/docs/OPEN_ITEMS_AUDIT_20260929.md` | B2 / C2 / C3 三项状态回填 |
| `910C/README.md` · `README.md`（主看板） | 加 B2 与「宿主副本已对齐」条目 |
| `P800/README.md` | 加「无宿主 git 副本」事实 + r6 复跑结果 |
| `P800/docs/KUNLUN_P800_SHARED_LAYER_RERUN_20260930.md` | 新增（P800 r6 复跑） |
| `prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md` | 新增（工作包 D） |
