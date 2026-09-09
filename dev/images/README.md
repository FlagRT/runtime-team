# dev/images —— 基座镜像归档与重建

> 状态：生效（2026-09-10）

## 1. 本目录的职责

对**已使用过的基座镜像**按血统与版本做归档、整理、索引，并保证可重建。只回答：

- 这个镜像由什么构成（血统、内置组件版本、官方 / 自定义边界）
- 如何从零重建它（Dockerfile、脚本、步骤、依赖清单、sha256）
- 它的离线归档在哪里（指针，不含实体）

**不回答**：为什么用它、给项目的哪个环节、什么使用约束、设备路线选型——这些由消费方文档（`dev/stack.lock.910c.*.yaml`、`docs/` 阶段目标文档）各自声明，不在本目录重复。

## 2. 结构

```
dev/images/
  README.md                  # 本文件：职责 + 规范 + 结构
  image_list.md              # 索引：所有归档镜像的 tag / 血统 / 内置版本 / repro_status / 链接
  .gitignore                 # 大文件与 *.local 护栏
  <name>/                    # 一个镜像血统一目录（去 registry 前缀与 tag 的稳定短名）
    v<N>/                    # 一个版本一目录，多版本共存
      Dockerfile             # 原始构造还原（逐字节复现目标；可能缺 owner 私有件）
      Dockerfile.repro       # 已实机验证可重建的配方（功能等价）
      build.sh               # 一键重建 + 校验
      lock.yaml              # 血统 + 内置版本 pin + 官方/自定义边界 + 重建缺口 + 重建验证结果
      REBUILD.md             # 重建步骤、踩坑、验证方法
      ARCHIVE.md             # 离线归档清单（tar 名 / 大小 / sha256 / 内容 / docker load 用法）— 机器无关，追踪
      BACKUP.local           # 本机归档实体的绝对路径 + 最近校验日期 — 机器相关，不追踪
      assets/               # sha256 清单 / requirements / verify 脚本 / patch / docker-history / provenance
```

官方镜像（非自建）：`v<N>/` 可只有 `lock.yaml` + `ARCHIVE.md`，无 Dockerfile。

## 3. 命名与版本化

- `<name>` = 稳定短名，同一血统的所有版本归其下。
- `v<N>/` 目录不可变，对应一个镜像 tag。当前生效版本见 `image_list.md`。
- 镜像有变更 → 新建 `v<N+1>/`，旧目录保留，旧 `lock.yaml` 顶部加 `superseded_by: v<N+1>`。
- 不影响已产出制品的措辞修正，直接改当前目录并记 `lock.yaml` changelog。

（与 `docs/` 阶段目标文档 §8 的文档版本化规则一致。）

## 4. 官方 / 自定义边界

`lock.yaml` 的 `layers:` 逐项标 `origin`，供重建时判断"跟随上游"还是"需按 pin 重现"：

| origin | 含义 | pin 方式 |
|---|---|---|
| `official` | 上游（华为昇腾 / 开源项目）原样产物：CANN、ubuntu、Python、vLLM-Ascend、triton-ascend 官方 wheel、MPICH 源码等 | digest 或官方版本号 |
| `custom` | FlagRT/FlagOS 自行组合或打补丁：Torch-FL、FlagGems、FlagCX、triton 补丁、verify 脚本、运行时开关 ENV | git commit + patch sha256 |

## 5. 入档门槛

一个镜像被本目录归档，其 `v<N>/` 须满足：

1. 有 `lock.yaml`（血统 + 内置版本 + 官方/自定义边界）；
2. `repro_status` ≥ 🟡；
3. 官方镜像可无 Dockerfile，但须记可拉取 digest + 上游来源。

`repro_status`：
- 🟢 = `Dockerfile.repro` + `assets` 已实机重建，且与原镜像比对通过（`pip freeze` 逐行 / 关键 `.so` 逐字节）
- 🟡 = 配方在手，未实机验证
- 🔴 = 黑盒，无配方（不得作为归档终态；须有 `ARCHIVE.md` 记录离线包）
