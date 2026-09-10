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

上游发布的镜像（华为昇腾官方 / 社区，非本组自建）：`v<N>/` 可只有 `lock.yaml` + `ARCHIVE.md`，无 Dockerfile。

## 3. 命名与版本化

- `<name>` = 稳定短名，同一血统的所有版本归其下。
- `v<N>/` 目录不可变，对应一个镜像 tag。当前生效版本见 `image_list.md`。
- 镜像有变更 → 新建 `v<N+1>/`，旧目录保留，旧 `lock.yaml` 顶部加 `superseded_by: v<N+1>`。
- 不影响已产出制品的措辞修正，直接改当前目录并记 `lock.yaml` changelog。

（与 `docs/` 阶段目标文档 §8 的文档版本化规则一致。）

## 4. 官方 / 自定义边界

### 4.0 术语："官方"在本目录指哪个主体（必须带主体名，不写裸"官方"）

| 简称 | 主体 | 在本目录出现的东西 | 性质 |
|---|---|---|---|
| **华为昇腾官方** | 华为技术有限公司昇腾计算 | `quay.io/ascend/vllm-ascend` 镜像、CANN toolkit、`torch_npu`、`triton-ascend` wheel | 厂商正式发布物，公共可拉 |
| **BAAI·FlagTree 官方** | 北京智源人工智能研究院 / FlagTree 开源项目 | 「FlagTree ascend 用户手册」wiki、ascend3.5 / ascend3.2 版本线定义 | 开源社区构建规范（本目录镜像仅作对标，未按其构建路径） |
| **BAAI 内部（非发布物）** | 智源内部构建，未对外发布 | `harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-*`（CANN 底座） | 人工手搭（`manual-` 前缀），仅存 BAAI harbor + 本机 + 离线包 |
| **通用开源上游** | ubuntu / python.org / MPICH 等各自项目 | ubuntu 22.04、CPython、MPICH 源码 | 各项目官方发行版 |

### 4.1 origin 标注

`lock.yaml` 的 `layers:` 逐项标 `origin`，供重建时判断"跟随上游"还是"需按 pin 重现"：

| origin | 含义 | pin 方式 |
|---|---|---|
| `official` | 上游原样产物（华为昇腾官方 / 通用开源上游）：CANN、ubuntu、CPython、`vllm-ascend`、华为发布的 `triton-ascend` wheel、MPICH 源码等 | digest 或上游版本号 |
| `custom` | FlagRT/FlagOS 自行组合或打补丁：Torch-FL、FlagGems、FlagCX、triton 补丁、verify 脚本、运行时开关 ENV | git commit + patch sha256 |

> 注：`origin: official` 只表示"该层是上游原样产物、按上游版本号/digest 跟随"，**不等于"华为发布"**——CANN 底座镜像本体是 **BAAI 内部**手搭的（见 4.0），只是其中的 CANN toolkit 用的是华为昇腾官方版本。

## 5. 入档门槛

一个镜像被本目录归档，其 `v<N>/` 须满足：

1. 有 `lock.yaml`（血统 + 内置版本 + 官方/自定义边界）；
2. `repro_status` ≥ 🟡；
3. 上游发布的镜像（华为昇腾官方 / 社区）可无 Dockerfile，但须记可拉取 digest + 上游来源。

`repro_status`：
- 🟢 = `Dockerfile.repro` + `assets` 已实机重建，且与原镜像比对通过（`pip freeze` 逐行 / 关键 `.so` 逐字节）
- 🟡 = 配方在手，未实机验证
- 🔴 = 黑盒，无配方（不得作为归档终态；须有 `ARCHIVE.md` 记录离线包）

上游发布的镜像（华为昇腾官方 `vllm-ascend`）/ BAAI 内部 CANN 底座（无自建 Dockerfile）用 🟢 表示"digest 锁定 + 已归档 + 已验证可拉"，与自建镜像的 `functional-repro` 是两层含义，`image_list.md` 的备注列会写明是哪一种。

## 6. 迁移前置

把归档整套迁到新机器，需要：① 仓内配方（本目录，随 git）② 各 `ARCHIVE.md` 指向的离线包（三个 `docker save` tar.gz + `recovered/` 资产，当前在 raid 盘，不随 git）③ 上游可达（`harbor.baai.ac.cn` 的 **BAAI 内部** CANN 底座、`quay.io` 的**华为昇腾官方** vLLM 镜像）。三者缺一：缺 ② 退化到"从 ① + ③ 功能等价重建"；缺 ③ 且缺 ② 则阻塞。逐字节复现另需 owner 私有 FlagCX commit + patch（见 `TODO.md`）。
