# FlagOS 运行时组 · 团队协调仓（runtime-team）

> 定位：FlagOS 运行时组协调仓，位于 **FlagRT 组织**（github.com/FlagRT/runtime-team）——文档、公共脚本、部署文件、项目骨架、版本事实源。**不含任何子库代码**（5 个子库在 FlagRT 组织下独立维护）。

## 当前状态

- 组织 FlagRT 已建：5 个子库（fork 自 flagos-ai，共同开发主干）+ runtime-team 公共仓，6 仓就位

## 内容规划

- `VERSIONS.md` —— 物资起始清单与版本事实源（新成员从这里起步）
- `deploy/` —— 启动配置中心（各子方向启动方式聚集处，结构见下节）
- `<子方向>/README.md` —— 各子方向目录入口（看板/任务/环境速查；细节各子方向自管，样式参照 `runtime-memory/README.md`）

## 启动配置中心（deploy/）

所有子方向的启动方式聚集于此：**各子方向成员维护自己的目录，其他成员 `git pull` 公共仓即可同步**，保证全组启动参数一致。

```
deploy/
├── clone_all.sh              # 新成员一键 clone FlagRT 5 仓（团队公共）
├── compose.base.yml           # 公共资源配置：镜像/19 设备/网络/内存/驱动挂载/公共环境变量（所有子方向共享）
└── <子方向>/                 # 如 memory/；未来 kv/ 等，对应成员维护
    ├── docker-compose.yml    # 本子方向专属：include 公共配置 + 容器名/专属环境变量
    └── .env.example          # 本子方向环境变量模板：cp 成 .env 后填个人路径
```

使用基准：
- 日常启动：`cd deploy/<子方向> && cp .env.example .env && 编辑 .env（个人路径）→ docker compose up -d`
- 公共配置变更（镜像 tag/设备/公共环境变量）：改 `deploy/compose.base.yml`，各子方向自动生效 → 同步 `VERSIONS.md`
- 子方向专属变更：改本子方向的 `docker-compose.yml` / `.env.example`
- `.env` 不入仓；需要 docker compose v2.24+（include 语法）

## 新成员起步

1. 读 `VERSIONS.md`（物资清单：fork 基线 + 镜像 + venv 组合 + 环境变量）
2. 按对应子方向 README + deploy/ 启动配置搭建（示例：memory 子方向 `runtime-memory/README.md`、`deploy/memory/`）

## 协作纪律

- 上游 flagos-ai 5 库（只读参照）→ **FlagRT 组织仓（共同开发主干，成员直接 clone，无需 fork）** → 本地分支 → push → 同仓 PR
- 日常开发：`git checkout -b <名字>/<功能>` → push 组织仓同名分支 → PR → `dev-1.0`（当前公共开发分支，宽松）；集成测试无误、阶段验收后 PR 合入 `main`（1 人 review，squash）
- 上游同步由组织仓 **Sync fork** 按钮统一负责，成员 `git pull` 即得最新

## 分支模型（GitFlow develop 模式）

- `main`：**稳定主线**。唯一变更来源 = ① fork sync（上游同步）② `dev-1.0` 的验收 PR
- `dev-1.0`：**团队公共开发分支**（develop 角色，公共版本线 1.0）。所有个人分支在此汇合、集成测试，验收后 PR 进 main。命名与子方向内部版本解耦（如 memory 子方向的 2.4 是其内部代号）；后续公共版本线依次建 `dev-2.0` 等，同样汇 main
- 铁律：共享分支（main / dev-1.0）只用 merge，不用 rebase

## 上游同步

- 组织仓保留 fork 状态：`main` 同步上游用网页 **Sync fork** 按钮；自建分支（`dev-1.0`）用 `git fetch upstream && git merge upstream/main`
- 纪律：每周至少一次；冲突当场解决并记录；共享分支只用 merge，不用 rebase
