# FlagOS 运行时组 · 团队协调仓（runtime-team）

> 定位：FlagOS 运行时组协调仓（FlagRT 组织，github.com/FlagRT/runtime-team）——文档、公共脚本、部署文件、项目骨架、版本事实源（`VERSIONS.md`）。**不含任何子库代码**（5 个子库 fork 自 flagos-ai，在 FlagRT 组织下独立维护）。

## 目录布局（嵌套布局：公共仓根 = 开发总目录）

clone 公共仓后，5 个子库由 `dev/clone_all.sh` 收拢进公共仓根，所有开发都在其中进行。

```
runtime-team/                  # 公共仓根 = 开发总目录
├── README.md / VERSIONS.md / .gitignore   # VERSIONS.md = 版本事实源（新成员起步点）
├── dev/                       # 开发配置中心（各子方向成员维护自己目录，pull 即同步）
│   ├── clone_all.sh           # 一键对齐 5 子库（缺失 → clone，已有 → 更新）
│   ├── compose.base.yml       # 公共资源配置：镜像/设备/网络/挂载/公共环境变量
│   └── <子方向>/              # 如 memory/：README.md 看板 + docker-compose.yml + .env.example
├── PyTorch-Plugin-FL/         # 5 个子库（独立 git 仓库，各自 fork 管理，不进本仓）
├── FlagCX/
├── FlagGems/
├── vllm-plugin-FL/
└── FlagTree/
```

要点：
- 宿主侧零配置：`compose.base.yml` 相对路径 `../` 自动解析公共仓根（按首个 -f 文件所在目录）；容器侧固定挂载 `/workspace`
- 日常启动：`cd dev/<子方向> && docker compose -f ../compose.base.yml -f docker-compose.yml up -d`（需 compose v2，后文件覆盖前文件）
- 公共配置变更 → 改 `compose.base.yml` 并同步 `VERSIONS.md`；子方向专属变更 → 改各自目录；`.env` 不入仓

## 新成员起步

**前置**：GitHub SSH key 已上传、已是 FlagRT 组织成员（`clone_all.sh` 自动预检认证，失败即退出）。

```bash
git clone git@github.com:FlagRT/runtime-team.git && cd runtime-team
bash ./dev/clone_all.sh      # 一键对齐 5 子库
cd dev/<子方向>               # 按该子方向 README 启动容器
```

- 验证：容器内 `ls /workspace` 应看到 5 子库 + 公共仓内容
- 起容器前 `npu-smi` 确认卡空闲；DrvMng 并发容器上限 ≈3，冲突时先 `docker stop` 旧容器
- 日常开发：`cd <子库> && git checkout dev-1.0`（FlagTree 用 `triton_v3.2.x`）→ 建个人分支 → push → 同仓 PR
- 细节与踩坑：各子方向看板（如 `dev/memory/README.md`）+ `VERSIONS.md`

## 分支模型（fork 三线）

<div align="center">

<img src="assets/git_fork_branch_discipline.svg" width="500">

</div>

我们 fork 了上游仓库，用三条线干活：

| # | 规则 | 大白话 |
|---|------|--------|
| 1 | **main 只同步、不开发** | main 专门用来 Sync fork 拉原仓库的新东西，永远别在它上面提交自己的代码 |
| 2 | **dev-1.0 是公共开发分支** | 所有人的开发分支最终都合并到 dev-1.0，大家在这条线上协作 |
| 3 | **上游新特性，看需要才合** | dev-1.0 想用上游的新功能，就把 main 合并进来；不想用就不合，完全没问题 |

**日常操作（三步）**

**① 同步上游（负责人定期做）**：GitHub 网页 → fork 仓库 → 点 **Sync fork** → **Update branch**。main 自动跟上原仓库，不用命令。

**② 开发（每个人）**：从 dev-1.0 拉最新 → 开自己的分支 → 开发完 → 发 PR 合回 **dev-1.0**。

**③ 要上游新特性（看团队需要）**：

```bash
git checkout dev-1.0 && git pull
git merge origin/main   # 把已同步的 main 合进来
git push origin dev-1.0
```

**记住两句**

- main 上**不能**直接开发 —— 它是同步专用的。
- 不合并 main **不会怎样** —— 只是 dev-1.0 暂时没有上游的新功能而已。

> 例外：FlagTree 的基准分支是 `triton_v3.2.x` 而非 main（上游按 triton 大版本分支维护，昇腾绑 3.2）；FlagGems 的同步分支名为 `master`（fork 默认分支非 main）。各仓基准分支以 `VERSIONS.md` §1 为准。
> 铁律：共享分支（main / dev-1.0）只用 merge，不用 rebase

## 红线

- 不改宿主机器配置/驱动；所有安装与实验都在容器内进行
- 多卡测试前 `npu-smi` 确认 16 卡空闲；DrvMng 并发容器上限 ≈3，超了先停旧容器
- 公共仓内容逐项审查后才上传；个人调试记录默认收拢 `personal/` 不上传
