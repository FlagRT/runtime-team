# Case 目录标准(`dev/images/<name>/v<N>/`)

`v<N>/` 目录同时承担两个角色:镜像重建档案(构成、配方、归档,见 `README.md` §2–§5)
与对外交付的 case 包(他组据此复现问题、验证修复)。本文件规定后者需要的目录构成与
规则。新建 `v<N>/` 或向他组交付 case 时,按本文件 §4 清单逐项核对。

本标准同时适用于 [`../issues/`](../issues/README.md) 下的跨方向缺陷 case 包——缺陷不随镜像版本走、
不归属任何子方向,单独立目录,**借用**兄弟镜像目录的镜像与 demo:镜像侧档案(`Dockerfile.repro` /
`lock.yaml` / `image.baseline.yaml` / `demo/`)不在 issue 目录内重复,改为在 README 的前置条件表中
声明所用镜像 tag、image id 与核对方式(`up.sh --strict`);§3 的三段标题相应读作
**问题与现状 / 复现 / 修复验收**,内容义务不变。

## 1. 目录构成

```
<name>/v<N>/
  README.md                  # case 入口,三段式(见 §3)
  Dockerfile.repro           # ┐
  build.sh                   # │ 重建档案,规范见 README.md §2
  lock.yaml                  # │ (技术口径权威:lock.yaml / REBUILD.md)
  REBUILD.md                 # │
  ARCHIVE.md / BACKUP.local  # ┘
  image.baseline.yaml        # 镜像基线(版本事实),规范见 dev/ENV-SPEC.md §3.1
  demo/                      # demo 运行环境目录:docker-compose.yml + env.baseline.yaml(dev/ENV-SPEC.md §3.2)
  NOTES-step-by-step.md      # 可选:从零到判定的逐步操作手记(容器生命周期、踩坑清单),README 只留单入口并链过去
  assets/
    <脚本>                   # 复现 / 验证 / 收数 三类,类别在 README 脚本表中标注;兼具复现与验收的标「复现 / 验证」
    patches/                 # 构建期应用的声明式补丁(各带 README 与 sha256)
    provenance/              # 收数产物(pip freeze 等)
    evidence/                # 报错证据截选(见 §2 规则 4)
```

- demo 运行环境目录可由同一血统的兄弟目录提供,此时 README 以相对路径引用,不复制。
  例:`ascend-operator-runtime/v3/` 的验证在完整镜像上执行,使用 `../../ascend-train-comm/v3/demo/`。
- 脚本三类定义:

| 类别 | 作用 | 判定要求 |
|---|---|---|
| 复现 | 触发并展示一个已知问题或行为 | 输出须明确"问题已复现 / 未复现" |
| 验证 | 对镜像或修复给出通过与否的结论 | 退出码 0 = PASS、非 0 = FAIL,末尾输出判定行 |
| 收数 | 采集版本、环境、性能等数据 | 不做判定;产物入 `provenance/` 或由报告引用 |

  构建输入类文件(如 `patch_triton_ascend_flagtree.py`、`shmem_stub.py`)不属于上述三类,
  在 README 中单列。

## 2. 四条硬规则

1. **单入口**:复现与验证各自一条命令(或一个脚本)即可完成;前置准备纳入脚本或
   demo 的 `docker-compose.yml`,不写"先手动执行若干步骤再运行"。**撤除规避型复现**
   (demo 已内置规避配置,复现须把它拆掉)仍压缩成一条命令,环境覆盖写在命令里,并在
   README 中点明这是关键一步——漏掉会看到假 PASS。
2. **外部依赖必须声明**:镜像与以可编辑方式挂载的代码仓记录在基线文件
   (`image.baseline.yaml` / `env.baseline.yaml`),由 `dev/lib/up.sh --strict` 机器核对;
   基线文件不覆盖的依赖(模型权重、数据集、所需卡数)在 README「复现」节的前置条件表中
   逐项列出,并给出获取方式。
3. **自判定**:验证脚本以退出码与判定行给出结论,执行者无需人工判读日志;README 写明
   关键判据(例:"all_gather+sync 两 rank 均 PASS")。一个脚本兼作复现与修复验收时,
   统一三态口径:`overall: PASS` / 退出码 0 = 缺陷不存在(验收通过);`overall: FAIL` / 1 =
   缺陷复现;`overall: ERROR` / 2 = 无法下判定(环境不满足或撞到另一个问题),不计入两者。
4. **日志不进仓**:完整运行日志留在执行机器,README 注明机器名与绝对路径;仓内只放
   `assets/evidence/` 下的截选(关键报错栈、判定输出),每份截选在文件头注明来源日志
   与截取范围。

## 3. README 三段式

README 是摘要与入口,技术细节以 `lock.yaml` / `REBUILD.md` 为准,README 只给结论与指针。

```markdown
# <name> v<N>

## 用途与现状
- 镜像 tag、image id、能力范围(发布物需在首句说明)
- repro_status 及升级条件
- 已知限制:每项一行,指向 lock.yaml known_issues 条目与需求文档

## 复现
- 前置条件表:镜像 / 卡数 / 代码仓 / 模型与数据,每项给获取方式
- 单入口命令
- 预期结果(问题复现时的关键输出)

## 验证
- 脚本表:脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据路径
```

## 4. 交付前检查清单

- [ ] README 三段齐全,首句说明镜像能力范围
- [ ] 镜像目录:有 `image.baseline.yaml`,`baseline.py check --image` 通过;demo 运行环境目录(或 README 引用的兄弟目录 demo)`baseline.py check` 通过。issue 目录:在前置条件表中声明 image id 与核对方式
- [ ] 模型、数据集、卡数等基线文件之外的依赖已在前置条件表列出并给出获取方式
- [ ] 复现与验证均为单入口,命令可直接复制执行
- [ ] 验证脚本自判定(退出码 + 判定行),README 写明判据
- [ ] 报错证据有 `assets/evidence/` 截选;完整日志的机器名与路径已注明
- [ ] 容器生命周期等多步操作有 `NOTES-step-by-step.md`(或 README 已覆盖)
- [ ] README 中引用的路径全部有效
