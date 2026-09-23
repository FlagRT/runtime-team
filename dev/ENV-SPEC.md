# 运行时开发环境规范（dev/ENV-SPEC.md）

> 状态文档（分类见 `README.md`「文档职责」）：规范本身——机制、schema、用法、现状。

## 0. 核心原则（锚点）

运行时组的目标：搭建一套**兼容多款国产芯片**的 runtime。

- **device-context** 方向负责屏蔽底层设备差异，提供统一运行时接口。
- **总组**收拢 device-context 已验证的能力和现状，制定供全组统一开发的基座；总组只对
  "基座怎么组装、怎么校验"负责。
- **memory / communication 等子方向**在这个统一基座上做各自的优化验证。
- **组外算子/编译组**产出 FlagTree / FlagGems 等代码库——我们是消费方。
- **模型组**做最终的训练/推理调度——我们是被依赖方。

**判据**：本规范的任何字段、任何工具，服务于"屏蔽设备差异 / 支撑全组统一基座"这两件事。

## 1. 适用范围

覆盖 `dev/` 下所有子方向的容器启动方式，以及 `dev/images/<name>/vN/` 下所有候选血统的
启动方式，建立在 docker compose 原生 `-f` 多文件合并之上，补充"版本声明"和"启动前
校验"两块。

某个芯片+版本该用哪个 compose 文件，以 `dev/images/<chip>/vN/` 下的
`docker-compose.<tag>.yml` 为准，配合阶段目标文档 / `dev/stack.lock.910c.vN.yaml`
确定当前候选或生效版本。

`dev/compose.base.yml` 状态为 LEGACY（见该文件头部说明）。

## 2. 两类文件，各管一件事

同一个"层"（layer）由一对文件组成，同名同目录，后缀不同：

| 文件 | 谁读 | 管什么 |
|---|---|---|
| `docker-compose.<tag>.yml` | docker compose（原生） | 这一层用什么镜像、什么挂载、什么环境变量、什么设备 |
| `pins.<tag>.yaml` | 本规范的校验脚本 | 这一层期望的版本是什么、如何核对 |

一层可以只有 `docker-compose.<tag>.yml`（不声明版本约束，如只改一个环境变量的探索层）。
版本声明必须挂在一个真实会被启动的层上。

**分文件的原因**：docker compose 的 `-f` 多文件合并对 `services:`/`volumes:`/
`environment:` 等原生字段深度合并，对自定义顶层扩展字段（`x-*`）整体替换（`docker
compose config` 实测确认：后声明的 `x-` 字段整体覆盖前面的，未提到的子字段一并丢失）。
版本声明因此用独立文件、独立的合并逻辑（见 §5）。

## 3. 目录与命名约定

- **官方候选血统层**（总组发布/收拢的版本）：放在 `dev/images/<name>/vN/` 下，与该血统
  的 `Dockerfile.repro`/`lock.yaml`/`REBUILD.md` 同目录。
- **子方向探索层**（各方向自己叠加、验证不同配置差异用）：放在 `dev/<子方向>/` 下，与该
  方向自己的 `docker-compose.yml` 同目录。
- `<tag>` 自由命名，用能看出用途的短语（如 `routeA`、`probe-flagcx-sync`）。层与层的组合
  方式由调用者显式列出文件路径决定（见 §8）。
- `pins.<tag>.yaml` 与对应 `docker-compose.<tag>.yml` 同目录、`<tag>` 部分完全一致；后缀
  `.yaml`/`.yml` 均可识别，同一层只保留一份。
- `docker-compose.<tag>.yml` 里挂载公共仓根用 `${WORKSPACE_ROOT}`（`dev/lib/up.sh` 会
  设置这个环境变量），不用相对路径 `../`——compose 的相对路径按"首个 `-f` 文件所在目录"
  解析，一个层被谁排在第几位调用会改变解析结果；`${WORKSPACE_ROOT}` 不受调用顺序影响。

## 4. `pins.<tag>.yaml` schema

```yaml
image:                     # 可选。声明了即做严格校验（没有 strict/loose 之分）。
  ref: <string>             # 完整 image reference（repo:tag）
  digest: <string>          # sha256:... ，来源：docker image inspect --format '{{.Id}}' <ref>，
                            # 或对应 dev/images/<name>/vN/lock.yaml 的 image_id 字段（补全 sha256: 前缀）

repos:                     # 可选。key = 仓库名（见下方"仓库名注册表"），value = 该仓库的声明
  <RepoName>:
    class: strict | loose   # 默认 loose
    expect_commit: <40 位 hex commit sha>   # class=strict 时必填
    expect_branch: <string>                  # class=loose 时可选，仅作提示，不阻断
```

**仓库名注册表**（`<RepoName>` 只能是下列六个之一，对应宿主目录固定，见
`dev/lib/verify_env.py` 的 `REPO_DIR`）：

| RepoName | 宿主目录（相对仓库根） |
|---|---|
| `FlagTree` | `FlagTree` |
| `FlagGems` | `FlagGems` |
| `Torch-FL` | `PyTorch-Plugin-FL` |
| `FlagCX` | `FlagCX` |
| `vllm-plugin-FL` | `vllm-plugin-FL` |
| `FlagPerf` | `FlagPerf` |

**关键澄清**：是否挂载、挂到哪，由对应 `docker-compose.<tag>.yml` 的 `volumes:` 决定；
`repos` 只核对宿主检出的代码是否等于声明的期望值，两者独立。

**声明 `repos` 条目的前提**：`Dockerfile.repro` 里这个库那一层是 `COPY <build-context
内的路径> ...`（从构建时的宿主目录复制进镜像，宿主检出的 commit 因此有意义，`repos`
声明的价值是"确认宿主当前检出与构建这个镜像时用的是同一个 commit"）。判断依据是
`Dockerfile.repro` 对应那几行的实际写法（`COPY` 还是构建期自带 `git fetch`/`clone` 指定
commit、不依赖构建上下文里的宿主目录），不是猜这个库是否被烘焙进镜像。

## 5. 合并规则

多层 `pins.<tag>.yaml` 按 `-f` 传入顺序做**字典递归合并**（后者覆盖前者同名字段）：

- 普通字段（字符串/布尔）：后者整体替换前者。
- 字典字段（如 `image`、`repos`）：递归合并，只覆盖后者提到的子键，未提到的子键保留前者
  的值。
- `repos` 本身是"以仓库名为 key 的字典"，因此天然按仓库名合并——一层只改
  `vllm-plugin-FL` 的声明，同一份 pins 里其他仓库的声明保持不变。

合并逻辑实现在 `dev/lib/verify_env.py`。

## 6. 校验规则

对合并后的 pins 结果，逐项检查：

1. **image**（如声明）：`docker image inspect --format '{{.Id}}' <ref>` 必须等于
   `digest`。不等 = 硬失败（不区分 strict/loose）。
2. **repos** 每一项：
   - `class: strict`：宿主目录须存在、是合法 git 仓库，且 `git rev-parse HEAD` 等于
     `expect_commit`（比较完整 commit SHA）。任一条件不满足 = 硬失败。
   - `class: loose`：宿主目录须存在、是合法 git 仓库；若声明了 `expect_branch`，当前分支
     不一致只警告，不阻断。
   - `git status --porcelain` 非空（有未提交改动）→ 警告，不阻断。
3. 汇总：有一条 strict/image 硬失败即整体 FAIL，退出码非 0；loose 类不一致只出现在报告
   里，不影响退出码。

每次校验把完整结果（合并后的 pins、逐项 PASS/WARN/FAIL、时间戳）写入
`dev/lib/.env_provenance.json`，供撰写验收报告时引用。

## 7. 工具

四个脚本，都在 `dev/lib/`，对所有芯片/所有子方向/所有迭代版本通用：

| 脚本 | 作用 | 退出码语义 |
|---|---|---|
| `verify_env.py` | 合并 pins + 执行 §6 校验 + 写 provenance + 打印报告 | 0=通过，1=有 strict 失败，2=用法/文件格式错误 |
| `verify_env.sh` | `verify_env.py` 的薄包装 | 同上 |
| `up.sh` | 先调用 `verify_env.sh`，通过才执行 `docker compose -f ... up -d` | 校验未通过时不启动，原样返回非 0 |
| `list_pins.py` | 扫描仓库内所有 `pins.*.yaml`，按仓库名分组列出每层的声明，标出同一仓库在不同层被声明成不同 commit 的情况 | 只读 |

`verify_env.sh`/`up.sh` 的参数是**一组 `docker-compose.<tag>.yml` 文件路径**（与最终传给
`docker compose -f` 的列表一致，顺序也一致）——每个路径按 §2 的命名约定推导出同目录下的
`pins.<tag>.yaml`（不存在则跳过）。

`list_pins.py` 不带参数列出全部，带仓库名参数过滤（如 `list_pins.py FlagCX FlagGems`），
加 `--images` 一并列出各层声明的镜像 ref/digest。对比数据来自脚本实时读取
`pins.*.yaml`，始终反映当前文件内容。

## 8. 用法

**只校验，不启动**（用于撰写验收报告前留证据）：

```bash
dev/lib/verify_env.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml
```

**校验并启动（总组发版对齐环境）**：

```bash
dev/lib/up.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml
```

**子方向自行探索**（同一个入口，多叠一层，只声明要改的部分）：

```bash
dev/lib/up.sh dev/images/ascend-operator-runtime/v3/docker-compose.routeA.yml \
              dev/communication/docker-compose.yml \
              dev/communication/docker-compose.<自己的探索tag>.yml
```

总组发版和子方向预研调用同一个脚本、同一份校验逻辑，区别只在叠了几层文件、每层文件里
写了什么。

## 9. 与既有文档的关系

- `dev/compose.base.yml`：LEGACY（见该文件头部）。
- `VERSIONS.md`：全组的人读版本事实源（新成员起步入口）。候选血统新增/切换 pins 时，
  `VERSIONS.md` §3 对应章节更新一行指向该血统的 `pins.<tag>.yaml` 路径，不在
  `VERSIONS.md` 里重复版本号本身。
- `dev/images/<name>/vN/lock.yaml`：镜像 digest 的权威来源（`image_id` 字段）；
  `pins.<tag>.yaml` 里的 `image.digest` 直接取自这里。
