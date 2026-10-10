# 运行时环境基线规范（dev/ENV-SPEC.md）

> 状态：生效（2026-10-10）。本文件是环境基线机制的唯一规范：概念、判定规则、文件格式、
> 工具与用法。设计依据见
> [黄金准则 GR3/GR5 与 §3](../docs/运行时基座建设-黄金准则与方案存档.v1.md)。

## 0. 结论

- **镜像**的版本事实记录在 `image.baseline.yaml`（与 `Dockerfile.repro` 同目录），由 image id 约束。
- **运行实例**由运行环境目录定义：`docker-compose.yml`（启动与装配）+ `env.baseline.yaml`（版本事实）。
  以可编辑方式挂载的 repo 由宿主 checkout 的 HEAD 约束。
- 两类基线文件均由 `dev/lib/baseline.py` 生成和核对，不手工编辑；统一以
  `dev/lib/up.sh <env-dir> [--strict]` 启动。
- 适用范围：`dev/images/` 下全部自建镜像，以及基于这些镜像启动的全部运行环境。
  `dev/compose.base.yml` 及仍沿用它的子方向 compose 属于 LEGACY，不在本机制内。
- 实例：镜像基线 [`images/ascend-train-comm/v3/image.baseline.yaml`](images/ascend-train-comm/v3/image.baseline.yaml)；
  运行环境目录 [`images/ascend-train-comm/v3/demo/`](images/ascend-train-comm/v3/demo/docker-compose.yml)。

## 1. 两类对象

| 对象 | 定义 | 基线文件回答的问题 |
|---|---|---|
| 镜像 | 构建完成即不可变的制品，由 image id 寻址 | 镜像固化了哪些资产，各自是什么版本 |
| 运行环境目录 | 一个目录 = 一套完备的运行环境，基于一个指定镜像创建一个运行实例 | 实例用的是哪个镜像，另外以可编辑方式挂载了哪些 repo，各自在哪个 commit |

- 基线文件不参与 compose 的创建、启动与运行，是全组遵循的环境记录。
- 每个镜像附带一个 demo 运行环境目录，记录该镜像验收时使用的运行配置，同时作为推荐用法。
- 镜像的使用方可在任意位置建立自己的运行环境目录（模板见 [`_template/`](_template/README.md)）。

## 2. 固化资产与挂载资产

一个 repo 属于哪一类，由**代码从哪里来**决定，与产出方、变更频率无关：

| | 固化资产（baked） | 挂载资产（mounted） |
|---|---|---|
| 定义 | 代码在构建期进入镜像文件系统 | 代码来自宿主目录，运行时经 volume 挂载并以可编辑方式生效 |
| 版本归属 | 镜像（由 image id 唯一确定） | 运行实例（由启动时宿主 repo 的 HEAD 确定） |
| 记录位置 | `image.baseline.yaml` | `env.baseline.yaml` |
| check | 只核对本地镜像 id，不核对宿主 repo | 核对宿主 repo 的 HEAD 与工作区状态 |
| load | 不处理 | 将宿主 repo 切换到基线中的 commit |

四层模型（L1 芯片底座 / L2 算子编译 / L3 Runtime 组件 / L4 挂载装配）描述资产的职责层级，
与交付方式正交：同一层的资产既可固化也可挂载。基线文件中每项以 `layer` 标注层级。

### 2.1 判定规则

- **镜像场景**：解析 `Dockerfile.repro` 中把代码带入镜像的指令，只收录真正固化的 repo（§4.2）。
  - 本组构建层不允许 editable 安装（代码固化时 editable 无意义；依赖挂载时镜像不再自包含），export 报错。
  - 父镜像继承来的 editable 安装，若源码位于镜像内，即为固化资产，标注
    `install: editable-in-image` 并给出提示。实例：L1 基座中的 vLLM，源码在 `/vllm-workspace/vllm`。
- **运行环境场景**：只收录 compose 中**显式声明**为可编辑挂载的 repo（§3.2）。
  仓根整体挂载到 `/workspace` 只使代码可见，不使其生效，不自动收录。

### 2.2 同一 repo 既固化又挂载

允许，用于在镜像基线之上调试某个组件：

- `env.baseline.yaml` 中该 repo 标注 `overrides_baked: true` 与 `baked_commit`；镜像内固化版本带补丁时
  另记 `baked_patches`，check 给出警告：挂载副本不含这些补丁。
- compose 的生效方式须确保挂载版本优先（§5）。

### 2.3 资产在两类之间迁移

变更频率决定资产应放在哪一类；跨越边界只通过以下两个显式动作：

| 动作 | 触发条件 | 结果 |
|---|---|---|
| 晋升（挂载 → 固化） | 该组件本阶段验证冻结 | 写入新版本镜像 `v<N+1>/` 的构建配方，构建后 export 镜像基线 |
| 下放（固化 → 挂载） | 需要在现有镜像上快速迭代该组件 | 在运行环境 compose 中声明挂载，形成 §2.2 的覆盖关系；镜像不变 |

## 3. 文件形态

### 3.1 镜像基线：`image.baseline.yaml`

位置：`dev/images/<name>/v<N>/image.baseline.yaml`。展开收录父镜像继承来的资产，一个文件即可看全
镜像的全部资产。实例见 [`images/ascend-train-comm/v3/image.baseline.yaml`](images/ascend-train-comm/v3/image.baseline.yaml)。

| 字段 | 内容 |
|---|---|
| `schema` / `kind` | `flagrt.baseline/v1` / `image` |
| `generated_by` / `generated_at` | 生成命令与时间 |
| `image.ref` / `image.id` | 镜像 ref 与 image id（`id` 是 check 项） |
| `image.recipe` / `image.demo` | 构建配方（含父层配方）；demo 运行环境目录 |
| `base_image` | 外部基座（L1）的 `ref`、`digest`、`provides`（CANN / Python / torch 等版本） |
| `baked_repos.<名称>` | `layer`、`from`（来源层与方式）、`commit`、`install`，可选 `patches`（文件名 + sha256）、`source_path`、`info` |
| `baked_resources[]` | 固化资源：`name`、`layer`、`url` 或 `path`、`sha256`；构建期未固定版本的记 `pin: unpinned (...)` |
| `packages` | 完整包清单（pip freeze）路径，仅供查阅 |

`info`（分支、提交时间）仅供阅读，不参与任何校验；无法获取时记为 `unavailable`。

### 3.2 运行环境目录

```
<任意位置>/<环境名>/
  docker-compose.yml      # 启动与装配，含 x-flagrt 声明
  env.baseline.yaml       # baseline.py export 生成，不手工编辑
  .last-check.json        # 每次 check 的结果（不追踪）
```

**声明**：在 `docker-compose.yml` 中用顶层扩展字段声明本环境。工具只读取本目录这个文件中的
声明，`extends` 引入的文件中的 `x-flagrt` 不生效；路径相对本目录。

```yaml
x-flagrt:
  image_baseline: ../image.baseline.yaml      # 相对本目录
  mounted_repos:                              # 无挂载时写 {}
    vllm-plugin-FL: {activation: pip-editable}  # 或 pythonpath，见 §5
```

`mounted_repos` 的键为仓库名。`host_path`（相对 `WORKSPACE_ROOT`）缺省按下表取值，
`container_path` 缺省由 compose 实际解析出的 bind 挂载推导，两者均可显式填写：

| 仓库名 | 宿主目录 | 仓库名 | 宿主目录 |
|---|---|---|---|
| `FlagTree` | `FlagTree` | `FlagCX` | `FlagCX` |
| `FlagGems` | `FlagGems` | `vllm-plugin-FL` | `vllm-plugin-FL` |
| `Torch-FL` | `PyTorch-Plugin-FL` | `FlagPerf` | `FlagPerf` |

**`env.baseline.yaml`**：`kind: container`；`image`（`ref`、`id`、`baseline`）抄录自镜像基线；
`baked_repos` 原样抄录、只读；`mounted_repos.<名称>` 记录 `host_path`、`container_path`、
`activation`、`commit`、`dirty`，覆盖固化版本时另记 `overrides_baked`、`baked_commit`、
`baked_patches`（§2.2）。实例见 [`images/ascend-train-comm/v3/demo/env.baseline.yaml`](images/ascend-train-comm/v3/demo/env.baseline.yaml)。

**复用镜像 demo**：运行环境目录可经 compose `extends`（按服务）复用镜像 demo，只写差异。
以下写法在本机 Docker Compose v2.29.7 上经 `docker compose config` 与 `baseline.py export/check`
验证（设备、挂载、环境变量按键合并，`command` 等标量字段覆盖）：

```yaml
name: flagrt-<环境名>                     # 必须与 demo 不同
x-flagrt:
  image_baseline: ../images/ascend-train-comm/v3/image.baseline.yaml
  mounted_repos: {}
services:
  runtime-dev:
    extends:
      file: ../images/ascend-train-comm/v3/demo/docker-compose.yml
      service: runtime-dev
    container_name: flagrt-<环境名>-910c   # 必须覆盖，否则与 demo 容器重名
```

compose `include` 不能覆盖被引入文件中已定义的服务，不用于这一场景。

## 4. 管理脚本：`dev/lib/baseline.py`

### 4.1 动作与场景

| 动作 | 镜像场景（`--image <image-dir>`） | 运行环境场景（`<env-dir>`） |
|---|---|---|
| `export` | 解析 Dockerfile 判定固化资产，以镜像标签取 commit，交叉核对后生成 `image.baseline.yaml`；`--demo <dir>` 写入 `image.demo`，`--image-ref` 指定镜像 ref（缺省取 `build.sh` 的 `-t`） | 抄录镜像基线的固化部分，读取 `x-flagrt.mounted_repos` 对应宿主 repo 的 HEAD、工作区状态与提交元数据，生成 `env.baseline.yaml`；`--require-clean` 时工作区不干净即拒绝导出 |
| `check` | 本地镜像 id 与 commit 标签是否与基线一致 | 本地镜像 id；每个挂载 repo 的 HEAD 是否等于基线 commit、工作区是否干净；结果写 `.last-check.json` |
| `load` | 不适用（镜像只能重新构建） | 将每个挂载 repo 切换到基线 commit；不拉取、不构建镜像 |
| `show` | 以表格打印资产、版本、层级与 info | 同左，并标出覆盖关系 |
| `list` | 列出仓内全部 `dev/images/*/v*/image.baseline.yaml` 的资产与版本 | — |

退出码：`0` 一致或成功；`1` 不一致（check）或核对、切换失败（export / load）；`2` 用法或文件格式错误；
`3` load 被安全规则中止。

### 4.2 镜像场景：Dockerfile 判定规则

只识别以下模式，其余一律报错，不做猜测：

1. `COPY src/<Repo>/ …`，随后对该路径执行非 editable 的 `pip install` → 固化 repo（构建上下文来源）。
2. `git remote add origin <url>` + `git fetch … ${<X>_COMMIT|REF}` → 固化 repo（构建期拉取来源）。
3. `FROM ${…}` → 父镜像：有自己的 `image.baseline.yaml` 则展开继承；外部基座则从镜像读取
   `pip list --editable` 与 pip freeze 补全 L1 资产。
4. `curl … ${<X>_URL}` / `COPY assets/<file>` → 固化资源。
5. 本层出现 `pip install -e` → 报错（§2.1）。

commit 以已构建镜像的 `io.flagrt.*.commit` 标签为准，Dockerfile 的 ARG 只用于交叉核对，不一致即报错。
配方写法变化导致 export 报错时，同步扩展规则，不放宽为猜测。

### 4.3 运行环境场景的行为

- **export**：工作区不干净时记录 `dirty: true` 并警告；验收或出结论的环境加 `--require-clean`。
- **check**：镜像 id 不一致、挂载 repo commit 不一致、容器内挂载来源与核对的宿主目录不一致，
  均报告为不一致；工作区不干净只警告；固化部分不与宿主比对。
- **load**：任一挂载 repo 工作区不干净 → 整体中止（退出码 3），不做部分切换；本地缺少目标 commit
  时先 `git fetch`；切换结果为 detached HEAD，并打印切换前的 commit 与分支以便回退；切换后
  检出不完整即报错。
- **`dev/lib/up.sh <env-dir> [--strict]`**：先执行 check，默认只报告不一致、不阻止启动；加
  `--strict` 时不一致即拒绝启动，用于验收与出结论的运行。然后执行
  `docker compose -f <env-dir>/docker-compose.yml up -d`。`--strict` 只约束启动时刻：
  宿主 checkout 在启动后的改动会直接进入运行中的实例。
- `up.sh` 以 `FLAGRT_WORKSPACE_ROOT`（缺省为本仓根）同时导出 compose 的 `WORKSPACE_ROOT`，保证
  check 核对的就是容器实际挂载的 checkout。绕开 `up.sh` 直接使用 `docker compose` 时须自行
  `export WORKSPACE_ROOT=<公共仓根绝对路径>`。

## 5. 挂载资产的生效方式（`activation`）

| 取值 | 做法 | 适用 |
|---|---|---|
| `pip-editable`（推荐） | 容器 `command` 先执行 `pip install --no-deps --no-build-isolation -e <挂载路径>`，再常驻 | 默认选择 |
| `pythonpath` | 在 `PYTHONPATH` 最前面加入挂载路径 | 仅在必须只读挂载时使用 |

`pip-editable` 的 compose 写法（vllm-plugin-FL 试点已端到端验证）：

```yaml
    environment:
      - PYTHONDONTWRITEBYTECODE=1   # 否则宿主 checkout 中出现 root 属主的 __pycache__
      - GIT_OPTIONAL_LOCKS=0        # 否则 setuptools-scm 使宿主 .git/index 变为 root 属主
    command: ["bash", "-c", "pip install --no-deps --no-build-isolation -e /workspace/vllm-plugin-FL && exec sleep infinity"]
```

- 安装过程只写入 setuptools-scm 的 `_version.py`（上游已 gitignore），其内容在重新安装前不更新。
- 代码改动在容器内直接生效；入口点、依赖或版本号的改动须重新安装（重启容器）。
- `pythonpath` 方式下，包元数据、入口点与版本号仍是镜像内固化版本的；且 compose 中须写出镜像
  `PYTHONPATH` 的完整值（§6）。

## 6. 注意事项

- **compose 的 `PYTHONPATH` 整体替换镜像内的值**：须写出完整值（含 CANN 路径，取自
  `docker image inspect` 的 `.Config.Env`），并在 YAML 中加引号，否则结尾的 `:` 会被解析为映射。
  写法见 demo 的 `docker-compose.yml`。
- **不要以部分克隆（partial clone）的再克隆作为挂载 repo**：缺失对象会导致检出不完整；load 会
  对此报错。
- **首次 `export --image` 会下载 URL 资源计算 sha256**（大 tarball，缓存于
  `~/.cache/flagrt-baseline`，可用 `FLAGRT_BASELINE_CACHE` 改位置）。npu1-11 上代理对
  `baai-cp-web.ks3-cn-beijing.ksyuncs.com` 限速，必要时将其加入 `no_proxy`。
- **`info: unavailable`**：经代理访问 GitHub 失败时无法取得分支与提交时间（当前 v3 的 `vllm`
  即如此）。不影响校验；在可联网的机器上重新 export 即可补全。
- **load 修改宿主 repo 状态**：共享机器上可能影响同一 checkout 的其他使用者，运行环境目录宜在
  个人 checkout 上使用。
- v1/v2 镜像没有 commit 标签，不纳入本机制。

## 7. 常用流程

```bash
# 新镜像版本：构建后生成并核对镜像基线
python3 dev/lib/baseline.py export --image dev/images/<name>/v<N> --demo dev/images/<name>/v<N>/demo
python3 dev/lib/baseline.py check --image dev/images/<name>/v<N>

# 运行环境：声明 x-flagrt → 导出 → 启动
python3 dev/lib/baseline.py export <env-dir>
dev/lib/up.sh <env-dir> [--strict]

# 复现他人的运行环境：按基线切换挂载 repo，再启动
python3 dev/lib/baseline.py load <env-dir>
dev/lib/up.sh <env-dir> --strict

# 查看
python3 dev/lib/baseline.py show <env-dir>          # 或 show --image <image-dir>
python3 dev/lib/baseline.py list
```

## 8. 与其他文档的关系

- [`images/README.md`](images/README.md) 与各 `lock.yaml`：`lock.yaml` 记录构建过程、已知问题与验证结果；
  版本事实以 `image.baseline.yaml` 为准。
- [`../VERSIONS.md`](../VERSIONS.md)：指向镜像基线与 demo 目录，不重复版本号。
- `dev/stack.lock.910c.*.yaml`：锁定声明，本机制不修改。
- `dev/compose.base.yml`：LEGACY（见该文件头部）。
