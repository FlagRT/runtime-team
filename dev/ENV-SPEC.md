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
- 镜像自带的运行环境目录分两个角色，变更性质相反，不并存于同一个目录：
  - `demo/` = 该镜像版本**验收时所用配置的历史记录**，永久冻结、内容不再改动，**不是推荐用法**
    （它是验收结论的对应物）；概念上是该镜像运行环境的第 0 版。
  - `env/v<N>/` = **现行推荐运行环境**，带修订号、可迭代；路径与编号约定见 §3.2。
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

**不可下放的固化资产**：从包索引装入的预编译发行包（§3.1 `baked_packages`）没有对应的源码
checkout，挂载源码也不会改变已装好的 wheel，因此不适用「下放」——把它写进
`x-flagrt.mounted_repos` 是配置错误，`export` 直接报错。需要改这类组件的代码时，要么换成
从源码构建的固化方式（晋升路径，落到 §4.2 模式 1/2），要么等上游发新版本后改 pin。
实例：昆仑芯的 `flagtree`（[`images/kunlun-operator-runtime/v2/image.baseline.yaml`](images/kunlun-operator-runtime/v2/image.baseline.yaml)）。

## 3. 文件形态

### 3.1 镜像基线：`image.baseline.yaml`

位置：`dev/images/<name>/v<N>/image.baseline.yaml`。展开收录父镜像继承来的资产，一个文件即可看全
镜像的全部资产。实例见 [`images/ascend-train-comm/v3/image.baseline.yaml`](images/ascend-train-comm/v3/image.baseline.yaml)。

| 字段 | 内容 |
|---|---|
| `schema` / `kind` | `flagrt.baseline/v1` / `image` |
| `generated_by` / `generated_at` | 生成命令与时间 |
| `image.ref` / `image.id` | 镜像 ref 与 image id（`id` 是 check 项） |
| `image.recipe` / `image.demo` / `image.env_current` | 构建配方（含父层配方）；验收记录目录（`demo/`，冻结）；**现行推荐运行环境目录**（`env/v<N>/`，`check` 项，§3.2） |
| `base_image` | 外部基座（L1）的 `ref`、`digest`、`provides`（无 commit 的版本事实：CANN / XTDK / XRE / XCCL / Python / torch 等）；能对应到 commit 的包只记在 `baked_repos`，不在此重复。取数来源是 `baseline.py` 的三张声明式探针表（`BASE_PROVIDES_ENV` 环境变量 / `BASE_PROVIDES_FILES` 镜像内文件 / `BASE_PROVIDES_PKGS` 包白名单），接新芯片只加数据 |
| `baked_repos.<名称>` | 有仓库身份、以 commit 为版本锚的固化资产：`layer`、`from`（来源层与方式）、`commit`、`install`，可选 `commit_source`、`version`、`patches`（文件名 + sha256）、`source_path`、`supersedes`、`info`。键一律是 §3.2 的仓库名，不是 pip 发行名 |
| `baked_packages[]` | 从包索引装入的预编译发行包（无仓库身份、无 commit，以 `version` 为版本锚）：`name`（pip 发行名）、`layer`、`version`、`index_url`、`install`、`pin`，可选 `wheel`（`filename`、`sha256`、`source`、`fetched_at`）。不可下放为挂载资产（§2.3） |
| `baked_resources[]` | 固化资源：`name`、`layer`、`url` 或 `path`、`sha256`；构建期未固定版本的记 `pin: unpinned (...)` |
| `packages` | 完整包清单（pip freeze）路径，仅供查阅 |

**版本锚**：每项固化资产至少要有一个可机器核对的版本锚——`baked_repos` 是 `commit`，
`baked_packages` 是 `version`，两者都以镜像标签为准（§4.2）。`check` 核对全部**存在**的锚。

**可复现性的六项版本事实**：挂载覆盖形态下真正运行的代码不在镜像里，只有镜像标识不足以让别人
复现到同一套栈。一套运行实例的版本事实必须含六项——**镜像标识**、**外部仓地址**、**提交号**、
**获取方式**、**校验方式**、**容器内挂载点**。镜像基线回答第一项（`image.ref` / `image.id`），
并以 `image.env_current` 指出现行推荐运行环境是哪个目录；其余五项由 `env.baseline.yaml` 的
`mounted_repos` 承载（依次是 `remote`、`commit`、`activation`、`dirty` / `pushed`、
`container_path`，§3.2）。**外部仓地址是必含项**：缺了它，`host_path` 只是本机的目录约定、
`commit` 无从寻址，别人照基线取不到同一份代码。

`commit` 有三档取值，不可混用：40 位 sha（核对项）、`unavailable`（已知不可得，export 告警一次，
版本事实退回由 image id 兜底——父镜像继承的 editable 安装可能取不到 commit）、以及配方写法
超出 §4.2 时的**报错**（不写文件）。`info`（分支、提交时间）仅供阅读，不参与任何校验；
无法获取时同样记为 `unavailable`。

`commit` 的取数口径有两个，由 `commit_source` 区分：**缺省不写该字段 = 取自 pip 元数据**
（`pip freeze` 的 `-e <url>@<40hex>#egg=…`）；`commit_source: in-image-git` = pip 元数据没记，
回退读镜像内 `source_path` 下 `.git` 的 HEAD。回退**只填空、不改写**——已有 commit 的条目一律
以 pip 元数据为准，否则换一个取数口径就会让既有基线漂移。这条回退把一部分
`commit: unavailable` 降级成了真正的版本锚（实例：昆仑芯 v2 的 `vllm_plugin_fl`，厂商内网仓
取不到 pip VCS 元数据，但镜像内带 `.git`）。

**`install: editable-in-image` 的第二版本锚**：这类资产的 commit 来自基座，而镜像标签只是
「本组对基座内容的断言」、标签不存在时不核对（见 §4.2）。`check --image` 因此额外比对
**镜像内 `source_path`/.git 的 HEAD 与基线 `commit`**——这是一条不依赖任何标签、镜像自证的锚：
不一致即 FAIL；镜像内没有 `.git`、或配方未声明 `ARG PYTHON_BIN`（取不到目标解释器）一律 INFO，
不 FAIL——**带 `.git` 不是对厂商基座的要求**。

`supersedes`：本层固化的资产与父层/基座继承来的同名资产重名时，就近者赢（本层覆盖父层），
基线只留本层这一条，被覆盖的那条记入 `supersedes` 留痕——否则会写出重复的 YAML 键，
`yaml.safe_load` 静默取后者。

**登记式基线（上游官方镜像的例外形态）**：当镜像整体是上游官方产物（本组无 `Dockerfile.repro`、
无法 `export`）时，允许手工登记 `image.baseline.yaml` —— 字段口径与上表一致，外加两条纪律：
① 全部条目 `from: base_image`（被动继承，无本组配方层），`check` 走 INFO 分支、由 `image.id`
（及 `image.digest`）兜底；② `generated_by` 必须写明 `manual-registration` 与依据的 ADR。
commit 可用 `commit_source: in-image-git` 的口径取自镜像内 `.git`（自证锚）。实例：
[`ascend-infer-vllm/v1/image.baseline.yaml`](images/ascend-infer-vllm/v1/image.baseline.yaml)（ADR-0007）。

**但首选是不产生第二份**：基座已带着某个组件、且版本正是本组要的，就直接用它，不要再固化
一遍同一 commit。重复声明同一事实会制造「哪份生效」的歧义，`supersedes` 只是把歧义记录下来，
并没有消除它。昆仑芯 v2 原先 `COPY` 固化了与基座同 commit 的 `FlagGems`，正是按这条口径
去掉的（见 [`images/kunlun-operator-runtime/v2/lock.yaml`](images/kunlun-operator-runtime/v2/lock.yaml)
`changelog` 2026-10-11）；`supersedes` 留给确有覆盖必要的场景（本组需要的版本与基座不同）。
基座版本不是本组要的、又只需在运行实例里临时改动时，用 §2.2 的挂载覆盖，不要改镜像。

#### 怎么读镜像基线

一条资产要同时读三个互相独立、不能互推的维度：**记在哪个块**（资产形态）、**`from`**（代码从哪来）、**`install`**（怎么装的）。

| 维度 A：记在哪个块 | 判别（有仓库身份？能拿到 commit？是 python 包？） | 版本锚 |
|---|---|---|
| `base_image.provides` | 无仓库身份、拿不到 commit 的 L1 事实（CANN / XTDK / Python / torch …） | 基座 digest |
| `baked_repos.<名称>` | 有仓库身份且能拿到 commit | `commit` |
| `baked_packages[]` | 无仓库身份、是 python 包（包索引的预编译 wheel） | `version` |
| `baked_resources[]` | 两者都不是（tarball、脚本、补丁素材） | `sha256` / `pin` |
| `packages`（单数） | 指向 pip freeze 快照的指针：不是资产块，也不是核对项 | — |

| 维度 B：`from` = 代码从哪来（也是基线里分组分割线的划分依据） | 维度 C：`install` = 怎么装的 |
|---|---|
| `base_image`：厂商基座自带，被动继承，本组配方未声明、改不了 | `pip (non-editable)`：源码装入，非 editable |
| `<name>/v<N> Dockerfile.repro(COPY 构建上下文)`：本组配方带入，源码在 `src/` 下 | `pip (non-editable, USE_*=1)`：同上，带构建开关 |
| `<name>/v<N> Dockerfile.repro(构建期 git fetch)`：本组配方带入，commit 由 ARG 固定 | `editable-in-image`：editable 安装，但源码在镜像内 |
| `… + 补丁`：同上，另有 `patches`（文件名 + sha256） | `pip (index wheel, --no-deps)`：从包索引装的 wheel |

父镜像是本仓自建镜像时，继承来的条目 `from` 是父层的 `Dockerfile.repro(...)`，仍属本组配方带入。两处易混：

- `install: editable-in-image` 讲的是 pip 的安装形态，不是「代码来自宿主挂载」：源码在**镜像内**（`source_path` 是容器内绝对路径），随 image id 不可变，**是固化资产**；
  代码来自宿主、运行时挂载的是 §3.2 `mounted_repos` 的 `activation: pip-editable`（其 `host_path` 相对 `WORKSPACE_ROOT`）。
- `layer`（L1–L4）与「从哪继承」**正交**：基座里装的组件该是哪层就是哪层（昆仑芯基座自带的 `FlagGems` 是 L2、`FlagCX` 是 L3），不因来自基座就变成 L1；
  `base_image.provides` 里只放**没有仓库身份、拿不到 commit** 的那部分 L1 事实。

### 3.2 运行环境目录

```
<任意位置>/<环境名>/
  docker-compose.yml      # 启动与装配，含 x-flagrt 声明
  env.baseline.yaml       # baseline.py export 生成，不手工编辑
  .last-check.json        # 每次 check 的结果（不追踪）
```

#### `dev/images/` 下的两类运行环境目录

| 路径 | 角色 | 变更性质 |
|---|---|---|
| `dev/images/<血统>/v<N>/demo/` | 该镜像版本验收时所用配置的历史记录（第 0 版） | 永久冻结，内容不动 |
| `dev/images/<血统>/v<N>/env/v<M>/` | 现行推荐运行环境 | 每版发布后不可修改，只能新增下一版 |

- 修订号 `v<M>` 是**单调递增整数，不用语义版本号**：运行环境叠加层兑现不了主/次/补丁的兼容性
  承诺。口径参照 Debian 的 upstream-revision（上游版本不动，修订号纯整数递增）。
  `export --image … --env-current` 只接受 `<image-dir>/env/v<整数>` 形态，其余写法报错。
- **「现行是哪一版」不靠目录名表达**：写在镜像索引（[`images/README.md`](images/README.md)）与
  现行锁定声明里，镜像基线的 `image.env_current` 记录它（§3.1）。`baseline.py` 只认显式声明，
  不按目录名推断最大修订号。`demo` 与 `env/v1` 的对应关系（第 0 版到第 1 版）记在镜像索引里。
- 已发布文档里指向 `demo/` 的命令与引用保持有效、不回改：冻结之后它们永久准确。
- 以上命名约束只约束 `dev/images/` 下的发布物。**各方向自建的运行环境目录自由命名**，不受此约束
  （它们也不进 `image.env_current`——那个字段只指镜像自带的现行推荐运行环境）；实例：
  `dev/device-context/910C/env/{train,infer}`（按路径命名）、
  `dev/communication/env/910c-bucketed`（按用途命名）。

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
`baked_repos` 原样抄录、只读；分割线「环境挂载（env）」之后的 `mounted_repos.<名称>` 记录 `host_path`、
`remote`、`container_path`、`activation`、`commit`、`dirty`、`pushed`（commit 是否已在某个远端分支上），覆盖固化版本时另记 `overrides_baked`、`baked_commit`、
`baked_patches`（§2.2）。实例见 [`images/ascend-train-comm/v3/demo/env.baseline.yaml`](images/ascend-train-comm/v3/demo/env.baseline.yaml)。

**`remote`（外部仓地址）是必含项**：它是该挂载 repo 宿主 checkout 的 `origin` 远端地址，由
`export` 自动产出，不在 `x-flagrt` 里声明。挂载覆盖形态下真正运行的代码不在镜像里，而
`host_path` 只是本机的目录约定、`mounted_repos` 的键只是仓库名——别人要取到同一份代码，必须
知道这个 commit 出自哪个仓。固定取 `origin`：多远端时「哪个才是权威来源」不是工具能判的；取不到
记 `unavailable` 并告警（版本事实此时留有缺口，不足以交他人复现）。这是 §3.1 六项版本事实里的
第二项。

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
| `export` | 解析 Dockerfile 判定固化资产，以镜像标签取 commit，交叉核对后生成 `image.baseline.yaml`；`--demo <dir>` 写入 `image.demo`，`--env-current <dir>` 写入 `image.env_current`，`--image-ref` 指定镜像 ref（缺省取 `build.sh` 的 `-t`） | 抄录镜像基线的固化部分，读取 `x-flagrt.mounted_repos` 对应宿主 repo 的 HEAD、工作区状态与提交元数据，生成 `env.baseline.yaml`；`--require-clean` 时工作区不干净或 commit 未推送即拒绝导出 |
| `check` | 本地镜像 id，与基线中每项固化资产的版本锚：`baked_repos` 核对 commit 标签、`baked_packages` 核对 version 标签。基线记了锚而镜像缺对应标签即不一致（FAIL）；基线记 `commit: unavailable` 的条目报 INFO，由 image id 兜底。另核对 `image.env_current` 指向的目录仍存在且仍是运行环境目录 | 本地镜像 id；每个挂载 repo 的 HEAD 是否等于基线 commit、工作区是否干净、`origin` 远端是否仍是基线记的 `remote`；结果写 `.last-check.json` |
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
   `pip list --editable` 与 pip freeze 补全 L1 资产。读镜像内 pip 元数据的入口是
   `ARG PYTHON_BIN`（目标解释器绝对路径），缺它即报错；conda 多环境基座须给出目标 env 内
   解释器的绝对路径，不能依赖 `PATH` 上的 `python3`（实测昆仑芯基座 `PATH` 上的是 miniconda
   base 的 3.13.11，不是目标 env 的 3.10.18）。
4. `curl … ${<X>_URL}` / `COPY assets/<file>` → 固化资源。
5. 本层出现 `pip install -e` → 报错（§2.1）。
6. `pip install <pkg>==<version>`（或 `===`）+ `--index-url|-i <url>` → 固化索引发行包
   （`baked_packages`，版本锚，无 commit）。未固定版本、或未写出来源索引，均报错。

**版本锚以已构建镜像的标签为准**，Dockerfile 的 ARG 只用于交叉核对，不一致即报错：
模式 1/2 用 `io.flagrt.*.commit`（登记在 `baseline.py` 的 `COMMIT_LABEL`），模式 6 用
`io.flagrt.*.version`（登记在 `VERSION_LABEL`）。模式 6 另有第三条边——镜像内实际安装的
dist 版本（经模式 3 的 `PYTHON_BIN` 读出）须同时相等，三方不一致即报错；这条边是 commit
情形做不到的（源码固化后可能已删），所以无 commit 的资产可核对性并不弱于有 commit 的资产。

**模式 6 的第四条边：wheel 的 sha256。** 版本号本身防不住「上游以同一版本号重发不同内容」。
索引页按 PEP 503 在 `href` 上直接公布 `#sha256=`，一次小 HTTP 请求即可取，不必下载 wheel
（昆仑芯的 flagtree wheel 是 3.3GB）。`export` 把它记进 `baked_packages[].wheel`，
`check --image --online` 与索引页当前内容比对。四条规则：

1. **定位 wheel 条目按解析匹配，不按字符串匹配。** 文件名由「dist 名 + 镜像内实装版本 +
   dist-info 的 `WHEEL: Tag`」从镜像内事实完整推出，但只作观测值：匹配时解析索引页每个条目的
   wheel 文件名、按 PEP 440 做版本相等比较。两个坑都是实测到的——子串匹配会误命中（真索引页上
   子串 `0.6.1` 命中 20 条，还有 `0.6.1a1+xpu3.6` 这种近邻干扰项）；而 PEP 427 规定文件名的
   version 段应把非字母数字串转义成 `_`，该索引实际未转义，所以「自己拼文件名再比字符串」会在
   上游换构建工具时失配。同版本多 wheel 时用 `Tag` 消歧，仍不唯一即报错，不猜。
   `dist-info` 的 `WHEEL` 有多条 `Tag` 行（压缩 tag 集）时同样不猜，只告警并不记 `wheel`。
2. **`export` 尽力而为，取不到只告警不报错**——构建机可能无外网，离线建得出镜像就该导得出基线；
   此时该资产只剩 `version` 一个锚，`pin` 字段写明这一点。
3. **`check` 默认不联网，核对须显式 `--online`**：`check` 在 `dev/lib/up.sh --strict` 的启动
   路径上，默认联网会破坏离线/气隙使用并可能挂住。`--online` 下取不到索引页记 FAIL（显式要求
   核对却无法下判定，静默放过会让这个开关失去意义）；基线里没有 `wheel` 块记 INFO（向后兼容，
   旧基线不因此 FAIL）。代理不在命令里硬编码，走 curl 的标准环境变量（需绕过时 `export no_proxy='*'`）。
4. **语义边界**：它发现的是「索引当前内容 ≠ 基线当初记的值」，**不证明镜像内装的字节就是这个
   指纹**——要证明得留着整个 wheel。镜像内字节的唯一性仍由 image id 承载（§2）。
   上游合法重建同版本 wheel 时这条会 FAIL，属需人工裁定的情形，不是缺陷。

**版本锚也可以是「对基座内容的断言」**：继承资产（`from: base_image`）本来没有本层标签——它不是本组
固化的。本组若为它在 `COMMIT_LABEL` 里登记并在配方中打出该标签，标签语义就从「本层固化资产的锚」
变为「本组对基座内容的断言」（类似 lockfile 对不受控上游的 digest 断言）：`export` 把基座 pip 元数据
读出的 commit 与标签值比对，不等即报错（厂商换了基座，须核对后更新标签）；标签不存在时不核对，
基座取不到 commit 时只告警并仍记 `commit: unavailable`，绝不以标签值冒充观测值。

配方写法变化导致 export 报错时，同步扩展规则，不放宽为猜测。**接入新芯片时，先对其配方跑一遍
`export --image`，把报错逐条当成规范缺口清单**——本规范的模式 6 与 `baked_packages` 就是这么
发现的（昆仑芯 v1 的 flagtree 是 pip 索引预编译 wheel，五种模式全不覆盖）。

### 4.3 运行环境场景的行为

- **export**：工作区不干净时记录 `dirty: true` 并警告；commit 不在任何远端分支上时记录 `pushed: false`
  并警告（他人 load 时无法取得）；宿主 repo 没有 `origin` 远端时记录 `remote: unavailable` 并警告。
  验收、出结论或交给他人复现的环境加 `--require-clean`，dirty 与未推送两者任一不满足即拒绝导出。
- **check**：镜像 id 不一致、挂载 repo commit 不一致、容器内挂载来源与核对的宿主目录不一致，
  均报告为不一致；工作区不干净、commit 未推送、`origin` 远端与基线 `remote` 不一致只警告；
  固化部分不与宿主比对。
- **向后兼容**：`image.env_current` 与 `mounted_repos.<名称>.remote` 都是后加的字段，不含它们的
  旧基线文件 `check` 一律报 INFO、不 FAIL（提示重新 `export` 即可补齐）。
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

- 上面的写法假定 `PATH` 上的 `pip` 就是目标解释器的 pip。**conda 多环境基座不成立**：须把
  `pip` 换成目标 env 解释器的绝对路径（与该镜像 `Dockerfile.repro` 的 `ARG PYTHON_BIN` 同一个值），
  例如昆仑芯基座写
  `/root/miniconda/envs/python310_torch29_cuda/bin/python3 -m pip install --no-deps …`；
  用 `conda activate` 亦可，但绝对路径少一层环境依赖。实例见
  [`images/kunlun-operator-runtime/v2/demo/docker-compose.yml`](images/kunlun-operator-runtime/v2/demo/docker-compose.yml)。
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
- **`info: unavailable`**：`repo_info()` 的浅取（`git fetch --depth 1`）取不到时无法记录分支与
  提交时间（当前 v3 的 `vllm` 即如此）。**不是 GitHub 不可达**：npu1-11 上实测可达，是代理的
  HTTP/2 处理在传输中途断流（`RPC failed; curl 92 HTTP/2 stream not closed cleanly`，已传
  数十 MB 后断，可重现）；同一条 fetch 改走 HTTP/1.1 即一次成功。各机器代理行为不同，按机器
  试即可。`info` 仅供阅读、不参与任何校验，故取不到不作为缺口跟踪。
- **load 修改宿主 repo 状态**：共享机器上可能影响同一 checkout 的其他使用者，运行环境目录宜在
  个人 checkout 上使用。
- **昇腾** `ascend-operator-runtime` / `ascend-train-comm` 的 v1/v2 没有 commit 标签，不纳入本机制；这是那两个具体目录的事实，不是「凡 v1/v2 都排除」。能提供版本锚的旧版本照样可纳入——例如 `kunlun-operator-runtime/v1` 带 `io.flagrt.flag_gems.commit`，
  但它的归档镜像与配方不一致（见该目录 `lock.yaml` known_issues: `v1-image-is-docker-commit-not-build`），接入工作放在 v2。

## 7. 常用流程

```bash
# 新镜像版本：构建后生成并核对镜像基线
python3 dev/lib/baseline.py export --image dev/images/<name>/v<N> \
    --demo dev/images/<name>/v<N>/demo/ --env-current dev/images/<name>/v<N>/env/v<M>/
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

- [`../docs/运行时基座-使用与维护指南.md`](../docs/运行时基座-使用与维护指南.md)：面向维护者与使用者的流程入口（发布新镜像、接入新芯片、分支开发与固化）。
- [`images/README.md`](images/README.md) 与各 `lock.yaml`：`lock.yaml` 记录构建过程、已知问题与验证结果；
  版本事实以 `image.baseline.yaml` 为准。
- [`../VERSIONS.md`](../VERSIONS.md)：指向镜像基线与 demo 目录，不重复版本号。
- `dev/stack.lock.910c.*.yaml`：锁定声明，本机制不修改。
- `dev/compose.base.yml`：LEGACY（见该文件头部）。

<!-- changelog: 2026-10-22
依据指挥台 ADR-0008《运行环境目录版本化（env/vN）与 demo 冻结》决定 1/2/3/5/6。
- §1 两类对象：原「每个镜像附带一个 demo 运行环境目录……同时作为推荐用法」一句把变更性质相反的
  两个角色挤在一个目录里（验收记录必须永不改动，推荐用法必然随缺陷修复而变），拆为
  demo/ = 冻结的验收记录（第 0 版，不是推荐用法）与 env/v<N>/ = 现行推荐运行环境两条。
- §3.1：`image.demo` 行扩为 `image.recipe` / `image.demo` / `image.env_current` 三项，新增
  `image.env_current`（现行推荐运行环境目录，check 项）；新增「可复现性的六项版本事实」段，
  写明外部仓地址是必含项及其分工（镜像基线一项，env.baseline.yaml 五项）。
- §3.2：新增「dev/images/ 下的两类运行环境目录」小节（路径形态、单调整数修订号及理由、发布后
  不可修改只能新增、现行版本记在镜像索引与锁定声明、已发布文档指向 demo/ 不回改、各方向自建
  目录自由命名并点名两个实例）；`mounted_repos` 字段清单加 `remote`，并新增「remote（外部仓
  地址）是必含项」段。
- §4.1 / §4.3 / §7：同步 baseline.py 的现行行为 —— export --env-current、check 核对
  env_current 与 remote、两个新字段缺失时报 INFO 不 FAIL、常用流程命令加 --env-current。
-->

