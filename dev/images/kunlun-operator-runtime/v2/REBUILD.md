# 重建 kunlun-operator-runtime v2

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`，走正规 `docker build`。产物
`flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`。

与 v1 的组件集合和 pin 完全相同（flagtree `0.6.1+xpu3.6` + FlagGems `73c5aff1`）；v2 改的是
**配方形态**，使其落在 [`../../../ENV-SPEC.md`](../../../ENV-SPEC.md) §4.2 的判定规则内、可被
`baseline.py` 机器核对。三处改动及理由：

| v1 的做法 | v2 的做法 | 为什么 |
|---|---|---|
| `cd /env/FlagGems && pip install .`（父镜像内的源码目录） | `COPY src/FlagGems/` + 非 editable `pip install`（§4.2 模式 1），上下文由 `build.sh` 从宿主 checkout 按 `ARG FLAG_GEMS_COMMIT` 做 `git archive` 自备 | §2 的判定依据是「代码从哪里来」。`/env/FlagGems` 的代码来自**父镜像**——本仓既不能寻址它、也无法在 export 时核验它。改成模式 1 后，commit 锚由宿主 checkout + ARG + 镜像标签三方自动成立，不再依赖父镜像是否恰好带着未被动过的 `.git` 目录 |
| `git rev-parse HEAD \| grep -q "^${FLAG_GEMS_COMMIT}$"` 构建期手工断言 | 删除；由 §4.2 的 ARG/标签交叉核对接管 | 那条断言是在手工补做交叉核对，前提（父镜像带 `.git`）脆弱且不在本组控制下 |
| 静态自检 `import torch, flag_gems, triton` | 不 import `flag_gems`，改断言 dist 版本 + `from triton._C.libtriton import xpu` + `flag_gems` 非 editable | `import flag_gems` 在无设备容器里必抛 `RuntimeError: No device were detected on your machine !`，而 `docker build` 不能传 `--device`——v1 那个自检层在普通构建中**不可能通过**（这也是 v1 归档镜像实为 `docker commit` 产物的证据之一，见 [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: v1-image-is-docker-commit-not-build`） |

## 构建前置

| 项 | 来源 |
|---|---|
| 基座镜像 | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（公开 harbor，本机已存在，image id `ea6d797a7d44`） |
| flagtree wheel | 构建期从 `resource.flagos.net/repository/flagos-pypi-hosted` pip 装（~3.3GB，见踩坑 1） |
| FlagGems 源码 | 宿主仓根 `FlagGems/` @ `73c5aff1`，由 `build.sh` 做 `git archive`（18.9MB）；该 commit 已在 `origin/master`、`origin/dev-1.0` 上，`build.sh` 会检查，未推送即拒绝构建 |
| 磁盘 | `/var/lib/docker` 需 ~15GB 余量（本层增量约 11GB，绝大部分是 flagtree wheel 解包后的体积） |

无需回收任何离线资产——每一层都能从公开可达源头联网安装。

## 踩坑

1. **flagtree wheel 经代理下载易 `IncompleteRead` 中断**（v1 已记录，v2 复核仍然适用）。该 wheel
   ~3.3GB，经 `HTTP_PROXY`/`ALL_PROXY` 转发时在 500MB~2.7GB 处不定点中断。`build.sh` 用
   `DOCKER_BUILDKIT=0 --network=host` 且不透传任何 `*_proxy` build-arg，构建期容器内无代理
   环境变量，即为直连；本轮实测一次成功。
   **⚠️ 若在此坑上重试 `pip install --force-reinstall`**：重试期间旧文件已卸载、新文件未下完，
   会留下「`pip show` 显示已安装但 `triton/_C/` 目录缺失」的损坏中间态——必须重试到成功，
   不能满足于 `pip show` 不报错。`Dockerfile.repro` 的静态自检里
   `from triton._C.libtriton import xpu` 就是拦这个的。
2. **两层 pip 安装都必须 `--no-deps`**。不加会让 pip 解析 `torch>=2.2.0` 在 x86_64+cuda 平台的
   可选依赖 `triton==3.5.0`（PyPI 通用 triton，与 flagtree 是两个 pip 发行名但共享 `triton/`
   顶层导入命名空间），**物理覆盖** flagtree 提供的 `triton/` 目录，`triton._C.libtriton.so`
   变成不含 xpu 扩展的版本，flagtree 整体不可用。详见
   [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: flaggems-kunlunxin-extras-missing`
   （含「`pip install .[kunlunxin]` 在该 commit 上静默退化」这个坑）。
3. **`ARG PYTHON_BIN` 必须是目标 conda env 解释器的绝对路径**，不能依赖 `PATH` 上的 `python3`：
   本基座 `PATH` 上的是 miniconda base 的 **3.13.11**，目标 env 的是 **3.10.18**。v2 用
   `/root/miniconda/envs/python310_torch29_cuda/bin/python3` 直达目标 env，不用 `conda activate`
   （少一层环境依赖）；`baseline.py` 读镜像内 pip 元数据时也以这个 ARG 为入口（§4.2 模式 3）。
4. **FlagGems 的构建依赖走默认索引**。`pip install --no-deps .` 仍会为源码包建构建环境
   （`Installing build dependencies`），需要默认索引可达；本轮实测正常。若目标机器不通，
   加 `--no-build-isolation`（基座已满足 setuptools/pybind11 等构建依赖）。

## 验证

### 静态（无设备，构建期已内置）

`Dockerfile.repro` 最后一个 RUN 即静态自检，断言 flagtree/flag_gems/triton 三件套版本、
`from triton._C.libtriton import xpu`、以及 `flag_gems` 为非 editable 安装。构建通过即这些成立。

单独复跑（容器内）：

```
/root/miniconda/envs/python310_torch29_cuda/bin/python3 \
  /workspace/dev/images/kunlun-operator-runtime/v2/assets/verify_kunlun_runtime.py --static
```

### 带卡（自判定，单入口）

见 [`README.md`](README.md)「验证」一节。先 `docker ps` + `xpu-smi` 看空闲卡，再
`dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/demo --strict`，然后在容器内跑
`assets/verify_kunlun_runtime.py`（退出码 0 = PASS）。证据截选在
[`assets/evidence/`](assets/evidence/)。

### pip freeze 对比

`build.sh` 构建成功后自动把 `pip freeze` 写到
`assets/provenance/repro-pipfreeze-v2.txt`，可与 v1 回收的基座基线
[`../v1/assets/provenance/GOLDEN-pipfreeze-base-official.txt`](../v1/assets/provenance/GOLDEN-pipfreeze-base-official.txt)
对比。预期差异仅两处：新增 `flagtree==0.6.1+xpu3.6`；`flag_gems` 从
`-e git+...@73c5aff1...#egg=flag_gems` 变为非 editable 安装（同 commit）。

## GPU / 机器纪律

共享机，起容器前先 `docker ps`；**绝不停止、重启或 exec 进入任何非本人创建的容器**；用完
`docker compose -f <demo>/docker-compose.yml down` 收干净。

P800 的设备模型与昇腾不同，**不要照抄昇腾「驱动对带卡容器独占」的话术**——完整结论与实测证据
写在 [`demo/docker-compose.yml`](demo/docker-compose.yml) 文件头。要点：多容器可并存、不需要
`--privileged`、`/dev/xpuctrl` 必需、不需要挂宿主驱动目录、设备按授予顺序从 0 密集重编号。

用卡选择（运行纪律，不是版本事实，故不写进 compose 默认值）：

- 先 `xpu-smi` 确认目标卡 0% 利用率、0MiB 显存占用，再用 `CUDA_VISIBLE_DEVICES` 指定。
- 多卡集合通信优先选**同一 NUMA 组内连续且空闲**的卡（本机 NUMA0 = 卡 0-3、NUMA1 = 卡 4-7）。
- **卡 1 历史上反复被共享机其他租户占用**，训练场景应优先避开，即便临跑前显示瞬时空闲；
  反面证据（卡 1,2 组合下集合通信初始化挂死）见
  [`../v1/assets/provenance/train-smoke-flagcx-initcomm-hang.txt`](../v1/assets/provenance/train-smoke-flagcx-initcomm-hang.txt)。
