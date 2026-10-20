# 重建 kunlun-operator-runtime v2

## 方式

`build.sh <构建上下文目录>` 驱动 `Dockerfile.repro`，走正规 `docker build`。产物
`flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`。

与 v1 的组件集合和 pin 完全相同（flagtree `0.6.1+xpu3.6` + FlagGems `73c5aff1`）；v2 改的是
**配方形态**。差别只剩两处：

| v1 的做法 | v2 的做法 | 为什么 |
|---|---|---|
| `cd /env/FlagGems && pip install .`——从**父镜像自带的源码目录重新安装一遍** | **连装都不装**：直接用父基座已装好的那份（`/env/FlagGems`，editable，源码在镜像内，按 [`../../../ENV-SPEC.md`](../../../ENV-SPEC.md) §2.1 即固化资产 `install: editable-in-image`） | 组内决策口径：不变更基座已有的资源，非必要不额外追加，保持镜像最简最完备。v1 那条 `pip install .` 既多出一层，又依赖 `/env/FlagGems` 这个**可变目录在构建期存在且未被动过**；基座本来就装好了同一个 commit，重装一遍不增加任何事实。改完后本层相对基座**只新增 flagtree 一个包**，重建也不再依赖宿主任何 checkout。代价（版本事实改由基座 pip 元数据承载、标签降格为「对基座内容的断言」）见 [`lock.yaml`](lock.yaml) `known_issues: flaggems-version-fact-comes-from-base` |
| 静态自检 `import torch, flag_gems, triton` | 不 import `flag_gems`，改断言 dist 版本 + `from triton._C.libtriton import xpu` + `flag_gems` **是**基座那份 editable-in-image 安装（`url == file:///env/FlagGems`） | `import flag_gems` 在无设备容器里必抛 `RuntimeError: No device were detected on your machine !`，而 `docker build` 不能传 `--device`——v1 那个自检层在普通构建中**不可能通过**（这也是 v1 归档镜像实为 `docker commit` 产物的证据之一，见 [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: v1-image-is-docker-commit-not-build`）。`flag_gems` 这一项的期望值与通常相反——§2.1 禁止的是**本组构建层**做 editable 安装，父镜像继承来的 editable 安装源码在镜像内即为固化资产；这条断言确认的是「用的确实是基座那份，没有被覆盖」 |

> v1 还有一条 `git rev-parse HEAD | grep -q "^${FLAG_GEMS_COMMIT}$"` 的构建期手工断言，
> v2 一并删除：它的前提（父镜像带着 `.git`）脆弱且不在本组控制下。同一件事现由
> `baseline.py export` 核对「基座 pip 元数据读出的 commit == 镜像标签
> `io.flagrt.flag_gems.commit`」完成。

## 构建前置

| 项 | 来源 |
|---|---|
| 基座镜像 | `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（公开 harbor，本机已存在，image id `ea6d797a7d44`） |
| flagtree wheel | 构建期从 `resource.flagos.net/repository/flagos-pypi-hosted` pip 装（~3.3GB，见踩坑 1） |
| FlagGems 源码 | **不需要**。本层不带入 FlagGems，用基座自带的 `/env/FlagGems`；重建不依赖宿主任何 checkout，也不要求任何 commit 已推送 |
| 磁盘 | 基座已在本机时 `/var/lib/docker` 留 **~12GB**：本层只加两个 layer，实测（2026-10-11 `docker history`）flagtree 层 **7.38GB**（容器内 `site-packages/triton` 6.9GB）、`rm -rf /root/.cache/pip` 层 0B，其余 ARG/LABEL/SHELL/WORKDIR 全 0B；构建期另有 ~3.3GB wheel 下载为瞬时占用。基座不在本机时另需为基座本体留出空间（`docker images` 显示 105GB） |

构建上下文只有一个文件（`build.sh` 从本目录拷进去的 `Dockerfile`）。
无需回收任何离线资产——每一层都能从公开可达源头联网安装。

## 踩坑

1. **flagtree wheel 经代理下载易 `IncompleteRead` 中断**（v1 已记录，v2 复核仍然适用）。该 wheel
   ~3.3GB，经 `HTTP_PROXY`/`ALL_PROXY` 转发时在 500MB~2.7GB 处不定点中断。`build.sh` 用
   `DOCKER_BUILDKIT=0 --network=host` 且不透传任何 `*_proxy` build-arg，构建期容器内无代理
   环境变量，即为直连；2026-10-11 首轮构建实测一次成功（同日配方简化后的重建命中 docker
   缓存，未重新下载）。
   **⚠️ 若在此坑上重试 `pip install --force-reinstall`**：重试期间旧文件已卸载、新文件未下完，
   会留下「`pip show` 显示已安装但 `triton/_C/` 目录缺失」的损坏中间态——必须重试到成功，
   不能满足于 `pip show` 不报错。`Dockerfile.repro` 的静态自检里
   `from triton._C.libtriton import xpu` 就是拦这个的。
2. **flagtree 的 pip 安装必须 `--no-deps`**（本层只有这一条 pip 安装；v1 时代是「两层都必须」，
   因为 v1 还重装了一遍 FlagGems）。不加会让 pip 解析 `torch>=2.2.0` 在 x86_64+cuda 平台的
   可选依赖 `triton==3.5.0`（PyPI 通用 triton，与 flagtree 是两个 pip 发行名但共享 `triton/`
   顶层导入命名空间），**物理覆盖** flagtree 提供的 `triton/` 目录，`triton._C.libtriton.so`
   变成不含 xpu 扩展的版本，flagtree 整体不可用。详见
   [`../v1/lock.yaml`](../v1/lock.yaml) `known_issues: flaggems-kunlunxin-extras-missing`
   （含「`pip install .[kunlunxin]` 在该 commit 上静默退化」这个坑）。
3. **`ARG PYTHON_BIN` 必须是目标 conda env 解释器的绝对路径**，不能依赖 `PATH` 上的 `python3`：
   本基座 `PATH` 上的是 miniconda base 的 **3.13.11**，目标 env 的是 **3.10.18**。v2 用
   `/root/miniconda/envs/python310_torch29_cuda/bin/python3` 直达目标 env，不用 `conda activate`
   （少一层环境依赖）；`baseline.py` 读镜像内 pip 元数据时也以这个 ARG 为入口（§4.2 模式 3）。

> 原「踩坑 4：FlagGems 的构建依赖走默认索引」（`pip install --no-deps .` 仍会为源码包建构建
> 环境，需要默认索引可达）**已不适用**——本层不再安装 FlagGems，没有源码包要建构建环境。
> 若将来有版本重新把 FlagGems 固化进镜像，这条坑会回来，加 `--no-build-isolation` 规避
> （基座已满足 setuptools/pybind11 等构建依赖）。

## 验证

### 静态（无设备，构建期已内置）

`Dockerfile.repro` 倒数第二个 RUN 即静态自检，断言 flagtree/flag_gems/triton 三件套版本、
`from triton._C.libtriton import xpu`、以及 `flag_gems` **是**基座那份 editable-in-image 安装
（`direct_url.json` 的 `dir_info.editable == true` 且 `url == file:///env/FlagGems`）。
构建通过即这些成立；2026-10-11 实测输出：

```
static check OK: 2.9.0+cu129 0.6.1+xpu3.6 4.2.1rc0 3.6.0 file:///env/FlagGems
```

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
对比。配方简化后**预期差异只剩一行**：新增 `flagtree==0.6.1+xpu3.6`；`flag_gems` 保持基座
原本的 `-e git+https://github.com/flagos-ai/FlagGems.git@73c5aff1…#egg=flag_gems` 形态。
2026-10-11 实测即如此（`diff` 仅 `> flagtree==0.6.1+xpu3.6` 一行，270 行 → 271 行）——
这正是「本层相对基座只新增一个包」的机器可核对证据。

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
