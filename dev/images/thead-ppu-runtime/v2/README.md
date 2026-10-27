# thead-ppu-runtime **v2** · 派生镜像（官方基座 + 固化 triton/flagtree 的 flagtree wheel）

> 版本轴（ADR-0008）：本目录 = **镜像版本 v2**；其下 `env/v1/` = 该镜像的**第一版运行环境**。
> 上一版 v1 = 官方基座的**登记式基线**（未改动任何字节），见 [`../v1/`](../v1/README.md)。

## 一、这一版解决什么（一句话）

官方基座**不含 `triton`**，而 vLLM 起服务需要它 ⇒ 服务化腿必挂；此前靠**手工 `docker cp`** 从共享目录补、
**容器重建即丢**。v2 用一个 **flagtree wheel** 把 `triton + flagtree` **pip 固化进镜像**，
做到「拉起镜像即可服务化」。

**证据（2026-10-27 真机，均为全新容器 + 零手工干预）**

| 检查 | 基座 v1 | **v2** |
|---|---|---|
| `import triton` | `ModuleNotFoundError` | `3.6.0` |
| `importlib.util.find_spec("flash_attn") / ("triton")` | `True / **False**` ⇒ 起服务必挂 | `True / **True**` |
| 真编 kernel（非空转） | 无法进行 | **`TRITON_PPU_SMOKE_PASS`**（1.50 s，最大误差 `0.000e+00`，边界 n=3000 一致） |
| 服务化端到端 | 失败 | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`** |
| 本方向运行容器直接拉起后跑判据 | 需先手工补 triton | **`s03 PASS` · `s06 CONTRACT_INVARIANTS_PASS 4/4`**（`rc=0`） |

## 二、版本锚

| 层 | 事实 |
|---|---|
| **本镜像（L1 派生）** | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`<br>image id `sha256:1ba7ead15378160ba9aa1dafcfb77686de6b16732590ffab062bab19c60c32b8` · **41.7GB**（基座 40.3GB + 派生层 ≈1.4GB） |
| 基座（被动继承，本组不可重建） | `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base`（id `sha256:f5366712d6a0…`，Harbor digest `sha256:50a906e6ed9d…`） |
| 派生层内容 | **一个 wheel**：`flagtree-0.7.0+ppu.gite223f42c-cp312-cp312-linux_x86_64.whl`<br>sha256 `b57daff7393972d9c5ff2487930caf2416d0cbd79d8f50b0fd6a5ca56c5378b4`（358MB） |
| 该 wheel 为何够用 | 实测 389 条目，顶层恰为 `triton/`、`flagtree/`、`flagtree-…dist-info/` ⇒ **同时提供 triton 与 flagtree**（与此前手工复制的三件同源） |
| 资产来源 | `/bmcp_lvm_fs/A/data/flagtree-src/dist/`（共享盘既有产物，**非网络下载**） |
| 运行时层 | **不在镜像内** —— 与 v1 同口径，仓内 `dev/device-context/prototype/runtime/`（`runtime ppu 0.1.0`）经 `WORKSPACE_ROOT` 挂载使用 |

## 三、与 v1 的差异清单（**只做加法**）

| # | 变化 | 说明 |
|---|---|---|
| 1 | `+ pip install flagtree wheel` | 一条命令同时装出 `triton 3.6.0` 与 `flagtree 0.7.0+ppu` |
| 2 | 三个 `LABEL org.flagrt.*` | 便于 `docker inspect` 识别派生层与基座来源 |

**未改动**：基座任何包版本、`NCCL_SOCKET_IFNAME` 等运行期参数（按 `env/v1/` 表达）、本组运行时（仍挂载）。

## 四、运行环境与用法

`env/v1/`（`docker-compose.yml` + `env.baseline.yaml`）：容器参数**逐项沿用** `../v1/demo/`，只换镜像
⇒ 两份环境可做 A/B 对照。两条 PPU 启动坑（`--add-host` 主机名、`NCCL_SOCKET_IFNAME=bond0`）**已内置**。

```bash
dev/lib/up.sh dev/images/thead-ppu-runtime/v2/env/v1 --strict      # 起容器（check 已实测 PASS）
```

本方向现役容器已切到 v2：`/bmcp_lvm_fs/hliu553/hliu553-dc-dev.run.sh` 的 `IMG=` 已指向本镜像
（原件备份 `…run.sh.bak-20261027`），**不再需要任何手工 `docker cp`**。

## 五、已知限制（如实登记）

1. **`pip show triton` 查不到**：元数据由 `flagtree-…dist-info` 提供（覆盖 triton/ 与 flagtree/ 两个目录），
   故 `pip show flagtree` 可见、`pip show triton` 不可见；但 `import triton` 与
   `find_spec("triton")` 均正常 ⇒ **排查时别用 `pip show triton`/`pip list` 判断**。
2. **本镜像未 push 到 registry**（与组内口径一致，`RepoDigests` 为空）⇒ 跨机分发需 `docker save`；
   ⚠️ 本仓已知问题：**巨型镜像 `docker save` 会静默导出空 tar**（[`../../TODO.md`](../../TODO.md) #21），
   分发前须按该条三项校验（tar 体量 / blob 数 / `tar -tf` 可读）。
3. **基线为登记式（ADR-0007）而非 `export` 生成**：`dev/lib/baseline.py` §4.2 暂不支持「wheel 单文件固化」
   形态（详见 [`image.baseline.yaml`](image.baseline.yaml) `gaps`）⇒ 已作为待办提总组。
   ⚠️ 两份基线已用 `check` 实测 **PASS（rc=0）**，可被 `up.sh --strict` 消费。
4. **派生层不修平台缺陷**：PPU 退出期弱引用悬垂仍由 device-context 方向在自己层里规避
   （不依赖弱引用语义），**不得**因为换了镜像就认为平台已修。

## 六、复现与验证

见 [`REBUILD.md`](REBUILD.md)（磁盘水位守卫 · 三条验证判据 · 5 条踩坑记录 · 2026-10-27 实测读数）。
