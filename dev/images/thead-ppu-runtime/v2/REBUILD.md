# thead-ppu-runtime v2 · 重建与验证

> 单入口：`bash dev/images/thead-ppu-runtime/v2/build.sh`（ENV-SPEC 四硬规则之「单入口」）
> 本文每条判据都在 **2026-10-27 真机**跑过，读数照抄，不写"预期"。

## 0 前置条件

| 项 | 要求 | 本机（PTG_0 / `ai-server`）现状 |
|---|---|---|
| 基座镜像 | `flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base` | ✅ 本地已有（40.3GB，id `f5366712d6a0`） |
| 资产 | `flagtree-0.7.0+ppu.gite223f42c-cp312-cp312-linux_x86_64.whl`（358MB） | ✅ `/bmcp_lvm_fs/A/data/flagtree-src/dist/` |
| docker | 可用 `build` | ✅ |
| **磁盘** | `/`（`/var/lib/docker`）**≥ 2.5G** 可用 | ⚠️ 2026-10-27 实测仅 **~3.1G**（99%）；`build.sh` 会守水位，不足 `exit 3` |
| 跑 `check` 的解释器 | ≥3.7（`baseline.py` 用了 `capture_output`）+ `pyyaml` | 宿主默认 `python3.6` **太旧**；用 `python3.11` + `python3.11 -m pip install --user pyyaml`（已做，用户级、不进仓） |

## 1 构建

```bash
bash dev/images/thead-ppu-runtime/v2/build.sh
# 覆盖项：REF=<镜像ref> · WHEEL_DIR=<wheel 目录> · MIN_FREE_GB=<水位阈值> · CTX=<上下文目录>
```

脚本做四件事（都是为了不把共享机写爆、且可复现）：

1. **水位守卫**（默认 ≥2.5G），不足直接 `exit 3`；
2. **资产自证**：打印 wheel 体量与 sha256，并用 `zipfile` 确认顶层含 `triton/` 与 `flagtree/`；
3. **在数据盘组装上下文** `$CTX/src/flagtree/<wheel>`（复制 358MB，不在 root 盘做 context）；
4. 构建后打印镜像 id / 体量 / 余量变化 / wheel sha256。

实测（10-27 10:05）：`41.7GB` 新镜像，`/` 可用 4G → 4G（`df -BG` 取整，基座层共享故净增被摊薄）。

## 2 验证判据（三条，缺一不算过）

> ⚠️ 必须「**全新容器 + 零手工干预**」。在有历史的开发容器里验证是**假证据**——
> 那个容器里的 triton 正是我们手工 `docker cp` 进去的。

### 判据 1 · 资产在位

```bash
docker run --rm <v2-ref> python3 -c "import triton, flagtree; print(triton.__version__, triton.__file__)"
```
**实测**：`3.6.0 /usr/local/lib/python3.12/site-packages/triton/__init__.py`

### 判据 2 · 真编 kernel（不是 import 就算过）

```bash
docker exec -e CUDA_VISIBLE_DEVICES=0 <容器> bash -lc \
  "cd /workspace/runtime-team/dev/device-context/prototype && \
   python3 /workspace/runtime-team/dev/device-context/PPU/probes/triton_smoke.py"
```
**实测**：`TRITON_PPU_SMOKE_PASS` · 1.50 s · 最大误差 `0.000e+00` · 边界 n=3000 一致

### 判据 3 · 服务化端到端（原来的失败点）

```bash
docker exec -e DC_BACKEND=ppu -e DEV=0 -e STOP_AFTER=1 -e PORT=8123 -e SERVE_FORM=embed \
  -e MODEL=/workspace/models/Qwen3-Embedding-0.6B -e CUDA_VISIBLE_DEVICES=0 \
  -e DC_OUT_DIR=/workspace/probes_ppu/v2_serve_1027 <容器> \
  bash -lc "cd /workspace/runtime-team/dev/device-context/prototype && bash scripts/serve_standard.sh"
```
**实测**：`[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)`

### 判据 4（本方向补充）· 现役容器切到 v2 后判据不回退

```bash
# hliu553-dc-dev.run.sh 的 IMG 已指向 v2，重建容器后：
cd dev/device-context/PPU/probes && RUNID=<新> bash run_acceptance_ppu.sh --only s03,s06
```
**实测**：`s03 PASS` · `s06 CONTRACT_INVARIANTS_PASS 4/4`，`rc=0`

## 3 踩坑记录（每条都是 10-27 实测，能省下一次白跑）

| # | 坑 | 现象 | 处置 |
|---|---|---|---|
| 1 | **`serve_standard.sh` 内部 `exec` 重定向** | 外层 `> log 2>&1` 得到 **0 字节**，误判"脚本没跑" | 传 `-e DC_OUT_DIR=<挂载数据盘目录>`，或 `docker exec cat /tmp/dc_serve/serve_standard.log` |
| 2 | **`STOP_AFTER=1` 的 `Killed` 是主动停机** | 日志出现 `Killed nohup vllm serve` ⇒ 误判崩溃 | 看 verdict 行 `SERVE_STANDARD_PASS (ready=1 smoke=1)` |
| 3 | **wheel 被改名 ⇒ pip 拒绝** | `Invalid wheel filename (wrong number of parts)` | **保留原 wheel 文件名**（PEP 427） |
| 4 | **`baseline.py` 只认 `src/<Repo>`（恰两段、目录）** | `src/flagtree/x.whl`、`src/flagtree-src/dist/x.whl` 都被判「不符合 src/<Repo>/ 模式」 | 用 `COPY src/flagtree/ <dir>/`（目录形态） |
| 5 | **`baseline.py` 不认「wheel 单文件安装」** | `pip install 的来源未登记为固化 repo` | 本代先按 **ADR-0007 登记式基线**入档 + 把工具缺口提总组（[`../../TODO.md`](../../TODO.md)） |
| 6 | **宿主 `python3.6` 跑不了 `baseline.py`** | `TypeError: __init__() got an unexpected keyword argument 'capture_output'` | 用 `python3.11`；它缺 pyyaml ⇒ `python3.11 -m pip install --user pyyaml` |
| 7 | **跨属主硬链接被拒** | `cp -al` 报 `Operation not permitted`（`fs.protected_hardlinks=1`） | 在数据盘用 `cp -a` 复制（358MB，数据盘余量充足） |
| 8 | **`--network host` 下主机名不解析 / 镜像写死 `NCCL_SOCKET_IFNAME=eth0`** | torchrun 静默挂 300 s；真实 `all_reduce` 报 `no socket interface found` | `--add-host` + `-e NCCL_SOCKET_IFNAME=bond0`（env/v1 已内置） |

## 4 回滚

`env/v1/docker-compose.yml` 把 `image:` 改回基座 ref 即可（容器参数无需改）；
或直接用 v1：`dev/lib/up.sh dev/images/thead-ppu-runtime/v1/demo --strict`。
⚠️ 回滚后**服务化会重新失败**（缺 triton），需手工 `docker cp` 补 —— 这正是 v2 要消除的动作。
