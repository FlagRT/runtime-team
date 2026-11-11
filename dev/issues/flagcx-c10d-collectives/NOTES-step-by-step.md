# 逐步操作：从零复现 FlagCX c10d collective 缺陷（2 卡）

面向第一次在本机操作的人。全部命令在**公共仓根**（npu1-11：`/home/xliu969/runtime-team`）执行，
按序复制即可，不需要额外手工准备。结论口径见 [`README.md`](README.md)「修复验收」。

## 0. 先确认卡没被占用

```bash
docker ps
```

**昇腾驱动对带卡容器独占：本机同时只能有一个持有 `/dev/davinci*` 的容器。**
输出里若有带卡容器（`docker inspect <name> --format '{{json .HostConfig.Devices}}' | grep davinci` 可确认），
且不是自己启动的，**停手**，与机器上的其他人约时间，不要动别人的容器。

## 1. 启动运行环境

```bash
FLAGRT_MODELS_DIR=/home/xliu969/models dev/lib/up.sh dev/images/ascend-train-comm/v3/demo --strict
```

- `FLAGRT_MODELS_DIR` **必须给**：demo 把它挂为容器内 `/workspace/models:ro`，缺省值 `<仓根>/models`
  在本机是指向宿主个人目录的软链，容器内无法解析，不给值容器起不来。本缺陷的复现不读模型，
  但挂载点仍需有效，指向任一实体目录即可。
- `--strict`：启动前按镜像基线核对 image id，不一致即拒绝启动（用于出结论的运行）。
  预期输出含 `[PASS] image: flagrt/ascend-operator-runtime-comm:...`。
- `up.sh` 会自动导出 compose 需要的 `WORKSPACE_ROOT`，这一步不用自己设。

```bash
docker ps   # 应看到 flagrt-v3-demo-910c，状态 Up
```

## 2. 跑复现（2 卡，20 轮）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagrt-v3-demo-910c \
  torchrun --nproc_per_node=2 dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py --iters 20
echo "exit=$?"
```

- `ASCEND_RT_VISIBLE_DEVICES=0,1`：用 0、1 两张卡。一个容器内的多项检查靠这个变量分区；
  若这两张卡另有用途，改成两个空闲卡号即可（`--nproc_per_node` 与卡数保持一致）。
- `--iters 20` 别减小：未同步组的失败是偶发的，单次可能恰好通过，20 轮才稳定给出同一判定。
- 约 1 分钟。过程中会打印各失败组的样例行（`[rank1] iter1 broadcast+nosync: got=... -> FAIL`），属正常输出。

## 3. 读判定

往上翻到 `=== VERDICT` 段（判定段在 torchrun 的报错之前打印），对照八行：

```
=== VERDICT (all ranks, iters=20) ===
  broadcast+nosync                 FAIL
  broadcast+sync                   PASS
  all_reduce+nosync                FAIL
  all_reduce+sync                  PASS
  all_gather_list+nosync           FAIL
  all_gather_list+sync             FAIL
  all_gather_into_tensor+nosync    FAIL
  all_gather_into_tensor+sync      PASS
overall: FAIL
exit=1
```

- `overall: FAIL` + `exit=1`，且八行与上面一致 → **两类缺陷均复现**（当前预期）：
  `+sync` 行 PASS 的三项是类别 ①（异步返回），`all_gather_list+sync` 的 FAIL 是类别 ②（结果错位）。
- `overall: PASS` + `exit=0` → 八组全通过，即**修复验收通过**（另需原生 DDP 双卡脚本，见 README）。
- 判定段之后 torchrun 必然打印 `ChildFailedError ... exitcode: 1`，那是退出码 1 的正常表现，不是另一个错误。
- 若判定段都没打印出来，才是脚本本身没跑到判定，看前面的报错。

## 4. 收尾：释放卡

```bash
WORKSPACE_ROOT=$PWD docker compose -f dev/images/ascend-train-comm/v3/demo/docker-compose.yml down
docker ps
```

**`WORKSPACE_ROOT=$PWD` 必须给**：`down` 绕开了 `up.sh`，compose 文件里的 `${WORKSPACE_ROOT}` 无人赋值
会报 variable is not set 并解析失败。`docker ps` 确认自己的容器已消失，卡交还给机器上其他人。

## 会踩到的坑清单

| 现象 | 原因与处理 |
|---|---|
| 容器启动报 EBUSY / 设备忙 | 本机已有带卡容器，昇腾驱动独占。`docker ps` 查明归属，不要强停别人的容器 |
| compose 报 `WORKSPACE_ROOT` 未设置 | 只在绕开 `up.sh` 时发生（例如 `down`），按第 4 步加前缀 |
| 容器起不来、挂载 `/workspace/models` 失败 | 没给 `FLAGRT_MODELS_DIR`，缺省路径是软链 |
| 判定段之后一大段 `ChildFailedError` 栈 | 退出码 1 的正常表现，判定以 `overall:` 行为准 |
| 退出期 `free(): invalid pointer` / SIGABRT | FlagCX 已知的退出期问题（low priority）。脚本在打印判定后直接 `os._exit`，不会把判定覆盖成 torchrun 失败 |
| `--iters` 调小后未同步组偶尔 PASS | 失败是偶发的，必须 20 轮；不要据单次结果改结论 |
| 想顺带验证另一个缺陷 | 同一个容器内接着跑，用 `ASCEND_RT_VISIBLE_DEVICES` 分区即可，不要再起第二个带卡容器（见 [`../shmem-missing/NOTES-step-by-step.md`](../shmem-missing/NOTES-step-by-step.md)） |

---

# 附：修复验收臂（挂载覆盖）的逐步操作（2026-10-16，npu1-27）

> 上面第 0–4 节是**复现臂**（镜像固化版，用镜像自带 demo）。本节是**修复验收臂**：
> 同一个镜像，额外把宿主 checkout 的 FlagCX 以 `pip-editable` 挂载覆盖固化版本。
> 两臂的判据、装置与结果见 [`README.md`](README.md)「修复验收」。

## F0. 两臂的关系（先看清，避免误读结论）

|  | 复现臂（第 0–4 节） | 修复验收臂（本节） |
|---|---|---|
| 运行环境 | 镜像 demo [`dev/images/ascend-train-comm/v3/demo/`](../../images/ascend-train-comm/v3/demo/docker-compose.yml) | device-context 方向的 [`dev/device-context/910C/env/train/`](../../device-context/910C/env/train/docker-compose.yml) |
| 容器内 `import flagcx` 解析到 | 镜像内 `site-packages/flagcx`（commit `4e0e0cbc`） | **宿主 checkout** `/workspace/FlagCX/plugin/torch/flagcx` |
| 预期判定 | `overall: FAIL`（rc=1，缺陷复现） | `overall: PASS`（rc=0，修复到货） |

**两臂相差的只有一个变量**：加载哪一份 FlagCX。⚠️ 所以**必须让脚本把实际加载路径打出来**
（`run_910c_p9_acceptance.sh` 的 `[0] 自证` 段会打印 `flagcx 包: <路径>` 与 `.so` sha256 前 16 位）——
否则很容易出现"换了一份没生效、却以为验证过了"的假 PASS。

## F1. 前置（四项，缺哪项都会以别的形式失败）

```bash
# ① 仓根（要挂进容器，故用绝对路径）
cd <公共仓根>

# ② 挂载资产：FlagCX 的宿主 checkout，位置固定为 <仓根>/FlagCX（ENV-SPEC §3.2 默认表）
git clone git@github.com:FlagRT/FlagCX.git FlagCX
git -C FlagCX checkout b576fa6a          # 已推送的干净 commit；--require-clean 依赖这一点
git -C FlagCX status --porcelain          # 必须为空

# ③ 卡：本机两个**实测空闲**的 die（不要在 npu-smi 进程表上肉眼挑）
bash dev/device-context/910C/probes/probe_all_dies_910c_20261016.sh
export DC_DIE_A=5 DC_DIE_B=0              # 用上一步输出的 FREE 列表

# ④ 模型目录（挂载点须有效，本缺陷不读模型）
mkdir -p <仓根>/models
```

## F2. 起容器

按规范（2026-10-16 起在 npu1-27 上**已可原样执行**——该机当日以用户级方式补齐了
compose v2.29.7 与 pyyaml 6.0.2，**未用 sudo**；步骤见
[`../../device-context/910C/env/README.md`](../../device-context/910C/env/README.md) §2）：

```bash
python3 dev/lib/baseline.py export dev/device-context/910C/env/train --require-clean
docker ps                                 # 确认本机无其他带卡容器
export DC_DIE_A=<die> DC_DIE_B=<die>       # 见 F1 ③
FLAGRT_MODELS_DIR=<实体模型目录> dev/lib/up.sh dev/device-context/910C/env/train
#   ⚠️ 不加 --strict：本机镜像 id 与基线记录的制造机 id 不同 ⇒ check 恒 FAIL、--strict 恒拒绝启动
#      （image_list.md 已登记"跨机 image id 不同"；这条后果 2026-10-16 实测钉死）。注明"未满足 --strict"。
```

> 当日实测（机制路径）：`export` 双腿 rc=0 · `up.sh` rc=0 起容器 · 挂载覆盖安装生效 ·
> **容器内两项验收 `overall: PASS`** · `down` rc=0 无残留。证据
> [`../../device-context/910C/probes/baseline_mechanism_20261016_out/SUMMARY.md`](../../device-context/910C/probes/baseline_mechanism_20261016_out/SUMMARY.md)。

另一种（前置未就绪时的退路）是用**与 compose 逐项等价**的 `docker run` 复刻其参数：

```bash
docker run -dit --name flagos-dc-train-910c-dev \
  --network host --ipc host --shm-size 512g --cap-add SYS_PTRACE \
  -v <仓根>:/workspace -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v <仓根>/models:/workspace/models:ro \
  --device /dev/davinci$DC_DIE_A --device /dev/davinci$DC_DIE_B \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e HCCL_WHITELIST_DISABLE=1 -e PYTHONDONTWRITEBYTECODE=1 -e GIT_OPTIONAL_LOCKS=0 \
  -e WORKSPACE_ROOT=<仓根> \
  flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 \
  bash -c 'USE_ASCEND=1 python3 -m pip install --no-deps --no-build-isolation -e /workspace/FlagCX && exec sleep infinity'
```

`command` 里那步就是 compose 声明的生效方式；它跑完（日志里出现 `Successfully installed`）才谈得上验收。

## F3. 跑验收（一条命令，两项）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0,1 flagos-dc-train-910c-dev \
  bash dev/device-context/910C/probes/run_910c_p9_acceptance.sh
echo "exit=$?"        # 0 = 两项均 PASS；1 = 至少一项 FAIL；2 = 无法下判定
```

要做 A/B 对照（强烈建议——否则分不清"判据变绿"是不是环境漂移）：

```bash
# A 臂：把镜像内固化版解出来当影子，PYTHONPATH 指过去，其余完全不变
docker run --rm --entrypoint bash -v <某目录>:/out <镜像> -lc \
  'cp -a /usr/local/python3.11.15/lib/python3.11/site-packages/flagcx /out/flagcx'
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e PYTHONPATH=/opt/baked-flagcx \
  flagos-dc-train-910c-dev bash dev/device-context/910C/probes/run_910c_p9_acceptance.sh
# 该臂应 FAIL —— 若它也 PASS，说明"挂载覆盖"根本没生效，先查 [0] 自证段
```

## F4. 收尾（释放卡）

```bash
docker stop flagos-dc-train-910c-dev && docker rm flagos-dc-train-910c-dev
ps -eo pid,cmd | grep -E '[t]orchrun|[d]dp_native|[f]lagcx_sync_test'   # 应为空
docker ps | grep -c flagos-dc-train                                       # 应为 0
```

用 `up.sh` 起的则：`docker compose -f dev/device-context/910C/env/train/docker-compose.yml down`。

## F5. 本节特有的坑（本轮踩到的）

| 现象 | 原因与处理 |
|---|---|
| `baseline.py` 报 `No module named 'yaml'` 或 `docker: 'compose' is not a docker command` | 该机缺机制前置（npu1-27 即如此）。要么补齐，要么用 F2 的等价 `docker run` 并**如实登记"未跑 --strict"** |
| compose 报 `required variable DC_DIE_A is missing a value` | `.env` 没填或没 export die 号。**这是刻意设计**：不给默认值，选卡必须每次按实测来 |
| 验收脚本打印 `overall: ERROR` 但两项其实是 FAIL | 脚本早期版本的判定取值 bug（把 `overall: FAIL` 整串与 `FAIL` 比较）。已修为只取判定词；若复现请先确认脚本版本 |
| 宿主侧找不到 `acceptance_out/` 证据文件 | `OUT` 传了**宿主**路径 ⇒ 容器里创建了同名但属另一文件系统的目录（宿主/容器路径混用的老坑）。证据目录必须在挂载内 |
| 容器里 `import flagcx` 仍指向 `site-packages` | `pip install -e` 没跑成（看容器日志）或 `PYTHONPATH` 把它盖住了。以 `[0] 自证` 段打印的路径 + `.so` sha256 为准 |
| `pip install -e` 编译期报 `fatal error: cuda_runtime_api.h` | 用的是修复之前的源码（`plugin/torch/flagcx/src/backend_flagcx.cpp` 无条件包含 `ATen/cuda/CUDAContext.h`）。`b576fa6a` 起已加 `USE_NVIDIA_ADAPTOR` 守卫 |
| `import flagcx` 报 `undefined symbol: aclnnInplaceAddGetWorkspaceSize` | 构建时把标准 CANN 的 `-lopapi` 也砍了。`makefiles/ascend.mk` 里只有 DAG 专用的 `-lcust_opapi` 该按开关去掉，`-lopapi`/`-lnnopbase` 必须保留（`b576fa6a` 起已条件化，默认值即正确） |
| `Successfully installed` 但行为没变 | pip 增量缓存导致"看着重编、实际复用旧产物"。判据不能只看 pip 退出码：看 `_C*.so` 的 sha256 是否变了、`libflagcx.so` 的 `DT_NEEDED` 是否含 `libopapi.so` |

