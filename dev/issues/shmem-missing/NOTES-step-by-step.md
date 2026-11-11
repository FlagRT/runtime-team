# 逐步操作：从零复现 shmem 缺失（1 卡）

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
  在本机是指向宿主个人目录的软链，容器内无法解析，不给值容器起不来。本缺陷的最小复现不读模型，
  但挂载点仍需有效，指向任一实体目录即可。
- `--strict`：启动前按镜像基线核对 image id，不一致即拒绝启动（用于出结论的运行）。
  预期输出含 `[PASS] image: flagrt/ascend-operator-runtime-comm:...`。
- `up.sh` 会自动导出 compose 需要的 `WORKSPACE_ROOT`，这一步不用自己设。

```bash
docker ps   # 应看到 flagrt-v3-demo-910c，状态 Up
```

## 2. 复现缺陷（撤除 shmem 占位模块）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0 flagrt-v3-demo-910c \
  bash -c 'PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:} python3 dev/issues/shmem-missing/assets/minirepro_shmem.py'
echo "exit=$?"
```

- `ASCEND_RT_VISIBLE_DEVICES=0`：只用 0 号卡。一个容器内的多项检查靠这个变量分区；
  如本机 0、1 号卡另有用途，改成空闲卡号即可。
- `PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:}`：去掉占位模块目录，还原未规避的原始环境。
  这是复现的关键一步，漏掉就会看到 PASS。
- 日志里的 `can not use command: npu-smi info` 是容器内无 npu-smi 的提示，与本缺陷无关。

## 3. 读判定

只看最后三行与退出码：

```
=== VERDICT (minirepro_shmem, 1 card) ===
  repro: CONFIRMED - for-loop kernel compilation hit ModuleNotFoundError: No module named 'shmem'
overall: FAIL
exit=1
```

- `overall: FAIL` + `exit=1` → **缺陷已复现**（当前预期）。
- `overall: PASS` + `exit=0` → 该环境不存在缺陷；若这是修复后的运行，即**验收通过**。
  若这是第 2 步的运行却得到 PASS，先检查第 2 步的 `PYTHONPATH` 是否真的去掉了 `/opt/flagrt-stub`：
  `docker exec flagrt-v3-demo-910c bash -c 'PYTHONPATH=${PYTHONPATH#/opt/flagrt-stub:} python3 -c "import shmem"'`
  应报 `ModuleNotFoundError`。
- `overall: ERROR` + `exit=2` → 撞到的不是本缺陷，看打印的 traceback，不要据此下结论。

## 4. 对照组（可选，证明规避有效）

```bash
docker exec -w /workspace -e ASCEND_RT_VISIBLE_DEVICES=0 flagrt-v3-demo-910c \
  python3 dev/issues/shmem-missing/assets/minirepro_shmem.py
echo "exit=$?"
```

与第 2 步只差 `PYTHONPATH` 一项，预期 `overall: PASS`、`exit=0`。

## 5. 收尾：释放卡

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
| compose 报 `WORKSPACE_ROOT` 未设置 | 只在绕开 `up.sh` 时发生（例如 `down`），按第 5 步加前缀 |
| 容器起不来、挂载 `/workspace/models` 失败 | 没给 `FLAGRT_MODELS_DIR`，缺省路径是软链 |
| 第 2 步得到 `overall: PASS` | `PYTHONPATH` 没真正去掉 `/opt/flagrt-stub`（见第 3 步的自查命令） |
| 改写脚本后缺陷不再复现 | Triton 编译缓存命中已编译产物，不再走 `visit_For`。本脚本每次运行自建全新 `TRITON_CACHE_DIR`，自行改写时须保留这个处理（见 README 末节） |
| 改写脚本后判定变成 ERROR、数值不符 | for 循环 kernel 的输入输出用了同一张量（原地读改写）在本后端数值错误，与本缺陷无关（见 README 末节） |
