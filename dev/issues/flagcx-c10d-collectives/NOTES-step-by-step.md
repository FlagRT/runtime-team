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
