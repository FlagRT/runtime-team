# vllm-plugin-FL pooling 输出类型漏判致崩溃

## 问题与现状

- **一句话**：FlagOS 基座组件 `vllm-plugin-FL` 的 `vllm_fl/worker/worker.py` 在
  `execute_model()` 里对 pooling/embed 任务的输出类型判断漏判 `AsyncModelRunnerOutput`，
  在 `current_platform.is_cuda_alike()` 为真的平台（如昆仑芯，torch.cuda 命名空间重写命中此判断）
  上跑任何 embedding/pooling 任务（如 Qwen3-Embedding）**必定崩溃**，不是边界条件。
- **归属方向**：不是运行时组任何子方向的缺陷，是 FlagOS 基座组件 `vllm-plugin-FL`
  （`github.com/FlagRT/vllm-plugin-FL`，sublib，在 `dev/repos.yaml` 登记 owner=总组，
  分支模型 main 只同步 / dev-1.0 公共开发 / 个人分支；我方对该仓 dev-1.0 有写权限）的代码缺陷。
- **状态（2026-11-22）**：**已在依赖源码层面修复**——一行 `isinstance` 判断补充，已合入
  `FlagRT/vllm-plugin-FL` 自己的 `dev-1.0`（合并提交 `039fb46`，父提交含修复本体 `0c833a4`）。
  **但现行发版镜像尚未重打包**，仍命中（见下方「版本集合」）。

### 根因

`vllm_fl/worker/worker.py` 的 `execute_model()`（修复前约第 849 行）：

```python
if isinstance(output, (ModelRunnerOutput, NoneType)):
    return output
assert isinstance(output, IntermediateTensors)
```

`model_runner._pool()`（`vllm_fl/worker/model_runner.py`）在 `is_cuda_alike()` 为真的平台上
返回 `AsyncGPUPoolingModelRunnerOutput`（定义于同文件 397 行，继承自 `AsyncModelRunnerOutput`，
**不是** `ModelRunnerOutput`）。上面的 `isinstance` 没认出它，于是掉进后面的
`assert isinstance(output, IntermediateTensors)`，必定失败。`AsyncModelRunnerOutput` 本身已在
本文件 `sample_tokens()` 的返回类型声明中出现、已 import，是合法的终态输出，不该被当成
pipeline-parallel 中间张量转发。

修复：把判断改成
`if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):`
——`AsyncModelRunnerOutput` 已在文件头部 import，不需要新增 import。

已核对上游 `flagos-ai/vllm-plugin-FL`（我方 fork 的来源仓）的 `main` 分支同样命中这个判断缺口，
没有另提 PR 给上游（两者是不同仓，上游我方无写权限）。

### 版本集合（分版表态）

坐标 = 镜像血统 + 运行环境修订（现行昆仑芯推理腿唯一一条运行环境）。状态取三态之一：
在该版仍命中 / 在该版已修（写明形态）/ 未在该版测过。

| 坐标 | 状态 | 依据 |
|---|---|---|
| `kunlun-infer-vllm/v1`（锚点 digest `sha256:e180a0dd4267ebdfbce4ea5434710870a6ea7e96118457538cc02442907d54b1`，镜像内打包 vllm-plugin-FL 为 `plugin0.2.0`） | **仍命中** | 2026-11-22 P800-C 容器 `flagrt-kunlun-infer-v1-env-v1-p800a` 实测：补丁前最小复现确认崩溃（`overall: FAIL`，报错原文见 [`assets/evidence/pooling-dispatch-20261122.txt`](assets/evidence/pooling-dispatch-20261122.txt)）。镜像内打包的 vllm-plugin-FL 版本早于修复提交 |
| `vllm-plugin-FL` 仓自身（依赖源码层面，不是某个镜像坐标） | **已修**（形态：上游分支直接修复，非挂载覆盖、非层包） | 2026-11-22：合并提交 `FlagRT/vllm-plugin-FL@039fb46`（父含 `0c833a4`）。同一容器内 `docker exec` 手工 `sed` 改该行、不重建镜像，重跑最小复现 `overall: PASS`，`EMBED_OK, dim= 1024`（见同一份证据截选） |
| `kunlun-operator-runtime/v2`（昆仑芯训练腿） | **未在该版测过** | 没有实测该血统是否也打包了 vllm-plugin-FL、是否同样命中；不表态 |
| 下次 `kunlun-infer-vllm` 镜像/插件版本升级 | 待核实 | 需确认新打包的 vllm-plugin-FL commit 是否 ≥ `039fb46`，到时候补一行分版表态 |

## 复现

### 前置条件

| 项 | 要求 | 获取方式 |
|---|---|---|
| 镜像 | 现行昆仑芯推理腿发布物 `kunlun-infer-vllm/v1`，digest `sha256:e180a0dd...`（镜像内 `plugin0.2.0`） | 按现行 `kunlun-infer-vllm/v1` 的 `env.baseline.yaml` 核对镜像 id（本目录不复制该血统的 `image.baseline.yaml`/`demo/`，按 CASE-TEMPLATE §0 借用兄弟镜像目录） |
| 运行环境 | 容器 `--network none`（生产隔离）；需 shell `export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1`（**必须用 shell export，脚本内 `os.environ.setdefault` 对 vllm spawn 的子进程不生效**——这是本 case 复现脚本的头号坑，漏掉会看到 `Unable to find address for: eth0` 而不是目标断言） | 随现行容器启动纪律 |
| 卡 | 1 × 昆仑芯 P800 | 启动前 `xpu-smi` 挑 0% 利用率 / 0MiB 显存的卡 |
| 模型与数据 | Qwen3-Embedding-0.6B（fp16），本地路径需剔除 `config_sentence_transformers.json` / `modules.json`（否则 vllm 配置加载会在另一处崩，与本缺陷无关） | 经 hf-mirror.com 下载，sha256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`；已有副本：P800-C `/data2/xliu969/scratch-bugverify/Qwen3-Embedding-0.6B/`（容器内挂载路径 `/home/models/Qwen3-Embedding-0.6B/`） |

### 单入口命令

```bash
# 容器内，已 export 上述三个网络变量
python3 dev/issues/vllm-plugin-fl-pooling-dispatch/assets/minirepro_pooling_dispatch.py \
  --model /home/models/Qwen3-Embedding-0.6B/
```

脚本自动建立一个剔除两个 sentence-transformers 配置文件的临时软链接视图，不修改原模型目录；
以子进程方式跑真正的 `llm.embed()` 调用，只认目标栈帧（`worker.py execute_model` 的那个
`assert isinstance(output, IntermediateTensors)`）为复现确认，其他断言/异常判为 `ERROR`
而非 `FAIL`，避免误判。

### 预期结果（问题存在时）

```
=== VERDICT (minirepro_pooling_dispatch, 1 card) ===
  repro: CONFIRMED - execute_model rejected pooling output
overall: FAIL
```

退出码 1。补丁后预期 `overall: PASS`、退出码 0、输出含 `EMBED_OK, dim= 1024`。

## 修复验收

**判据**：容器内 `vllm_fl/worker/worker.py` 的 `execute_model()` 判断补上
`AsyncModelRunnerOutput` 后，上面单入口命令应 `overall: PASS`，退出码 0。

| 脚本 | 类别 | 命令 | 通过判据 | 最近一次结果与证据 |
|---|---|---|---|---|
| [`assets/minirepro_pooling_dispatch.py`](assets/minirepro_pooling_dispatch.py) | 复现 / 验证 | 见「单入口命令」 | 退出码 0 且 `overall: PASS`，输出含 `EMBED_OK, dim= 1024` | 2026-11-22 P800-C：补丁前 `overall: FAIL`（退出码 1，`repro: CONFIRMED`）；容器内 `docker exec sed -i` 手工打补丁、不重建镜像后，重跑 `overall: PASS`（退出码 0，`EMBED_OK, dim= 1024`）。证据：[`assets/evidence/pooling-dispatch-20261122.txt`](assets/evidence/pooling-dispatch-20261122.txt) |

修复到货到现行发版镜像后（下次 `kunlun-infer-vllm` 版本升级，打包的 vllm-plugin-FL commit
≥ `039fb46`），上方「版本集合」需新增一行分版表态，并将该坐标标为「已修」。

### 复现脚本的坑（已在脚本内处理，自行改写时注意）

1. **三个网络变量必须 shell export，不能用脚本内 `os.environ.setdefault`**：容器
   `--network none`，vllm 的 EngineCore 走 spawn 子进程，脚本内设置的环境变量不会被子进程
   继承；不这样做会看到 `Unable to find address for: eth0`，而不是目标断言，容易误判为
   "环境问题、不是这个缺陷"。
2. **模型目录不能直接用原路径**：带 `config_sentence_transformers.json` / `modules.json`
   两个文件的本地路径会在 vllm 配置加载的另一处代码先崩，跟本缺陷无关；脚本用临时软链接视图
   规避，不修改原模型目录。
3. **只认目标栈帧，不认任意 `AssertionError`**：脚本用正则精确匹配
   `worker.py execute_model` 那个具体的断言栈帧，其余异常判 `ERROR`，避免撞到别的问题时
   误报"本缺陷已复现"或"已修复"。

<!-- changelog:
2026-11-22 —— 首次登记：根因由 2026-11-22 排查定位（AsyncModelRunnerOutput 漏判，非最初怀疑的
  PoolerOutput 分支缺失），修复当日合入 FlagRT/vllm-plugin-FL dev-1.0@039fb46，P800-C 真机
  验证补丁前后对照（崩溃→EMBED_OK dim=1024）。源起 inbox/kunlun芯显存问题排查.md（袁先乙提交的
  另一份厂商侧缺陷报告）排查过程中的附带发现，与该报告的两个厂商缺陷不是同一机制。
-->
