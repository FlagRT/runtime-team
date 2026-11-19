# FlagScale × FlagPerf 最佳实践

> 读者：FlagScale 检索训练（`feat/retrieval-flagos`）的开发者
> 版本：FlagPerf `zhiyu/perf-train`（适配器 `inference/tools/flagscale_adapter.py`）
> 本文所有输出都来自实机跑：主要是**在 FlagScale 自身训练入口
> （`flagscale/train/train_retrieval.py`）里真实跑通的一次**，外加 FlagPerf
> 引擎侧的 8B FSDP2 演示。平头哥 PPU（torch 2.10，双卡）。

---

## 第一部分 我们交付什么

### 1.1 交付形态：一个 Python 包 + 一个适配器文件

我们不改你们的框架、不要求换训练引擎。交付两样东西：

1. **`flagprof` 包**（纯 Python，无守护进程）：
   ```bash
   # 方式 A（已验证）：加路径
   export PYTHONPATH=/path/to/FlagPerf/inference:$PYTHONPATH
   # 方式 B：本地安装（需要构建工具；我们的容器里未验证过，建议优先 A）
   pip install -e /path/to/FlagPerf/inference
   ```
2. **`flagscale_adapter.py`**（单文件，约 90 行）：拷到你们训练脚本可 import
   的位置即可。它负责：建采集会话、厂商会话自动识别、通信探针安装、相位
   标记、进程内收尾。

依赖只有 PyTorch 本身（采集用的都是 `torch.profiler` 与
`torch.distributed` 公开 API），没有额外三方依赖。

### 1.1b 运行镜像（按厂商，名称 + ID）

| 平台 | 镜像名称 | 镜像 ID |
|---|---|---|
| 平头哥 PPU | `flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64` | `1ba7ead15378` |
| 昆仑芯 P800 | `flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64` | `cac5ddae5f62` |
| 昇腾 910C | `flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64` | `62bcaae3f839` |

（910C 另有 comm 变体 `flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64`，ID `e13a15d0d02b`，本文未用它。）

### 1.2 支持的功能点

| 功能 | 说明 |
|---|---|
| 训练相段计时 | data / forward / backward / optimizer / step 五段区间（`obs.phase(...)`） |
| 通信探针 | 19 个 torch.distributed 入口（集合通信 + send/recv/isend/irecv/batch_isend_irecv），逐调用记录 op/字节/对端/时长 |
| 按 rank 归集 | 每个 rank 独立 store 与 trace，报表逐 rank 分列 |
| 等待 vs 带宽 | 同主机时钟域内跨 rank 对齐，wait（到达偏斜）与带宽分开报（按厂商标注可信度） |
| 相段占比/同并行度对比 | 每步五相时长与 step 墙钟占比；并行度入档，只允许同 world_size 对比 |
| 官方 profiler trace | 厂商会话：PPU / Ascend 910C / NVIDIA / 昆仑芯；含 phase 与 comm 区间 |
| 零漂移对照 | 观察开/关同参运行，比较最终权重（PPU/P800 逐位；FSDP2 比数值不比文件字节） |
| 训练任务与协议 | 五类任务（分类/LM/CTR 14 架构/embedding/rerank）、AdamW/SGD、梯度累积、检查点与恢复、AMP、DDP、FSDP2 |
| 单算子微基准 | `inference/tools/op_micro.py`（14 算子；PPU 与 910C 均已实跑） |
| 算子级 profiling | 官方 profiler 设备内核时间线（内核名/时长/次数），可按训练相归因；`comm-report` 汇总内核数与通信内核占比 |
| 通信画像报表 | `comm-report`：通信分列（次数/字节）、设备节（内核与通信占比）、等待带宽、相段占比，一份 JSON + stdout |
| 训练结果数值对比 | `flagprof.training.compare`：比较两跑 loss/梯度/参数/优化器状态，逐张量给出误差（0 容差可用） |

---

## 第二部分 支持的模型列表

> 图例：✅ 实测通过　○ 支持、该模型未单独实测　✗ 不支持　— 不适用。
> "性能分析"指观察采集链路；它跟模型结构无关，已在 8B 与 bge-m3 上完整
> 跑通（本文第三部分即 bge-m3 真跑）。

| 模型/架构 | 任务类型 | 单卡训练 | 多卡 DDP | FSDP2 | 断点续训 | AMP | 性能分析（观察态） | 备注 |
|---|---|---|---|---|---|---|---|---|
| BAAI/bge-m3 | embedding（XLMRoberta） | ✅ | ✅（真 FlagScale 双卡，第三部分） | ✅（2 卡实测，per-block+root） | ○ | ○ | ✅ | 分块发现覆盖 `model.encoder.layer`（有回归测试） |
| Qwen/Qwen3-Reranker-8B | rerank（causal_yesno） | ✗ 显存 | ✗ 显存 | ✅（2 卡 10 步实测） | ✗ FSDP+断点暂不支持 | ○ | ✅ | FP32 训练态约 128GB，必须分片；P800 16 卡、910C 4 卡也跑过 |
| openai/clip-vit-large-patch14 | 多模态 | ✗ | ✗ | ✗ | — | — | — | 我们的训练入口没有视觉/图文任务（FlagScale 仓内的 CLIP 配置属他们自己的实现路径） |
| DIN | CTR | ✅ | ✅ | ○ | ✅ | ✅ | ○ | DDP 实测至 16 卡/100 步稳定 |
| DeepFM | CTR | ✅ | ✅ | ○ | ○ | ○ | ○ | DDP16×100 步稳定 |
| DCN-Mix | CTR | ✅ | ✅ | ○ | ✅（本次补测：resume 与连续跑逐位一致） | ○ | ○ | DDP8×150 步 + DDP16×100 步 |
| DSSM | CTR | ✅ | ✅ | ○ | ○ | ○ | ○ | DDP16×100 步稳定 |
| LightGCN | CTR | ✅ | ○ | ○ | ○ | ○ | ○ | 单卡实测 |
| SASRec | CTR | ✅ | ✅（本次补测 2 卡） | ○ | ○ | ○ | ○ | 单卡 + 双卡 DDP 实测 |

---

## 第三部分 从零配置并在 FlagScale 里集成（实跑案例）

本章是从零把一个真实 FlagScale 训练跑出结果的完整过程，照着做可以复现
我们拿到的每一个输出。

### 3.1 环境准备（从零开始）

**① 两边的代码与包**

```bash
# 你们的仓库（本文用分支 feat/retrieval-flagos）
git clone https://github.com/wxj630/FlagScale.git && cd FlagScale
git checkout feat/retrieval-flagos
# 我们的包（只需要 inference 子目录）
git clone https://github.com/FlagRT/FlagPerf.git /path/to/FlagPerf
git -C /path/to/FlagPerf checkout zhiyu/perf-train
export PYTHONPATH=/path/to/FlagPerf/inference:$PYTHONPATH
cp /path/to/FlagPerf/inference/tools/flagscale_adapter.py flagscale/train/
```

**② 依赖检查**（我们实测的容器里已有；缺什么补什么）：
`torch`、`transformers`、`omegaconf`、`pyarrow`、`PIL`、`numpy`。
Hydra 只有走 `flagscale.run` 的 CLI 才需要；直接调 `train_retrieval.py`
不需要。

**③ store 目录**：采集产物落盘位置，必须在本机持久文件系统（ext4/xfs）。
网络盘、overlay、tmpfs 会被拒绝并给出明确报错。

**④ 数据**：`train_retrieval.py` 的 triplet 读取器要 parquet（列：
`anchor`、`positive`、`negative_1..K`）。没有 T2Ranking 时可以用一个小
parquet 跑通链路（我们就是这么做的，120 行）：

```python
# fs_make_data.py
import pyarrow as pa, pyarrow.parquet as pq
rows = [("anchor text %d about retrieval topic" % i,
         "positive passage %d" % i, "negative passage %d" % i) for i in range(120)]
pq.write_table(pa.table({"anchor": [r[0] for r in rows],
                         "positive": [r[1] for r in rows],
                         "negative_1": [r[2] for r in rows]}),
               "/data/fs-tiny/data.parquet")
```
```bash
python fs_make_data.py      # 生成后 data.path 指向 /data/fs-tiny
```

**⑤ 模型**：把 `train.model.model_path` 指向本地 bge-m3 目录即可。

**⑥ 配置文件**（standalone YAML，绕开 Hydra 也能直接被入口读取）：

```yaml
train:
  name: bge-m3-demo
  task: embedding
  seed: 42
  batch_size: 2
  gradient_accumulation_steps: 1
  epochs: 1
  log_every: 5
  temperature: 0.05
  max_grad_norm: 1.0
  output_dir: /workspace/fs-run/out
  model:
    model_type: bge_m3
    model_path: /models/bge-m3          # 换成你们的本地路径
    max_length: 64
    torch_dtype: bf16
  data:
    path: /data/fs-tiny                  # 上面生成的 parquet 目录
    validation_path: /data/fs-tiny
    test_path: /data/fs-tiny
    train_split: train
    validation_split: validation
    test_split: test
    eval_max_samples: 16
    split_modulo: 20
    shuffle_buffer: 0
    num_workers: 0
    num_negatives: 1
    max_samples: 60
  optimizer:
    lr: 2.0e-5
    weight_decay: 0.01
  hardware:
    device_type: cuda
    distributed_backend: auto
  parallel: ddp
  master_dtype: bf16
  save_every_epoch: false
```

### 3.2 在 FlagScale 训练入口集成适配器（逐处解释）

改动全部落在 `flagscale/train/train_retrieval.py`，共六处。下面每处都说
清楚"改什么、为什么、你们自己的循环怎么对应"。

**改动 1：顶部加一个可选 import**（放在 `import torch` 之前）：

```python
try:  # 可选依赖：只有设置了 FLAGPROF_TRAIN_OBSERVE 才会真正工作
    from flagscale_adapter import ObservationSession
except Exception:  # 没装/路径没配时不使用采集，训练不受影响
    ObservationSession = None
```

为什么：把采集做成"有就用、没有也可以"的可选项，避免绑定依赖。

**改动 2：加一个相位标记小工具**（放在 `log()` 之后）：

```python
def _phase(obs, name: str, step: int):
    """相位标记；没开采集时是空上下文，行为完全不变。"""
    return obs.phase(name, step=step) if obs is not None else contextlib.nullcontext()
```

**改动 3：`train_epoch()` 的参数表加一个 `obs=None`**——保持默认调用兼容。

**改动 4：在训练循环里包五个相位**（核心改动；下面是包好后的骨架，
改动处用 `# ←` 标出）：

```python
    with _phase(obs, 'data', 0):                       # ← data：建迭代器与首个小批
        stream = iter(loader)
        batch = next(stream, None)
    while batch is not None:
        with _phase(obs, 'data', micro_batches):       # ← data：取下一个小批
            nxt = next(stream, None)
        with _phase(obs, 'forward', micro_batches):    # ← forward：前向 + loss
            with autocast_context(device, dtype_name):
                raw_loss = loss_for_batch(model, processor, task, batch, device, temperature)
                loss = raw_loss / grad_accum
        ...
        is_window_end = (micro_count + 1) >= grad_accum or nxt is None
        with _phase(obs, 'backward', micro_batches):   # ← backward：反向
            with contextlib.nullcontext() if is_window_end else parallel_utils.gradient_sync_disabled(model):
                loss.backward()
        ...
        with _phase(obs, 'optimizer', micro_batches):  # ← optimizer：裁剪+step+调度器
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if lr_scheduler is not None: lr_scheduler.step()
            optimizer_steps += 1
        with _phase(obs, 'step', micro_batches):       # ← step：步边界
            pass
```

对应关系：你们自己的框架里，把"取 batch / 前向+loss / backward /
optimizer.step / 步结束"五个位置包上同名 phase 即可。`step` 用累计的
microbatch 序号，保证每个相位计数可对齐。

**改动 5：`train()` 里创建会话并包住 epoch 循环**：

```python
    obs = ObservationSession() if ObservationSession is not None else None
    if obs is not None:
        log('flagprof observation session active')
    with (obs if obs is not None else contextlib.nullcontext()):   # ← 会话生命周期
        for epoch in range(1, epochs + 1):
            train_loss = train_epoch(..., obs=obs)                 # ← 传下去
```

为什么包在 epoch 循环外：会话要在所有训练步开始前 attach、结束后统一
finalize；`with` 退出时自动关闭厂商 profiler 并封存 store。

**改动 6：启动时用环境变量打开**（不要改代码）：

```bash
export FLAGPROF_TRAIN_OBSERVE=/data/profiles/fs-run1
```

不开这个变量，`ObservationSession()` 内部完全空转——我们用同一份代码跑了
开/关两次，最终权重逐字节相同（见 3.4-④）。

### 3.3 启动命令（我们实际用的方式 + docker）

我们在这台 PPU 机器上**用的是 docker**。镜像与完整命令如下（可直接抄）：

```bash
IMG=flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64
# 同上镜像的 ID：1ba7ead15378（两种写法等价；下文命令里的 $IMG 即此）
docker run --rm --shm-size=16g \
  --device=/dev/alixpu_ctl --device=/dev/alixpu \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  -v /host/models:/models:ro \
  -v /host/flagscale:/workspace/FlagScale -w /workspace/FlagScale \
  -v /host/fstag:/data \
  -e PYTHONPATH=/workspace/FlagScale:/workspace/FlagPerf/inference:/workspace/FlagPerf/inference/tools \
  -e FLAGPROF_SCRATCH=/data/scratch \
  -e FLAGPROF_TRAIN_OBSERVE=/data/fs-obs-store \
  $IMG bash -c "cd /workspace/FlagScale && \
      torchrun --standalone --nproc-per-node=2 \
      flagscale/train/train_retrieval.py --config-file /data/fs_demo_config.yaml"
```

说明：
- 你们仓库自己的文档（`docs/retrieval/training/15-launch-commands.md`）记录
  的是宿主机 venv / `python -m flagscale.run` 的跑法；那套命令在我们的
  容器里同样适用，只是把 `python` 换成镜像里的 `python3`、把数据/模型路径
  换成容器内挂载路径。**两种跑法的差别只是包装，训练入口与配置完全一致。**
- 用我们 FlagPerf 引擎跑（不接 FlagScale 时）的 docker 命令见附录。
- P800 用镜像 `cac5ddae5f62`、XPU 设备节点、`--network=host`；910C 用
  `flagrt/ascend-operator-runtime:…torch2.10`、`--privileged`、davinci 设备。

### 3.4 实机结果（真 FlagScale 训练 + 全部采集功能演示）

这次跑的是 FlagScale 自己的 `train_retrieval.py`、自己的循环、自己的 DDP。
**所有输出如下，均实机生成。**

**① 训练输出（stdout 节选）**

```text
flagprof observation session active
epoch=1 optimizer_step=5  running_loss=0.455176 window_loss=0.455176 lr=2.000e-05
epoch=1 optimizer_step=10 running_loss=0.288562 window_loss=0.121948 lr=2.000e-05
epoch=1 optimizer_step=15 running_loss=0.201626 window_loss=0.027753 lr=2.000e-05
epoch=1 optimizer_step=20 running_loss=0.154145 window_loss=0.011703 lr=2.000e-05
epoch=1 optimizer_step=25 running_loss=0.124305 window_loss=0.004947 lr=2.000e-05
epoch=1/1 train_loss=0.116918 validation_loss=0.000107 device=cuda:0
saved checkpoint to .../fs-run/out
```

**② 通信与计算分列**（store 读回；两个 rank）

```text
rank-0 manifest= complete comm= {'model/allreduce': 2} phases= {'data': 28, 'forward': 27, 'backward': 27, 'optimizer': 27, 'step': 27}
rank-1 manifest= complete comm= {'model/allreduce': 2} phases= {'data': 28, 'forward': 27, 'backward': 27, 'optimizer': 27, 'step': 27}
```

**③ 按 rank 归集**：上面两行就是——两 rank 完全对称。

**④ 同并行度对比 + 零漂移**（同配置跑一次不带 `FLAGPROF_TRAIN_OBSERVE`
的对照，两跑同为 2 rank）：

| 对比项 | 观察态跑 | 未观察跑 | 判读 |
|---|---|---|---|
| world_size | 2 | 2 | 同并行度，可对比 |
| 最终权重文件 sha256 | `4b346bb79adcd23e…` | `4b346bb79adcd23e…` | **逐字节一致** |
| train/validation loss | 0.116918 / 0.000107 | 0.116918 / 0.000107 | 一致 |

**怎么读这几个数字**：
- `comm={'model/allreduce': 2}`：这两条来自你们训练代码里的评测（validation/
  test 各一次 loss 归约），不是梯度同步。**DDP 的梯度 allreduce 发生在 C++
  Reducer 里，Python 层探针按设计看不到**——所以没有梯度通信条目，这是已知
  边界，不是漏采（见 4.2 的说明与替代方案）。
- `phases`：forward/backward/optimizer/step 各 27 = 27 个 microbatch；
  data 28 = 首次取批（step 0）+ 每轮取下一批，比其它相多 1 是 lookahead
  结构造成，属正常。
- `train_loss 0.455→0.117`、`validation 0.000107`：小样本演示数据上的收敛
  曲线，用于证明训练确实在走，不代表真实数据效果。

**⑤ 等待与带宽**：该样例中 DDP 无可见集合通信，故无 wait/bandwidth 行；
真实数值演示见 4.4（rerank06b 双卡 FSDP 的实测输出）。

### 3.5 并行策略：支持情况与验证状态（如实）

| 策略 | 我们跑过什么 | 状态 |
|---|---|---|
| 单进程 | 五类任务、14 CTR 架构、embedding/rerank/8B | ✅ 三平台 |
| DDP | 真 FlagScale 双卡（本章）；自研引擎 2/4/8/16 卡、100+ 步稳定性 | ✅ |
| FSDP2 | 自研引擎：0.6B/8B 双卡 10 步、bge-m3 2 卡（per-block+root）、P800 16 卡 8B | ✅（FlagScale 侧配 `parallel=fsdp2` 走同一函数，未在你们仓里实跑） |
| Megatron PP | 你们仓有 `_train_megatron` 路径（依赖 Megatron-LM-FL 与你们的 vendor env）；我们的探针覆盖其底层 p2p 原语（send/recv/batch_isend_irecv）。**我们尝试用 PyTorch 官方流水引擎（`torch.distributed.pipelining`）在 PPU 与 P800 两个厂商栈上实跑，均卡在第一步（无报错、NCCL RAS 转储），属厂商栈兼容性问题，已留证**；因此不宣称引擎级流水并行已验证 | ⚠️ 未验证（厂商栈问题） |

---

## 第四部分 输出与解读（按功能分节）

> 每节结构：功能是什么 → 怎么开 → 实际输出 → 怎么读。
> 产物根目录：`<store>/rank-N/<run-id>/`。

### 4.1 功能：训练相位区间（spans）

- **是什么**：把每步切成 data/forward/backward/optimizer/step 五段计时。
- **怎么开**：代码里 `obs.phase('<相>', step=<序号>)`，见 3.2 改动 4。
- **实际输出**：`phases= {'data': 28, 'forward': 27, 'backward': 27,
  'optimizer': 27, 'step': 27}`（本次真跑）。
- **怎么读**：forward/backward/optimizer 计数应等于 microbatch 数；data 因
  lookahead 多 1。某相占比异常 → 该方向排查（data 高=数据管线；forward
  高=算子/访存）。跨卡只在同并行度下比较。

### 4.2 功能：通信与计算分列（通信事件流）

- **是什么**：每次集合通信/点对点调用单独一条记录（op、role、字节、对端、
  host 耗时），与算子内核分列。这是检索组 §5.5 第一条。
- **怎么开**：自动。world>1 时适配器在 gloo control group 上装 19 入口探针。
- **实际输出**：`comm= {'model/allreduce': 2}`（本次真跑——评测的两次 loss
  归约）。另一个真实样本（自研引擎 8B FSDP2，10 步）：
  `comm= {'model/allgather': 730, 'model/reducescatter': 370}`。
- **怎么读**：
  - 逐事件详情在 store（op/role/bytes/peer/group_seq/host_duration_ns）。
  - **DDP 梯度 allreduce 不可见**（C++ Reducer）。要测梯度通信：改用
    FSDP2（funcol 可见），或在关键处显式写 `dist.all_reduce`（会被记录）。
  - FSDP2 计数核对：每步 allgather≈块数×2+1、reducescatter≈块数+1（8B：73/37
    与 36 块吻合）。
  - p2p（Megatron PP 的底层原语）以 `p2p/send`、`p2p/recv`、`p2p/batch_p2p`
    单列。

### 4.3 功能：按 rank 归集（per-rank store）

- **是什么**：每 rank 独立 store 与 trace；报表按 rank 分节。§5.5 第二条。
- **怎么开**：自动（每 rank 各自建会话）。
- **实际输出**：本次真跑两行逐字段相同（见 3.4-②）。
- **怎么读**：SPMD 集合通信两 rank 应对称；PP 各 stage 天生不同，只在同
  stage 组内比；某 rank 偏少先查 manifest 是否 complete。

### 4.4 功能：等待与带宽分开报（wait_bandwidth）

- **是什么**：跨 rank 对齐同一调用的设备内核起点，wait=到达偏斜，
  transfer=内核墙钟−wait，带宽=字节/transfer。§5.5 第三条。
- **怎么开**：自动；同主机时钟域才计算，跨主机标 `unsynchronized-clock`。
- **实际输出**（PPU 实机存档：rerank06b 双卡 FSDP 的封存 trace 分析跑）：
  ```text
  wait/bw allgather wall=0.7ms wait_med=6.2ns wait_share=0.8411 bw_med=78833.0GB/s
        [vendor-decomposed: wall not wire time]
  ```
- **怎么读**（分级）：
  1. 只有 `alignment=aligned` 的行有数字；`p2p`/`unaligned`/
     `kernel-event-count-mismatch`/`unsynchronized-clock` 各有说明。
  2. `wait_med` 永远可信（PPU 实测 5.1–523.7 ns）：它只是时钟差。
  3. 行尾 `[vendor-decomposed: wall not wire time]`（PPU 触发条件：host
     enqueue 中位 > 10× 内核墙钟中位）时，**带宽与 wait_share 不能当线上值**
     ——PPU 把集合通信拆成多个微内核 + DMA。只看 wait_med。
  4. P800：厂商 kineto 无设备内核事件 → 该节报 `no-device-intervals`/
     `kernel-event-count-mismatch`，等待/带宽不可得；通信计数与字节账仍完整。
  5. 910C：平台逐跑非确定，wait 量级可参考、精确断言需重复测量。

### 4.5 功能：同并行度对比（segments + 并行度元数据）

- **是什么**：相段占比与每步墙钟入档，并行度一并记录；只允许同 world_size
  对比。§5.5 第四条。
- **怎么开**：自动；每次运行独立 store 目录。
- **实际输出**（PPU 存档，rerank06b 双卡 FSDP）：
  ```text
  rank-0 segments: {'data': 0.3983, 'forward': 0.6828, 'optimizer': 0.2061, ...}
         data_ns= 2378541569
  ```
  本次真 FlagScale 跑的同并行度对比见表 3.4-④（同为 2 rank）。
- **怎么读**：同并行度比相段占比/每步墙钟/wait；跨并行度只比总吞吐与总成本，
  不比单算子毫秒。并行度来源：`<run-id>/manifest.json → identity.world_size`。

### 4.6 功能：检查点与恢复

- **是什么**：训练中断后从检查点续训，模型/优化器/RNG 状态都恢复。
- **怎么开**（引擎跑法）：`--stop-after N --checkpoint <path>` 先停，
  `--resume <path>` 续跑。FlagScale 侧用你们自己的 checkpoint 机制（本次
  真跑里 `save_every_epoch: false`，结束时保存了 `model.safetensors`）。
- **实际输出**（本次补测 dcnmix）：stop/resume 与不中断连续跑
  `final_model_sha256` 逐位一致（`9ccdc377…`；din 此前同样一致）。
- **怎么读**：sha 一致即恢复正确；FSDP2 分片检查点暂不支持（明确报错）。

### 4.7 功能：AMP（自动混合精度）

- **是什么**：bf16/fp16 混合精度训练，loss 区固定 FP32（不改变数值语义）。
- **怎么开**：引擎 `--amp bfloat16`；FlagScale 侧是配置里的 `torch_dtype`
  与 `master_dtype`（本次真跑用了 bf16）。
- **实际输出**：引擎补测 `training-result.json → protocol."amp": "bfloat16"`，
  rc=0；本次 FlagScale 真跑日志中 autocast 正常、loss 轨迹正常。
- **怎么读**：确认协议/配置里的 dtype 与预期一致；loss 轨迹应与全精度同量级。

### 4.8 功能：零漂移对照（观察是否影响训练）

- **是什么**：采集开/关各跑一次，比最终权重。这是"观察零扰动"的直接证据。
- **怎么开**：开=设置 `FLAGPROF_TRAIN_OBSERVE`；关=unset 后再跑（其余不变）。
- **实际输出**：
  - 真 FlagScale 本次：两跑 `model.safetensors` 逐字节相同
    （`4b346bb7…`）。
  - 8B FSDP2 演示：两次保存的文件字节不同（DTensor 元数据），但 **399/399
    个张量值逐一相同**、loss 序列逐位相同。
- **怎么读**：
  - 普通张量模型：直接比 `final_model_sha256` 或权重文件 sha。
  - **FSDP2：比数值不比文件字节**——保存的是 DTensor，pickle 带运行时布局
    元数据，文件 sha 天然不同。
  - 910C：平台逐跑非确定，此判据不适用，改用 loss 轨迹/完成状态。

### 4.9 功能：封存 trace 与采集完整性（manifest / producer）

- **是什么**：官方 profiler 的 trace（相位/通信/设备内核）+ 采集状态清单。
- **怎么开**：随会话自动。
- **实际输出**（本次真跑）：`manifest.json` `status=complete`；store 共
  1.6 GB；trace 位于 `capture-staging/<id>/official/trace.json`；
  `producer.json` 记录 `status: closed`。
- **怎么读**：`manifest.status=complete` 是完整性第一凭据；失败看
  `producer.json` 的 errors（适配器也会把错误冒泡到 stderr）。

### 4.10 功能：单算子微基准（op_micro）

- **是什么**：14 个代表性算子（B8 S512 H1024 FFN4096 vocab151669，bf16）
  的毫秒中位数，跨芯片对照用。
- **怎么开**：`python3 inference/tools/op_micro.py --out ops.json --label <名>
  --device cuda:0`（910C 用 `--device npu:0`）。
- **实际输出**（本次在 910C 实跑）：mm_lm_head 7.893 ms、flash_attn_fwd
  1.872 ms、mm_ffn 0.263 ms、gelu 0.054 ms……（14 项全出，JSON 落盘；
  与检索组文档的 910C 数字互证：mm_lm_head 7.901 vs 我们 7.893）。
- **怎么读**：这组数只用于单卡算子快慢对照，不混入训练采集。

### 4.11 功能：算子级 profiling（设备内核时间线）

- **是什么**：官方 profiler 记录每个设备内核（名字、时长、调用次数、所属流），
  配合训练相区间可以回答"每个训练阶段里哪些算子跑了、各花多久"。
- **怎么开**：随观察会话自动采集；产物在
  `capture-staging/<id>/official/trace.json` 的 `cat='kernel'` 事件；
  `comm-report` 的 `device` 节给出内核总数与集合通信内核占比。
- **实际输出**（PPU 实机存档，rerank06b 双卡 FSDP）：
  ```text
  device rank-0: {"kernels": 11208, "comm_share": 0.2631, ...}
  ```
  即该跑共 11208 个设备内核，设备时间中 **26.3% 花在集合通信内核上**
  （与通信事件流的 172 次调用互为印证）。
- **怎么读**：热点算子按内核名与调用次数排序（内核名到语义的映射见 §5.3 的
  类别归一；更细的逐算子微基准用 4.10 的 op_micro）。
  注意两类平台限制：P800 的厂商 kineto 不产设备内核事件 → 该层不可得；
  PPU 的集合通信被厂商分解成多个微内核 → 内核墙钟不等于线上时间。

### 4.12 功能：通信画像报表（comm-report 汇总）

- **是什么**：把 writer store（通信事件、相段）与封存 trace（设备内核）汇成
  一份 JSON 报表 + stdout 摘要，四个小节正好对应 §5.5 的四条需求。
- **怎么开**：`python3 train_observe_comm_report.py <store> <post-dir> <out.json>`；
  适配器采集的 store 先过封存/导入后链（`train_observe_post.py` 等价步骤），
  否则报表只有通信事件与相段、没有设备内核节。
- **实际输出**（PPU 存档，rerank06b 双卡）：
  ```text
  comm bytes rank-0: {"model/allgather": 4144975872, "model/reducescatter": 4766220288}
  device rank-0:     {"kernels": 11208, "comm_share": 0.2631, ...}
  wait/bw allgather  aligned wait_med=6.2ns wait_share=0.8411 [vendor-decomposed]
  segments rank-0:   {"data": 0.3983, "forward": 0.6828, "optimizer": 0.2061, ...}
  ```
- **怎么读**：四节各对应一条需求——`comm_by_op`（分列与字节账）、每 rank 分键
  （归集）、`wait_bandwidth`（等待/带宽，分级读法见 4.4）、`segments`（同并行度
  对比）。字段拼写与含义固定，可直接进交付报告。

### 4.13 功能：训练结果数值对比（compare）

- **是什么**：比较两次**已完成**训练的数值结果——loss、梯度、模型参数/buffer、
  优化器状态，逐张量给出是否一致与最大绝对误差；协议或数值后端（TF32 等）
  不同的两跑会被直接拒绝。
- **怎么开**：
  ```bash
  python3 -m flagprof.training.compare <run-a> <run-b> --out <新目录>       [--atol 0 --rtol 0] [--declare backend] [--declare world_size]
  ```
  输出 `comparison.json`（机器读）与 `comparison.md`（人读）。
- **实际输出**（本次实测：dcnmix 同一配置跑两遍，0 容差）：
  ```text
  | pass | 0.0 | 0.0 | 224 |
  observed loss, gradients, model buffers/parameters and optimizer state; not convergence certification
  | 0 | loss | pass | 0 | 0.0 |
  | 0 | snapshot/model/net.crossnet.u_list.0 | pass | 0 | 0.0 |
  ...
  ```
  224 项全部 pass、不匹配元素 0、最大误差 0.0。
- **怎么读**：`status=pass` 即数值一致（0 容差下的逐位一致）；`different` 时
  看 rows 里哪些张量不匹配、误差多大；atol/rtol 只在明确接受差异时使用，且
  报告会把容差写进结论。`scope` 字段写明它不做收敛性结论。
  另注意：覆盖步数不同的两跑（如 resume 与从头跑）会被拒绝——这是设计，防止
  拿不等价的运行做比较。

## 第五部分 验收表格：检索组功能需求 vs 实际完成情况

| # | 需求（文档出处） | 完成情况 | 证据 |
|---|---|---|---|
| 1 | §5.5 通信与计算分列 | 完成：19 入口探针、独立事件流、p2p 单列 | 3.4-②（FlagScale 真跑）、4.2（8B：730/370） |
| 2 | §5.5 按 rank 归集 | 完成：每 rank 独立 store/报表 | 3.4-②两行对称 |
| 3 | §5.5 区分等待与带宽 | 完成（分级）：wait 有效；带宽按厂商标注 | 4.4（aligned/vendor-decomposed/P800 无内核事件） |
| 4 | §5.5 同并行度对比 | 完成：并行度入档、跨并行度禁比 | 4.5 + 3.4-④（同 2 rank 对照） |
| 5 | §5.2 每芯片 profiler 调通 | PPU ✅ / 910C ✅ / P800 部分（设备内核事件厂商缺） | 三平台交付报告 |
| 6 | §5.3 kernel 名映射 | 类别归一（pccl/nccl/xccl/aten 前缀族）；语义映射仍人工/厂商 | launcher.md |
| 7 | §5.1 只发现不定位根因 | 同边界：给身份与占比，根因交算子/编译层 | 交付报告 §二 |
| 8 | §5.6 移交清单中"多卡通信与计算分解" | 由我们交付（即 1–4 条） | 本文 |
| 9 | 单卡阶段拆解/微基准（他们的脚本） | 不重复造：op_micro 可做同类微基准（910C 已实跑）；阶段拆解由他们脚本做 | 4.10 |

---

## 第六部分 附录：九个模型的运行与集成

> 通用 docker 模板（PPU；镜像 `1ba7ead15378`）：
> ```bash
> docker run --rm --shm-size=16g \
>   --device=/dev/alixpu_ctl --device=/dev/alixpu \
>   --device=/dev/alixpu_ppu<A> --device=/dev/alixpu_ppu<B> \
>   -v <宿主机 FlagPerf>:/workspace/FlagPerf \
>   -v <数据目录>:/data -v <模型目录>:/models:ro \
>   -w /workspace/FlagPerf \
>   -e PYTHONPATH=/workspace/FlagPerf/inference \
>   -e FLAGPROF_SCRATCH=/data/scratch \
>   -e FLAGPROF_TRAIN_OBSERVE=/data/store-<名字> \   # 不开采集就删这行
>   1ba7ead15378 bash -c "<下面的引擎命令>"
> ```
> PPU 镜像：`flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64`（ID `1ba7ead15378`）。
> P800 镜像：`flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64`
> （ID `cac5ddae5f62`），XPU 设备（`--device=/dev/xpu3` 等）、`--network=host`。
> 910C 镜像：`flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64`
> （ID `62bcaae3f839`），`--privileged`、davinci 设备。

### 6.1 BAAI/bge-m3（embedding）

- FlagScale 真跑（本文第三部分）：`train_retrieval.py --config-file <yaml>`，
  `parallel: ddp`、`torch_dtype: bf16`。
- 我们的引擎跑法（套上 docker 模板）：
  ```bash
  python3 -B training/run.py --task embedding --model /models/bge-m3 \
      --inputs /data/emb-groups.jsonl --optimizer adamw --steps 3 \
      --device cuda:0 --outputs /data/out/bgem3
  ```
- FSDP2：`--sharding fsdp` + `torchrun --nproc-per-node=2`（本次实测
  `fsdp_wrapping=per-block+root`）。

### 6.2 Qwen/Qwen3-Reranker-8B（rerank）

```bash
# 必须 FSDP2（FP32 训练态约 128GB）
torchrun --standalone --nproc-per-node=2 training/run.py \
    --task rerank --model /models/Qwen3-Reranker-8B \
    --package /pkgs/qwen3-reranker-causal --rerank-score causal_yesno \
    --optimizer adamw --steps 2 --mode performance --batch-size 1 \
    --ddp --backend nccl --device cuda --sharding fsdp --outputs /data/out/8b
```
- 实测：PPU 2 卡 10 步（本文 4.8 的 399 张量对照即此）；P800 2/4/16 卡；
  910C 4 卡。
- 集成到你们循环：同 3.2，FSDP2 分块包装用你们自己的 36 层发现。

### 6.3 openai/clip-vit-large-patch14（多模态）

- 我们的训练入口不支持（没有视觉/图文任务族）；FlagScale 仓里有 CLIP 配置，
  属他们自己的实现路径，不在本包范围内。接入我们的采集需要先有文本侧模型
  循环（双塔前向可以包相位与通信探针，但训练任务是新开发项）。
- 结论：**不支持，需要新任务族**，不要在交付里含糊。

### 6.4 DIN（CTR）

```bash
python3 -B training/run.py --task ctr --ctr-arch din --package /pkgs/ctr-train-1024 \
    --optimizer adamw --steps 4 --device cuda:0 --outputs /data/out/din
# 多卡（实测至 16 卡/100 步）
torchrun --standalone --nproc-per-node=4 training/run.py --task ctr --ctr-arch din \
    --package /pkgs/ctr-train-1024 --optimizer adamw --steps 10 --mode performance \
    --ddp --backend nccl --device cuda --outputs /data/out/din-ddp4
```

### 6.5 DeepFM / DCN-Mix / DSSM / LightGCN / SASRec

同上，换 `--ctr-arch`：`deepfm` / `dcnmix` / `dssm` / `lightgcn` / `sasrec`。
- 本次补测：dcnmix 的 stop/resume 与连续跑逐位一致；sasrec 双卡 DDP 通过；
  din 双卡 FSDP 通过（`fsdp_wrapping=root-only`，CTR 模型无块结构属正常）。
- 多卡建议：要通信画像就用 FSDP2（DDP 的 Reducer 不可见）。

### 6.6 通用注意事项

1. `--device` 带索引（`cuda:0`/`npu:0`）；910C 加 `--backend hccl`。
2. 容器内可见设备数必须等于 `--nproc-per-node`。
3. store 放本地 ext4/xfs；网络盘/tmpfs 会被拒绝。
4. 两套跑法产物格式一致：引擎 `training/run.py` 与适配器外部循环都写同一
   种 store/报表。
5. 每次跑完最少核对三样：`manifest.status=complete`、相段计数=步数、
   comm 计数符合并行策略。
