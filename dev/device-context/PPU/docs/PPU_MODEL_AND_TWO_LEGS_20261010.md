# PPU（平头哥 真武 ZW810E）模型资产与两条腿验证（2026-10-10 · 第 4 家实例）

> 承接《PPU 后端接入报告》（同目录 `PPU_BACKEND_ONBOARDING_20261010.md`）里「§6 未完成项 2/3」——
> 那两项的**共同前置是模型资产**。本文记录：模型从哪来、怎么验、两条腿结果、以及本轮暴露的 2 个新问题。
> 证据目录：`../probes/model_and_legs_20261010_out/`（30 个文件，124 KB）。

## 0. 一句话结论

**模型资产已就位并完成交叉校验，训练腿与推理腿的判据均全通过** ——
训练腿 **`TRAIN_LEG_PASS 6/6`**（2 卡 **50 步**，loss 15.4498→**11.1530**，**5167 tok/s** 两卡合计）、
推理腿 **`INFER_LEG_PASS 13/13`**（dim 1024，135.77 句/s，p50 22.15 ms，同/异主题区分度 0.6391）。
**但两个进程都在判据通过之后、解释器退出阶段段错误（SIGSEGV）** —— 已定位到与
`runtime.create_stream()` 相关，逐条排除过程见 §4；**判据本身不受影响**（结果 JSON 已落盘）。

## 1. 模型从哪来：三条路的实测选择

| 路 | 可行性实测 | 判定 |
|---|---|---|
| **A. 从另三台机搬**（910C / P800 / MLU590） | 910C、MLU590 本机**不可达**；P800 可达，其副本在宿主 `/data1/dinghaisong/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…` | 可做，但要经本机中转 ≈1.2 GB，慢且引入中转误差 |
| **B. 从 HuggingFace 官方下载** | 本机 `https://huggingface.co/` **不通**（curl 20 s 超时，HTTP 000） | ❌ |
| **C. 经 hf-mirror 下载**（`HF_ENDPOINT=https://hf-mirror.com`） | 容器内实测 **HTTP 200 / 0.30 s**；12 文件 1.2 GB **25 s 下完** | ✅ **采用** |

**关键判据（为什么 C 可以替代 A）**：hf-mirror 返回的仓库 revision =
**`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`**，与 P800 / MLU590 上
`models--Qwen--Qwen3-Embedding-0.6B/snapshots/<hash>` 的**目录名逐字相同**
⇒ 拿到的是**同一个 revision**，跨实例可比性成立。
（P800 那份正是三实例验收在用的副本，故用它做 sha256 基准最有说服力。）

- ⚠️ **模型是 `Qwen/Qwen3-Embedding-0.6B`（Embedding 版），不是通用 `Qwen3-0.6B`**。
  三台机验收统一用 embedding 版（`../../prototype/scripts/serve_standard.sh` 头部注释
  「2026-09-22 补：验收模型统一为 Qwen3-Embedding-0.6B」），换通用版会破坏跨实例可比性。
- 落盘位置（容器内可见）：宿主 `/bmcp_lvm_fs/hliu553/models/Qwen3-Embedding-0.6B`
  ↔ 容器 **`/workspace/models/Qwen3-Embedding-0.6B`**（目录 `models/` 由容器内 root 创建，
  所以属主是 root；`/data` 已 98% 满，故未放 `/data`）。
- HF API 清单快照：`…/models_manifest/Qwen3-Embedding-0.6B.hf_api.json`（12 个文件）。

## 2. 步 0：模型完整性验证（接入手册「先验模型完整性」）

### 2.1 sha256 与 P800 逐字对照（6 个关键文件）

| 文件 | 本机 sha256（前 16 位） | P800 同文件 | 判定 |
|---|---|---|---|
| `model.safetensors`（1 191 586 416 B） | `0437e45c94563b09` | `0437e45c94563b09` | ✅ 一致 |
| `tokenizer.json` | `def76fb086971c78` | `def76fb086971c78` | ✅ 一致 |
| `vocab.json` | `ca10d7e9fb3ed185` | `ca10d7e9fb3ed185` | ✅ 一致 |
| `merges.txt` | `8831e4f1a0444713` | `8831e4f1a0444713` | ✅ 一致 |
| `config.json` | `b5bf1f51fc45be47` | `b5bf1f51fc45be47` | ✅ 一致 |
| `tokenizer_config.json` | `253153d0738ceb4c` | `253153d0738ceb4c` | ✅ 一致 |

完整清单 + 全部文件 sha256 见 `…/model_manifest_sha256.txt`。
（P800 那份缺 `generation_config.json` / `.gitattributes`，本机 12 个文件与 HF API 清单**数量一致**。）

> **为什么必须做这一步**：910C 上发生过共享模型资产**静默损坏**（同路径 09-30 好、10-08 坏，
> 而 mtime/ctime 未变），当时两条腿同时 `loss=nan`。**「能跑完 50 步」不能读成「模型没问题」。**

### 2.2 可加载 + 数值健全（`step0_loadcheck.log`）

| 判据 | 读数 |
|---|---|
| `AutoConfig` | `architectures=Qwen3ForCausalLM` · `hidden_size=1024` · `num_hidden_layers=28` · `torch_dtype=bfloat16` |
| Tokenizer | `vocab_size=151643`；`input_ids` 形状 `[2, 6]` |
| CPU 前向 | `last_hidden [2,6,1024]` · **`all_finite=True`** · `absmax=21.718658` · `mean=-0.035541` |
| PPU 前向（`cuda:0`） | `last_hidden [2,6,1024]` · **`all_finite=True`** · `absmax=21.718630` · `mean=-0.035541` |
| CPU↔PPU 可比性 | `mean_abs_diff=4.34e-05`（float32 正常量级，非逐位比较）· 形状一致 |
| **判定** | **`VERDICT=PASS`**（无 NaN / 无 Inf） |

环境：torch `2.10.0` · transformers `4.57.0` · `cuda.is_available()=True` · `device_count=2` · `PPU-ZW810E`。

## 3. 两条腿结果（模型路径就位后）

### 3.1 训练腿（2 卡，`TRAIN_LEG_PASS 6/6`）

| 项 | 读数 |
|---|---|
| 后端 / 进程组 | `backend=ppu` · `dist=nccl` |
| 判据 | **6/6 全通过**（`train_leg_result_rank0.json` / `rank1.json`） |
| loss | **15.4498 → 11.1530**（**50 步**，BATCH 4 × SEQ 128 —— ⚠️ 首轮误用 20 步，与手册规定的 50 步不符，**2026-10-10 自审后已重跑更正**） |
| 吞吐 | rank0 `5167.3` · rank1 `5150.9` tok/s（两卡合计） |
| 集合通信对照 | `all_reduce`（期望 3.0）/ `all_gather`（`[0,1]`）/ P2P（rank0↔rank1 收发校验）三组均通过 |
| 启动方式 | `python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py`（**cwd = 原型根 `prototype/`**） |

**独立复现佐证**：另写一份**不 import 原型**的脚本（`e5_noruntime_full_loop.py`）复刻同一训练循环，
得到**同轨迹的损失曲线**（该复刻脚本为 20 步口径，末值 11.2841；50 步正式轮末值 **11.1530**，与三家 910C/P800/MLU590 的 11.1479/11.1481/11.1479 同轨迹）
⇒ 设备侧计算是真实的，判据不是空转。

### 3.2 推理腿（单卡，`INFER_LEG_PASS 13/13`）

| 项 | 读数 |
|---|---|
| 判据 | **13/13 全通过**（`proto_infer_leg_result.json`） |
| 向量质量 | dim `1024` · **同主题相似度 0.7821 / 异主题 0.1430** · 区分度 **0.6391** · 范数 1.0 · 无 NaN |
| 性能 | **135.77 句/s** · p50 **22.15 ms** · avg 22.1 ms |
| 设备上下文 | `count=1` · `total_mb=98304 used_mb=171 free_mb=98132` |
| 错误分级 | `L2_PARAM → raise`（`graded_by=message_hint`）；**如实跳过 `vendor_code_map`**（本栈错误只给错误名、无数值码） |
| 启动方式 | `CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu python3 runtime/proto/proto_infer_leg.py` |

## 4. ⚠️ 本轮新问题（两条腿都触发，必须记录）

### 4.1 现象：判据通过后、解释器退出阶段段错误

| 腿 | 判据 | 进程退出 |
|---|---|---|
| 训练腿（torchrun，2 rank） | `TRAIN_LEG_PASS 6/6` | 两个 rank 均 **`exitcode -11`（SIGSEGV）**；torchrun 因此报 `ChildFailedError`，外层 RC=1 |
| 推理腿（单进程） | `INFER_LEG_PASS 13/13` | **rc=139**（128+11） |

崩溃形态（`PYTHONFAULTHANDLER=1`）：

```
Fatal Python error: Segmentation fault
Current thread 0x… (most recent call first):
  Garbage-collecting          ← 训练腿（两 rank 皆然）
  <no Python frame>           ← 推理腿
```

崩溃**发生在**：结果 JSON 已落盘、`print` 判据行已输出、NCCL communicator 已
`Destroy COMPLETE`、`dist.destroy_process_group()` 已返回之后。
**判据与产物不受影响**，但进程退出码不干净 —— 属「看起来通过」家族的**退出路径**变体，故按纪律记录而非放过。

### 4.2 已做的对照矩阵（每组 1 次运行，`…/segv/*.txt`）

| 编号 | 对照内容 | 结果 |
|---|---|---|
| E11 | **逐字副本**（原样复制 `proto_train_leg.py` 到别处运行） | **复现**（RC=1） |
| E12a | 副本去掉 `init_process_group(..., timeout=...)` | 仍复现 |
| E12b | 副本去掉 `_preflight_env_check()` | 仍复现 |
| **E13b** | 副本**只把 `st = runtime.create_stream()` 改成 `st = None`** | **干净 RC=0** ✅ |
| E13a | 副本整块移除原型运行时调用 | 干净 RC=0（判据退化为 5/6，预期） |
| E14a | 同位置同生命周期改用**原生 `torch.cuda.Stream()`** | **干净 RC=0** ✅ |
| E14b | 同位置同生命周期用我方 `runtime.create_stream()` | **复现** |
| E1 | 纯 torch + nccl 最小两进程（无模型无原型） | 干净 RC=0 |
| E2 | 仅原型（`use`/`device_count`/`priority_range`/`context_query`），无模型 | 干净 RC=0 |
| E3 | 仅模型（tokenizer + AutoModel 前向反传），无原型 | 干净 RC=0 |
| E4 | `all_reduce` + `all_gather` + P2P（无模型无原型） | 干净 RC=0 |
| E5 | **完整训练循环**（CausalLM + AdamW + clip_grad + 20 步梯度 all_reduce），无原型 | 干净 RC=0 |
| E6 | 完整循环 + 裸 `ctypes.CDLL("libcuda.so.1")`（含 `RTLD_NODELETE` 变体） | 干净 RC=0 |
| E7 | 完整循环 + 原型，三档深度（仅 import / +`device_count` / +`synchronize`） | 三档均干净 RC=0 |
| E15 | 显式 `del st; gc.collect()`（把释放提前到可观测处） | 回收 **0 个对象**（**无引用环**），此后**仍在退出期崩** |
| E16 | 换「建进程组」与「建流」的先后顺序 | 两序**都崩** |

### 4.3 目前的定位与**明确未定论**的部分

- **支持性结论**：崩溃与「**我方统一流包装**」相关 —— 去掉那一行即干净（E13b），
  换成原生 `torch.cuda.Stream()` 也干净（E14a）；且**两条腿（训练/推理）都崩**，与是否用 torchrun/NCCL 无关。
- **已排除**：CDLL 被 dlclose（E6）· 引用环被循环 GC 回收（E15）· 建流与建组的先后（E16）·
  `init_pg` 的 timeout / 前置检查（E12）· 模型、NCCL、P2P、训练循环本身（E1–E5、E7）。
- **尚未定论**：精确触发条件**还没收敛到不变量**（E10 那种「模块级 + 末尾 `runtime.synchronize`」的组合曾干净、
  E14/E16 的等价形态却复现 ⇒ 存在未识别的第二因素）。
  容器内**无 gdb / eu-stack**，无法取 C 栈；要收口需要带调试器的镜像或厂商协助。
- **纪律记录**：**不得**把本条写成「PPU 平台缺陷」或「我方 bug」——
  证据只支持「与我方 `create_stream()` 强相关」，**归属未定**。

## 5. 容器级阻塞项（已固化修法）

| 现象 | 根因（实测） | 修法 |
|---|---|---|
| `torchrun --standalone` **静默挂 5 分钟**、无任何子进程输出，然后报 `The client socket has timed out after 300000ms while trying to connect to (ai-server, 44927)`；其前身是反复的 `[c10d] The IPv6 network addresses of (ai-server, 44927) cannot be retrieved (gai error: -2 - Name or service not known)` | `--network host` 时 **docker 不代管 /etc/hosts**（不加容器自己的主机名），而宿主主机名 `ai-server` 也不在 DNS（`hostname -f` 直接 `Name or service not known`，`getent hosts ai-server` 为空）⇒ `--standalone` 的 c10d rendezvous 端点用的是主机名 ⇒ TCPStore 自连超时 | 启动脚本加 **`--add-host=ai-server:127.0.0.1`**（已写入 `/bmcp_lvm_fs/hliu553/hliu553-dc-dev.run.sh`）；重建后 `getent hosts ai-server → 127.0.0.1`，最小两进程脚本由 **RC=124 / 零输出** 变为 **RC=0 / 两 rank 各自正常** |

⚠️ **对照记忆**：同一容器里 `mp.spawn` + `init_method=tcp://127.0.0.1:<port>` **不受影响**
（早先的 2 卡 allreduce 探针正是这样跑通的）—— 所以这个坑**只在 torchrun / 主机名形式的 rdzv 端点**上暴露。

⚠️ 排查中顺手确认：**宿主 `/etc/hosts` 未被改动**（容器内 `/etc/hosts` 是私有副本，
与宿主 `device:inode` 不同；宿主 mtime 仍是 2018 年、内容无 `ai-server` 行）。

## 6. 未完成项（按时序，非缺口掩盖）

| # | 项 | 阻塞 / 前置 |
|---|---|---|
| 1 | **推理腿服务化**（`serve_standard.sh`） | 需给脚本补 `ppu` 分支（现仅 ascend/kunlun/cambricon）；容器内已带 vLLM `0.19.0` |
| 2 | 退出期段错误收口 | 需带调试器的镜像（容器内无 gdb）或厂商侧协助 |
| 3 | 多流 Stream 16 项基线 | 无需新前置，可直接跑 |
| 4 | 错误闭环四类注入 | 需先确认注入面（本栈无数值错误码） |
| 5 | 职责响应审计（78 项口径） | 需两条腿就绪后跑（**两腿判据已就绪**） |
| 6 | 多卡多进程故障恢复压测（A2） | 本后端**未声明 `recovery_real`** ⇒ 如实跳过 |

## 7. 复现命令（容器内，容器内路径为准）

```bash
# 步 0：模型完整性 + 可加载
python3 -u probes_ppu/step0b_loadcheck.py /workspace/models/Qwen3-Embedding-0.6B

# 训练腿（cwd 必须是原型根；DC_ROOT 必须显式给）
cd /workspace/runtime-team/dev/device-context/prototype
CUDA_VISIBLE_DEVICES=0,1 DC_BACKEND=ppu DC_DIST_BT=nccl \
DC_ROOT=/workspace/runtime-team/dev/device-context/prototype \
DC_MODEL=/workspace/models/Qwen3-Embedding-0.6B \
DC_OUT_DIR=/workspace/probes_ppu/out_train_leg \
MAX_STEPS=20 BATCH=4 SEQ=128 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py

# 推理腿
CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu \
DC_ROOT=/workspace/runtime-team/dev/device-context/prototype \
DC_MODEL=/workspace/models/Qwen3-Embedding-0.6B \
DC_OUT_DIR=/workspace/probes_ppu/out_infer_leg DC_ROUNDS=5 \
python3 -u runtime/proto/proto_infer_leg.py
```

（模型下载命令：容器内 `HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-Embedding-0.6B --local-dir /workspace/models/Qwen3-Embedding-0.6B`）
