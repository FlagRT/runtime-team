# 壁仞 BR_0：集合通信后端名探测（2026-11-11）

> 采集时刻 2026-11-11 16:35–16:51 CST ｜ 机器 `wz-server008` ｜ 容器 `dc-biren-hliu553`（卡 `card_5` / `card_7`）
> 证据目录：`BR0/probes/biren_dist_20261111_out/`（含 MANIFEST 与 md5）

## 0 结论

| 项 | 值 | 依据 |
|---|---|---|
| **集合通信后端名** | **`sccl`** | `torch_br/__init__.py:179`：`Backend.register_backend("sccl", create_fn, extended_api=True, devices=["supa","cpu"])` |
| 后端实现 | `torch_br._C._distributed_c10d.ProcessGroupSCCL`（C++，注册于 `libtorch_br.so`） | 同上 + `nm -D` 见 `c10d_br::ProcessGroupSCCL` |
| **版本** | **SCCL `10.8.0`** | `libtorch_br.so` 符号 `c10d_br::getScclVersion`（ctypes 直呼，见 §3） |
| 功能判据 | **2-rank `all_reduce` PASS**（supa 双卡，值精确匹配） | `BIREN_DIST_ALLREDUCE_PASS`，rc=0 |
| 错名行为 | **报错、无静默回落** | `bccl` → `AssertionError: Unknown backend type bccl` |
| 训练腿用法 | `DC_DIST_BT=sccl`（或 `init_process_group(backend="sccl")`） | — |

按本方向纪律「集合通信后端名必须探测、未探测宁可报错不给兜底」，本项**已完成探测并实测**，训练腿可以直接用 `sccl`，不需要任何兜底。

## 1 探测方法（三层）

### 1.1 注册层（谁的注册表说了算）

候选名逐个问 `dist.is_backend_available()`（16 个常见厂商名）：

```
gloo  True      ← torch 内置
nccl  False     hccl False    cncl False   bccl False   bcl  False
supa  False     biren False   br  False    rccl False   accl False
pccl  False     xccl False    xpu False    mlu  False   npu False
```

⚠️ **`sccl` 不在我最初的候选表里**——猜名是猜不到的。权威来源是 grep `register_backend`：

```python
# torch_br/__init__.py:179（import torch_br 即触发注册）
torch.distributed.Backend.register_backend("sccl", create_fn, extended_api=True, devices=["supa", "cpu"])
```

注册条件：`torch_br/distributed/__init__.py:59` 的 `is_available()`（查 `torch_br._C._c10d_supa_init`）
为真才注册——所以**必须先 `import torch_br`**，否则 torch 根本不认识 `sccl`（与「拼厂商设备串前先触碰设备」同族的懒加载纪律）。

### 1.2 功能层（2-rank 真 all_reduce）

```
BACKEND_NAME=sccl (torch_br/__init__.py:179, devices=[supa,cpu])
GOT     = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
EXPECT  = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
MATCH   = True
BIREN_DIST_ALLREDUCE_PASS     rc=0
```

用法要点（与 MLU590 的 `DC_DIST_BT=cncl` 同构，BR_0 用 `DC_DIST_BT=sccl`）：
`torch_br.supa.set_device(rank)` → `init_process_group(backend="sccl", init_method="tcp://…")` → supa 张量 `all_reduce`。

### 1.3 错名行为（无静默回落）

```
init_process_group(backend="bccl", …) → AssertionError: Unknown backend type bccl
```

torch 对未注册名**就地断言失败**，不会回落 gloo —— 这是本方向要的语义（「未探测宁可报错」在平台侧天然成立）。
注意 `gloo` 本身可用（`is_backend_available("gloo")=True`），但那是显式选择，不是静默回落。

## 2 三条边界（如实登记，不并入结论）

1. **`all_reduce` 拒绝 CPU 张量**：`RuntimeError: Tensors must be SUPA and dense`
   （栈在 `torch_br/distributed/distributed_c10d.py:1325`）。注册声明 `devices=["supa","cpu"]`，
   但集合算子只吃 supa 张量 ⇒ **实用面 = supa-only**。CPU 侧收集请自行 `.cpu()` 后处理。
2. **world=1 单卡也通**（`SUPA_W1_OK`，值原样）——单卡训练腿不会被后端挡住。
3. **SCCL 版本号 Python 面不暴露**（`_distributed_c10d` 只导出 `ProcessGroupSCCL`/`SCCLOptions`）；
   `10.8.0` 是经 `libtorch_br.so` 的 `c10d_br::getScclVersion` 符号 ctypes 直呼读到的（方法见 `scripts/sccl_ver6.py`：
   先 `import torch_br` 让依赖全就位，再 `CDLL` 同一 `.so` 复用句柄）。

## 3 首版翻车记录（保留作反证）

首版脚本在 `mp.spawn` 的 worker 里写 `os.environ["BIREN_VISIBLE_DEVICES"]="0,1"` ⇒ **三段全部 124 超时**，
连「错名」段都不报错——因为卡点根本不在集合通信，而在**进程内改容器级注入变量导致设备上下文挂住**。
分步打印（`biren_dist_dbg.sh`）定位：两个 worker 都 entered、都 import 成功、都 `before init` 后无声挂住；
去掉那行后全通。

**教训**：`BIREN_VISIBLE_DEVICES` 是**容器级**变量（`-e` 注入），进程内改它不是「没效果」而是「挂住」；
选卡交给容器参数，进程内只用 `torch_br.supa.set_device(rank)`。

## 4 复跑

```bash
# 宿主侧（Biren-1）
bash BR0/probes/biren_dist_20261111_out/scripts/biren_dist_recon.sh    # 注册层侦察
bash BR0/probes/biren_dist_20261111_out/scripts/biren_dist_final.sh    # 主判据：all_reduce + 版本
# 期望：SCCL_VERSION= 10.8.0 / MATCH= True / BIREN_DIST_PROBE_DONE / rc=0
# 错名对照期望：AssertionError: Unknown backend type bccl
```
