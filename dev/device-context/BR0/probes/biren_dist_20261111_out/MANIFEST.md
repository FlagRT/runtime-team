# 证据索引：壁仞 BR_0 集合通信后端名探测（2026-11-11）

> 采集时刻：2026-11-11 16:35–16:51 CST ｜ 载体：`wz-server008` 容器 `dc-biren-hliu553`（卡 `card_5`/`card_7`）
> 镜像：`birensupa-smartinfer-vllm:26.07.09-…-br1xx`（ID `sha256:081634a3…`）

## 结论一览

| 判据 | 结果 | 判定串 |
|---|---|---|
| 后端名（注册层） | **`sccl`** —— `torch_br/__init__.py:179` `Backend.register_backend("sccl", create_fn, extended_api=True, devices=["supa","cpu"])` | `is_backend_available("sccl")=True` |
| 后端版本 | **SCCL `10.8.0`**（`libtorch_br.so` 符号 `c10d_br::getScclVersion`，ctypes 直呼） | `getScclVersion()='10.8.0'` |
| 功能判据（2-rank all_reduce，supa 双卡） | **PASS**（值逐元素精确匹配） | `BIREN_DIST_ALLREDUCE_PASS`，`rc=0` |
| 错名必须报错 | **PASS**（无静默回落） | `bccl` → `AssertionError: Unknown backend type bccl`（单进程就地报错） |
| CPU 张量行为 | **如实登记：`all_reduce` 拒绝 CPU 张量**（`RuntimeError: Tensors must be SUPA and dense`）⇒ 实用面=supa-only | — |

## 目录

| 文件 | 内容 |
|---|---|
| `biren_dist_final_20261111_165043/dist_final.log` | **主判据**：2-rank all_reduce + SCCL 版本（md5 `2c508acef7d12d9368b5f8e4996a41d1`） |
| `biren_dist3_20261111_164519/dist3_sccl.log` | 功能判据首过（md5 `94f7274f7f5d0b738b4ee216f6cf2482`） |
| `biren_dist3_20261111_164519/dist3_cpu.log` | CPU 张量拒绝的完整栈（`torch_br/distributed/distributed_c10d.py:1325`） |
| `biren_dist_20261111_163650/` | 首版脚本（`mp.spawn` 内改 `BIREN_VISIBLE_DEVICES` 挂住 ⇒ 三段全超时）——**保留作反证**：教训=容器级变量不许进程内改 |
| `biren_dist4_20261111_164613/` | world=1 supa OK / world=1 CPU FAIL 对照 |

## 复跑脚本（`scripts/`）

| 脚本 | 作用 |
|---|---|
| `biren_dist_recon.sh` | 注册层侦察（`is_backend_available` × 16 候选名 + grep `register_backend` + 库实体） |
| `biren_dist_dbg.sh` | 错名就地报错 + `mp.spawn` 逐步打印（定位首版挂住根因） |
| `biren_dist_fn3.sh` | 修正版功能判据（2-rank supa all_reduce + CPU 行为） |
| `biren_dist_final.sh` | **汇总判据**：all_reduce + SCCL 版本一体（期望 `BIREN_DIST_PROBE_DONE`，rc=0） |
| `sccl_ver6.py` | 版本读取方法（`import torch_br` 后 `ctypes.CDLL` 复用已加载句柄直呼 `getScclVersion`） |
| `sccl_reg2.py` | 注册表口径（`Backend.SCCL` 属性、devices 声明） |

## 运行前置与期望读数

```bash
# 容器内（必须先 source 厂商环境脚本）
. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set
python3 -u /tmp/d_final.py        # 见 scripts/biren_dist_final.sh 内嵌
# 期望：SCCL_VERSION= 10.8.0 / MATCH= True / BIREN_DIST_PROBE_DONE / rc=0
# 错名对照：init_process_group(backend="bccl") → AssertionError: Unknown backend type bccl
```

> ⚠️ 教训（首版三段全超时的根因）：不要在 worker 进程内改 `os.environ["BIREN_VISIBLE_DEVICES"]`
> —— 该变量是**容器级注入**，进程内改会导致设备上下文挂住（连错名路径都超时而非报错）。
> 选卡交给容器 `-e BIREN_VISIBLE_DEVICES=...`，进程内只用 `torch_br.supa.set_device(rank)`。
