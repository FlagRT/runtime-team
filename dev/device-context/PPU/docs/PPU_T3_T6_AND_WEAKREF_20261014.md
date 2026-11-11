# PPU（平头哥）· T3 服务化 · T4 错误闭环 · T6 调度效果 与一处跨后端缺陷的定位与修复

> 日期：2026-10-14 ｜ 实例：**第四实例 PPU**（容器 `hliu553-dc-dev`）
> 承接：执行单 **T3 / T4 / T6**（`prototype/docs/TODO_NEXT_WEEK_20261012.md`）
> 证据目录：`../probes/t3_serve_20261014_out/` · `../probes/t456_20261014_out/` · `../probes/t5_20261014_out/` · `../probes/repro_release_recreate_20261014/`

---

## 0. 结论摘要

| 项 | 判据 | 结果 | 证据 |
|---|---|---|---|
| **T3** 服务化 | `SERVE_STANDARD_PASS (ready=1 smoke=1)` | ✅ **通过**（就绪 35 s；维度 1024、范数 1.000000） | `../probes/t3_serve_20261014_out/form/serve_standard.log` |
| **T3** 消费方 | `SERVE_LEG_PASS` | ✅ **10/10**（103.15 句/s、p50 28.8 ms、区分度 0.4118） | `../probes/t3_serve_20261014_out/consumer/proto_infer_serve_result.json` |
| **T4** 错误闭环 | `ERROR_RECOVERY_LOOP_PASS` | ✅ **闭环 5 / 跳过 0 / 失败 0**（与三家同口径） | `../probes/t456_20261014_out/t4_errorloop_ppu.json` |
| **T6** 调度效果 | `STREAM_PRIORITY_SCHED_EFFECT_PASS` | ✅ 通过；**`effect_observed=False`（如实记录：未观测到实质效果）** | `../probes/t456_20261014_out/t6_sched_effect_ppu_fixed.log` |
| ⭐ **途中发现并修复** | 跨后端缺陷（本层误拦） | ✅ 已修 + 修前/修后交替对照 + 非空转判据 | `../probes/repro_release_recreate_20261014/{before,after}/` |

**两个必须一起读的边界（不掩盖）**：

1. **T4 与 T6 的进程在判据通过、结果落盘之后，于解释器退出期段错误**（`rc=139`）——与执行单 **T11** 同源；
   **判据本身全绿**，但"判据通过 ≠ 进程干净退出"（本方向纪律）。
2. **T3 能跑起来自身依赖一个非镜像自带的前置**：镜像缺 `triton`（详见 §1），本轮按"拉齐"原则补齐。

---

## 1. 执行环境与两个前置缺口（**先说前置，再谈结果**）

### 1.1 缺口 A：镜像**不含 `triton`** ⇒ vLLM 引擎起不来

首跑 `serve_standard.sh` 的失败形态（容器内 `$DC_OUT_DIR/form/vllm_serve.log`）：

```
(EngineCore) File ".../vllm/model_executor/layers/rotary_embedding/common.py", line 138
             from flash_attn.ops.triton.rotary import apply_rotary
ModuleNotFoundError: No module named 'triton'
```

- 该导入由 vLLM 的 `if find_spec("flash_attn") is not None:` 门控 —— 本容器**装了 `flash_attn` 却没装 `triton`** ⇒ 必然触发。
- **同镜像的兄弟容器对照**（关键证据，说明这不是镜像内容而是"每容器手动补"）：

  | 容器 | 镜像 | `import triton` |
  |---|---|---|
  | `flagos-retrieval` | **与本容器同一 tag** | ✅ `3.6.0`（带 `backends/ppu`、`FLAGTREE_BACKEND=ppu`） |
  | `flagos-adrec` | **与本容器同一 tag** | ❌ `ModuleNotFoundError` |
  | `hliu553-dc-dev`（本容器） | 同上 | ❌ 同缺口 |
  | `zhenghaojia-…-vllm-0_24_0` | `flagos-dev/vllm-plugin-fl:v0.24.0-thead-ci` | ✅ 3.6.0（`triton-3.5.0+gitcc5446cf.dist-info`） |

- **补法（拉齐，非自定）**：共享 wheel 目录 **`/bmcp_lvm_fs/A/data/whl/`** 内含
  `triton/`（1.1 GB / 379 文件）+ `flagtree/` + `flagtree-0.7.0+ppu.gite223f42c.dist-info`。
  与 `flagos-retrieval` 已装的那份**逐字节相同**（`triton/__init__.py` sha `df309217…`、`_flagtree_backend.py` sha `489f24dd…`）⇒ 按 `docker cp` 复制进 site-packages。
- ⚠️ **容器内 `pip` 不可用**（不是网络问题）：其 index 指向不可达的内网地址
  `aiext-pypi.mirrors.aliyuncs.com` ⇒ `pip index versions triton` 全部 `ConnectTimeout`，
  `pip install` 报 `No matching distribution`。故**只能走"复制现成包"这条路**。
- ✅ **非空转验证**（"能 import ≠ 能用"）：`probes/triton_smoke.py` 真编一个 triton kernel 并在 PPU 上执行：

  ```
  triton 3.6.0 | torch 2.10.0 | dev0 = PPU-ZW810E
  [编译+执行] 耗时 1.45s  最大误差 0.000e+00  一致=True
  [边界尺寸 n=3000] 一致=True
  TRITON_PPU_SMOKE_PASS
  ```

**⇒ 服务化的可复现前置（本实例）**：容器内 `site-packages` 需有 **FlagTree 的 `triton`(3.6.0, `backends/ppu`) + `flagtree`**；
来源 `/bmcp_lvm_fs/A/data/whl/`；**该前置不在镜像内**（属接入期应记入实例手册的环境事实）。

### 1.2 缺口 B：卡 0 被他人占用 ⇒ 必须选空闲卡（并说明默认参数为何会直接失败）

- 容器挂 `alixpu_ppu0` + `alixpu_ppu1`；实测 **dev0(ppu0) 已用 30.7 GiB**（多容器共享同一物理卡），dev1(ppu1) **96.0 GiB 全空**。
- 若在 dev0 上起，vLLM 默认 `--gpu-memory-utilization 0.9` 需 86.4 GiB > 65.3 GiB 可用 ⇒ **直接 `ValueError`**（不是"服务起不来"，是参数与现况不匹配）。
- 处置：**取空闲卡 dev1**（`DEV=1`），保持默认利用率口径不变（便于跨实例对照）。

### 1.3 踩坑（已记入本次脚本注释）

- ⚠️ **宿主/容器路径映射写错（本方向第 8 次同族）**：v1 脚本把日志目录写成
  `HO=/bmcp_lvm_fs/hliu553/dc_t3_20261014`，而容器内 `DC_OUT_DIR=/workspace/probes_ppu/dc_t3_20261014`
  的宿主侧是 `/bmcp_lvm_fs/hliu553/**probes_ppu**/dc_t3_20261014` ⇒ 判据全 grep 在空目录上，
  屏幕只有 `rc=0`、verdict 一片空白。**规则：`HO` 必须由 `O` 按挂载映射推出来，不得手写。**

---

## 2. T3 服务化结果

### 2.1 形态验证（`serve_standard.sh`，`SERVE_FORM=embed · DEV=1 · EAGER=1 · STOP_AFTER=1`）

```
  vllm = /usr/local/bin/vllm ｜ python3 = 3.12.3
--- 用卡前 ---  [torch.cuda:0] free=96.00GiB / total=96.00GiB
############ 启动 vllm serve ############
  ✅ 服务就绪 t=35s
  维度=1024 范数=1.000000
  （冒烟耗时 5s；超时上限 SMOKE_TIMEOUT=600s）
[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)
non-default args: {'model': '/workspace/models/Qwen3-Embedding-0.6B', 'runner': 'pooling',
                   'convert': 'embed', 'max_model_len': 4096, 'enforce_eager': True,
                   'served_model_name': ['qwen3-embedding-0.6b']}
```

### 2.2 消费方（`proto_infer_serve.py`，本层最后一个 `create_stream()` 消费方）

```
[1] 设备上下文: 后端=ppu 设备数=2
[1b] 同卡跨流计算: 3.0 ✅
[2] 服务就绪: /v1/models 200, model=qwen3-embedding-0.6b
[3] 向量: 维度=1024 有限=True 范数=1.000000
[4] 语义: 同主题 0.7000 / 异主题 0.2883 / 区分度 0.4118
[5] 吞吐: 103.15 句/s | p50 28.8ms | max 30.7ms
[6] 错误注入: HTTP 400 | 超长输入被拒 → L2_PARAM / raise
[7] 业务继续: ✅
=== SERVE_LEG_PASS | 10/10 ===
```

### 2.3 为跑通所做的**唯一**源码改动：`serve_standard.sh` 补 `ppu` 分支

要点（均在脚本内注释）：

| 项 | PPU 分支的做法 | 与其他实例的差别 |
|---|---|---|
| 设备 API | **路径 B**：走 `torch.cuda` 命名空间，vLLM 自识别 `NvmlCudaPlatform` | **不需要**任何厂商插件/平台环境变量（不踩 910C 的 `VLLM_PLUGINS`、P800 的 `PYTHONPATH/FlagGems`） |
| 选卡 | `CUDA_VISIBLE_DEVICES`（同 kunlun） | 与 ascend(`ASCEND_RT_*`)/cambricon(`MLU_*`) 不同 |
| 默认模型 | `/workspace/models/Qwen3-Embedding-0.6B`（**容器内**视角） | 沿用"容器内路径"纪律 |
| 形态 | `SKIP_EXTRA=0`（embedding：`--runner pooling --convert embed`） | 与 kunlun/cambricon 同口径 |
| NCCL 前置 | 由容器启动带入（`NCCL_SOCKET_IFNAME=bond0`；镜像默认写死 `eth0` 而本机真网卡是 `bond0`） | TP=1 单卡不吃这条，但**多卡路径必须** |

---

## 3. T4 错误闭环（四类注入）

```
后端: ppu | 设备数: 1
[基线] 业务正常: 786432.0
[shape_mismatch(L2_PARAM 期望)]  L2_PARAM / raise  / action=raise_to_caller（参数类，重试无意义） / 业务继续=True
[oom(L1_RESOURCE 期望)]          L1_RESOURCE / retry / action=retry_ok                        / 业务继续=True
[stream_timeout(L3_EXECUTION 期望)] L3_EXECUTION / replay / 业务继续=True / 进程内捕获 TimeoutError，进程存活；
                                    异常消息内无厂商码（本后端为「超时上报」语义，不产生设备侧数字码）；已等待设备归零
[l4_by_code(码表内他厂码·诚实降级负向测试)] L3_EXECUTION / replay / action=recover_probe=ok + replay_ok
                                    / 业务继续=True / 期望核对=✅

=== ERROR_RECOVERY_LOOP_PASS | 闭环 5 / 跳过 0 / 失败 0 ===
```

- 第 4 类走的是**诚实降级负向测试**（PPU 未声明 `error_map`，与 kunlun/cambricon 同路径）：
  注入昇腾码 `507015` ⇒ 必须 `mapped=False` / `error_code=None`，**不得**声称码表命中 —— 实测 ✅。
- ⚠️ **`rc=139`**：判据通过、JSON 落盘之后进程在**解释器退出期段错误**（同 §1 与 T11 一族）。**结论不受影响，但必须与判据一起看**。

---

## 4. ⭐ 本轮抓到并修复的跨后端缺陷（**平台侧弱引用悬垂 ⇒ 本层误拦**）

### 4.1 现象（T6 首跑的失败形态）

```
[FAIL] V1_gate_actually_gated     RuntimeError: 该流已被 `release_stream()` 释放（厂商句柄 0x564330547310）⇒ 不可继续使用
[FAIL] G1_gate_high_priority_wins 同上
[FAIL] G4_naive_submit_order_control / G5 / G7 / G8  同上（0x564330547310 或 0x564330562f50）
```

- 探针 `run_group()` **每组自建自释放**（契约 §1.10 规则 3 要求），却在**后续组的第一次使用点**被判"已释放"；
- 6 个场景**全部拿不到读数** ⇒ T6 主判据整体失效（而 910C/P800 为 `NOT_APPLICABLE`、MLU590 曾 `PASS`，故这是 PPU 侧新暴露）。

### 4.2 定位（三步，全部可复现）

**① 稳定复现器**（`../probes/repro_release_recreate_20261014/probe_release_recreate_loop.py`）：
按"建 → 用 → 释放"节奏循环 60 轮 ⇒ **5 轮内即命中 3 次**（`REPRODUCED`）。命中瞬间现场：

```
native: id=140245613396384 type=ExternalStream handle=0x5573bec4e220
_released_streams[id]: 句柄=0x5573bec4e220 持有器类型=ReferenceType
   holder() is native = True            ← 为**旧对象**建的 weakref 解析出了**新对象**
_owned_streams: 有条目（本层认为拥有）
```

**② 2×2 判据实验**（唯一变量 = 是否插入 `gc.collect()`，各 8 轮；`probe_weakref_2x2.py`）：

| 条件 | 同 id | `w_old()` 已置死 | **`w_old()` 跟随新对象** | 使用点失败 |
|---|---|---|---|---|
| 不 gc | 8/8 | 0/8 | **8/8** | **8/8** |
| `gc.collect()` | 6/8 | 2/8 | **6/8** | 7/8 |

⇒ **平台侧 `weakref` 悬垂**：PPU 的 torch 流对象被回收后其 weakref **未随对象死亡置死**；
当地址被后续对象复用时，该 weakref **解析出那个新对象**。

**③ 结论修正（对 10-10 的旧记录）**：10-10 曾以"最小实验显示 weakref 被正确置死"**否证**悬垂假说 ——
本轮证明那次抽样只落在**少数不跟随的路径**（2/8）上，**悬垂成立**。旧注释已按"改结论必须连注释一起改"同步。

### 4.3 本层的暴露面

`_released_streams` 以 `id()` 为键、以"弱引用解析结果 `is` 当前对象"为身份复核：

- 悬垂弱引用 ⇒ 复核**被绕过**（解析出的正是"当前对象"）；
- 且新旧对象**类型相同**（同为 `ExternalStream`）⇒ 10-10 加的**类型复核挡不住**；
- ⇒ **新建的合法流被判"已释放"**，使用点硬失败。
- ⚠️ 这**否证**了此前"该表误判只是**误拦**（保守、可恢复）"的判断 —— 实测误拦是**硬失败**。

### 4.4 修法（**机制无关**，不依赖弱引用语义）

| # | 改动（`prototype/runtime/backends/base.py`） | 作用 |
|---|---|---|
| ① **主防线** | `_register_owned_stream()` 内新增 `self._reg("_released_streams").pop(id(native_stream), None)` —— **新建登记使同 id 的旧「已释放」记录失效** | 语义精确：能落在该 id 上的旧记录必然属于一条**已死**对象（否则其地址不可能被复用）；且**不削弱**原保护——被释放的流若仍被调用方持有，其地址不会被复用，这条清除不会命中它 |
| ② 纵深 | `_released_streams` 的值由 2 元组升为 **3 元组**（句柄 + 持有器 + **释放时类型名**）；`check_stream_usable()` 增加**类型复核** | 挡"跨类型地址复用"造成的误拦（同类型复用由 ① 负责） |

### 4.5 验证（**交替对照 + 非空转**）

- **修前/修后对照**（同一机器、同一时段；做法：备份修好版 → 把那行换成 `pass` → 跑 → 恢复 → 再跑）：

  | 复现器 | 修前（`before/`） | 修后（`after/`） |
  |---|---|---|
  | 循环 60 轮 | **3 次命中 / 5 轮** | **0 次命中 / 60 轮** |
  | 8 轮使用点 | 2 可用 / **6 失败** | **8 可用 / 0 失败** |
  | 2×2 使用失败 | **8/8 · 7/8** | **0/8 · 0/8** |

- ⭐ **机制未变、本层免疫**：修后 2×2 里 `w() is new=True` **仍是 8/8**（平台的悬垂照旧），
  但**使用点失败降为 0** ⇒ 证明修法是**对本层暴露面的直接免疫**，不是"把平台问题藏起来"。
- **离线判据（台账第 38 条，新增）**：`../../prototype/scripts/backend_offline_check.py` 的 `H)` 项
  「悬垂弱引用：新建登记后，使用点**不得**被旧「已释放」记录误拦」。
  **非空转**：把 ① 那行换成 `pass` ⇒ 该项立即 `[FAIL] … 误拦：RuntimeError`（114 通过 / 1 失败）；
  恢复后 `115 通过 / 0 失败`。
- **四家离线回归**：`ascend 90/0/1` · `kunlun 111/0/1` · `cambricon 100/0/0` · `ppu 115/0/1`（0 失败）。

### 4.6 边界（如实登记）

1. 修法针对**可修的点**（本层不再依赖弱引用语义），**不修平台**：PPU 的 weakref 悬垂依旧存在（修后 8/8 仍跟随）；
2. 「为何不 gc 时 8/8 跟随、gc 后减为 6/8」的**平台内部细节未深究** —— 本层不再依赖该细节，故不再下探；
3. 该改动落**共享层**（`base.py`）⇒ **破坏面覆盖在本轮之前已完成**（三家真机 78 项复跑：`73/0/5` · `68/0/10` · `62/0/16`，均零回归）；
   ⚠️ 但**本轮这次改动本身**同样需要三家复跑确认 —— 已列入 `TODO_NEXT_WEEK_20261012.md`（与 T1 同法，一个窗口）。

---

## 5. T6 调度效果结论（简版；16 项口径见 `THEAD_PPU_STREAM_BASELINE_16_20261014.md`）

- `STREAM_PRIORITY_SCHED_EFFECT_PASS`，`applicable=True · instrument_valid=True · **effect_observed=False**`；
- 仪器有效性有**正对照**背书（G2：同优先级、工作量差 2× ⇒ 8/8 先完成、中位差 34.07 ms）；
- **gated（两流同时就绪）**：胜率 6/8（重跑 8/8）**刚好达线**，但**中位差 0.48–0.51 ms ≪ 阈值 8.7 ms** ⇒ **无实质效果**；
- **ungated（一条已运行、另一条才提交）2×2**：G4/G8/G7 三种情形一致指向**「提交顺序主导」**⇒ 汇总**如实 False**；
- 与 MLU590（同为 `(0,-3)` 多档）结论**不同**（MLU590 在"有排队争用"场景测到 5.3–8.1 ms）⇒ **如实记录差异，不为对齐改写**；PPU 上该场景本轮**未构造出可判别差异**（列入遗留）。

---

## 6. 遗留与下一步

| # | 项 | 说明 |
|---|---|---|
| 1 | **本轮 base.py 改动需三家复跑** | 同 T1 方法（真机 78 项）；窗口需求 1 次/台 |
| 2 | **T3 前置应记入实例手册** | 镜像缺 `triton`（来源 `/bmcp_lvm_fs/A/data/whl/`）+ 容器 pip index 不可达 |
| 3 | **T11 退出期段错误** | 本轮 T4/T6 均复现 `rc=139`（判据已通过）；需带调试器镜像或厂商协助 |
| 4 | PPU「有排队争用」场景的调度效果 | 需另设计"同一条流串行排队"对照（本轮不下结论） |
| 5 | T7 MLU590 多卡 TP / 非 eager 服务化 · T9 证据索引 · T10 命名对齐 | 仍待做 |

## 7. 复现命令

```bash
# T3（宿主侧；容器内 cwd=原型根）
docker exec -e DC_BACKEND=ppu -e SERVE_FORM=embed -e DEV=1 -e EAGER=1 -e STOP_AFTER=1 \
  -e PORT=8100 -e MODEL=/workspace/models/Qwen3-Embedding-0.6B \
  -e DC_OUT_DIR=/workspace/probes_ppu/dc_t3_out hliu553-dc-dev \
  bash -lc "cd /workspace/runtime-team/dev/device-context/prototype && bash scripts/serve_standard.sh"
#   ⇒ 判据读 $DC_OUT_DIR/serve_standard.log（脚本内部 exec 重定向；外层拿不到）

# T4 / T5 / T6：见 ../probes/run_t456_ppu_20261014.sh 与 ../probes/run_t5_extra_20261014.sh
# triton 非空转验证：../probes/triton_smoke.py
# 缺陷复现与修前/修后对照：../probes/repro_release_recreate_20261014/{probe_*.py,stage_evidence2.sh}
```
