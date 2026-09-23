# 统一原型 · 三芯片职责验收与发布结论

> 状态：✅ **验收完成（2026-09-22）** ｜ 作者：Kistich（hliu553）
> 口径：**厂商官方 torch 插件统一原型**（Route A）下，逐芯片核对「设备上下文 + 多流 Stream」全部职责。
> 定位：**验收记录**（读者＝本方向与下游对接人）。职责本体见 `INTERFACE_CONTRACT_DC_20260908.md`；
> 多流基线见 `RUNTIME_DC_STREAM_PLAN_20260907.md` §5；逐条复核清单见 `VERIFICATION_MANIFEST_20260920.md`。

---

## 一、结论

**原型（`runtime-v0.2.0` 线）在 910C 与 P800 两个实例上完成了全部职责，可发布。**
本轮 10 项判定在三芯片实例上按**同一套判据、同一份脚本**重跑：910C **10/10 通过**、P800 **10/10 通过**；
**寒武纪 MLU590 两台主机当日 SSH 无响应**（同刻 910C/P800 正常），其接入阶段的 6 项已完成，
**推理腿与服务化 2 项未做**，故第 3 家**本轮不参与发布判定**（详见 §3.3）。

---

## 二、职责定义（本次验收的判据来源）

**§2.1 职责边界**（`RUNTIME_DC_STREAM_PLAN_20260907.md` §1）——我们**做**与**不做**：

| 归属 | 模块 | 我们是否实现 |
|---|---|---|
| 本层 | Backend 插件机制 + 注册表 | ✅ |
| 本层 | 设备上下文（设备 / 内存 / 执行句柄） | ✅ |
| 本层 | 多流 Stream（流 / 事件 / 同步语义） | ✅ |
| 本层 | 错误码翻译与分级 | ✅ |
| 本层 | 设备状态恢复 | ✅ |
| 本层 | 通信（集合通信适配） | ❌（通信方向） |
| 上层 | 模型转换器 / 算子库 | ❌（模型、算子方向） |
| 下层 | 多机多卡分布式训练 / 推理 | ❌（但我们提供底座与验证脚本） |

**§2.2 接口面**（`INTERFACE_CONTRACT_DC_20260908.md`）：
**13 个 `@abstractmethod`**（设备 4 · 多流 7 · 错误 1 · 恢复 1）
+ 三个多流支撑方法（`stream_context` / `synchronize_stream`（有界）/ `wait_event_host`（有界））
+ 可选能力由 `supports()` **如实声明**（不支持就不声明）。

**§2.3 两条硬纪律**（违反即数据错乱）：
① 跨流传缓冲必须 `record_stream`；② 错误隔离分层——API 级失败只影响该次调用，
**芯片级故障影响该设备全部流，流级重试无效，必须走设备级 `recover_device`**。

**§2.4 验收项**（→ 本文 §3 逐项判定）：① 离线性契约自检 ② 跨后端对称性 ③ 冒烟自检
④ conformance 13 例 ⑤ conformance 推理 6 例 ⑥ 多流 16 项基线 ⑦ 训练腿 2 卡 ⑧ 推理腿前向
⑨ 推理腿服务化 ⑩ 错误注入 → 恢复闭环。

---

## 三、逐芯片判定

图例：✅ 通过 · ⏹ 如实跳过/不支持（不计失败）· ⏳ 未做 · ❌ 失败（**本次无**）

| # | 验收项 | 910C（`ascend` / `torch_npu`） | P800（`kunlun` / XPytorch） | MLU590（`cambricon` / `torch_mlu`） |
|---|---|---|---|---|
| 1 | 离线契约自检 | ✅ **35/0**（1 跳过） | ✅ **39/0**（1 跳过） | ✅ 39/0（本机离线，历史） |
| 2 | 跨后端对称性 `--all` | ✅ **5/0** | ✅ **5/0** | ✅ 5/0（历史） |
| 3 | 冒烟自检 | ✅ **52/0** | ✅ **46/0** | ✅ 42/0（历史） |
| 4 | conformance 13 例 | ✅ **13/13** `CONFORMANCE_PASS` | ✅ **13/13** `CONFORMANCE_PASS` | ✅ 13/13（历史） |
| 5 | conformance 推理 6 例 | ✅ **6/6** `CONFORMANCE_PASS` | ✅ **6/6** `CONFORMANCE_PASS` | ✅ 6/6（历史） |
| 6 | 多流 16 项基线 | ✅ 语义 **8/8** · 图捕获 **4/4** · 配额 **3/3** · S12 支持 | ✅ 语义 **8/8** · 图捕获 **4/4** · 配额 **3/3** · ⏹ S12 不支持（如实） | ✅ 15 通过 / 1 不适用（历史） |
| 7 | 训练腿 2 卡 | ✅ **`TRAIN_LEG_PASS 6/6`** · loss 15.4498→11.1479 · **4075.4 tok/s** · `dist=hccl` | ✅ **`TRAIN_LEG_PASS 6/6`** · loss 15.4488→11.1481 · **3533.5 tok/s** · `dist=cpu:gloo,cuda:flagcx` | ✅ 6/6 · 2957.8 tok/s（历史） |
| 8 | 推理腿前向 | ✅ **`INFER_LEG_PASS 14/14`** · dim 1024 · 77.49 句/s · p50 38.31 ms | ✅ **`INFER_LEG_PASS 13/13 + 1 跳过`** · dim 1024 · 53.28 句/s · p50 56.12 ms | ⏳ 未做 |
| 9 | 推理腿服务化 | ✅ **`SERVE_STANDARD_PASS`** | ✅ **`SERVE_STANDARD_PASS`** | ⏳ 未做 |
| 10 | 错误注入 → 恢复闭环 | ✅ **`ERROR_RECOVERY_LOOP_PASS` 闭环 5 / 跳过 0 / 失败 0** | ✅ **`ERROR_RECOVERY_LOOP_PASS` 5 / 0 / 0** | ✅ 5/0/0（历史） |

### 3.1 910C · 第 1 家（昇腾）

- 设备：16 × Ascend910，`acl.init rc=0`、`get_device_count=(16,0)`、逐卡 free ≈ **60.9–61.1 GiB / 61.27 GiB**（全空闲）
- 口径：`DC_BACKEND=ascend`（`torch_npu`）+ `DC_DIST_BT=hccl`；解释器 `venv-infer-a`（torch 2.11.0 / torch_npu 2.11.0）
- 容器：**`flagos-proto-train-910c`**（训练腿，锁定镜像，挂 `/mnt/raid/hliu553`）+ **`flagos-infer-910c`**（服务化，`vllm-ascend:v0.20.2rc1-a3`）
- 关键量化：训练腿 **4075.4 tok/s**；推理腿前向 **77.49 句/s / p50 38.31 ms / 区分度 0.6391**；
  服务化 **35s 就绪** + embedding 冒烟 **维度 1024 / 范数 1.000000**（`SERVE_FORM=embed`，`DEV=4`，在 `flagos-infer-910c` 内）
- ⚠️ 执行顺序：**先停训练容器再起服务**（设备共享冲突，见 §4.1）；验收后两容器均已 `stop`，设备名额已释放

### 3.2 P800 · 第 2 家（昆仑芯）

- 设备：8 × XPU（98304 MiB/卡），驱动 5.0.21.47；用卡前挑空闲卡（验收用 5/6/7）
- 口径：`DC_BACKEND=kunlun`（`torch.cuda` 兼容层 XPytorch）+ `DC_DIST_BT=cpu:gloo,cuda:flagcx`、`FLAGCX_ADAPTOR=klx`
- 容器：`hliu553-device-context-p800`；解释器 conda `python310_torch29_cuda`（torch 2.9.0+cu129）
- 关键量化：训练腿 **3533.5 tok/s**；推理腿前向 **53.28 句/s / p50 56.12 ms / 区分度 0.6392**；服务化 **25s 就绪** + embedding 冒烟通过
- ⏹ **S12 流优先级如实不支持**（上游缺陷，后端主动拦截，统一 API 返回 `None`）——不计为失败

### 3.3 MLU590 · 第 3 家（寒武纪）—— 本轮未连通，如实标注

- **两台主机（10.1.1.21 / 10.1.1.22）SSH 均超时**（`Operation timed out`，同刻 910C 与 P800 正常）
- 已完成（2026-09-22 稍早，历史证据）：环境打通 · 镜像定档 · `cambricon` backend 落地 ·
  离线自检 · smoke 42/0 · conformance **13/13 + 6/6** · 多流 16 项（15 通过 / 1 不适用）·
  训练腿 **6/6**（2957.8 tok/s）· 错误闭环 **5/0/0**
- **未做**：推理腿前向 · 服务化（前置已就绪）
- ⇒ **本轮不参与"三实例全部完成"的发布判定**；`cambricon` 后端的**代码层与判据层已就绪**，
  待主机可达后按同一套命令补齐 2 项即可并入。

---

## 四、本轮验收暴露并修复的 3 处缺陷

> 均为**同一根因家族的第三次暴露**：实现/文档在"看起来通过"的地方与真实前置条件不一致。

| # | 缺陷 | 性质 | 现象（实测） | 修复 |
|---|---|---|---|---|
| **16** | 三个后端无关探针取设备命名空间前**没有触碰设备** | 框架层（原型） | 容器关 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 时，`use()` 之后 `hasattr(torch,"npu")` 仍为 **False** ⇒ 模块级 `getattr(torch, DEV_API)` 直接 `AttributeError`，**三个探针同时崩** | `resolve_dev_api()` 内改为先 `backend.device_count()` 触碰设备再取 `device_type`（最小触碰，不写厂商模块名）。修后 **8/8 / 4/4 / 3/3 全绿** |
| **17** | 两条腿脚本的 **P800 用法示例**错两处 | 文档/可用性 | ① `DC_MODEL` 给了 HF **缓存根目录** ⇒ `ValueError: Unrecognized model in …`（须给到 `snapshots/<hash>`）；② 脚本路径少一级（`proto_train_leg.py` → 实为 `runtime/proto/proto_train_leg.py`）⇒ `No such file or directory`。**照文档抄即失败** | 两处示例改正，并把"为什么"写进注释（与 `serve_standard.sh` 里已有的同款提示对齐） |
| **18** | `serve_standard.sh` **不能起同形态服务** | 标准脚本 | `SKIP_EXTRA` 按后端分支写死（ascend=生成形态、kunlun/cambricon=embedding）⇒ 验收模型统一为 `Qwen3-Embedding-0.6B` 后，**910C 无法用同一份标准脚本起 embedding 服务**，横向比对缺一角；此前只能绕开标准脚本手工起 vLLM（违反"禁止各自维护启动脚本"） | 新增 `SERVE_FORM=embed\|generate` 覆盖开关，**留空＝各后端现状（零行为变更）**；`bash -n` 校验通过 |

**16、17 的共性教训**：凡是"前置条件"（厂商扩展已加载、路径给到快照层、形态与模型匹配），
**必须在脚本/文档里显式写出并加断言**，否则会以"看起来通过"或"跑不起来"的形式反复出现。

### 4.1 另一条实测确认的**并发纪律**（非缺陷，但本轮踩到）

**910C 上训练容器与推理容器不能同时持卡** —— 二者都挂全部 16 个 davinci 设备，
同时 Up 时后起的一方服务/引擎打不开设备：

```text
Failed to obtain the console log level. The possible causes are as follows:
1. Different containers share the same device;      ← 命中此条
...
terminate called after throwing an instance of 'std::logic_error'
RuntimeError: Engine core initialization failed.
```

**本条与"并发上限 3"是两件事**：本轮同时只有 **2 个**带卡容器（≤3，**名额没超**），
冲突来自**设备共享**而非名额。⇒ **纪律：910C 上的两条腿串行**——
训练容器 `docker stop` 释放设备后，再起推理容器；反之亦然。
（`910C/distributed_training/README.md` 已写过"与推理腿串行"，本次是**再次踩到并留下原文证据**。）

**顺带一条容器事实**：**训练容器自带 `vllm` 入口但缺包**（`/usr/local/python3.11.15/bin/vllm` 存在，
`import vllm` 报 `ModuleNotFoundError`）⇒ 服务化**必须用推理容器**
`flagos-infer-910c`（`vllm-ascend:v0.20.2rc1-a3`）；`serve_standard.sh` 的"找不到 vllm 就激活 conda"兜底对 910C 不适用。

---

## 五、发布结论

| 判定 | 结论 |
|---|---|
| 职责覆盖 | **五域 + 三个多流支撑方法 + 两条硬纪律**在 910C / P800 上**全部有实测证据**，无缺口 |
| 判据一致性 | 三实例共用**同一套判据集**（conformance 13+6）与**同一份脚本**（两条腿 / 探针 / `serve_standard.sh`），**换芯片只改 `DC_BACKEND`** |
| 可发布范围 | **910C 与 P800 可发布**（`runtime-v0.2.0` 线，本次 10/10 × 2） |
| 未纳入 | MLU590 —— 主机不可达，推理腿/服务化 2 项未做；**代码层与判据层已就绪**，补齐后并入 |
| 遗留（不影响发布） | ① P800 的 S12 流优先级（上游缺陷，已如实声明）· ② D11 real 多卡多进程压测 · ③ 芯片级错误真实触发 · ④ 训练侧完整 epoch 吞吐复测 |

---

## 六、复跑命令（三实例同构，换芯片只改 `DC_BACKEND`）

```bash
cd <prototype 目录>            # 910C: /mnt/raid/hliu553/runtime-team/dev/device-context/prototype
                              # P800: /workspace/prototype
export DC_BACKEND=ascend      # 或 kunlun / cambricon

# ①~③ 无设备/轻量
python3 scripts/backend_offline_check.py --backend $DC_BACKEND
python3 scripts/backend_offline_check.py --all
python3 runtime/smoke_runtime.py --backend $DC_BACKEND

# ④⑤ conformance
python3 runtime/conformance/runner.py --backend $DC_BACKEND
python3 runtime/conformance/runner.py --backend $DC_BACKEND --cases infer_cases

# ⑥ 多流 16 项
DC_BACKEND=$DC_BACKEND python3 probes/probe_stream_semantics_full.py --rounds 5
DC_BACKEND=$DC_BACKEND python3 probes/probe_graph_capture_stream_v2.py
DC_BACKEND=$DC_BACKEND python3 probes/probe_stream_quota.py

# ⑦ 训练腿（2 卡）
DC_BACKEND=$DC_BACKEND DC_DIST_BT=<hccl | cpu:gloo,cuda:flagcx | cncl> \
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py

# ⑧ 推理腿前向
DC_BACKEND=$DC_BACKEND python3 runtime/proto/proto_infer_leg.py

# ⑨ 服务化（三实例同形态）
DC_BACKEND=$DC_BACKEND SERVE_FORM=embed STOP_AFTER=1 bash scripts/serve_standard.sh   # 期望 SERVE_STANDARD_PASS

# ⑩ 错误注入 → 恢复闭环
DC_BACKEND=$DC_BACKEND python3 runtime/proto/proto_error_recovery_loop.py --backend $DC_BACKEND
```

> 证据落点：`910C/probes/accept_*_20260922.*`（910C）· `P800/probes/accept_*_20260922.*`（P800）。
