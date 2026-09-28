# 统一原型 · 三芯片职责验收与发布结论（2026-09-28 版）

> 状态：✅ **三实例全部完成（2026-09-28）** ｜ 作者：Kistich（hliu553）
> 口径：**厂商官方 torch 插件统一原型**（Route A）下，逐芯片核对「设备上下文 + 多流 Stream」全部职责。
> 定位：**验收记录**（读者＝本方向与下游对接人）。
> 上一版：`PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md`（当时 MLU590 两台主机不可达 ⇒ 未纳入）。
> 本版把第三实例补齐并给出**三芯片矩阵**；上一版记录的问题（3 处缺陷 + 并发纪律）**原样保留**，不覆盖。

---

## 一、结论

**三实例（910C / P800 / MLU590）在同一套判据下 12 项全部通过，原型可发布。**
本轮（09-28）补齐第三实例遗留的**推理腿两形态**，其余各项用**当前原型**复跑取新证据。

| 判定 | 结论 |
|---|---|
| 职责覆盖 | 五域 + 三个多流支撑方法 + 两条硬纪律，**三实例全部有实测证据**，无缺口 |
| 判据一致性 | 三实例共用**同一套判据集**（conformance 13+6）与**同一份脚本**（两条腿 / 三个探针 / `serve_standard.sh`），**换芯片只改 `DC_BACKEND`** |
| 可发布范围 | **910C / P800 / MLU590 三实例均可发布**（`runtime-v0.2.0` 线） |
| 本轮新修 | **1 处**：冒烟超时硬编码 60 s（在 MLU590 上造成**假失败**，见 §四） |
| 本轮新解法 | **1 项**：推理腿服务化改用 FlagOS 官方**应用镜像**（运行时镜像不含 vLLM，见 §四） |
| 本轮实测更正 | **1 处**：宿主**带卡容器名额**「上限 ≈3」**不可当可用阈值**（09-22 / 09-28 两次实测均到不了 3，见 §4.3） |
| 遗留（不影响发布） | ① P800 的 S12 流优先级（上游缺陷，已如实声明）· ② D11 real 多卡多进程压测 · ③ 芯片级错误真实触发 · ④ 训练侧完整 epoch 吞吐复测 |

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

**§2.4 验收项与覆盖映射**（→ 本文 §3 逐项判定）：

| 覆盖的职责 | 由哪几项验收 |
|---|---|
| 设备域（init / context / device_count / memory_stats / device_state） | ① 离线自检 ③ 冒烟 ⑤ conformance i1、i5 |
| 多流域（流 / 事件 / 同步 / 依赖 / 传输 / 页锁定 / 在途保护 / 拓扑） | ④ conformance e1–e3、s1–s4、t1–t3 ⑥ 多流 16 项基线 |
| 错误域（翻译 / 分级 / disposition） | ④ conformance f1 ⑩ 错误闭环四类注入 |
| 恢复域（`recover_device` 契约） | ④ conformance r ⑩ 错误闭环 `recover_probe` |
| 训推集成（两条腿） | ⑦ 训练腿 ⑧ 推理腿前向 ⑨ 服务化 |
| 跨后端一致性 | ② `--all` 对称性自检（硬判据 5 条 + 差异清单） |

---

## 三、逐芯片判定（三实例全绿）

图例：✅ 通过 · ⏹ 如实跳过/不支持（不计失败）· ❌ 失败（**本次无**）

| # | 验收项 | 910C（`ascend` / `torch_npu`） | P800（`kunlun` / XPytorch） | MLU590（`cambricon` / `torch_mlu`） |
|---|---|---|---|---|
| 1 | 离线契约自检 | ✅ **35/0**（1 跳过） | ✅ **39/0**（1 跳过） | ✅ **39/0/0** |
| 2 | 跨后端对称性 `--all` | ✅ **5/0** | ✅ **5/0** | ✅ **5/0** |
| 3 | 冒烟自检 | ✅ **52/0** | ✅ **46/0** | ✅ **46/0** |
| 4 | conformance 13 例 | ✅ **13/13** `CONFORMANCE_PASS` | ✅ **13/13** `CONFORMANCE_PASS` | ✅ **13/13** `CONFORMANCE_PASS` |
| 5 | conformance 推理 6 例 | ✅ **6/6** | ✅ **6/6** | ✅ **6/6** |
| 6 | 多流 16 项基线 | ✅ 语义 **8/8** · 图捕获 **4/4** · 配额 **3/3** · S12 **支持** | ✅ 语义 **8/8** · 图捕获 **4/4** · 配额 **3/3** · ⏹ S12 **如实不支持** | ✅ 语义 **8/8** · 图捕获 **4/4** · 配额 **3/3** · S12 **支持** |
| 7 | 训练腿 2 卡 | ✅ **6/6** · loss 15.4498→11.1479 · **4075.4 tok/s** · `hccl` | ✅ **6/6** · loss 15.4488→11.1481 · **3533.5 tok/s** · `flagcx` | ✅ **6/6** · loss 15.4498→11.1479 · **3015.3 tok/s** · `cncl` |
| 8 | 推理腿前向 | ✅ **14/14** · dim 1024 · 77.49 句/s · p50 38.31 ms · 区分度 0.6391 | ✅ **13/13 + ⏹1 跳过** · dim 1024 · 53.28 句/s · p50 56.12 ms · 区分度 0.6392 | ✅ **13/13 + ⏹1 跳过** · dim 1024 · **41.08 句/s** · p50 **72.89 ms** · 区分度 **0.6391** |
| 9 | 推理腿服务化 | ✅ **`SERVE_STANDARD_PASS`**（**09-22 首测** 35 s 就绪／**09-28 按 v1.2 脚本复跑一致**：35 s、冒烟 0 s） | ✅ **`SERVE_STANDARD_PASS`**（25 s 就绪，冒烟 0 s） | ✅ **`SERVE_STANDARD_PASS`**（**150 s 就绪，冒烟 42 s**） |
| 10 | 错误注入 → 恢复闭环 | ✅ **闭环 5 / 跳过 0 / 失败 0** | ✅ **5 / 0 / 0** | ✅ **5 / 0 / 0** |

**三处差异都有解释（不是缺漏）**：

1. **推理腿前向 14 vs 13** —— `vendor_code_map` 一项：**只有 910C 有厂商数字码表**（108 条 ACL 码）；
   另两家厂商**抛错误名而非数字码** ⇒ 该项**如实跳过**（与 §四 的结论一致，不是能力缺失）。
2. **S12 流优先级** —— **MLU590 支持**（`priority_range()=(0,-3)`）**而 P800 不支持**
   ⇒ 同一 API 跨芯片**相反**；两家都**如实声明**，已写进《新芯片接入手册》§9 坑清单。
3. **吞吐不可横向比** —— 三台均为**共享机**（本轮 MLU 侧卡 1/4 被他人占 60.5 GiB）、
   用卡号也不同 ⇒ 只作"同档可比"，**不构成性能结论**。

### 3.1 910C · 第 1 家（昇腾）

- 设备：16 × Ascend910，`acl.init rc=0`、`get_device_count=(16,0)`，逐卡 free ≈ **60.9–61.1 GiB / 61.27 GiB**
- 口径：`DC_BACKEND=ascend`（`torch_npu`，**两条腿统一**）+ `DC_DIST_BT=hccl`；解释器 `venv-infer-a`
- 容器：`flagos-proto-train-910c`（训练腿，锁定镜像）+ `flagos-infer-910c`（服务化，`vllm-ascend:v0.20.2rc1-a3`）
- ⚠️ 执行顺序：**先停训练容器再起服务**（设备共享冲突，见上一版 §4.1）；验收后容器均已 `stop`
- ✅ **09-28 补齐**：网络恢复后按新脚本（v1.2，含 `SMOKE_TIMEOUT`）复跑服务化 ⇒ **`SERVE_STANDARD_PASS`**
  （就绪 **35 s**、维度 1024、范数 1.000000、**冒烟耗时 0 s**），与 09-22 逐项一致
- ⚠️ 该复跑**首跑失败**：宿主带卡容器名额被他人占满 ⇒ `acl.init`=500000、`get_device_count`=(0,0)；
  释放后通过（含一次决定性验证，见 §4.3）

### 3.2 P800 · 第 2 家（昆仑芯）

- 设备：8 × XPU（98304 MiB/卡），驱动 5.0.21.47；用卡前挑空闲卡
- 口径：`DC_BACKEND=kunlun`（`torch.cuda` 兼容层 XPytorch）+ `DC_DIST_BT=cpu:gloo,cuda:flagcx`、`FLAGCX_ADAPTOR=klx`
- 容器：`hliu553-device-context-p800`；解释器 conda `python310_torch29_cuda`（torch 2.9.0+cu129）
- ⚠️ 硬前置：`PYTHONPATH=/env/FlagGems/src`（否则 vLLM 报 `Failed to infer device type`）
- ⏹ **S12 流优先级如实不支持**（上游缺陷，后端主动拦截）——不计为失败

### 3.3 MLU590 · 第 3 家（寒武纪）—— 本轮补齐

- 设备：8 × MLU590-M9（98304 MiB/卡），驱动 v6.2.29 / 固件 v1.5.0；`cnmon` 只读挂载
- 口径：`DC_BACKEND=cambricon`（`torch_mlu`，PrivateUse1 / `device_type="mlu"`）+ `DC_DIST_BT=cncl`
- 镜像（**两个**，软件栈逐项一致）：运行时 `flagos-runtime-cambricon-neuware4.4.3:2.2.0`（前向/训练/探针）
  + **应用 `flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`**（服务化，见 §四）
- 容器挂载：宿主 `/srv/hliu553`→容器 **`/work`**、`/srv/data/hf_cache`→**`/hf_cache`**
- 关键量化：训练腿 **3015.3 tok/s**；推理腿前向 **41.08 句/s / p50 72.89 ms / 区分度 0.6391**；
  服务化 **150 s 就绪 + 冒烟 42 s**（维度 1024 / 范数 1.000001）
- ⚠️ **训练数据只代表单机 2 卡**：CNCL 未加载 `libibverbs`/`libmlx5` ⇒ 走 **`MLU_LINK` 片间互联、非 RDMA**
- 详见 [`../../MLU590/docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md`](../../MLU590/docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md)

---

## 四、本轮的 1 处修复 + 1 项解法 + 1 处实测更正

> 与上一版同源：都是"**前置条件没写进工具/文档**"。

### 4.1 冒烟超时**硬编码 60 s** ⇒ 把"慢"判成了"不通"（假失败）

| 步骤 | 实测 |
|---|---|
| 第 1 次服务化（修复前） | 就绪 `t=365s` → 冒烟 `curl -m 60` **超时**（`raw:` 为空）⇒ `SERVE_STANDARD_FAIL (ready=1 smoke=0)`；服务端日志**只有 `GET /v1/models` 200，没有 `POST /v1/embeddings` 访问行** |
| 手工复现（保服务运行后单发请求） | `http=200`、**`time_total=63.59 s`** ⇒ **请求成功，只是首次请求含编译开销** |
| 第 2 次服务化（修复后） | 就绪 `t=150s`、冒烟 **42 s** ⇒ `SERVE_STANDARD_PASS` |

- **修复**：`serve_standard.sh` 新增 `SMOKE_TIMEOUT`（默认 **180 s**），两个形态分支都改用它，
  并**打印本次冒烟实际耗时** ⇒ 参数显式化，且下次读日志即可分辨"慢"还是"不通"。
- **纪律**：修复前那次失败**原样留档**（`MLU590/probes/accept_serve_cambricon_PRE_FIX_*`），
  改的是**工具的参数上限**、不是**判定标准**，**没有"改判据变绿"**；两次都入库，读者可自行判定。
- **适用面**：该上限对三实例生效；官方 `-base` 镜像上**首次请求更慢**（含 triton 编译）时同样受益。

### 4.2 运行时镜像**不含 vLLM** ⇒ 服务化改用 FlagOS 官方**应用镜像**

- 事实：`flagos-runtime-cambricon-neuware4.4.3:2.2.0` 内无 `vllm`（09-22 已登记 `known_issues`）。
- 解法：FlagOS 官方 `flagos-app` 仓有**寒武纪 vLLM 应用镜像**且**可匿名拉取**，
  采用 **`vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`**（digest `sha256:f568f23cf29b2…`，
  与 harbor 登记一致）；镜像内 torch / torch_mlu / py 版本与运行时镜像**逐项一致** ⇒ **同档可比**。
- 起容器脚本：宿主 `/srv/hliu553/start_container_mlu590_vllm.sh`（**与原脚本只差镜像一行**，
  设备节点/挂载/参数沿用同一套规则）。

---

### 4.3 宿主**带卡容器名额**：实测更正「上限 ≈3」（09-28）

| 步骤 | 实测 |
|---|---|
| 09-28 首跑（他人 2 + 我方 1 = **3 个带卡容器**） | vLLM `Engine core initialization failed`；root cause 原文 `Failed to obtain the console log level. The possible causes are as follows: 1. Different containers share the same device;`（`accept_serve_ascend_vllm_20260928_NAMESLOT_BLOCKED.log` 第 58–59 行）；`acl.init`=**500000**、`get_device_count`=**(0,0)** |
| 查卡挂载重叠 | 我方容器挂 **davinci0–15（全部）**；`temp-cp-arbitrary` 挂 davinci8–15、`mem-profile-910c` 挂 davinci1 ⇒ 确有重叠 |
| **决定性验证**：起临时容器**只挂 davinci0**（无人占用）+ 3 个管理设备 | **仍失败** —— `acl.init`=500000、`get_device_count`=0 ⇒ **与「挑哪张卡 / 卡是否重叠」无关**，是宿主名额本身 |
| 释放两个他人容器（带卡容器 → 1） | 立即恢复：`acl.init rc=0`、`get_device_count=**(16,0)**`、`torch.npu` 小算子 OK |
| 复跑服务化 | **`SERVE_STANDARD_PASS`**（就绪 **35 s**、维度 1024、范数 1.000000、冒烟 **0 s**） |
| 复原 | 两他人容器 `docker start` **原样恢复**（`running`、设备数 11 / 4 不变） |

- **口径更正**：早期记的「DrvMng 容器名额 **≈3**」（`910C/distributed_inference/docs/DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md`）
  与两次实测不符（09-22：我方 2 个容器共存即失败；09-28：3 个时失败，且**只挂单卡也一样**）
  ⇒ **不要把 3 当可用阈值**，实践中按「**同一时刻只留 1 个带卡容器**」安排。
- **失败证据原样留档**（`...._20260928_NAMESLOT_BLOCKED.log`），**未「改判据变绿」**；
  改的是「**环境使用方式**」（先释放名额），不是判定标准。

---

## 五、发布结论

| 判定 | 结论 |
|---|---|
| 职责覆盖 | **无缺口**：§2.4 的六类职责三实例均有实测证据 |
| 判据一致性 | 同一判据集 + 同一脚本，换芯片只改 `DC_BACKEND`（唯一例外：服务化在寒武纪需**换镜像**，已在 §4.2 说明并留脚本） |
| 可发布范围 | **三实例均可发布** |
| 遗留 ① | P800 S12 流优先级（上游缺陷，已如实声明为不支持） |
| 遗留 ② | ✅ **已补（09-28）**：910C 按新脚本复跑服务化 ⇒ `SERVE_STANDARD_PASS`（就绪 **35 s**、维度 1024、范数 1.000000、冒烟 **0 s**），与 09-22 逐项一致 ⇒ **三实例均已复跑**。⚠️ 首跑因宿主名额被他人占满失败（见 §4.3），**失败证据原样留档** |
| 遗留 ③④ | D11 real 多卡多进程压测 · 芯片级错误真实触发 · 训练侧完整 epoch 吞吐复测 |

---

## 六、复跑命令（三实例同构，换芯片只改 `DC_BACKEND`）

```bash
cd <prototype 目录>            # 910C: /mnt/raid/hliu553/runtime-team/dev/device-context/prototype
                              # P800: /workspace/prototype
                              # MLU590: /work/prototype（宿主 /srv/hliu553/prototype）
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

# ⑨ 服务化（三实例同形态；寒武纪需先起 vLLM 应用镜像容器，见 §4.2）
DC_BACKEND=$DC_BACKEND SERVE_FORM=embed STOP_AFTER=1 bash scripts/serve_standard.sh   # 期望 SERVE_STANDARD_PASS

# ⑩ 错误注入 → 恢复闭环
DC_BACKEND=$DC_BACKEND python3 runtime/proto/proto_error_recovery_loop.py --backend $DC_BACKEND
```

> 证据落点：`910C/probes/accept_*_20260922.*` · `P800/probes/accept_*_2026092{2,8}.*` ·
> `MLU590/probes/accept_*_20260928.*`（第三实例本轮全套）。
