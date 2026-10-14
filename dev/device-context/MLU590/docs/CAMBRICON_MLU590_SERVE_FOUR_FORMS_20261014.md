# MLU590 服务化四形态补全（TP∈{1,2} × EAGER∈{1,0}）+ 本层消费方腿

> 日期：**2026-10-14** ｜ 对应执行单 `../../prototype/docs/TODO_NEXT_WEEK_20261012.md` **T7**
> 与审计 `../../prototype/docs/OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md` **§3 A3**
> 证据：`../probes/t7_serve_20261014_out/` ｜ 脚本：见 §7
> 前置容器：`dc-mlu590-vllm-hliu553`（**vLLM 应用镜像** `flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`，`--network host`，8 卡设备节点齐）
> ⚠️ 未取到证据的一律标「未验证」；本报告**不声称为性能结论**（见 §6）。

---

## 0 结论先行

| 项 | 结果 |
|---|---|
| **四形态矩阵** | ✅ **全部 `SERVE_STANDARD_PASS (ready=1 smoke=1)`** —— `TP=1/EAGER=1` · `TP=1/EAGER=0` · **`TP=2/EAGER=1`** · **`TP=2/EAGER=0`** |
| **形态真的生效（正证据）** | vLLM 自身 config dump 逐形态吻合：`enforce_eager=True/False` 与 `tensor_parallel_size=1/2`、`world_size=1/2`（见 §3 末列） |
| **与 910C 对称性** | 910C 于 09-29 补齐四形态 ⇒ 本轮补齐后**两台实例的多卡推理形态覆盖对称**（此前 MLU590 只有 `TP=1 + eager` 单形态） |
| **本层消费方腿** | ✅ **主腿（`TP=1/EAGER=1`）：`SERVE_LEG_PASS 10/10`**（维度 1024 · 范数 1.000001 · 区分度 0.4091）—— 与 10-08 同口径可比（§4） |
| ⚠️ **如实登记 #1** | **消费方客户端观测窗口硬编码 120 s 不足**：`TP=1/EAGER=1` 实测首请求 **85 s**（已压在窗口边缘）；`TP=2/EAGER=0` 首跑**第 2 个 embedding 请求超时**。已加显式开关 `SERVE_REQ_TIMEOUT`（**默认值不变**），并单独补跑定性（§4.2） |
| ⚠️ **如实登记 #2** | **停机清理漏杀 `VLLM::Worker_TP*`**（大写进程名不被大小写敏感的 `pkill -f "[v]llm"` 命中）⇒ 首跑后 cnmon 实测卡 0/1 **各占 89 GiB 未释放**；按 `VLLM::` 前缀清后**立即回落 94.6/94.8 GiB**。脚本已修（§5） |
| **边界（不声称）** | 未测 `TP>2`；**不给性能结论**（吞吐被首请求编译主导）；**CNCL RDMA 仍未验证**（§6）；停机期出现的 CNCL 心跳 `c10::Error` 属**停机顺序噪声**，不是本次判据失败（§6） |

---

## 1 为什么做

`910C` 已于 **2026-09-29** 补齐四形态（`TP=1/2 × EAGER=1/0` 全 PASS），而 MLU590 的服务化证据**只有 `TP=1` + eager 单形态**（`MLU590/probes/m1_serve_20261008.log`，`DEV=2`）。两家实例的**多卡推理形态覆盖不对称** ⇒ 任何"多卡可用性"的横向比较都缺少本家数据。本任务把这一格补齐。

---

## 2 环境与前置（都可复核）

| 项 | 实测值 |
|---|---|
| 机器 / 容器 | `Mlu-1`（`tza-0a06-ai01-em9`）· `dc-mlu590-vllm-hliu553`（**我方资产**，启动前 `Exited (137) 5 days`） |
| 网络模式 | **host**（⇒ 宿主 `curl 127.0.0.1:8100` 直连容器内服务，判据轮询才成立） |
| 栈 | vLLM **0.20.2** · torch **2.7.1** · torch_mlu **1.29.2** · `device_config=mlu` |
| 模型 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…/`（**容器内**解析 + `config.json` 存在性断言） |
| 选卡 | `TP=1` → `DEV=0`；`TP=2` → `DEV=0,1`（容器内 0 基） |
| 原型版本自证 | 容器内 `runtime/backends/base.py` sha256 前 16 位 = **`4dca3469d99ebf81`**（= 本机当前版本，含 T11 修法） |
| 服务形态 | `SERVE_FORM=embed` ⇒ `--runner pooling --convert embed`（三实例同口径） |
| 就绪/冒烟预算 | `READY_BUDGET=900` · `SMOKE_TIMEOUT=600`（历史教训：首条请求含编译，硬编码短超时会把**成功**判成失败） |

---

## 3 四形态矩阵（每形态**独立进程**，跑前先 `cleanup` 并核残留）

| 形态 | 就绪耗时 | 冒烟耗时 | 维度 | 范数 | verdict | 形态生效的正证据（vLLM config dump） |
|---|---|---|---|---|---|---|
| `TP=1/EAGER=1` | **175 s** | 45 s | 1024 | 1.000001 | ✅ `SERVE_STANDARD_PASS (ready=1 smoke=1)` | `enforce_eager=True` · `tensor_parallel_size=1` · `world_size=1` |
| `TP=1/EAGER=0` | **530 s** | 254 s | 1024 | 1.000001 | ✅ 同上 | `enforce_eager=False` · `tps=1`（`compilation_config.custom_ops` 由 `['all']` → `['none']`、`ir_enable_torch_wrap` 由 `False` → `True`） |
| **`TP=2/EAGER=1`** | **345 s** | 117 s | 1024 | 1.000001 | ✅ 同上 | `non-default args {… 'enforce_eager': True, 'tensor_parallel_size': 2}` · `tps=2` · `world_size=2` |
| **`TP=2/EAGER=0`** | **440 s** | 142 s | 1024 | 1.000001 | ✅ 同上 | `enforce_eager=False` · `tps=2` · `world_size=2` |

**读法（三态纪律）**：
- 「就绪」= `/v1/models` 返回 200；「冒烟」= 真发一条 embedding 并校验维度与范数 —— **只看就绪会掩盖"服务起来了但算不出"**，故两者都进 verdict。
- 每形态的 `vllm_serve.log` 都保留**非默认参数行**（`non-default args` / engine config），这是"形态真的进了设备"的**正证据**，不是推测。
- ⚠️ 两形态数字**不可直接比性能**：`EAGER=0` 走编译路径，就绪与冒烟耗时被**编译**主导（530 s vs 175 s），与本任务的判据无关（§6）。

**残留与用卡**：矩阵跑完后残留进程 = 0；卡释放至 **dev0 94.7 GiB / dev1 94.8 GiB**（98304 MiB 卡，基线量级）。

---

## 4 本层消费方腿（`runtime/proto/proto_infer_serve.py`）

### 4.1 主腿 `TP=1/EAGER=1` —— ✅ `SERVE_LEG_PASS 10/10`

| 检查项 | 本轮（DEV=0） | 10-08 历史（DEV=2，同形态） |
|---|---|---|
| `device_context` | ✅ 后端=cambricon 设备数=8 已绑定 device 0 | ✅ 同（设备数=8） |
| `stream_on_serving_device` | ✅ 服务运行同卡上跨流计算 = 3.0 | ✅ 同 |
| `service_ready` | ✅ `/v1/models` 200 · served model = qwen3-embedding-0.6b | ✅ 同 |
| `embedding_valid` | ✅ 维度=1024 有限=True 首条范数=1.000001 | ✅ 同 |
| `semantic_order` | ✅ 同主题 0.6975 > 异主题最小 0.2884 | ✅ 同 |
| `semantic_gap` | ✅ 区分度 **0.4091**（判据 ≥0.10） | ✅ 0.4091 |
| `throughput` | ⚠️ **0.17 句/s** · p50 667.7 ms | ⚠️ 0.09 句/s · p50 678.5 ms |
| `error_injection` | ✅ HTTP 400 → `L2_PARAM / raise` | ✅ 同 |
| `error_grading_consistency` | ✅ 期望 `L2_PARAM/raise`，实测一致 | ✅ 同 |
| `business_continues` | ✅ 注入后仍可正常请求、维度一致 | ✅ 同 |

⚠️ **可比性**：语义类读数（0.6975 / 0.2884 / 0.4091）与 10-08 **逐字相同**；**吞吐不可作结论** —— 两次都只有 5 轮、且首条请求含编译（本轮 max=**85269 ms**），量级由冷启动主导（0.09 vs 0.17 句/s 的差异不足以说明任何设备差异）。

### 4.2 对照腿 `TP=2/EAGER=0`：首跑失败 → 定性为**客户端观测窗口不足**

- **首跑形态**（`../probes/t7_serve_20261014_out/consumer_tp2_eager0/`）：服务 `/v1/models` = **200（就绪）**，但消费方在**第 2 个** embedding 请求（`SAMPLE+DIFF` 两条文本）上抛
  `TimeoutError: timed out` ⇒ 进程 rc=1、未落 JSON。
- **定性依据（三条并置，不是单一猜测）**：
  ① 同一形态的 `serve_standard.sh` 冒烟在 `SMOKE_TIMEOUT=600` 下**通过**（§3）⇒ 服务侧能算；
  ② `TP=1/EAGER=1` 主腿实测**首条请求 85269 ms**，已紧贴客户端硬编码 **120 s**；
  ③ 报错发生在**请求等待**上（`http_post` 的 socket 读超时），不是 HTTP 错误码、也不是服务进程消失。
- ⇒ 处置：给消费方加**显式**观测窗口开关 `SERVE_REQ_TIMEOUT`（默认值**保持 120 s 不变**，见 §5），
  并**只放宽客户端窗口**重跑一次定性（服务侧判据一字未改）。

**补跑结果（`SERVE_REQ_TIMEOUT=600`，其余一字未改）**：

| 项 | 结果 |
|---|---|
| 判据 | ✅ **`SERVE_LEG_PASS 10/10`**（`consumer_tp2_eager0_retry/proto_infer_serve_result.json`） |
| 关键读数 | 吞吐 0.09 句/s · p50 **5109.3 ms** · **max 87467.3 ms（≈87 s）** · 错误注入仍 `L2_PARAM/raise` · 业务继续 ✅ |
| 就绪 | 220 s（`/v1/models` 200） |
| 停机后本侧残留 | **0**（`ps` 括号法计数 0；`cnmon` 进程表 `No running processes found`） |

⇒ **定性成立**：`TP=2/EAGER=0` 的服务侧无问题，首跑失败属**客户端观测窗口不足**
（`max 87467.3 ms` 已经逼近原 120 s 窗口；而失败那次是在**第 2 个**请求上撞线）。
⚠️ 该形态同时也说明了**为什么不能靠猜**：若不看"服务 200 + 同形态冒烟 PASS + 首请求 85 s"这三条并置证据，
很容易把一次客户端超时写成"TP=2 服务化不可用"。

---

## 5 两处如实登记的问题与修法（都属"看起来通过/看起来失败"家族）

| # | 问题 | 现场证据 | 修法 |
|---|---|---|---|
| **1** | **客户端观测窗口硬编码 120 s** | `proto_infer_serve.py::http_post` 原为 `timeout=120`；`TP=1/EAGER=1` 实测首请求 85 s、`TP=2/EAGER=0` 第 2 请求直接超时 | 新增 `REQ_TIMEOUT = int(os.environ.get("SERVE_REQ_TIMEOUT", "120"))`，`http_post(..., timeout=None)` 时取它。⚠️ **默认值不变 ⇒ 既有证据口径不动**；定性边界写进代码注释：请求超时是**观测窗口**，不是判据本身。与 `serve_standard.sh` 已固化的同类教训（2026-09-28：MLU590 首请求 63.6 s vs 旧硬编码 60 s）**同族**，这次落在客户端侧 |
| **2** | **停机漏杀 `VLLM::Worker_TP*`** | 首跑后 `cnmon` 进程表：`17383 VLLM::Worker_TP0 → 卡0 89351 MiB`、`17384 VLLM::Worker_TP1 → 卡1 89353 MiB`；两卡仅剩 7.4 GiB 空闲 | 根因：**进程名是大写 `VLLM::`**，而清理用的是大小写敏感的 `pkill -9 -f "[v]llm serve"`（只覆盖主进程与 `VLLM::EngineCore`），孤儿 Worker 继续持卡。改为按 **`VLLM::` 前缀**清、残留计数改 `grep -aiE`。**实测**：手工清掉后卡立即回落 **94.6 / 94.8 GiB** |

> 另记（不算问题）：容器内累积了一批 `[python]/[VLLM::EngineCor] <defunct>` 僵尸（ppid=1 未回收），**不占设备内存**；容器重启即清。首跑还有一条 **zombie `[vllm] <defunct>`** 同类。

---

## 6 边界与**不声称**的部分

1. **不给性能结论**：本任务判据是"能不能起、能不能算"，吞吐读数被首请求编译主导（max 85269 ms），**不得**用于比较形态优劣或与另两家横比。
2. **未测 `TP>2`**（8 卡机器，但本任务口径只要求 `TP>1`）。
3. **CNCL 的 RDMA 路径仍未验证**（审计 **A4**）：当前走 `MLU_LINK` 片间互联 ⇒ 本轮多卡数据**只代表单机 2 卡**，发结论时须连同该告警一起引用。**非本层缺口**。
4. **停机期的 CNCL 噪声**：`TP=2` 形态停机时日志出现
   `torch_mlu::ProcessGroupCNCL::heartbeatMonitor()` → `c10d::TCPStore::check` 抛 `c10::Error` 的 C++ 回溯，
   伴随 `Failed to check the "should dump" flag on TCPStore, (maybe TCPStore server has shut down too early)`。
   ⇒ 这是**停机顺序**（TCPStore 先关、心跳线程后查）产生的噪声，**发生在判据通过之后**；
   与 T11 的退出期 SIGSEGV 不是一回事（本次 `rc=0`、崩溃痕迹行数实测 0）。**如实记录，不掩盖、也不升级为结论**。
5. 本轮的"形态生效"证据来自 vLLM 自身 config dump 与 `ps` 实参；**未**做"TP=2 时真的跨卡切分"的算子级验证（不在本任务口径内）。

---

## 7 复现命令与证据索引

```bash
# 0) 容器（我方资产）：启动 vLLM 应用镜像容器
docker start dc-mlu590-vllm-hliu553

# 1) 四形态矩阵（宿主 Mlu-1 执行；每形态独立进程 + 跑前清理 + 形态生效正证据）
bash /srv/hliu553/run_t7_mlu590_serve_matrix_20261014.sh
#    ⇒ 证据 /srv/hliu553/dc_t7_20261014/{tp1_eager1,tp1_eager0,tp2_eager1,tp2_eager0}/
#      · 判据行只在 serve_standard.log 内（脚本开头 exec 重定向；外层重定向得到空文件）

# 2) 消费方腿（主腿 TP=1/EAGER=1 + 对照腿 TP=2/EAGER=0，消费方 stdout 必须重定向）
bash /srv/hliu553/run_t7_mlu590_consumer_20261014.sh
# 3) TP=2 对照腿定性补跑（只放宽客户端观测窗口）
bash /srv/hliu553/run_t7_mlu590_consumer_tp2retry_20261014.sh
```

| 证据 | 文件（`../probes/t7_serve_20261014_out/`） |
|---|---|
| 四形态矩阵汇总 | `form_matrix.log`（每行 `TP/EAGER/rc/verdict`） |
| 各形态判据与读数 | `tp{1,2}_eager{1,0}/serve_standard.log` |
| 形态生效正证据 | 各形态 `vllm_serve.log` 的 `non-default args` / engine config 行 |
| 消费方主腿 | `consumer_tp1_eager1/{proto_infer_serve_result.json,consumer_stdout.log,serve_standard.log,vllm_serve.log}` |
| 消费方对照腿首跑（**失败现场，勿当当前状态**） | `consumer_tp2_eager0/consumer_stdout.log`（`TimeoutError` 栈） |
| 消费方对照腿**补跑（定性）** | `consumer_tp2_eager0_retry/{proto_infer_serve_result.json,consumer_stdout.log,serve_standard.log,vllm_serve.log}`（`SERVE_LEG_PASS 10/10`） |
| 腿汇总 | `consumer_legs.log`（⚠️ 首版把"命中行数"标成 `rc=1`，已在脚本里改成 `verdict=` 标签） |
| 驱动脚本 | `../probes/run_t7_mlu590_serve_matrix_20261014.sh` · `run_t7_mlu590_consumer_20261014.sh` · `run_t7_mlu590_consumer_tp2retry_20261014.sh` |

---

## 8 本轮教训（写入运行手册）

1. ⭐ **停机清理必须按 `VLLM::` 前缀**（大小写敏感会漏掉 `VLLM::Worker_TP*`）⇒ 孤儿 Worker 会**继续持卡**，症状是"进程数=0 但卡没释放"。**排查顺序：先看 `cnmon` 进程表，再看 `ps` 计数**。
2. ⭐ **消费方（客户端）也必须重定向 stdout**：首跑没重定向，失败现场随外层管道丢失，只剩"没有结果文件"的空目录 ⇒ 无法定性。
3. ⭐ **"没有结果"不等于"服务不可用"**：本例服务 `200`、同形态冒烟 PASS，失败在**客户端观测窗口**。
   修法是**加显式开关、默认不变**，并在证据里注明用了哪个窗口 —— 不得靠"默认放宽"让判据变绿。
4. ⭐ **残留计数别把自己数进去**：把 `grep -E "[v]llm"` 改成 `grep -aiE "vllm"` 后，**`grep` 进程自身命令行含 `vllm`**
   ⇒ 计数恒 ≥1（"进程数=0 却报 1"）。正确写法是**括号法 + `-i`**：`grep -aiE "[v]llm|[s]erve_standard"`。
   同一轮里还犯了它的反面：把 `grep -ac` 的**命中行数**写成 `rc=`（日志出现 `rc=1` 会被读成失败）⇒ 标签必须名副其实。
5. **用卡读数只作复读，不作归因**：停机后 `dev0 94.69 / dev1 93.05 GiB`（基线 94.8）—— 本侧残留进程为 **0**
   且 `cnmon` 进程表为空，同机卡 3/4/5/6/7 上有他人占用（util 19–56%）⇒ **不做归因**，只如实记录数字。
