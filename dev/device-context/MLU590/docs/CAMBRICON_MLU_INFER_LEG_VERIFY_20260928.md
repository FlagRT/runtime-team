# 寒武纪 MLU590 · 推理腿（前向 + 服务化）验收报告

> 日期：**2026-09-28** ｜ 分支：`kistich/device-context` ｜ 方向：设备上下文（device-context）
> 判据依据：《运行时层接口约定（设备上下文章节）》五域 +《多流 Stream 验收基线》16 项
> **定位**：补齐 **09-22 两台主机 SSH 超时**导致未做的最后 2 项 ⇒ **第三实例 12 项判定全部完成**。

---

## 0. 结论

**寒武纪 MLU590 的推理腿两形态（单卡前向 + vLLM 服务化）本轮补齐，12 项判定全部通过。**
与前两实例同判据、同脚本、同验收模型（`Qwen3-Embedding-0.6B`，快照 `97b0c614…`）。

| # | 判定项 | 本轮结果（2026-09-28 真机） |
|---|---|---|
| 1 | 离线契约自检（无设备工具） | **39 通过 / 0 失败 / 0 跳过** |
| 2 | 跨后端对称性自检 `--all` | **5 通过 / 0 失败** |
| 3 | 冒烟自检（smoke） | **46 通过 / 0 失败** |
| 4 | conformance 设备上下文与多流 | **13/13（`CONFORMANCE_PASS`）** |
| 5 | conformance 推理 | **6/6** |
| 6 | 多流语义基线（探针） | **`STREAM_SEMANTICS_PASS 8/8`** |
| 7 | 图捕获流语义 | **契约内 4/4**（另 1 项契约外用法＝**宽容度观察项**，容忍，不计入判定） |
| 8 | 流配额 | **2000 流 `STREAM_QUOTA_PASS 3/3`** |
| 9 | 训练腿（2 卡 DDP） | **`TRAIN_LEG_PASS 6/6`** · loss **15.4498 → 11.1479** · **3015.3 / 3017.5 tok/s** · `dist=cncl` |
| 10 | **推理腿（单卡前向）** | **`INFER_LEG_PASS 13/13`**（+1 项 `vendor_code_map` **如实跳过**）· dim **1024** · **41.08 句/s** · p50 **72.89 ms** · 区分度 **0.6391** |
| 11 | **推理腿（服务化）** | **`SERVE_STANDARD_PASS (ready=1 smoke=1)`** · 就绪 **150 s** · 维度 **1024** · 范数 **1.000001** · 冒烟耗时 **42 s** |
| 12 | 错误注入 → 恢复闭环 | **`ERROR_RECOVERY_LOOP_PASS`：闭环 5 / 跳过 0 / 失败 0** |

> 第 10 项的 `13/13 + 1 跳过` 不是差一项：**910C 有厂商数字码表**故 14 项全跑；
> 寒武纪 **CNRT 抛的是错误名而非数字码**（`RuntimeError: CNRT error: invalid argument.`）⇒
> `vendor_code_map` 一项**如实跳过**（与 P800 同构，见 §4）。

---

## 1. 环境与前置（本轮实际使用的值）

| 项 | 值 |
|---|---|
| 主机 / 容器（前向 + 训练 + 探针） | `Mlu-1`（`tza-0a06-ai01-em9`）· 容器 `dc-mlu590-hliu553`（Up，09-22 起） |
| 主机 / 容器（服务化） | `Mlu-1` · 容器 **`dc-mlu590-vllm-hliu553`**（本轮新建，见 §3.1） |
| 镜像（前向/训练/探针） | `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0` |
| 镜像（服务化） | `harbor.baai.ac.cn/flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2`（digest `sha256:f568f23cf29b2…`，**与 harbor 登记一致**） |
| 软件栈（两镜像一致） | py **3.10.20** / Ubuntu 22.04.5 / torch **2.7.1+cpu** / torch_mlu **1.29.2+torch2.7.1** / **8 卡可见** |
| 模型 | `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`（共享缓存，**只读**） |
| 用卡 | 前向 `mlu:0`；冒烟/conformance/探针 `MLU_VISIBLE_DEVICES=6`、`6,7`；服务化 `DEV=2`；错误闭环 `7` |

**容器挂载（易踩）**：宿主 `/srv/hliu553` → 容器 **`/work`**，宿主 `/srv/data/hf_cache` → 容器 **`/hf_cache`**。
⇒ 容器内原型在 **`/work/prototype`**（不是 `/srv/hliu553/prototype`）。

---

## 2. 逐项证据（全部入库，命名与两实例同规范）

| 判定项 | 证据文件（`probes/`） |
|---|---|
| 离线自检 | `accept_offline_cambricon_20260928.log` · `accept_offline_all_20260928.log` |
| 冒烟 | `accept_smoke_cambricon_20260928.log` |
| conformance 13 / 推理 6 | `accept_conf13_cambricon_20260928.{json,log}` · `accept_confinfer6_cambricon_20260928.{json,log}` |
| 多流三项 | `accept_stream_semantics_cambricon_20260928.log` · `accept_graph_capture_cambricon_20260928.log` · `accept_stream_quota_cambricon_20260928.log` |
| 训练腿 | `accept_train_cambricon_20260928/{run.log, train_leg_result_rank0.json, train_leg_result_rank1.json}` |
| 推理腿前向 | `accept_inferleg_cambricon_20260928.log` · `accept_inferleg_result_cambricon_20260928.json` |
| 服务化 | `accept_serve_cambricon_stdout_20260928.log` · `accept_serve_cambricon_vllm_20260928.log` |
| 服务化（**修复前那次失败**，留档不改判） | `accept_serve_cambricon_PRE_FIX_20260928.log` · `accept_serve_cambricon_PRE_FIX_stdout_20260928.log` |
| 错误闭环 | `accept_errorloop_cambricon_20260928.{json,log}` |

---

## 3. 两条要点（都属「前置条件没写进工具」家族）

### 3.1 运行时层镜像**不含 vLLM** ⇒ 服务化改用 FlagOS 官方**应用镜像**

- **现象**：`flagos-runtime-cambricon-neuware4.4.3:2.2.0` 内 `command -v vllm` 无结果、
  `import vllm` 报 `ModuleNotFoundError`（该事实 09-22 已登记入 `cambricon.known_issues`）。
- **实测定档（本轮）**：FlagOS 官方 `flagos-app` 仓**确有寒武纪 vLLM 应用镜像**，且**可匿名拉取**：

| 仓 | tag | 大小 | 与本方向档位 |
|---|---|---|---|
| `flagos-app/vllm0.20.2-cambricon-neuware4.4.3` | **`2.2.0-0.2.2rc2.post2`** | 2.91 GiB | ✅ **采用**（4.4.3 档、版本 2.2.0） |
| `flagos-app/vllm0.24.0-cambricon-neuware4.4.3` | `2.2.0-0.3.0rc2.post2` | 2.91 GiB | 未采用（同一档、vLLM 版本更高，留作候选） |

- 拉取后**软件栈与运行时镜像逐项一致**（py3.10.20 / torch 2.7.1+cpu / torch_mlu 1.29.2 / 8 卡）
  ⇒ 结论与其余判定项**同档可比**，不需要附加条件标注。
- 起容器脚本落在宿主 `/srv/hliu553/start_container_mlu590_vllm.sh`：**与 `start_container_mlu590.sh`
  只差镜像一行**，设备节点 / 挂载 / 容器参数（`--net=host --shm-size=64g`、18 个设备节点、`cnmon`）
  全部沿用同一套规则（禁止"各自维护启动脚本"）。
- ✅ **服务化完成后容器已 `docker stop`**（容器保留，`docker start` 即可复用）；卡 2 复查为 `0 MiB`。

### 3.2 冒烟超时**硬编码 60 s** ⇒ 把"慢"判成了"不通"（假失败）

| 步骤 | 实测 |
|---|---|
| 第 1 次服务化（修复前） | 就绪 `t=365s` → 冒烟 `curl -m 60` **超时**（原始日志里 `raw:` 为空）⇒ `SERVE_STANDARD_FAIL (ready=1 smoke=0)`；**服务端 `vllm_serve.log` 只有 `GET /v1/models` 200，没有 `POST /v1/embeddings` 的访问行** |
| 手工复现（保服务运行后单发请求） | `http=200`、**`time_total=63.59 s`**、`size=21900` ⇒ **请求本身是成功的，只是首次请求含编译开销** |
| 第 2 次服务化（修复后） | 就绪 `t=150s`、冒烟 **42 s** ⇒ `SERVE_STANDARD_PASS` |

- **根因**：`serve_standard.sh` 的冒烟用的是 `curl -s -m 60`（硬编码），而 MLU590 上
  **首次 embedding 请求实测在 42–64 s 区间**（含首次编译）⇒ 冷启动那次恰好越过 60 s ⇒ **误判失败**。
  这是「**测试工具参数退化会让结论反向**」的又一例：与"参数退化静默跳过产生假证据"同族，方向相反。
- **修复（已落脚本）**：新增 `SMOKE_TIMEOUT`（默认 **180 s**），两个形态分支都改用它，
  并**打印本次冒烟实际耗时** ⇒ 判据参数显式化、且下一次读日志即可看出"是慢还是不通"。

```bash
# 修复后（两处）：
    SMOKE_T0=$(date +%s)
    OUT=$(curl -s -m "$SMOKE_TIMEOUT" -X POST "http://$HOST:$PORT/v1/embeddings" ...)
    echo "  （冒烟耗时 $(( $(date +%s) - SMOKE_T0 ))s；超时上限 SMOKE_TIMEOUT=${SMOKE_TIMEOUT}s）"
```

- **纪律**：修复前那次失败**原样留档**（`*_PRE_FIX_*`），**没有"改判据变绿"** ——
  改的是"工具的参数上限"，不是"判定标准"；且改动前后两次都入库，读者可自行判定。

---

## 4. 三实例对照（本轮 + 09-22 两实例）

| 判定项 | 910C（`ascend`） | P800（`kunlun`） | MLU590（`cambricon`） |
|---|---|---|---|
| 离线自检 | 35/0（1 跳过） | 39/0（1 跳过） | **39/0/0** |
| 对称性 `--all` | 5/0 | 5/0 | **5/0** |
| 冒烟 | 52/0 | 46/0 | **46/0** |
| conformance 13 + 推理 6 | 13/13 + 6/6 | 13/13 + 6/6 | **13/13 + 6/6** |
| 多流语义 / 图捕获 / 配额 | 8/8 · 4/4 · 3/3 | 8/8 · 4/4 · 3/3 | **8/8 · 4/4 · 3/3** |
| 多流 16 项基线 | 16/16 可判 | 14 通过 / 1 不支持 / 1 不适用 | **15 通过 / 1 不适用 / 0 不支持** |
| 训练腿 2 卡 | 6/6 · **4075.4** tok/s · `hccl` | 6/6 · **3533.5** tok/s · `flagcx` | **6/6 · 3015.3 tok/s · `cncl`** |
| 推理腿前向 | **14/14** | 13/13（+1 跳过） | **13/13（+1 跳过）** |
| 推理腿服务化 | `SERVE_STANDARD_PASS` | `SERVE_STANDARD_PASS` | **`SERVE_STANDARD_PASS`** |
| 错误闭环 | 5/0/0 | 5/0/0 | **5/0/0** |

**逐项差异都有解释（不是缺漏）**：

1. **推理腿前向 14 vs 13**：`vendor_code_map` 一项 —— **910C 有厂商数字码表**（108 条 ACL 码），
   另两家厂商**抛错误名不抛数字码** ⇒ 该项**如实跳过**（P800 同）。
2. **多流 S-12 流优先级**：**MLU590 支持（`priority_range()=(0,-3)`）而 P800 不支持**
   ⇒ 同一 API 跨芯片**相反**，已写进《新芯片接入手册》§9 坑清单。
3. **吞吐不可横向比**：三台都是**共享机**（本轮卡 1/4 曾被他人占 60.5 GiB），
   且卡号不同（910C 0,1 / P800 6,7 / MLU 6,7）⇒ 只作"同档可比"，**不作性能结论**。

---

## 5. 边界与未覆盖（如实标注，不外推）

1. **集合通信走 `MLU_LINK` 片间互联、非 RDMA**：CNCL 未加载 `libibverbs`/`libmlx5`
   （日志 `Failed to open libibverbs.so[.1] …`）⇒ **训练腿数据只代表单机 2 卡**，跨机/RDMA 路径未验证。
2. **服务化只验了单实例单卡**（`TP=1`、`--enforce-eager`）；多卡 TP / 关闭 eager 的形态未跑。
3. **910C 已按新脚本复跑（09-28 补齐）**：网络恢复后复跑 ⇒ `SERVE_STANDARD_PASS`（就绪 35 s、
   维度 1024、范数 1.000000、冒烟耗时 0 s），与 09-22 逐项一致 ⇒ **三实例均已按新脚本复跑**。
   （首次复跑因宿主带卡容器名额被他人占满失败，释放后通过。）
4. **档位条件**：全部结论在 **`neuware4.4.3` 档**（py3.10 / torch 2.7.1 / torch_mlu 1.29.2）取得；
   引用时须连同档位一起引（同 P800 的"KL3 未设置条件下取得"）。

---

## 6. 复现命令

```bash
# 0) 进容器（前向/训练/探针用运行时镜像容器）
docker exec -it dc-mlu590-hliu553 bash
PR=/work/prototype; OUT=/work/MLU590/probes; cd $PR
M=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

# 1) 推理腿前向（单卡，约 1 分钟）
DC_BACKEND=cambricon DC_ROUNDS=5 DC_ROOT=$PR DC_MODEL=$M DC_OUT_DIR=$OUT \
  python3 runtime/proto/proto_infer_leg.py

# 2) 服务化（**需 vLLM 应用镜像容器**，见 §3.1；宿主机上先跑）
#    bash /srv/hliu553/start_container_mlu590_vllm.sh   →  再进容器：
DC_BACKEND=cambricon SERVE_FORM=embed DEV=2 EAGER=1 STOP_AFTER=1 READY_BUDGET=900 \
MODEL=$M SERVED_NAME=qwen3-embedding-0.6b DC_OUT_DIR=/work/MLU590/serve_run \
  bash scripts/serve_standard.sh
#    ⇒ 判据：日志末尾 `[verdict] SERVE_STANDARD_PASS (ready=1 smoke=1)`
```

---

## 7. 下一步

- 910C 可达后按新脚本补跑一次服务化（§3.2 的口径一致性要求）。
- 第三实例已具备与前两实例**同口径、同判据、同证据规范**的完整数据集 ⇒
  可整体并入 **`../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md`** 的三芯片验收矩阵。
- 报告体例与证据命名对齐 `../P800/docs/` 与 `../910C/`，便于横向比对。
