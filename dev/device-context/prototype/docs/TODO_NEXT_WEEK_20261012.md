# 下周待办执行单（2026-10-12 起）

> **本文件是"执行单"**：怎么跑、判什么、要不要卡、按什么顺序。**结论与证据仍在原报告里**，本文件不重复结论。
> 来源 = `prototype/docs/OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md`（2026-10-10 四家复合核验）的「收尾条件」列。
> 关联提醒：**2026-10-12 09:30** 的一次性提醒（原名「PPU 基类改动补三台机真机复跑（准备网络环境）」）已把本文件登记为执行入口。
> ⚠️ **相对路径基准 = `dev/device-context/`**（同 `VERIFICATION_MANIFEST_20260920.md` 惯例）；本文件自身位于 `prototype/docs/`。
> ⚠️ 逐机挂载/解释器/模型路径见 skill `flagos-shared-machine-runbook` §1；**容器内仓库是 rsync 工作副本，不是 git 检出**。

---

## 0 三条最该先做

| 序 | 项 | 一句话理由 |
|---|---|---|
| **1** | ✅ **三家真机复跑 78 项职责审计**（T1）—— **三家全部完成、均无回归**（910C/P800 = 10-13 · **MLU590 = 10-14**） | 2026-10-10 的跨后端**基类改动**破坏面 = 共享层 ⇒ **910C / P800 / MLU590 三家全在破坏面内**，而三家最后取数都在 **10-08** ⇒ 现有结论**不覆盖当前版本**。**10-13 实测**：910C `73/0/5` · P800 `68/0/10`（均 PASS、FAIL=0），与 10-08 基线**逐项状态零差异** |
| **2** | **PPU 推理腿服务化**（T3） | 同时解锁「逐芯片职责验收 12 项」整行；容器内已带 vLLM `0.19.0` |
| **3** | **PPU 错误闭环四类注入**（T4） | 手册 §7 第 6 步，四家验收的固定项（PPU 是唯一缺的） |

---

## 1 待办总表

| # | 项 | 类别 | 入口 | 验收判据 | 需卡 | 窗口 |
|---|---|---|---|---|---|---|
| **T1** | 三家真机复跑 78 项职责审计（跨后端基类改动） | 🔴 最高 | `scripts/duty_response_audit.py --backend <b>` | 三家各 `DUTY_RESPONSE_PASS`（**0 FAIL**）；重点看 `L1` 流归属与释放 | ✅ ×3 台（910C/P800=10-13 · **MLU590=10-14**） | 1 次/台 |
| **T1-a** | ✅ **910C 已完成（10-13）** | — | 见 §2.5 执行记录 | `DUTY_RESPONSE_PASS` **73 / 0 / 5**（与 10-08 基线逐项状态零差异）· 契约不变式 **4/4** | — | — |
| **T1-b** | ✅ **P800 已完成（10-13）** | — | 见 §2.5 执行记录 | `DUTY_RESPONSE_PASS` **68 / 0 / 10**（与 10-08 基线逐项状态零差异）· 契约不变式 **4/4** | — | — |
| **T1-c** | ✅ **MLU590 已完成（10-14）** | — | 见 §2.5 执行记录 | `DUTY_RESPONSE_PASS` **62 / 0 / 16**（与 d2 基线逐项状态零差异）· 契约不变式 **4/4** | — | — |
| **T2** | ✅ **三家全部完成**（910C/P800=10-13 · MLU590=10-14） | 🟠 需卡（只读） | `bash scripts/preflight_env.sh` | ✅ 三台均 `rc=0`：`910C/probes/t2_preflight_env_ascend_20261013.log`（12.9 KB）· `P800/probes/t2_preflight_env_kunlun_20261013.log`（11.7 KB）· `MLU590/probes/t2_preflight_env_cambricon_20261014.log`（11.4 KB） | ✅ 三家已闭环 | 已闭环 |
| **T3** | ✅ **已完成（10-14）** | — | `serve_standard.sh` 已补 **`ppu` 分支**（脚本内注释含路径 B 与前置） | ✅ `SERVE_STANDARD_PASS (ready=1 smoke=1)`（就绪 35 s · dim 1024 · 范数 1.000000）· 消费方 **`SERVE_LEG_PASS 10/10`**（103.15 句/s） | — | — |
| **T4** | ✅ **已完成（10-14）** | — | 同左 | ✅ **`ERROR_RECOVERY_LOOP_PASS` 闭环 5 / 跳过 0 / 失败 0**（与三家同口径）；⚠️ 判据通过后 `rc=139`（退出期段错误，见 T11） | — | — |
| **T5** | ✅ **已完成（10-14）** | 无需额外 | 同左（另补跑 S-13 两卡 / S-16 配额 / S-7 图捕获） | ✅ `PPU/docs/THEAD_PPU_STREAM_BASELINE_16_20261014.md`：**16 项全部有 PPU 读数**（10 项本轮新测 + conformance/两条腿/错误闭环覆盖 + 1 项三家一致不适用）；⚠️ 唯一差异在 S-12 的**调度效果**（见 T6） | — | — |
| **T6** | ✅ **已完成（10-14）** | — | 同左 | ✅ `STREAM_PRIORITY_SCHED_EFFECT_PASS` · **`effect_observed=False`**（如实记录：gated 胜率临界但幅度 0.48–0.51 ms ≪ 阈值 8.7–8.9 ms；ungated 由**提交顺序主导**、两向不一致）；⭐ **途中抓出并修复一处跨后端缺陷**（平台侧弱引用悬垂 ⇒ 本层误拦，见 `PPU/docs/PPU_T3_T6_AND_WEAKREF_20261014.md` §4） | — | — |
| **T7** | MLU590 多卡 TP / 关闭 `--enforce-eager` 服务化 | 🟡 中 | 走 `serve_standard.sh`（**需 vLLM 应用镜像容器**） | `TP>1` 起服务 + 冒烟通过；`EAGER=0` 至少一轮 | ✅ | 需应用镜像 |
| **T8** | ✅ **P800 + MLU590 均完成**（P800=10-13 · MLU590=10-14） | 🟢 低 | `probes/recover_multiproc_stress.py --backend <b>` | ✅ 两后端均 `A2_BOUNDARY: SKIP_UNSUPPORTED`（`rc=3`，**未做任何设备操作**）⇒ `P800/probes/t8_a2_skip_kunlun_20261013.log` · `MLU590/probes/t8_a2_skip_cambricon_20261014.log`（未声明 `recovery_real` ⇒ 如实跳过留档） | ✅ 已闭环 | 已闭环 |
| **T9** | ✅ **已完成（10-14）** | 🟢 低（不需卡） | 纯文档 | ✅ `PPU/docs/EVIDENCE_INDEX_PPU.md`（按 T 号索引 + 判读纪律三条 + 已知 `rc=139` 覆盖表 + 命名现状） | ❌ | 已闭环 |
| **T10** | 🔶 **部分完成**（如实登记偏离） | 🟢 低（不需卡） | 纯文档 | 任务级产物已按 §5 命名（`t1_duty_cambricon_20261014.json` · `t2_preflight_env_cambricon_20261014.log` · `t8_a2_skip_cambricon_20261014.log`）；⚠️ **目录名未逐字对齐**（用 `<T号>_<用途>_<YYYYMMDD>_out`）⇒ 已在 `EVIDENCE_INDEX_PPU.md` §5 登记现状，**约定统一后再改** | ❌ | 待约定 |
| **T11** | PPU 退出期段错误收口 | ⚪ 需外部 | — | 需**带调试器的镜像**或厂商协助 | ⚠️ | 依赖外部 |
| **T12** | 契约修订建议剩余 2 条（`vendor` 字段 / smoke 打印 `known_issues`） | 🟢 低（不需卡） | 纯代码 | 第 6 条 3 行打印即可；第 1 条**先定"是否还要 `vendor`"** | ❌ | 任意 |

**需人定（不占技术工）**：P800「上报渠道待确认」（⏳ 待定，用户已裁定）· MLU590 CNCL RDMA / 档位升级（厂商与网络层，非本层缺口）。

---

## 2 T1 详细步骤（三家真机复跑）—— 本次重点

### 2.5 执行记录（**910C / P800 · 2026-10-13 · 已完成**）

> 结果：**两台均无回归**。判据全绿，且与各家 10-08 基线**逐项状态零差异**。

| 机 | 容器（本轮新建/启动） | 栈（实测） | 主判据 | 契约不变式 | 证据 |
|---|---|---|---|---|---|
| 910C | **新建** `dc-t1-910c-20261013`（只挂 `davinci5` + 3 个管理设备） | py3.12.13 · torch **2.11.0+cu130** · torch_npu **2.11.0** · `Ascend910_9382` | `DUTY_RESPONSE_PASS` **73 / 0 / 5** · `L1`=**OK** | **4/4**（i1 覆盖 16/17） | `910C/probes/t1_duty_ascend_20261013.json` |
| P800 | 启动既有 `hliu553-device-context-p800`（Exited 4 天） | py3.10.18 · torch **2.9.0+cu129** · 8 卡可见 | `DUTY_RESPONSE_PASS` **68 / 0 / 10** · `L1`–`L5` 全 **OK** | **4/4**（i1 覆盖 14/17） | `P800/probes/t1_duty_kunlun_20261013.json` |

**逐项比对（不许只看总数）**：状态差异 **0 项**（910C vs `d2_duty_ascend.json` / `e1_duty_ascend.json`；
P800 vs `r12_duty_kunlun.json` / `d2_duty_kunlun.json`）。3 处 `detail` 差异**均可解释**：
① `A1/A2` 的 `discover/available` 列表多了 **`ppu`**（10-10 接入第 4 家后端后的**预期变化**，非回归）；
② `B4` 显存读数（环境量）；③ `L5` 文案措辞随契约引用更新。

**本轮两个环境事实（会影响后续「跑的是当前版本」的判断）**：
1. ⚠️ **910C 的 `venv-infer-a` 在宿主上已断链** —— 其基座 `/usr/local/python3.12.13` 被宿主移除
   （宿主现仅 `/usr/bin/python3.9` 与 `/usr/local/python3.7.5`）。**但该 venv 在容器内仍完全可用**
   （镜像自带 `/usr/local/python3.12.13`）⇒ **必须用容器内解释器**（在宿主上会报 `No such file or directory`）。
2. ⚠️ 910C 的**名额**必须逐次重判：本轮 `dc-lean-910c-20260929`（挂 `davinci1,2,3`）**不可用** ——
   卡 1 被 `evalx-910c` 占、卡 2/3 被 `kelvin-w4-lzy` 占（NPU 上有活跃 `VLLMEngineCor` 进程）
   ⇒ 按 `910C/docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md` 改挂**空闲卡 `davinci5`**。
   （当次普查：仅 NPU 2 有进程；无人挂载的卡 = 0 / 5 / 7 等。）

**可复现命令**：`910C/probes/run_t1_910c_20261013.sh` · `P800/probes/run_t1_p800_20261013.sh`
（容器内执行；含版本自证 → 设备可见性 → 契约不变式 → 78 项审计 → 汇总五段）。

**同窗口顺带完成的两项**（原计划"与 T1 同窗口最省"）：

| 项 | 机 | 结果 | 证据 |
|---|---|---|---|
| **T2** 环境普查留档 | 910C / P800 / MLU590 | ✅ 均 `rc=0`（MLU590=10-14） | `910C/probes/t2_preflight_env_ascend_20261013.log`（12.9 KB）· `P800/probes/t2_preflight_env_kunlun_20261013.log`（11.7 KB）· `MLU590/probes/t2_preflight_env_cambricon_20261014.log`（11.4 KB） |
| **T8** A2 如实跳过留档 | P800 / MLU590 | ✅ 均 `rc=3`、`A2_BOUNDARY: SKIP_UNSUPPORTED`（**未做任何设备操作**；MLU590=10-14） | `P800/probes/t8_a2_skip_kunlun_20261013.log` · `MLU590/probes/t8_a2_skip_cambricon_20261014.log` |

> T2 顺带拿到两条与"环境漂移"相关的旁证：910C 宿主 `python3` 只剩 **3.9.9**
> （`/usr/local/python3.12.13` 确已被移除 ⇒ 与 §2.5 第 1 条互相印证）；
> 两台 docker 版本分别为 **20.10.8**（910C）与 **29.1.3**（P800）。

### 2.1 为什么必须做

2026-10-10 对 `prototype/runtime/backends/base.py` 做了**跨后端基类改动**：
`_owned_streams` 登记表值结构 **2 元组 → 3 元组**（句柄 + 持有器 + **类型名**），
`_owned_handle` / `release_stream` 增加**类型复核**。

- 破坏面 = **共享层** ⇒ **所有实例**都在破坏面内（"按破坏面覆盖"要求三家各跑一次真机）；
- 三家最后一次 78 项审计均在 **10-08**（`910C/probes/d2_20261008_out/d2_duty_ascend.json` ·
  `P800/probes/r12_20261008_out/r12_duty_kunlun.json` · `MLU590/probes/m1_20261008_out/m1_duty_cambricon.json`）
  ⇒ **不在改动后的版本内**；
- ⚠️ **离线自检四家无回归**（`90/0/1` · `108/0/1` · `97/0/0` · `114/0/1`）**不等于真机无回归**。

### 2.2 前置（先做这一步，别跳）

1. **网络环境** ⛔ **这是本次唯一的真前置** —— 三台里有**两台当前不可达**。**带时间戳的实测现状**：

   | 机 | SSH | 我们的容器 | 现状（**2026-10-10 17:10 实测**） |
   |---|---|---|---|
   | 910C | `910C` | 本轮用 **新建** `dc-t1-910c-20261013`（只挂 `davinci5`） | ✅ **10-13 已打通并完成 T1**（`npu1-27`）· ⚠️ 旧容器 `flagos-*` 仍 `Exited`，**建新容器是常规做法** |
   | P800 | `P800` | `hliu553-device-context-p800` | ✅ **10-13 已 `docker start` 并完成 T1**（当时 `Exited (137) 4 days ago`，未删）；⚠️ 机上另有他人在跑的容器（`mwy-dev-xpu*` / `liyanz-flaggems-p800` / `evalx-p800` / `flagos-p800-fullstack-test-20260823` 等）⇒ 用卡前先看资源占用（**本轮 8 卡全空**，取 `CUDA_VISIBLE_DEVICES=0`） |
   | MLU590 | `Mlu-1` / `Mlu-2` | `dc-mlu590-hliu553` | ❌ **两台皆 SSH 超时**（`10.1.1.21` / `10.1.1.22`）⇒ 需先打通网络 |
   | PPU（参照） | `PPU-2` | `hliu553-dc-dev` | ✅ **Up 6 hours**（另机上还有 4 个他人的 PPU 容器） |

   ⚠️ **「不可达」类结论必须带时间戳**（上面这行即是）；下次执行时**先重测**，不要照抄本表。
   ⚠️ P800 容器 `Exited (137)`（= SIGKILL）：**先确认不是被他人/宿主清理**，再决定是否 `docker start`；
   **共享机上他人/既有容器的启停一律先问用户**，不擅自操作。
2. **名额**：910C 是**带卡名额**机器 —— 同一时刻**只留 1 个带卡容器**；动手前 `docker ps -a | grep <自己前缀>`
   清点，症状会**伪装成设备故障**（`acl.init` 500000 / `get_device_count` (0,507899)）。
3. **同步当前原型**（容器内是 rsync 副本）：
   ```bash
   # 本机执行，源必须是本机路径
   rsync -a --exclude '__pycache__' --exclude '.DS_Store' \
     dev/device-context/prototype/ <host>:<宿主映射路径>/runtime-team/dev/device-context/prototype/
   ```
4. **自证"跑的是当前版本"**（关键修复点在不在）：
   ```bash
   grep -c "tname" <容器内>/runtime/backends/base.py        # 期望 4（类型复核在）
   grep -n '_KNOWN_BACKENDS' <容器内>/runtime/backends/registry.py
   ```

### 2.3 逐台怎么跑（容器内，cwd = 原型根）

| 机 | SSH | 容器 | 容器内原型根 | 选卡变量 |
|---|---|---|---|---|
| 910C | `910C` | `flagos-proto-train-910c` / `flagos-infer-910c` | `/mnt/raid/hliu553/runtime-team/dev/device-context/prototype`（同宿主路径） | `ASCEND_RT_VISIBLE_DEVICES` |
| P800 | `P800` | `hliu553-device-context-p800` | `/workspace/runtime-team/dev/device-context/prototype` | `CUDA_VISIBLE_DEVICES`（⚠️ 避开故障卡 `dev1`，UUID `b3509946`） |
| MLU590 | `Mlu-1` / `Mlu-2` | `dc-mlu590-hliu553` | `/work/runtime-team/dev/device-context/prototype` | `MLU_VISIBLE_DEVICES` |

> ⚠️ 三台**当前只有 P800 的 SSH 通**（且其容器已停）⇒ 本步骤的实际起点是「把网络与环境打通」，
> 不是「直接跑命令」。打通后再按上表逐台执行。

**每家跑这三条**（`<b>` = `ascend` / `kunlun` / `cambricon`）：

```bash
cd <容器内原型根>

# ① 主判据：职责响应审计 78 项（重点看 L1 流归属与释放）
python3 scripts/duty_response_audit.py --backend <b> \
  --out <实例>/probes/duty_audit_<b>_20261012.json

# ② 契约不变式真机 4/4（守"声明⇒有入口"/"禁止伪造"/"失效受管"/"降级可观测"）
python3 runtime/conformance/runner.py --backend <b> --cases contract_invariants

# ③ 回传证据到仓库（本机执行）
#    rsync -a <host>:<宿主路径>/…/<实例>/probes/duty_audit_<b>_20261012.json \
#      dev/device-context/<实例>/probes/
```

**判据**：① `DUTY_RESPONSE_PASS` 且 **FAIL = 0**（SKIP 均为"如实不具备"，可接受）；
② `CONTRACT_INVARIANTS_PASS 4/4`；③ 证据**必须 `git ls-files` 实测入库**（`probes/.gitignore` 有 `!*.log` 例外）。

### 2.4 三个易踩点（都踩过）

1. **`L1` 是本次的重点用例** —— 它守的正是这次改动的对象（`release_stream` 不得越权销毁厂商流）。
   历史上它**单独跑通过、只有跑完整序列才复现** ⇒ **必须跑完整 78 项**，不能只跑 `--only L1`。
2. **别用"改前 FAIL、改后 PASS"当根因证据**：跨条件对照必须**交替（A/B/A/B）或同期并行**，否则分不清
   "变量效应"与"时间/环境漂移"。
3. **判据通过 ≠ 进程干净退出**：PPU 已知退出期 SIGSEGV，**必须同时看判据与退出码**；若三家某家出现
   新段错误，按"同位置换原生对象是否仍崩"判归属。

---

## 3 PPU 未对齐项（T3–T6、T9、T10）

### T3 推理腿服务化
`scripts/serve_standard.sh` 当前只有 `ascend` / `kunlun` / `cambricon` 三个分支（`BACKEND=${DC_BACKEND:-ascend}`）⇒
**补 `ppu` 分支**要点：
- 设备变量走 `CUDA_VISIBLE_DEVICES`（PPU 属 `torch.cuda` 命名空间，**不是** `ASCEND_RT_*`/`MLU_*`）；
- 容器前置照旧：`-e NCCL_SOCKET_IFNAME=bond0`（镜像写死 `eth0`，本机无 `eth0`）；
- 形态按 embedding：`--runner pooling --convert embed`（该版本 vLLM **没有 `--task` 参数**）；
- ⚠️ 冒烟上限用可配的 `SMOKE_TIMEOUT`（默认 180 s）并**打印本次实际耗时**——首条请求含编译可能几十秒，
  硬编码短超时会把**成功**判成失败；
- ⚠️ 停机**连 `EngineCore` 子进程一起 `kill -9`**（残留会持续占卡）。

### T4 错误闭环四类注入
`runtime/proto/proto_error_recovery_loop.py --backend ppu`。
⚠️ 先确认注入面：**本栈无数值错误码**（厂商给的是错误名）⇒ 若某类注入无面可打，**如实标 SKIP 并写原因**，
**不得**"改判据变绿"。调用纪律：`translate_error` 必须传**完整原始异常**，截断会退化成 `L3_EXECUTION/replay`。

### T5 多流 16 项基线逐项比对
8 项探针已有（`STREAM_SEMANTICS_PASS 8/8`）⇒ 还差 **S-1…S-16 逐项**（含需要补测的 S-5 并发重叠性能、
S-6 TP 排他、S-9 设备级错误隔离等）。写法参照 `P800/docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md`
（其 §7「未覆盖/待办」的写法可照抄，把"测不出/不适用"如实列出）。

### T6 D2 调度效果实证
`probes/probe_stream_priority_sched_effect.py --backend ppu`（**芯片无关现成探针**，三家均已跑）。
⚠️ 性能类判据：判定 = **胜率 ∧ 幅度**，必须配**正对照 + 分辨力判据 + 同取值基线**；
⚠️ **「同一个派发量子」不得归因于优先级**（同优先级基线三台同量级）。

---

## 4 不需卡的收尾（T9 / T10 / T12）

- **T9** `EVIDENCE_INDEX_PPU.md`：照 `910C/docs/EVIDENCE_INDEX_910C.md` 的写法（含判读纪律：
  **带 `PRE_FIX` 字样的日志是失败现场留档，不是当前结论**）。
- **T10** 命名对齐 manifest §5：新证据按 `<层标识>_<项目>_<后端>_<条件>_<YYYYMMDD>.<ext>`。
- **T12** 契约修订建议剩余 2 条：第 6 条（`smoke_runtime.py` 未打印 `known_issues` 摘要，`grep -c` = 0）可直接补；
  第 1 条（`vendor` 字段）**先定要不要**再动。

---

## 5 窗口需求汇总（一次上机能收多少）

| 窗口 | 能做的项 | 串行约束 |
|---|---|---|
| **W1 · 三台各一次** | T1（78 项）× 3 · T2（preflight，只读）· T8（A2 留档，只读能力声明） | 910C **同一时刻只留 1 个带卡容器**（T2 可与 T1 同窗口，但**不能与训推腿并行**） |
| **W2 · PPU 一次** | T3（服务化）· T4（错误闭环）· T5（16 项基线）· T6（D2 调度效果） | 训练腿与推理服务化**须串行**（挂载/设备集相交）；服务化停机要连子进程清干净 |
| **W3 · MLU590 一次** | T7（TP>1 / `EAGER=0` 服务化） | 需 **vLLM 应用镜像**容器（`dc-mlu590-vllm-hliu553`） |
| **任意时刻** | T9 · T10 · T12 | 不需卡 |

---

## 6 执行完毕怎么闭环（不许"跑了但没回填"）

每完成一项，**同轮**完成这三件事，否则会再长出一批"口径与代码矛盾"：

1. **证据回传入库**：`rsync` 回 `<实例>/probes/`，再 `git ls-files <path>` **实测确认入库**
   （不能凭 `git add` 没报错就认定）；
2. **回填状态列**：`OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md` 的 A/B/C 表 + 本文件总表 ——
   状态列**最容易僵尸化**（历史教训：G6/G7 停在登记当日，与专项报告矛盾了 10 天）；
3. **看板三层同步**：主看板 `README.md` / `STATUS.md` / 各芯片 `README.md` ——
   含 `prototype/README.md` 的**矩阵行**（⚠️ 矩阵**缺一行 = 那一步永远不会被跑**，这正是 PPU 漏跑契约不变式的根因）。

**一键复核**（照 `OPEN_ITEMS_AND_PARITY_AUDIT_20261010.md` §5 的 8 条真跑一遍即可）。
