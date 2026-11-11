# 两条腿的 `DC_MODEL` 口径统一 + 两实例复验（2026-10-27）

**结论**：同一个 `DC_MODEL`，原来**只有推理腿会解析**（HF 缓存根 → `snapshots/<hash>`），
训练腿不解析 ⇒ 同一台机器、同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B`：
推理腿 13/13 通过、训练腿直接 `ValueError: Unrecognized model in …`。
现已把解析抽成**两腿共用的唯一实现**（`runtime/proto/model_path.py`），并加**静态同源守卫**
（`scripts/model_path_parity_check.py`，任一条腿自建 `resolve_model` 即 FAIL）。
本机可复验的两台实例（P800 / MLU590）均 **`LEGPARITY_PASS` 9 通过 / 0 失败**；
910C 与 PPU **未验证**（2026-10-27 11:38–12:47 三次探测均 `Connection refused`，见 §7）。

## 1. 缺陷（改前证据）

10-27 MLU590 共享层复跑时，同一轮里两条腿对同一份环境变量给出不同结果：

| 腿 | 传入 | 结果 |
|---|---|---|
| 推理腿 `proto_infer_leg.py` | `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B` | 解析到 `…/snapshots/97b0c614…`，`INFER_LEG_PASS 13/13` |
| 训练腿 `proto_train_leg.py` | **同一份** | `ValueError: Unrecognized model in /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B` |

当时的处置是**绕过**（在容器内先把 `DC_MODEL` 手工解析成 snapshot 路径再跑），
理由是不把共享层改动夹带进回归记录 —— 原始读数见
`MLU590/docs/CAMBRICON_MLU_RERUN_20261027.md` L44–L45。

**这不是"少写一行"，是"同一份声明只在一半实现里兑现"**：
变量名相同、语义（"三种传法都算合法"）却只落在推理腿；
上层看到"两条腿用同一份配置"会自然以为等价。同族缺陷在本方向已登记为
「声明即承诺」那一类（见《新芯片接入手册》§9 坑 20）。

## 2. 修法：共用一份实现，而不是复制一份

| 做法 | 结论 |
|---|---|
| 把 `resolve_model()` 复制到训练腿 | ❌ 两份实现必然漂移：下次谁改了 `refs/main` 逻辑，另一份不会跟着改 |
| **抽成 `runtime/proto/model_path.py`，两腿 import 同一函数** | ✅ 采用 |
| 加 AST 静态守卫：两腿不得自定义 `resolve_model`/`pick_model`/`require_model`、不得直接读 `DC_MODEL` 环境变量 | ✅ 采用（`scripts/model_path_parity_check.py` 的 G2 组） |

新增/改动文件：

| 文件 | 变化 |
|---|---|
| `runtime/proto/model_path.py` | **新增**（唯一解析实现）：`resolve_model` / `pick_model` / `require_model` |
| `runtime/proto/proto_infer_leg.py` | 删掉本文件的 `resolve_model` 定义与硬编码的候选表，改 import 共用模块 |
| `runtime/proto/proto_train_leg.py` | 补上解析（原先直接 `MODEL = os.environ.get("DC_MODEL", <910C 路径>)`），并去掉 910C 专有默认值 |
| `scripts/model_path_parity_check.py` | **新增**（离线自检 23 项：解析行为 10 + 同源守卫 9 + 早失败 5 —— 见 §3 G1/G2/G3） |
| `scripts/run_legparity_dcmodel.sh` | **新增**（四实例共用的复验驱动，判据 P1–P4，见 §3） |
| `910C|P800|MLU590|PPU/probes/run_*_legparity_20261027.sh` | **新增**（薄封装，只给本实例的容器/解释器/选卡变量/模型路径） |

`DC_MODEL` 现在**三种传法都合法**（两条腿一致）：

1. 普通模型目录（自身含 `config.json`）→ 原样使用
2. `snapshots/<hash>` 目录 → 原样使用
3. HF 缓存根 `models--<org>--<name>` → 解析到其中一个 snapshot，**优先听 `refs/main`**；
   无 `refs/main` 或它指向的 hash 不在 `snapshots/` 下 ⇒ 取排序最后一个**并打印告警**

⛔ 仍不支持给 `/hf_cache/hub`（所有模型的父目录）：既无 `config.json` 也无 `snapshots/`，
刻意**不替调用方猜"你想用哪个模型"**（猜对了没人知道，猜错了更难查）。

## 3. 判据：为什么不是"跑通就算"

`scripts/run_legparity_dcmodel.sh` 的四组判据，每组都对着一类"看起来通过"：

| 判据 | 内容 | 挡的是哪种假通过 |
|---|---|---|
| **P1** | 离线口径自检 `model_path_parity_check.py` 全过（23 项，含 AST 同源守卫） | "两条腿都能跑"但其实是两份实现碰巧行为相同 |
| **P2** | **同一份缓存根** ⇒ 推理腿与训练腿打印的解析路径**逐字相同**，且等于 `refs/main` 指向的真 hash | 各自解析成功、但落在不同目录（例如一个解析到别的 snapshot） |
| **P3** | **诱饵**：合成缓存里放一个字典序更大的 `zzzz…` 空目录；正确实现必须听 `refs/main` | 实现退化成"排序取最后一个"——那种写法在只有一个 snapshot 的机器上**永远不会暴露** |
| **P4** | 换「普通目录」传法（路径不同的软链）两腿仍解析到同一路径；两次训练腿 50 步的 loss 对比 | 修了缓存根、弄坏了原有传法；或两条传法其实喂了不同权重 |

判据落地细节（都是本次吃过亏才写进去的）：

- **同一世界断言**：开工前先做标记文件**双向回读**（宿主写、容器读），确认 `HOST_WORK ≡ WORK`
  是同一目录的两种视角。这条坑本方向累计踩过 9 次以上，表现是"日志空、屏幕只有 rc=0"。
- **`MAX_STEPS=50` 不改小**：训练腿的规定口径是 50 步（挂死类缺陷在第 3–4、41–50 次通信才出现），
  为省时间改小就不是同一条判据了。
- **证据按 RUNID 分目录、只追加**：本轮实际跑了 A/B/C 三轮（原因见 §6），三轮证据全部保留。
- **抽取路径只用路径字符集**（`[A-Za-z0-9._/-]+`），不用 `.*` —— 见 §6。

## 4. 读数（本机可复验的两台实例）

`run_legparity_dcmodel.sh` 的 roundC 读数，证据在
`P800/probes/legparity_20261027_out/`、`MLU590/probes/legparity_20261027_out/`：

| 判据 / 段 | P800（昆仑芯） | MLU590（寒武纪） |
|---|---|---|
| P1 离线口径自检 | **23 / 0 / 0** | **23 / 0 / 0** |
| P2-a 推理腿 × 缓存根 | `INFER_LEG_PASS 13/13` · 区分度 **0.6392** · 27.16 句/s · p50 111.09 ms | `INFER_LEG_PASS 13/13` · 区分度 **0.6391** · 40.67 句/s · p50 73.13 ms |
| P2-b 训练腿 50 步 × 缓存根 | `TRAIN_LEG_PASS 6/6` · loss **15.4488→11.1481** · 2995 tok/s | `TRAIN_LEG_PASS 6/6` · loss **15.4498→11.1479** · 3000 tok/s |
| P4-a 推理腿 × 普通目录 | `13/13` · 0.6392 · 42.84 句/s | `13/13` · 0.6391 · 40.24 句/s |
| P4-b 训练腿 50 步 × 普通目录 | `6/6` · 15.4488→11.1481 · 2915 tok/s | `6/6` · 15.4498→11.1479 |
| P2 两腿解析同一 snapshot | **PASS** | **PASS** |
| P3 未选中诱饵 | **PASS** | **PASS** |
| P4 两种传法解析一致 | **PASS** | **PASS** |
| 两次训练腿 `\|Δloss\|` | **0.000000** | **0.000000** |
| 总判 | **`LEGPARITY_PASS` 9 / 0** | **`LEGPARITY_PASS` 9 / 0** |

**与既有归档的对齐**（说明"没有因为改共享层而动了别的读数"）：

| 指标 | 归档基线 | 本轮 roundC |
|---|---|---|
| P800 推理腿区分度 | 0.6392（10-26 复跑） | **0.6392** |
| P800 训练腿 loss | 15.4488→11.1481 | **15.4488→11.1481** |
| MLU590 推理腿区分度 | 0.6391（10-27 复跑） | **0.6391** |
| MLU590 训练腿 loss | 15.4498→11.1479 | **15.4498→11.1479** |

⚠️ **吞吐不可直接比**：P800 单卡段与其它用户的作业同卡共跑（普查时卡 0 已用 12.2 GiB / util 19%，
卡 7 11.5 GiB / 0%），27.16 与 42.84 句/s 的差来自**同机并发**，不是版本差异；
MLU590 本轮 8 张卡均无他人进程（`cnmon` 无 running processes）。

⚠️ **P800 选卡**：单卡段用卡 0（与 10-26 同口径）；训练腿用卡 0,7。
**卡 1 是已登记的已知故障卡**（0 MiB 但不干活），全程避开；卡 2/3 普查时已 78 GiB / 100% 利用率，不碰。

## 5. 复现命令

```bash
# 离线（无设备，任何机器）
python3 prototype/scripts/model_path_parity_check.py        # 期望 23 通过 / 0 失败

# 上机（在各自宿主上跑，脚本自己 docker exec）
bash dev/device-context/P800/probes/run_p800_legparity_20261027.sh
bash dev/device-context/MLU590/probes/run_mlu590_legparity_20261027.sh
# 910C / PPU 同款封装已入库，网络恢复后直接跑（见 §7）
```

## 6. 判据自己误判过一次（如实记录）

roundB 的 MLU590 报 `LEGPARITY_FAIL`（8 通过 / 1 失败），失败项是 P2 —— **但被测对象没问题**：

- 原因：训练腿是 torchrun **双进程**，两个 rank 的 `[model] …` 行**被交错写进同一物理行**
  （`rank1` 那行尾部紧跟 `rank0` 的 `[model]` 字样）。原抽取写法是 `grep -oE '→ 解析 .*'`，
  `.*` 把后一个 rank 的文字一起吞进"路径"，于是"路径"不等于期望值。
- 修法：抽取改为只认路径字符集 `[A-Za-z0-9._/-]+`，并把这行教训写进脚本注释。
- 轮次关系：roundA（无 RUNID，判据标签有瑕疵）→ roundB（命中上述交错误判）→ **roundC = 官方读数**。
  三轮证据全部保留，不覆盖。

**归因边界**：这属于**观测工具的脆弱**，不是被测代码的行为；P800 的 roundB 同版本判据是 PASS
（说明它是**概率性**交错，不是必然复现）——所以不能拿"P800 过了"反证"MLU 也没问题"，
必须看具体日志行。

## 7. 结论边界

**已证明**：

- 同一份 `DC_MODEL`（缓存根 / 普通目录两种传法）在 P800、MLU590 上**两条腿解析到同一路径**；
- 训练腿自本轮起接受 HF 缓存根（改前 `Unrecognized model`），且 loss/区分度与归档**逐字相同**；
- 离线 23 项口径自检在四家实例上均可跑（无设备依赖）。

**未验证 / 不在本轮范围**：

| 项 | 状态 |
|---|---|
| 910C、PPU 上的两腿复验 | **未验证**：2026-10-27 11:38 / 12:07 / 12:47 三次探测 `10.120.72.27:22`、`192.168.5.112:22(及 2222)` 均 `Connection refused`（PPU 另需先连 VPN）。封装脚本已入库，网络恢复即跑 |
| 离线契约自检（P800 114 / MLU590 103 / 910C 92 / PPU 118 项）、smoke、契约不变式 I1–I4、职责审计 78 项 | **未重跑**：本次只改 `runtime/proto/` 的两条腿与新增脚本，**不触碰** `runtime/__init__.py`、`runtime/api/`、`runtime/backends/`、`scripts/duty_response_audit.py` ⇒ 这些段的输入没变。这是**范围裁定的声明，不是"已重跑证明无回归"**；如需整轮回归请走各实例的 `run_*_rerun_*.sh` |
| 910C 区分度 0.6380 与另两机 0.6391/0.6392 的 0.0011 差 | **未收口**（三机模型副本全量 sha256 对账仍未做）——本文件不涉及该结论 |
